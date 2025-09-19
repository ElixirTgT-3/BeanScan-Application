import os
import json
import math
import warnings
import random
import logging
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .custom_models import DefectDetectorFasterRCNN

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

# Optional TorchMetrics for mAP
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore
except Exception:  # pragma: no cover
    MeanAveragePrecision = None  # type: ignore


def setup_logger(log_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("defect_trainer")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# Defect classes provided by the user
DEFECT_CLASS_NAMES = [
    "insect_damage",
    "nugget",
    "quaker",
    "roasted-beans",
    "shell",
    "under_roast",
]


class BeanDefectDataset(Dataset):
    """Dataset that reads BeanScan annotations and yields images with Faster R-CNN targets."""

    def __init__(self, data_root: str, split: str = 'train', class_names: Optional[List[str]] = None):
        self.data_root = data_root
        self.split = split
        self.images_dir = os.path.join(data_root, split, 'images')
        self.annotations_path = os.path.join(data_root, split, f'{split}_annotations.json')
        self.class_names = class_names or []

        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Note: Avoid geometric transforms unless you also transform boxes.
        # Use only photometric jitter that doesn't invalidate boxes.
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        image_path = os.path.join(self.images_dir, ann['image_id'])
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        img_w, img_h = image.size

        boxes = []
        labels = []
        for defect in ann.get('defects', []):
            bbox = defect.get('bbox', None)
            if bbox is None or len(bbox) != 4:
                warnings.warn(f"Skipping annotation with missing/invalid bbox for image {ann.get('image_id')}")
                continue
            # Validate numeric and non-NaN
            try:
                bx = [float(b) for b in bbox]
            except Exception:
                warnings.warn(f"Non-numeric bbox values {bbox} in image {ann.get('image_id')}, skipping")
                continue
            if any(math.isnan(v) or math.isinf(v) for v in bx):
                warnings.warn(f"NaN/Inf in bbox {bbox} for image {ann.get('image_id')}, skipping")
                continue
            # COCO [x,y,w,h] -> [x1,y1,x2,y2], clamp to image, reject tiny/invalid
            x, y, w, h = bx
            if w <= 1.0 or h <= 1.0:
                warnings.warn(f"Zero/negative/tiny bbox {bbox} in {ann.get('image_id')}, skipping")
                continue
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(x + w, float(img_w))
            y2 = min(y + h, float(img_h))
            if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                warnings.warn(f"Degenerate bbox after clamp {[x1,y1,x2,y2]} in {ann.get('image_id')}, skipping")
                continue
            boxes.append([x1, y1, x2, y2])
            # Map class label from annotation
            defect_type = str(defect.get('type', '')).strip()
            try:
                label_idx = self.class_names.index(defect_type) + 1  # +1 for background
            except ValueError:
                # Skip unknown classes
                warnings.warn(f"Unknown defect class '{defect_type}' in {ann.get('image_id')}, skipping")
                label_idx = None
            if label_idx is not None:
                labels.append(label_idx)

        if len(boxes) == 0 or len(labels) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }

        return image_tensor, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def _discover_classes(data_root: str, split: str = 'train') -> List[str]:
    ann_path = os.path.join(data_root, split, f'{split}_annotations.json')
    with open(ann_path, 'r') as f:
        anns = json.load(f)
    classes = []
    for a in anns:
        for d in a.get('defects', []):
            t = str(d.get('type', '')).strip()
            if t and t not in classes:
                classes.append(t)
    return classes


def train(
    data_root: str = './data',
    device_str: str = 'cpu',
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = '../models',
    use_amp: bool = False,
    weight_decay: float = 1e-4,
    scheduler: Optional[str] = None,  # 'step' or 'plateau'
    scheduler_kwargs: Optional[Dict] = None,
    log_dir: Optional[str] = 'runs/defect_detector',
    use_tensorboard: bool = False,
    seed: Optional[int] = 42,
    resume_from: Optional[str] = None,
    config_path: Optional[str] = None,
):
    device = torch.device(device_str)

    # Reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        try:
            import numpy as np  # type: ignore
            np.random.seed(seed)
        except Exception:
            pass
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    # Dynamically discover defect class names from dataset
    class_names = _discover_classes(data_root, 'train')
    if len(class_names) == 0:
        raise RuntimeError('No defect classes found in annotations. Check dataset paths.')
    print('Using defect classes:', class_names)

    train_ds = BeanDefectDataset(data_root, 'train', class_names=class_names)
    val_ds = None
    if os.path.exists(os.path.join(data_root, 'val', f'val_annotations.json')):
        val_ds = BeanDefectDataset(data_root, 'val', class_names=class_names)

    pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin, collate_fn=collate_fn
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin, collate_fn=collate_fn)

    model = DefectDetectorFasterRCNN(
        num_classes=len(class_names) + 1,  # include background
        pretrained=True,
        class_names=class_names
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # LR scheduler
    sched = None
    if scheduler == 'step':
        kw = scheduler_kwargs or {'step_size': 3, 'gamma': 0.5}
        sched = optim.lr_scheduler.StepLR(optimizer, **kw)
    elif scheduler == 'plateau':
        kw = scheduler_kwargs or {'mode': 'min', 'factor': 0.5, 'patience': 2}
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kw)
    scaler = GradScaler(enabled=(use_amp and device.type == 'cuda'))

    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    writer = None
    if use_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir)
    logger = setup_logger(os.path.join(output_dir, 'train.log'))

    start_epoch = 0
    # Resume support (model-only checkpoints or full state)
    if resume_from and os.path.exists(resume_from):
        try:
            ckpt = torch.load(resume_from, map_location=device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'], strict=False)
                if 'optimizer_state' in ckpt and ckpt['optimizer_state'] and optimizer is not None:
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                if 'scheduler_state' in ckpt and sched is not None and ckpt['scheduler_state']:
                    sched.load_state_dict(ckpt['scheduler_state'])
                if 'scaler_state' in ckpt and isinstance(scaler, GradScaler):
                    try:
                        scaler.load_state_dict(ckpt['scaler_state'])
                    except Exception:
                        pass
                start_epoch = int(ckpt.get('epoch', 0))
                best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
                logger.info(f"Resumed from {resume_from} at epoch {start_epoch}")
            else:
                # Backward compatibility: model-only weights
                model.load_state_dict(ckpt, strict=False)
                logger.info(f"Loaded model weights from {resume_from}")
        except Exception as e:
            logger.warning(f"Failed to resume from {resume_from}: {e}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(use_amp and device.type == 'cuda')):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            if not torch.isfinite(loss):
                logger.warning("Non-finite train loss; skipping batch. Losses=%s", {k: float(v.detach().cpu()) for k, v in loss_dict.items()})
                continue
            scaler.scale(loss).backward()
            # Gradient clipping for stability
            if use_amp and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        avg_val_loss = None
        val_map = None
        if val_loader is not None:
            # Torchvision detection models only return losses in train() mode.
            # Use no_grad() to avoid gradients while switching to train() to get loss dict.
            val_loss_accum = 0.0
            model_state_was_train = model.training
            with torch.no_grad():
                model.train()
                for images, targets in tqdm(val_loader, desc='Validate'):
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    if not torch.isfinite(loss):
                        logger.warning("Non-finite val loss; skipping batch. Losses=%s", {k: float(v.detach().cpu()) for k, v in loss_dict.items()})
                        continue
                    val_loss_accum += loss.item()
            if not model_state_was_train:
                model.eval()
            avg_val_loss = val_loss_accum / max(1, len(val_loader))

            # Optional mAP evaluation (separate inference pass)
            if MeanAveragePrecision is not None:
                metric = MeanAveragePrecision()
                model.eval()
                with torch.no_grad():
                    for images, targets in tqdm(val_loader, desc='Eval mAP'):
                        images = [img.to(device) for img in images]
                        outputs = model(images)  # list of dicts
                        # Move tensors to CPU for metric
                        preds = []
                        for out in outputs:
                            preds.append({
                                'boxes': out['boxes'].detach().cpu(),
                                'scores': out['scores'].detach().cpu(),
                                'labels': out['labels'].detach().cpu(),
                            })
                        gts = []
                        for t in targets:
                            gts.append({
                                'boxes': t['boxes'].detach().cpu(),
                                'labels': t['labels'].detach().cpu(),
                            })
                        metric.update(preds, gts)
                mp = metric.compute()
                val_map = float(mp.get('map', 0.0))
                val_map50 = float(mp.get('map_50', 0.0))
                val_map75 = float(mp.get('map_75', 0.0))

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f'defect_frcnn_epoch_{epoch+1}.pth')
        # Save full training state for resuming
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': sched.state_dict() if sched is not None else None,
            'scaler_state': scaler.state_dict() if isinstance(scaler, GradScaler) else None,
            'best_val_loss': best_val_loss,
            'class_names': ['__background__'] + class_names,
        }, ckpt_path)

        # Save best
        if avg_val_loss is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(output_dir, 'defect_frcnn_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': sched.state_dict() if sched is not None else None,
                'scaler_state': scaler.state_dict() if isinstance(scaler, GradScaler) else None,
                'best_val_loss': best_val_loss,
                'class_names': ['__background__'] + class_names,
            }, best_path)

        msg = f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}"
        if avg_val_loss is not None:
            msg += f", val_loss={avg_val_loss:.4f}"
        if val_map is not None:
            msg += f", val_mAP={val_map:.4f}"
        print(msg)
        logger.info(msg)

        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch+1)
            if avg_val_loss is not None:
                writer.add_scalar('Loss/val', avg_val_loss, epoch+1)
            if val_map is not None:
                writer.add_scalar('mAP/val', val_map, epoch+1)
                if 'val_map50' in locals():
                    writer.add_scalar('mAP/val_50', val_map50, epoch+1)
                if 'val_map75' in locals():
                    writer.add_scalar('mAP/val_75', val_map75, epoch+1)

        # Step schedulers
        if sched is not None:
            if scheduler == 'plateau' and avg_val_loss is not None:
                sched.step(avg_val_loss)
            else:
                sched.step()


if __name__ == '__main__':
    # Use project-root execution; point to backend data and models
    train(
        data_root='backend/data',
        device_str='cuda' if torch.cuda.is_available() else 'cpu',
        epochs=10,
        batch_size=4,
        lr=1e-4,
        output_dir='backend/models',
        use_amp=False,
        scheduler='plateau',
        use_tensorboard=True,
        seed=42,
    )


