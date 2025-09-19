import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class DefectDetectorFasterRCNN(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, class_names: list = None):
        super().__init__()
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        
        # Replace the classifier with a new one for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.class_names = ["__background__"] + (class_names if class_names else [])
        
        if len(self.class_names) != num_classes:
            raise ValueError(f"class_names length ({len(self.class_names)}) must match num_classes ({num_classes})")
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

class BeanDefectDataset(Dataset):
    def __init__(self, data_root, split='train', transforms=None):
        self.data_root = data_root
        self.split = split
        self.transforms = transforms
        
        # Load annotations
        ann_file = os.path.join(data_root, split, f'{split}_annotations.json')
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get unique defect classes
        defect_types = set()
        for ann in self.annotations:
            for defect in ann.get('defects', []):
                defect_types.add(defect['type'])
        
        self.defect_classes = sorted(list(defect_types))
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.defect_classes)}
        
        print(f"Found {len(self.defect_classes)} defect classes: {self.defect_classes}")
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.data_root, self.split, 'images', ann['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Get targets
        targets = []
        labels = []
        
        for defect in ann.get('defects', []):
            bbox = defect['bbox']  # [x, y, w, h]
            x, y, w, h = bbox
            
            # Convert to [x1, y1, x2, y2] and clamp to image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image.width, x + w)
            y2 = min(image.height, y + h)
            
            # Skip invalid boxes
            if x2 > x1 and y2 > y1 and (x2 - x1) > 1 and (y2 - y1) > 1:
                targets.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[defect['type']])
        
        # Convert to tensors
        if targets:
            targets = torch.tensor(targets, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            targets = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target_dict = {
            'boxes': targets,
            'labels': labels
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target_dict

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)

def train():
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = BeanDefectDataset('dataset', 'train', train_transforms)
    val_dataset = BeanDefectDataset('dataset', 'val', val_transforms)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Data loaders
    batch_size = 8  # Good for Colab GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    num_classes = len(train_dataset.defect_classes) + 1  # +1 for background
    model = DefectDetectorFasterRCNN(num_classes, pretrained=True, class_names=train_dataset.defect_classes)
    model.to(device)
    
    print(f"Model created with {num_classes} classes")
    print(f"Defect classes: {train_dataset.defect_classes}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += losses.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                else:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Best val loss: {best_val_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved as 'best_model.pth'")

if __name__ == '__main__':
    train()
