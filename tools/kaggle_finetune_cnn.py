"""
Fine-tune BeanClassifierCNN on Excelsa & Liberica while preserving Arabica/Robusta weights.

This script is tailored for Kaggle environments where:
* The original 2-class checkpoint (Arabica/Robusta) sits at OLD_MODEL_PATH.
* The COCO-style dataset with Excelsa and Liberica annotations lives under DATA_DIR.

We:
1. Load the base MobileNetV3 classifier architecture (4 output classes).
2. Transfer all compatible weights from the 2-class checkpoint.
3. Manually copy the old logits for Arabica/Robusta into the new 4-way head.
4. Freeze every parameter except the final linear head and only update the rows
   corresponding to the new classes (Excelsa, Liberica) by zeroing their gradients.
5. Train on the COCO splits, evaluate, and export metrics + confusion matrix.
"""

import os
import sys
import json
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================================================
# Paths (update if your Kaggle input directories differ)
# ============================================================
DATA_DIR = "/kaggle/input/coffeebean"
OLD_MODEL_PATH = "/kaggle/input/cnn1/pytorch/cnn/1/cnn_best.pth"
CUSTOM_MODEL_DIR = "/kaggle/input/custom_model/pytorch/default/1"

sys.path.append(CUSTOM_MODEL_DIR)
from custom_models import BeanClassifierCNN  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/kaggle/working"
SAVE_MODEL = os.path.join(SAVE_DIR, "fine_tuned_cnn.pth")
SAVE_REPORT = os.path.join(SAVE_DIR, "classification_report.csv")
SAVE_CM = os.path.join(SAVE_DIR, "confusion_matrix.png")

# Class ordering must match the old checkpoint (first two entries) so the transferred
# logits stay aligned with their original semantics.
BASE_CLASSES = ["Arabica", "Robusta"]
NEW_CLASSES = ["Excelsa", "Liberica"]
ALL_CLASSES = BASE_CLASSES + NEW_CLASSES
LABEL_MAP = {name.lower(): idx for idx, name in enumerate(ALL_CLASSES)}


# ============================================================
# Dataset
# ============================================================
class CocoBeanDataset(Dataset):
    """COCO-formatted dataset that filters to specified bean classes."""

    def __init__(self, folder: str, transform=None, focus_classes: List[str] = None):
        self.folder = folder
        self.transform = transform
        self.focus_classes = [c.lower() for c in (focus_classes or NEW_CLASSES)]

        ann_path = os.path.join(folder, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = {img["id"]: img["file_name"] for img in data["images"]}
        categories = {cat["id"]: cat["name"].lower() for cat in data["categories"]}

        samples: List[Tuple[str, int]] = []
        for ann in data["annotations"]:
            cat_name = categories[ann["category_id"]]
            if cat_name not in self.focus_classes:
                continue
            image_name = images[ann["image_id"]]
            if cat_name not in LABEL_MAP:
                continue
            samples.append((image_name, LABEL_MAP[cat_name]))

        # Drop duplicate (image, label) tuples while preserving order
        seen = {}
        for img_name, label_idx in samples:
            seen.setdefault((img_name, label_idx), None)
        self.samples = list(seen.keys())

        print(f"[INFO] {len(self.samples)} samples loaded from {folder} (focus={self.focus_classes})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label_idx = self.samples[idx]
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


# ============================================================
# Transforms
# ============================================================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ============================================================
# Data loaders
# ============================================================
train_ds = CocoBeanDataset(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = CocoBeanDataset(os.path.join(DATA_DIR, "valid"), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)


# ============================================================
# Load base model & transfer old weights
# ============================================================
model = BeanClassifierCNN(num_classes=len(ALL_CLASSES), pretrained=False)
model_state = model.state_dict()

if not os.path.exists(OLD_MODEL_PATH):
    raise FileNotFoundError(f"Pretrained checkpoint not found at {OLD_MODEL_PATH}")

old_state = torch.load(OLD_MODEL_PATH, map_location="cpu")

# Load any tensors whose shapes still align (all except final classifier head).
compatible = {k: v for k, v in old_state.items()
              if k in model_state and model_state[k].shape == v.shape}
model_state.update(compatible)
model.load_state_dict(model_state)
print(f"[OK] Restored {len(compatible)}/{len(model_state)} tensors from base checkpoint.")

# Manually copy logits for the preserved classes into the expanded head.
if "classifier.6.weight" in old_state and "classifier.6.bias" in old_state:
    old_w = old_state["classifier.6.weight"]
    old_b = old_state["classifier.6.bias"]
    preserved_rows = old_w.shape[0]
    model.classifier[6].weight.data[:preserved_rows] = old_w
    model.classifier[6].bias.data[:preserved_rows] = old_b
    print(f"[OK] Copied logits for {BASE_CLASSES} into 4-class head.")
else:
    print("[WARN] Base checkpoint missing classifier head; Arabica/Robusta weights not copied.")
    preserved_rows = 0

model.to(DEVICE)

# Freeze everything except the expanded classifier head.
for param in model.parameters():
    param.requires_grad = False

model.classifier[6].weight.requires_grad = True
model.classifier[6].bias.requires_grad = True

# Optimizer only updates the final layer.
optimizer = optim.Adam(model.classifier[6].parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 20
PATIENCE = 5
best_val_loss = float("inf")
no_improve = 0


def zero_base_class_grads(linear_layer: nn.Linear, locked_rows: int) -> None:
    """Prevent updates to logits corresponding to the preserved base classes."""
    if locked_rows <= 0:
        return
    if linear_layer.weight.grad is not None:
        linear_layer.weight.grad[:locked_rows] = 0
    if linear_layer.bias.grad is not None:
        linear_layer.bias.grad[:locked_rows] = 0


# ============================================================
# Training loop
# ============================================================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        zero_base_class_grads(model.classifier[6], len(BASE_CLASSES))

        optimizer.step()
        running_loss += loss.item()

    avg_train = running_loss / max(1, len(train_loader))

    # Validation -------------------------------------------------------------
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_val = val_loss / max(1, len(val_loader))
    acc = accuracy_score(y_true, y_pred)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train {avg_train:.4f} | Val {avg_val:.4f} | Acc {acc:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        no_improve = 0
        torch.save(model.state_dict(), SAVE_MODEL)
        print(f"[OK] Saved best model at epoch {epoch + 1}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("[STOP] Early stopping triggered.")
            break


# ============================================================
# Evaluation
# ============================================================
print("\n[INFO] Evaluating best checkpoint...")
model.load_state_dict(torch.load(SAVE_MODEL, map_location=DEVICE))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"[RESULT] Validation accuracy: {acc:.4f}")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(ALL_CLASSES))),
    output_dict=True,
    target_names=ALL_CLASSES,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(SAVE_REPORT)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ALL_CLASSES))))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ALL_CLASSES, yticklabels=ALL_CLASSES)
plt.title(f"Confusion Matrix (Acc={acc:.2f})")
plt.tight_layout()
plt.savefig(SAVE_CM)
plt.close()

print("\n[INFO] Artifacts written to /kaggle/working:")
print(" - fine_tuned_cnn.pth")
print(" - classification_report.csv")
print(" - confusion_matrix.png")
