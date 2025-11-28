import torch
import torch.nn as nn
from torchvision import models, transforms


class CoffeeNetCNN(nn.Module):
  """
  MobileNetV3 Small classifier head tailored for coffee defect classification.
  Mirrors the architecture used in the provided training script.
  """

  def __init__(self, num_classes: int, pretrained: bool = True):
    super().__init__()

    base = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )

    self.backbone = nn.Sequential(
        base.features,
        base.avgpool,
    )

    in_features = 576

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )

  def forward(self, x):
    x = self.backbone(x)
    return self.classifier(x)


def build_inference_transform():
  """Transform stack matching the training-time preprocessing."""
  return transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
