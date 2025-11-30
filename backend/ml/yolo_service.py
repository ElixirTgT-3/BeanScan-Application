"""Lightweight YOLOv8 segmentation inference helper for FastAPI."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image
from ultralytics import YOLO


def _to_python_number(val: Any) -> float:
    if hasattr(val, "item"):
        return float(val.item())
    if isinstance(val, (int, float)):
        return float(val)
    return float(val)


class YOLOSegmentationService:
    """Wrapper that loads a YOLOv8 segmentation model once and runs predictions."""

    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(self.model_path)
        try:
            self.model.to(self.device)
        except Exception:
            # Some backends do not expose .to(); rely on internal handling
            pass

        # Fuse model for slightly faster inference when supported
        try:
            self.model.fuse()
        except Exception:
            pass

    def _class_map(self, names: Any = None) -> Dict[int, str]:
        names = names or getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(name) for i, name in enumerate(names)}
        return {}

    def _convert_detections(self, result) -> List[Dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        names = self._class_map(getattr(result, "names", None))

        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        for idx in range(len(boxes)):
            box = boxes[idx]
            xyxy = box.xyxy[0].tolist()
            xywh = box.xywh[0].tolist()
            cls_id = int(_to_python_number(box.cls[0]))
            det: Dict[str, Any] = {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": _to_python_number(box.conf[0]),
                "bbox": {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "width": float(xywh[2]),
                    "height": float(xywh[3]),
                },
                "center": {"x": float(xywh[0]), "y": float(xywh[1])},
                "area": float(xywh[2] * xywh[3]),
            }

            if masks is not None and getattr(masks, "xy", None) is not None:
                if idx < len(masks.xy):
                    polygon = masks.xy[idx]
                    det["mask"] = {
                        "format": "polygons",
                        "points": [[float(x), float(y)] for x, y in polygon.tolist()],
                    }
            detections.append(det)

        return detections

    def predict(
        self,
        image_bytes: bytes,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        if not image_bytes:
            raise ValueError("No image data provided")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        if not results:
            return {
                "detections": [],
                "classes": self._class_map(),
                "image_size": {"width": width, "height": height},
                "model": {
                    "path": str(self.model_path),
                    "device": self.device,
                    "task": getattr(self.model, "task", "detect"),
                },
            }

        result = results[0]
        detections = self._convert_detections(result)

        return {
            "detections": detections,
            "classes": self._class_map(getattr(result, "names", None)),
            "image_size": {"width": width, "height": height},
            "model": {
                "path": str(self.model_path),
                "device": self.device,
                "task": getattr(self.model, "task", "detect"),
                "has_masks": bool(getattr(result, "masks", None)),
            },
        }
