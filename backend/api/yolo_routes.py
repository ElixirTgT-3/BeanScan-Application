"""FastAPI routes for YOLOv8 segmentation inference (backend-hosted)."""

from pathlib import Path
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ml.yolo_service import YOLOSegmentationService

router = APIRouter()

# Resolve model path relative to backend root by default
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"
DEFAULT_DEFECT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "defect_best.pt"
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", DEFAULT_MODEL_PATH))
DEFECT_MODEL_PATH = Path(os.getenv("YOLO_DEFECT_MODEL_PATH", DEFAULT_DEFECT_MODEL_PATH))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")

try:
    yolo_service = YOLOSegmentationService(model_path=MODEL_PATH, device=YOLO_DEVICE)
    print(f"[OK] Loaded YOLO model from {MODEL_PATH} on device={yolo_service.device}")
except FileNotFoundError as exc:
    print(f"[ERROR] {exc}")
    yolo_service = None
except Exception as exc:  # pylint: disable=broad-except
    print(f"[ERROR] Failed to load YOLO model: {exc}")
    yolo_service = None

try:
    yolo_defect_service = YOLOSegmentationService(model_path=DEFECT_MODEL_PATH, device=YOLO_DEVICE)
    print(f"[OK] Loaded YOLO defect model from {DEFECT_MODEL_PATH} on device={yolo_defect_service.device}")
except FileNotFoundError:
    print(f"[WARN] Defect model not found at {DEFECT_MODEL_PATH}; skipping defect inference")
    yolo_defect_service = None
except Exception as exc:  # pylint: disable=broad-except
    print(f"[ERROR] Failed to load defect YOLO model: {exc}")
    yolo_defect_service = None


def _is_likely_image(upload: UploadFile) -> bool:
    """Be permissive about image detection (content_type OR extension)."""
    if upload.content_type and upload.content_type.startswith("image/"):
        return True
    filename = upload.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in {"jpg", "jpeg", "png", "bmp", "gif", "webp", "tif", "tiff"}


@router.get("/yolo/info")
async def yolo_info():
    """Return model metadata and available classes."""
    if yolo_service is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded on server")

    return {
        "model": {
            "path": str(MODEL_PATH),
            "device": yolo_service.device,
        },
        "classes": yolo_service._class_map(),  # pylint: disable=protected-access
    }


@router.post("/yolo/predict")
async def yolo_predict(
    image: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
):
    """Run YOLOv8 segmentation on an uploaded image."""
    if yolo_service is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded on server")

    if not _is_likely_image(image):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image; got content_type={image.content_type} filename={image.filename}",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        result = yolo_service.predict(image_bytes=image_bytes, conf=conf, iou=iou)
        defect_result = None
        if yolo_defect_service is not None:
            defect_result = yolo_defect_service.predict(image_bytes=image_bytes, conf=conf, iou=iou)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {exc}") from exc

    return JSONResponse(
        content={
            "success": True,
            "detections": result.get("detections", []),
            "classes": result.get("classes", {}),
            "image_size": result.get("image_size", {}),
            "model": result.get("model", {}),
            "defect": defect_result,  # optional defect model output
        }
    )
