import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from .custom_models import DefectDetectorFasterRCNN
from .defect_classifier_mobilenet import CoffeeNetCNN, build_inference_transform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_class_names(label_path: Path, fallback: List[str]) -> List[str]:
    """
    Load class names from a JSON file. Accepts either a list or a dict mapping.
    Falls back to provided list on error.
    """
    try:
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                # Accept {"0": "cls", ...} or {"classes": [...]}
                if "classes" in data and isinstance(data["classes"], list):
                    return [str(x) for x in data["classes"]]
                # Assume mapping of idx->name
                return [v for k, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    except Exception as exc:  # pylint: disable=broad-except
        logging.getLogger(__name__).warning("Failed to load class names: %s", exc)
    return fallback


class DefectDetectionService:
    """Service for detecting coffee bean defects using trained Faster R-CNN model"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.class_names = [
            "__background__",
            "insect_damage",
            "nugget", 
            "quaker",
            "roasted-beans",
            "shell",
            "under_roast"
        ]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self._load_model()
    
    def _load_model(self):
        """Load the trained defect detection model"""
        try:
            logger.info(f"Loading defect detection model from {self.model_path}")
            
            # Create model with correct number of classes (including background)
            # The trained model has 7 classes total (6 defect types + 1 background)
            self.model = DefectDetectorFasterRCNN(
                num_classes=7,  # 6 defect types + 1 background
                pretrained=False,
                class_names=self.class_names[1:]  # Exclude background from class names
            )
            
            # Load trained weights
            if Path(self.model_path).exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("✅ Defect detection model loaded successfully")
            else:
                logger.warning(f"Model file not found at {self.model_path}, using untrained model")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load defect detection model: {e}")
            # Don't raise the error, just log it and continue with untrained model
            logger.warning("Continuing with untrained model due to loading error")
            self.model = None
    
    def detect_defects(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect defects in a coffee bean image
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Check if model is available
            if self.model is None:
                logger.warning("Defect detection model not available, returning empty results")
                return {
                    'success': False,
                    'error': 'Defect detection model not available',
                    'detections': [],
                    'summary': {
                        'total_defects': 0,
                        'defect_types': {},
                        'defect_percentage': 0,
                        'quality_score': 1.0,
                        'quality_grade': 'Unknown'
                    }
                }
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Process predictions
            detections = []
            if predictions and len(predictions) > 0:
                pred = predictions[0]  # Get first (and only) prediction
                
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                
                # Filter by confidence threshold
                valid_indices = scores >= confidence_threshold
                
                for i, (box, score, label) in enumerate(zip(boxes[valid_indices], 
                                                          scores[valid_indices], 
                                                          labels[valid_indices])):
                    detection = {
                        'bbox': box.tolist(),
                        'confidence': float(score),
                        'defect_type': self.class_names[label],
                        'coordinates': {
                            'x1': float(box[0]),
                            'y1': float(box[1]), 
                            'x2': float(box[2]),
                            'y2': float(box[3])
                        },
                        'area': float((box[2] - box[0]) * (box[3] - box[1])),
                        'center': {
                            'x': float((box[0] + box[2]) / 2),
                            'y': float((box[1] + box[3]) / 2)
                        }
                    }
                    detections.append(detection)
            
            # Calculate summary statistics
            total_defects = len(detections)
            defect_types = {}
            for detection in detections:
                defect_type = detection['defect_type']
                if defect_type not in defect_types:
                    defect_types[defect_type] = 0
                defect_types[defect_type] += 1
            
            # Calculate defect percentage (rough estimate)
            image_area = image.size[0] * image.size[1]
            total_defect_area = sum(d['area'] for d in detections)
            defect_percentage = (total_defect_area / image_area) * 100 if image_area > 0 else 0
            
            # Determine overall quality
            quality_score = self._calculate_quality_score(detections, defect_percentage)
            
            return {
                'success': True,
                'detections': detections,
                'summary': {
                    'total_defects': total_defects,
                    'defect_types': defect_types,
                    'defect_percentage': round(defect_percentage, 2),
                    'quality_score': quality_score,
                    'quality_grade': self._get_quality_grade(quality_score)
                },
                'image_info': {
                    'width': image.size[0],
                    'height': image.size[1],
                    'format': image.format
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting defects: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'summary': {
                    'total_defects': 0,
                    'defect_types': {},
                    'defect_percentage': 0,
                    'quality_score': 0,
                    'quality_grade': 'Unknown'
                }
            }
    
    def _calculate_quality_score(self, detections: List[Dict], defect_percentage: float) -> float:
        """Calculate overall quality score based on defects"""
        if not detections:
            return 1.0  # Perfect quality if no defects
        
        # Base score starts at 1.0
        score = 1.0
        
        # Penalty for defect percentage
        score -= min(0.5, defect_percentage / 100)  # Max 50% penalty for area
        
        # Penalty for number of defects
        score -= min(0.3, len(detections) * 0.05)  # 5% penalty per defect, max 30%
        
        # Penalty for specific defect types (severity)
        for detection in detections:
            defect_type = detection['defect_type']
            if defect_type in ['insect_damage']:
                score -= 0.1  # High severity
            elif defect_type in ['quaker', 'under_roast']:
                score -= 0.05  # Medium severity
            else:
                score -= 0.02  # Low severity
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C+'
        elif score >= 0.4:
            return 'C'
        elif score >= 0.3:
            return 'D'
        else:
            return 'F'

class DefectClassificationService:
    """
    Lightweight defect classifier using MobileNetV3 Small (final stable version).
    Uses classification instead of bounding boxes for defect identification.
    """

    def __init__(self, model_path: str, class_names: List[str], device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.class_names = class_names
        self.transform = build_inference_transform()
        self.model = CoffeeNetCNN(num_classes=len(class_names), pretrained=False).to(self.device)
        self._load_model()

    def _load_model(self):
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logging.getLogger(__name__).info("Loaded defect classifier from %s", self.model_path)
        except Exception as exc:  # pylint: disable=broad-except
            logging.getLogger(__name__).error("Failed to load defect classifier: %s", exc)
            self.model = None

    def detect_defects(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        if self.model is None:
            return {
                "success": False,
                "error": "Defect classifier not available",
                "detections": [],
                "summary": {
                    "total_defects": 0,
                    "defect_types": {},
                    "defect_percentage": 0,
                    "quality_score": 1.0,
                    "quality_grade": "Unknown",
                },
            }

        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                confidence, pred_idx = probs.max(dim=1)

            confidence = confidence.item()
            pred_idx = pred_idx.item()
            predicted_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"class_{pred_idx}"

            is_clean = predicted_class.lower() in {
                "healthy",
                "no_defect",
                "background",
                "none",
                "clean",
            }

            detections = []
            if not is_clean and confidence >= confidence_threshold:
                detections.append(
                    {
                        "bbox": None,
                        "confidence": float(confidence),
                        "defect_type": predicted_class,
                        "coordinates": None,
                        "area": 0.0,
                        "center": None,
                    }
                )

            total_defects = len(detections)
            defect_types = {predicted_class: 1} if total_defects > 0 else {}
            defect_percentage = 0 if is_clean else round(confidence * 100, 2)
            quality_score = 1.0 if is_clean else max(0.0, 1.0 - (defect_percentage / 100) * 0.6)

            return {
                "success": True,
                "detections": detections,
                "summary": {
                    "total_defects": total_defects,
                    "defect_types": defect_types,
                    "defect_percentage": defect_percentage,
                    "quality_score": round(quality_score, 3),
                    "quality_grade": self._get_quality_grade(quality_score),
                },
                "image_info": {
                    "width": image.size[0],
                    "height": image.size[1],
                    "format": image.format,
                },
            }
        except Exception as exc:  # pylint: disable=broad-except
            logging.getLogger(__name__).error("Error running defect classifier: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "detections": [],
                "summary": {
                    "total_defects": 0,
                    "defect_types": {},
                    "defect_percentage": 0,
                    "quality_score": 0,
                    "quality_grade": "Unknown",
                },
            }

    def _get_quality_grade(self, score: float) -> str:
        if score >= 0.9:
            return "A+"
        if score >= 0.8:
            return "A"
        if score >= 0.7:
            return "B+"
        if score >= 0.6:
            return "B"
        if score >= 0.5:
            return "C+"
        if score >= 0.4:
            return "C"
        if score >= 0.3:
            return "D"
        return "F"


def create_defect_detector(model_path: str = "models/best_model.pth", device: str = "cpu"):
    """
    Create a defect detector. Prefer the MobileNetV3 classifier (defect_mobilenet_best.pth)
    when available; otherwise fall back to Faster R-CNN detector.
    """
    # Prefer explicit env override
    classifier_override = os.getenv("DEFECT_CLASSIFIER_PATH")
    candidates = []

    # Root directory for model assets (default: backend/models relative to this file)
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"

    if classifier_override:
        candidates.append(Path(classifier_override))

    # Prefer colocated MobileNet weight beside provided model_path
    candidates.append(Path(model_path).with_name("defect_mobilenet_best.pth"))
    # Also look in backend/models explicitly
    candidates.append(models_dir / "defect_mobilenet_best.pth")

    for candidate in candidates:
        if candidate.exists():
            labels_path = candidate.with_suffix(".json")
            default_classes = [
                "insect_damage",
                "nugget",
                "quaker",
                "roasted-beans",
                "shell",
                "under_roast",
            ]
            class_names = _load_class_names(labels_path, fallback=default_classes)
            return DefectClassificationService(str(candidate), class_names, device=device)

    return DefectDetectionService(model_path, device)
