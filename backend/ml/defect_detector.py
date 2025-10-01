import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from .custom_models import DefectDetectorFasterRCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def create_defect_detector(model_path: str = "models/best_model.pth", device: str = "cpu") -> DefectDetectionService:
    """Create and return a defect detection service instance"""
    return DefectDetectionService(model_path, device)
