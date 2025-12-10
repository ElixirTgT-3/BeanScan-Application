import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np

from .defect_classifier_mobilenet import CoffeeNetCNN

# Defect labels to ignore (excluded from scoring, severity, and returned detections)
# Keep this limited to "clean" tokens so defects like quaker/nugget/shell still count.
IGNORED_DEFECT_TYPES = {
    'good_bean',
    'goodbean',
    'good',
    'healthy',
    'clean',
    'background',
    'no_defect',
}

class MobileNetV3Backbone(nn.Module):
    """Custom MobileNetV3 backbone for feature extraction - matches trained model architecture"""
    
    def __init__(self, pretrained: bool = True, width_mult: float = 1.0):
        super().__init__()
        # Load pretrained MobileNetV3
        if pretrained:
            self.backbone = mobilenet_v3_small(pretrained=True)
        else:
            self.backbone = mobilenet_v3_small(pretrained=False)
        
        # Extract features from different layers
        self.features = self.backbone.features
        
        # Feature dimensions for different scales
        self.feature_channels = [16, 24, 40, 48, 96, 576]
        
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [2, 4, 6, 8, 10, 12]:  # Key feature layers
                features.append(x)
        return features

class BeanClassifierCNN(nn.Module):
    """CNN for bean type classification using MobileNetV3 backbone"""
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)

        default_class_names = ["Liberica", "Arabica", "Robusta", "Excelsa"]  # CoffeeNet order
        if class_names is not None:
            if len(class_names) == 0:
                raise ValueError("class_names must contain at least one entry")
            self.class_names = class_names
            num_classes = len(class_names)
        else:
            if num_classes <= len(default_class_names):
                self.class_names = default_class_names[:num_classes]
            else:
                extra = [f"Class_{i}" for i in range(len(default_class_names), num_classes)]
                self.class_names = default_class_names + extra
        
        # Classification head (increased dropout ~0.3 to mitigate overfitting)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        # Use the last feature map for classification
        x = features[-1]  # Last feature layer
        x = self.classifier(x)
        return x
    
    def predict(self, x, threshold: float = 0.45):
        """Predict bean type with confidence using light test-time augmentation"""
        self.eval()
        with torch.no_grad():
            if x.dim() != 4:
                raise ValueError(f"Expected input tensor of shape (N, C, H, W), got {x.shape}")

            augmentations = [
                lambda t: t,
                lambda t: torch.flip(t, dims=[-1]),  # horizontal flip
                lambda t: torch.flip(t, dims=[-2]),  # vertical flip
                lambda t: torch.rot90(t, k=1, dims=(-2, -1)),  # 90° rotation
                lambda t: torch.rot90(t, k=3, dims=(-2, -1)),  # -90° rotation
            ]

            augmented_batches = [aug(x) for aug in augmentations]
            stacked_augmented = torch.cat(augmented_batches, dim=0)

            logits = self.forward(stacked_augmented)
            probabilities = F.softmax(logits, dim=1)

            # Reshape to (num_augs, batch, num_classes) then average
            num_augs = len(augmentations)
            num_samples = x.shape[0]
            probabilities = probabilities.view(num_augs, num_samples, -1).mean(dim=0)

            confidence, predicted = torch.max(probabilities, 1)

            predictions = []

            for i in range(len(predicted)):
                top_class_index = predicted[i].item()
                top_probability = confidence[i].item()
                class_name = self.class_names[top_class_index]
                predictions.append({
                    'class': class_name,
                    'confidence': top_probability,
                    'probabilities': probabilities[i].tolist(),
                    'is_low_confidence': bool(top_probability < threshold),
                })

            return predictions

class DefectDetectorMaskRCNN(nn.Module):
    """Faster R-CNN for defect detection using MobileNetV3 - matches trained model architecture"""
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        
        # Use MobileNetV3 backbone with FPN (matches your trained model)
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        
        # Customize box predictor for defect classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 for background
        
        # Defect types (matching your training data)
        self.defect_types = ["insect_damage", "nugget", "quaker", "roasted-beans", "shell", "under_roast"]
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def detect_defects(self, image, confidence_threshold: float = 0.5):
        """Detect defects in bean image"""
        self.eval()
        with torch.no_grad():
            # Prepare image - Faster R-CNN expects a list of images
            if len(image.shape) == 4:  # Already batched
                image_list = [image.squeeze(0)]  # Convert to list
            elif len(image.shape) == 3:  # Single image
                image_list = [image]  # Convert to list
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # Get predictions
            predictions = self.forward(image_list)
            image_height = image_list[0].shape[-2] if hasattr(image_list[0], 'shape') else None
            image_width = image_list[0].shape[-1] if hasattr(image_list[0], 'shape') else None
            
            # Process results
            defects = []
            for pred in predictions:
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']
                
                # Faster R-CNN doesn't have masks, so we'll calculate area from bounding box
                for i in range(len(scores)):
                    if scores[i] >= confidence_threshold:
                        # Calculate area from bounding box (width * height)
                        x1, y1, x2, y2 = boxes[i]
                        area = (x2 - x1) * (y2 - y1)
                        
                        defect = {
                            'bbox': boxes[i].tolist(),
                            'confidence': scores[i].item(),
                            'defect_type': self.defect_types[labels[i].item() - 1],  # -1 for background
                            'area': area.item(),
                            'image_height': float(image_height) if image_height is not None else None,
                            'image_width': float(image_width) if image_width is not None else None,
                            'image_size': {
                                'width': float(image_width) if image_width is not None else None,
                                'height': float(image_height) if image_height is not None else None,
                            },
                            'coordinates': {
                                'x1': boxes[i][0].item(),
                                'y1': boxes[i][1].item(),
                                'x2': boxes[i][2].item(),
                                'y2': boxes[i][3].item()
                            }
                        }
                        defects.append(defect)
            
        return defects


class DefectClassifierAdapter(nn.Module):
    """
    Adapter to use MobileNetV3 Small defect classifier with the same interface
    as the detector's detect_defects method.
    """

    def __init__(self, classifier: CoffeeNetCNN, class_names: List[str], device: torch.device):
        super().__init__()
        self.classifier = classifier
        self.class_names = class_names
        self.device = device

    def detect_defects(self, image, confidence_threshold: float = 0.5):
        self.classifier.eval()
        try:
            if isinstance(image, torch.Tensor):
                tensor = image.to(self.device)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
            else:
                raise ValueError("Expected image tensor for defect classification")

            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = F.softmax(logits, dim=1)
                conf, idx = probs.max(dim=1)

            confidence = conf[0].item()
            pred_idx = idx[0].item()
            predicted_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"class_{pred_idx}"

            # Classes considered "clean"
            clean_tokens = {"healthy", "no_defect", "clean", "none", "background"}
            is_clean = predicted_class.lower() in clean_tokens

            detections = []
            if not is_clean and confidence >= confidence_threshold:
                detections.append(
                    {
                        "bbox": None,
                        "confidence": float(confidence),
                        "defect_type": predicted_class,
                        "area": 0.0,
                        "coordinates": None,
                        "center": None,
                        "image_width": None,
                        "image_height": None,
                        "image_size": {"width": None, "height": None},
                    }
                )

            return detections
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] DefectClassifierAdapter failed: {exc}")
            return []

class DefectDetectorFasterRCNN(nn.Module):
    """Faster R-CNN detector (bounding boxes only) for bean defects"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 class_names: Optional[List[str]] = None):
        super().__init__()
        # num_classes should include background (>=2)
        self.num_classes = max(2, num_classes)
        self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        default_classes = [
            "insect_damage",
            "nugget",
            "quaker",
            "roasted-beans",
            "shell",
            "under_roast"
        ]
        self.class_names = ["__background__"] + (class_names or default_classes)
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def detect(self, image, confidence_threshold: float = 0.5):
        self.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            outputs = self.forward(image)
            detections = []
            for pred in outputs:
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']
                for i in range(len(scores)):
                    if scores[i] >= confidence_threshold:
                        detections.append({
                            'bbox': boxes[i].tolist(),
                            'score': scores[i].item(),
                            'label': self.class_names[labels[i].item()]
                        })
            return detections

class RuleBasedShelfLife:
    """Rule-based shelf life prediction matching RULE_BASED_SEVERITY.md banded logic."""

    def __init__(self):
        # Base shelf life (days) per bean type; matches the “Perfect green beans” table rows.
        self.base_shelf_life_days = {
            'arabica': 1095,    # 36.5 months (specialty)
            'liberica': 1095,
            'excelsa': 900,     # 30 months
            'robusta': 720,     # 24 months (commodity)
            'other': 1095,      # align unknown to the clean specialty baseline
        }

        # Defect severity weights (table-aligned; higher = worse)
        self.defect_weights: Dict[str, float] = {
            'insect_damage': 8.0,
            'insect': 8.0,
            'borer': 8.0,
            'quaker': 7.0,
            'nugget': 5.0,
            'discoloration': 6.0,
            'physical_damage': 4.0,
            'broken': 4.0,
            'broken_cut': 4.0,
            'cut': 4.0,
            'chip': 4.0,
            'crack': 4.0,
            'shell': 3.0,
            'under_roast': 2.0,
            'roasted_beans': 7.0,
            'roasted-beans': 7.0,
            'unknown': 1.0,
        }

        # Scenario profiles (directly from the table)
        self.scenario_profiles = {
            'clean_specialty': {
                'months_range': (24.0, 36.0),
                'severity': 'normal',
                'base_confidence': 0.95,
            },
            'clean_commodity': {
                'months_range': (12.0, 24.0),
                'severity': 'normal',
                'base_confidence': 0.92,
            },
            'general_mixed': {
                'months_range': (12.0, 24.0),
                'severity': 'moderate',
                'base_confidence': 0.82,
                'reduction': (0.10, 0.60),
            },
            'fully_black': {
                'months_range': (10.0, 14.0),
                'severity': 'severe',
                'base_confidence': 0.70,
                'reduction': (0.40, 0.60),
            },
            'insect_damage': {
                'months_range': (12.0, 16.0),
                'severity': 'moderate',
                'base_confidence': 0.75,
                'reduction': (0.30, 0.50),
            },
            'broken_cut': {
                'months_range': (18.0, 27.0),
                'severity': 'mild',
                'base_confidence': 0.82,
                'reduction': (0.10, 0.25),
            },
            'roasted_defect': {
                'months_range': (7.0, 12.0),
                'severity': 'severe',
                'base_confidence': 0.68,
                'reduction': (0.50, 0.70),
            },
        }

        self.clean_tokens = set(IGNORED_DEFECT_TYPES)

    def predict_shelf_life(self, defect_sequence, bean_type: str = 'Arabica', confidence_threshold: float = 0.7):
        """
        Predict shelf life using the table-driven scenarios:
        clean / general mixed / fully black / insect damage / broken-cut / roasted-bean defect.
        """
        # Normalize inputs
        if isinstance(defect_sequence, list):
            defects = defect_sequence
        elif hasattr(defect_sequence, 'tolist'):
            defects = defect_sequence.tolist()
        else:
            defects = []

        bean_key = (bean_type or 'other').strip().lower()
        base_days = self.base_shelf_life_days.get(bean_key, self.base_shelf_life_days['other'])
        base_months = base_days / 30.0

        total_defect_score = 0.0
        defect_counts: Dict[str, int] = {}
        total_detected = 0
        cumulative_confidence = 0.0

        for defect in defects:
            if isinstance(defect, dict):
                defect_type = defect.get('type', 'unknown')
                confidence = defect.get('confidence', 0.5)
                count = defect.get('count', 1)
            else:
                defect_type = str(defect)
                confidence = 1.0
                count = 1

            norm_type = self._normalize_defect_key(defect_type)
            if norm_type in self.clean_tokens:
                continue

            weight = self.defect_weights.get(norm_type, self.defect_weights['unknown'])
            impact = weight * float(confidence) * float(count)
            total_defect_score += impact
            defect_counts[norm_type] = defect_counts.get(norm_type, 0) + int(count)
            total_detected += int(count)
            cumulative_confidence += float(confidence) * float(count)

        avg_detection_confidence = (cumulative_confidence / total_detected) if total_detected > 0 else 0.0
        normalized_score = min(total_defect_score / 45.0, 1.5)
        defect_percentage = self._clamp(normalized_score * 100.0, 0.0, 100.0)

        # Clean / normal branch
        if total_detected == 0 or defect_percentage <= 0:
            estimated_months = round(base_months, 1)
            return {
                'predicted_days': int(base_days),
                'estimated_months': estimated_months,
                'estimated_months_range': {'min': estimated_months, 'max': estimated_months},
                'base_shelf_life': int(base_days),
                'category': 'normal',
                'quality_grade': 'Grade A',
                'severity': 'normal',
                'severity_position': 0.0,
                'confidence': 0.95,
                'defect_percentage': 0.0,
                'defect_score': round(total_defect_score, 3),
                'defect_counts': defect_counts,
                'defect_categories': {},
                'total_defects_detected': total_detected,
                'average_detection_confidence': round(avg_detection_confidence, 3),
                'raw_prediction': base_days,
                'profile_used': 'clean',
            }

        scenario_key = self._select_scenario(defect_counts, total_detected, bean_key)
        profile = self.scenario_profiles.get(scenario_key, self.scenario_profiles['general_mixed'])
        months_min, months_max = profile['months_range']
        severity = profile['severity']

        # Adjust range for bean grade on clean rows
        if scenario_key.startswith('clean'):
            if bean_key in ('arabica', 'liberica', 'excelsa'):
                months_min, months_max = 24.0, 36.0
            else:
                months_min, months_max = 12.0, 24.0

        # Intensity drives position toward the lower end
        intensity = self._clamp(defect_percentage / 100.0, 0.0, 1.0)
        position = self._clamp(0.25 + 0.7 * intensity, 0.0, 1.0)
        predicted_months = months_max - (months_max - months_min) * position
        predicted_months = max(0.1, predicted_months)
        predicted_days = int(predicted_months * 30)

        # Special guard for roasted defect: enforce heavy reduction
        if scenario_key == 'roasted_defect':
            cap = months_max * 0.5
            predicted_months = min(predicted_months, cap)
            predicted_days = int(predicted_months * 30)

        # Confidence
        confidence = profile.get('base_confidence', 0.8)
        confidence -= 0.15 * position
        if total_detected > 1:
            confidence -= min(0.08, 0.01 * (total_detected - 1))
        if total_detected:
            confidence *= (0.85 + 0.15 * avg_detection_confidence)
        confidence = self._clamp(confidence, 0.2, 0.96)
        if confidence < confidence_threshold:
            confidence = max(confidence, confidence_threshold - 0.05)
            category_label = 'Uncertain'
        else:
            category_label = 'Excellent' if severity == 'mild' or severity == 'normal' else (
                'Good' if severity == 'moderate' and defect_percentage <= 40 else
                'Warning' if severity == 'moderate' else 'Critical'
            )

        quality_grade = 'Grade A'
        if severity == 'moderate':
            quality_grade = 'Grade B' if defect_percentage <= 40 else 'Grade C'
        elif severity == 'severe':
            quality_grade = 'Grade D'

        severity_position = position
        est_range_min = round(months_min, 1)
        est_range_max = round(months_max, 1)

        return {
            'predicted_days': int(predicted_days),
            'estimated_months': round(predicted_months, 1),
            'estimated_months_range': {'min': est_range_min, 'max': est_range_max},
            'base_shelf_life': int(base_days),
            'category': category_label,
            'quality_grade': quality_grade,
            'severity': severity,
            'severity_position': round(severity_position, 3),
            'confidence': round(confidence, 4),
            'defect_percentage': round(defect_percentage, 1),
            'defect_score': round(total_defect_score, 3),
            'defect_counts': defect_counts,
            'defect_categories': {},  # scenario-based; not breaking out here
            'total_defects_detected': total_detected,
            'average_detection_confidence': round(avg_detection_confidence, 3),
            'raw_prediction': predicted_days,
            'profile_used': scenario_key,
        }

    def _normalize_defect_key(self, defect_type: str) -> str:
        return str(defect_type or '').strip().lower().replace('-', '_').replace(' ', '_')

    def _select_scenario(self, defect_counts: Dict[str, int], total_detected: int, bean_key: str) -> str:
        if total_detected == 0:
            return 'clean_specialty' if bean_key in ('arabica', 'liberica', 'excelsa') else 'clean_commodity'

        def has(key):
            return defect_counts.get(key, 0) > 0

        # Roasted-bean defect dominates
        roasted = has('roasted_beans') or has('roasted-beans')
        black = any(k for k in defect_counts if 'black' in k and defect_counts.get(k, 0) > 0)
        insect = has('insect_damage') or has('insect') or has('borer')
        broken = any(k for k in defect_counts if any(token in k for token in ('broken', 'cut', 'chip', 'crack', 'physical')) and defect_counts.get(k, 0) > 0)

        categories_present = sum(1 for flag in (roasted, black, insect, broken) if flag)

        if roasted:
            return 'roasted_defect'
        if categories_present > 1:
            return 'general_mixed'
        if black:
            return 'fully_black'
        if insect:
            return 'insect_damage'
        if broken:
            return 'broken_cut'
        return 'general_mixed'

    @staticmethod
    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

class BeanScanEnsemble(nn.Module):
    """Ensemble model combining CNN, Mask R-CNN, and Rule-based Shelf Life"""
    
    def __init__(self, cnn_model: BeanClassifierCNN, 
                 defect_model: DefectDetectorMaskRCNN,
                 shelf_life_model: RuleBasedShelfLife):
        super().__init__()
        self.cnn_model = cnn_model
        self.defect_model = defect_model
        self.shelf_life_model = shelf_life_model
        
    def forward(self, image, defect_sequence=None):
        """Complete bean analysis pipeline"""
        results = {}
        
        # 1. Bean type classification
        bean_type = self.cnn_model.predict(image)
        results['bean_classification'] = bean_type
        
        # 2. Defect detection
        raw_defects = self.defect_model.detect_defects(image)
        # Filter out ignored defect types
        defects = []
        for defect in raw_defects:
            defect_type = (defect.get('defect_type') or '').lower() if isinstance(defect, dict) else ''
            if defect_type and defect_type in IGNORED_DEFECT_TYPES:
                continue
            defects.append(defect)
        results['defect_detection'] = defects
        
        # 3. Shelf life prediction (always compute; derive sequence from defects when not provided)
        # Determine bean type string for rule-based prediction
        bean_type_name = results.get('bean_classification', [{}])[0].get('class', 'Arabica') if results.get('bean_classification') else 'Arabica'

        # Build a defect sequence if none provided, based on detected defects
        derived_defect_sequence = defect_sequence
        if derived_defect_sequence is None:
            derived_defect_sequence = []
            if defects:
                for defect in defects:
                    derived_defect_sequence.append({
                        'type': defect.get('defect_type', 'unknown'),
                        'confidence': defect.get('confidence', 0.5),
                        'count': 1
                    })

        # Always compute shelf life using rule-based model (handles empty sequences)
        shelf_life = self.shelf_life_model.predict_shelf_life(derived_defect_sequence, bean_type_name)
        results['shelf_life_prediction'] = shelf_life
        
        # 4. Calculate overall health score
        health_score = self._calculate_health_score(bean_type, defects)
        results['health_score'] = health_score
        
        return results
    
    def _calculate_health_score(self, bean_type, defects):
        """Calculate overall bean health score"""
        # Base score from bean type confidence
        base_score = bean_type[0]['confidence'] if bean_type else 0.5
        
        # Penalty for defects
        defect_penalty = 0
        if defects:
            for defect in defects:
                # Higher penalty for more severe defects
                if defect['defect_type'] == 'Insect_Damage':
                    defect_penalty += 0.25
                elif defect['defect_type'] == 'Discoloration':
                    defect_penalty += 0.15
                elif defect['defect_type'] == 'Physical_Damage':
                    defect_penalty += 0.1
                
                # Additional penalty based on defect area
                defect_penalty += min(0.2, defect['area'] / 10000)  # Normalize area
        
        # Calculate final health score
        health_score = max(0.0, min(1.0, base_score - defect_penalty))
        
        return {
            'score': health_score,
            'percentage': health_score * 100,
            'grade': self._get_health_grade(health_score),
            'defect_count': len(defects) if defects else 0
        }
    
    def _get_health_grade(self, score):
        """Health grade simplified to a single neutral label (no letter grading)"""
        return 'ungraded'

def _load_defect_class_names(weights_path: Path) -> List[str]:
    default_classes = [
        "insect_damage",
        "nugget",
        "quaker",
        "roasted-beans",
        "shell",
        "under_roast",
    ]
    try:
        label_path = weights_path.with_suffix(".json")
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                if "classes" in data and isinstance(data["classes"], list):
                    return [str(x) for x in data["classes"]]
                return [v for k, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARNING] Failed to load defect class names: {exc}")
    return default_classes


# Utility functions
def create_models(device: str = 'cpu'):
    """Create and initialize all models"""
    device = torch.device(device)
    models_dir = Path(__file__).resolve().parent.parent / "models"
    
    # Initialize models
    cnn = BeanClassifierCNN(num_classes=4, pretrained=True)
    shelf_life_model = RuleBasedShelfLife()  # Rule-based instead of LSTM

    # Choose defect model: prefer MobileNet classifier if weights exist
    mobilenet_defect_path = models_dir / "defect_mobilenet_best.pth"
    if mobilenet_defect_path.exists():
        class_names = _load_defect_class_names(mobilenet_defect_path)
        defect_backbone = CoffeeNetCNN(num_classes=len(class_names), pretrained=False)
        defect_detector = DefectClassifierAdapter(defect_backbone, class_names, device)
        print(f"[INFO] Using MobileNet defect classifier: {mobilenet_defect_path}")
    else:
        defect_detector = DefectDetectorMaskRCNN(num_classes=6, pretrained=True)
        print("[INFO] Using Faster R-CNN defect detector (no MobileNet weights found)")
    
    # Move to device (rule-based model doesn't need device)
    cnn.to(device)
    defect_detector.to(device)
    
    # Create ensemble
    ensemble = BeanScanEnsemble(cnn, defect_detector, shelf_life_model)
    ensemble.to(device)
    
    # Load trained weights if available
    models = {
        'cnn': cnn,
        'defect_detector': defect_detector,
        'shelf_life_model': shelf_life_model,  # Updated key name
        'ensemble': ensemble
    }
    
    # Load saved weights
    load_models(device=device, models=models, model_dir=str(models_dir))
    
    return models

def save_models(models: Dict, save_dir: str = './models'):
    """Save all models"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        # Skip saving rule-based model (no state to save) and ensemble (composed of submodels)
        if name == 'shelf_life_model':
            print(f"[OK] Rule-based {name} model (no state to save)")
            continue
        if name == 'ensemble':
            print("[OK] Skipping saving 'ensemble' (composed of cnn + defect; no standalone weights needed)")
            continue
        
        torch.save(model.state_dict(), os.path.join(save_dir, f'{name}.pth'))
        print(f"[OK] Saved {name} model")

def load_models(device: str = 'cpu', models: Dict = None, model_dir: str = './models'):
    """Load all models"""
    import os
    device = torch.device(device)
    
    if models is None:
        # This should not happen in our current usage
        print("[WARNING] No models provided to load_models")
        return {}
    
    # Load saved weights if available
    for name, model in models.items():
        # Skip loading rule-based model (no state to load)
        if name == 'shelf_life_model':
            print(f"[OK] Rule-based {name} model (no state to load)")
            continue
        # Skip loading ensemble weights to avoid architecture mismatch; it's composed from submodels
        if name == 'ensemble':
            print("[OK] Skipping loading weights for 'ensemble' (composed of cnn + defect); using submodels' weights")
            continue
        
        # Map model names to actual file names
        model_file_map = {
            'cnn': 'cnn_best.pth',
            'defect_detector': 'defect_mobilenet_best.pth' if isinstance(model, DefectClassifierAdapter) else 'best_model.pth'
        }
        
        model_filename = model_file_map.get(name, f'{name}.pth')
        model_path = os.path.join(model_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                if isinstance(model, DefectClassifierAdapter):
                    model.classifier.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[OK] Loaded {name} model from {model_path}")
            except RuntimeError as e:
                print(f"[WARNING] Architecture mismatch for {name} model: {str(e)[:100]}...")
                print(f"[WARNING] Using initialized weights for {name} (trained model has different architecture)")
        else:
            print(f"[WARNING] No saved weights found for {name} ({model_filename}), using initialized weights")
    
    return models
