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
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        
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
        
        # Bean type names
        self.class_names = ["Arabica", "Robusta", "Liberica", "Excelsa"]
        
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
            
            # Filter by confidence threshold
            mask = confidence >= threshold
            predictions = []
            
            for i in range(len(predicted)):
                if mask[i]:
                    predictions.append({
                        'class': self.class_names[predicted[i].item()],
                        'confidence': confidence[i].item(),
                        'probabilities': probabilities[i].tolist()
                    })
                else:
                    predictions.append({
                        'class': 'Unknown',
                        'confidence': confidence[i].item(),
                        'probabilities': probabilities[i].tolist()
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
    """Rule-based shelf life prediction based on defect analysis"""
    
    def __init__(self):
        # Defect severity weights (higher = more critical)
        self.defect_weights = {
            'insect_damage': 8.0,
            'discoloration': 6.0,
            'physical_damage': 4.0,
            'quaker': 7.0,
            'shell': 3.0,
            'under_roast': 2.0,
            'roasted-beans': 1.0,
            'nugget': 5.0
        }
        
        # Base shelf life by bean type (in days) -- modelled as multi-month storage
        self.base_shelf_life = {
            'Arabica': 240,   # ≈8 months
            'Robusta': 210,   # ≈7 months
            'Liberica': 225,  # ≈7.5 months
            'Excelsa': 210,   # ≈7 months
            'Other': 180      # ≈6 months fallback
        }
        
        # Shelf life categories
        self.shelf_life_categories = ["Expired", "Critical", "Warning", "Good", "Excellent"]

        # Severity bands (percentage ranges with peak values and scaling information)
        self.severity_bands = [
            {
                'name': 'mild',
                'percent_range': (0.0, 22.0),
                'percent_peak': 8.0,
                'month_edges': (0.60, 0.82),   # scale at range edges
                'month_peak': 0.96,            # scale near peak
                'confidence_edges': (0.78, 0.88),
                'confidence_peak': 0.95,
            },
            {
                'name': 'moderate',
                'percent_range': (18.0, 78.0),
                'percent_peak': 45.0,
                'month_edges': (0.28, 0.52),
                'month_peak': 0.62,
                'confidence_edges': (0.42, 0.64),
                'confidence_peak': 0.74,
            },
            {
                'name': 'severe',
                'percent_range': (70.0, 100.0),
                'percent_peak': 90.0,
                'month_edges': (0.08, 0.20),
                'month_peak': 0.18,
                'confidence_edges': (0.18, 0.38),
                'confidence_peak': 0.5,
            },
        ]
    
    def predict_shelf_life(self, defect_sequence, bean_type='Arabica', confidence_threshold: float = 0.7):
        """Predict shelf life based on defect analysis using rule-based approach"""
        
        # Handle different input formats
        if isinstance(defect_sequence, list):
            defects = defect_sequence
        elif hasattr(defect_sequence, 'tolist'):
            defects = defect_sequence.tolist()
        else:
            defects = []
        
        # Start with base shelf life for the bean type
        base_days = self.base_shelf_life.get(bean_type, self.base_shelf_life['Other'])
        base_months = base_days / 30.0
        predicted_days = base_days
        
        # Calculate defect impact
        total_defect_score = 0
        defect_counts = {}
        total_detected = 0
        cumulative_confidence = 0.0
        
        # Count and score defects
        for defect in defects:
            if isinstance(defect, dict):
                defect_type = defect.get('type', 'unknown')
                confidence = defect.get('confidence', 0.5)
                count = defect.get('count', 1)
            else:
                # Handle simple defect type strings
                defect_type = str(defect).lower()
                confidence = 1.0
                count = 1
            
            # Get defect weight
            weight = self.defect_weights.get(defect_type, 1.0)
            
            # Calculate impact (weight * confidence * count)
            impact = weight * confidence * count
            total_defect_score += impact
            
            # Track defect counts
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + count
            total_detected += count
            cumulative_confidence += confidence * count
        
        avg_detection_confidence = (cumulative_confidence / total_detected) if total_detected > 0 else 0.0
        
        # Translate raw defect score into a 0-100% indicator
        normalized_score = min(total_defect_score / 45.0, 1.5)  # allow slight spillover for severe cases
        defect_percentage = max(0.0, min(100.0, normalized_score * 100.0))
        
        band = self._select_severity_band(defect_percentage)
        severity = band['name']
        percent_low, percent_high = band['percent_range']
        span = band['span']
        severity_position = min(1.0, max(0.0, (defect_percentage - percent_low) / span))
        peak_position = band['peak_position']
        
        edge_low, edge_high = band['month_edges']
        peak_scale = band['month_peak']

        severity_scale = self._interpolate_with_peak(
            severity_position,
            peak_position,
            edge_low,
            edge_high,
            peak_scale,
        )
        predicted_months = max(0.1, base_months * severity_scale)
        predicted_days = int(predicted_months * 30)
        
        # Additional guard rails for extreme insect damage
        if defect_counts.get('insect_damage', 0) > 2:
            predicted_months = min(predicted_months, base_months * 0.2)
            predicted_days = int(predicted_months * 30)
        
        predicted_days = max(0, predicted_days)
        
        confidence = self._interpolate_with_peak(
            severity_position,
            peak_position,
            band['confidence_edges'][0],
            band['confidence_edges'][1],
            band['confidence_peak'],
        )
        
        if total_detected:
            confidence -= min(0.18, (total_detected - 1) * 0.02)
            confidence *= (0.85 + 0.15 * avg_detection_confidence)
        
        confidence = max(0.2, min(0.96, confidence))
        
        months_scale_candidates = [edge_low, edge_high, peak_scale, severity_scale]
        valid_scales = [s for s in months_scale_candidates if s is not None]
        months_min = max(0.1, base_months * min(valid_scales))
        months_max = max(months_min, base_months * max(valid_scales))
        
        # Categorise shelf life and quality grade
        if severity == "mild":
            category = "Excellent"
            quality_grade = "Grade A"
        elif severity == "moderate":
            category = "Warning" if defect_percentage > 40 else "Good"
            quality_grade = "Grade B" if defect_percentage <= 30 else "Grade C"
        else:
            category = "Critical"
            quality_grade = "Grade D"
        
        # Ensure confidence thresholding behaviour
        if confidence < confidence_threshold:
            category = "Uncertain"
            confidence = max(0.15, confidence_threshold - 0.05)
        
        return {
            'predicted_days': max(0, int(predicted_days)),
            'category': category,
            'confidence': round(min(0.96, confidence), 4),
            'raw_prediction': predicted_months * 30,
            'defect_score': round(total_defect_score, 3),
            'defect_counts': defect_counts,
            'defect_percentage': round(defect_percentage, 1),
            'average_detection_confidence': round(avg_detection_confidence, 3),
            'total_defects_detected': total_detected,
            'severity': severity,
            'severity_position': round(severity_position, 3),
            'estimated_months': round(predicted_months, 1),
            'estimated_months_range': {
                'min': round(months_min, 1),
                'max': round(months_max, 1)
            },
            'quality_grade': quality_grade,
            'base_shelf_life': base_days
        }

    def _select_severity_band(self, percentage: float) -> dict:
        # Choose the first band containing the percentage; if none, pick the closest by range distance.
        for band in self.severity_bands:
            low, high = band['percent_range']
            if low <= percentage <= high:
                return self._prepare_band(band)

        # Fallback: choose band whose range midpoint is closest to percentage
        def distance(b):
            low, high = b['percent_range']
            midpoint = (low + high) / 2
            return abs(percentage - midpoint)

        band = min(self.severity_bands, key=distance)
        return self._prepare_band(band)

    def _prepare_band(self, band: dict) -> dict:
        prepared = dict(band)
        low, high = prepared['percent_range']
        span = max(1.0, high - low)
        peak = prepared['percent_peak']
        peak_position = (peak - low) / span
        prepared['span'] = span
        prepared['peak_position'] = min(1.0, max(0.0, peak_position))
        return prepared

    def _interpolate_with_peak(self, position: float, peak_position: float, edge_low: float, edge_high: float, peak_value: float) -> float:
        position = min(1.0, max(0.0, position))
        peak_position = min(1.0, max(0.0, peak_position))

        if peak_position == 0.0:
            # Avoid division by zero, fall back to edge_high trajectory
            return peak_value + (edge_high - peak_value) * position
        if peak_position == 1.0:
            return edge_low + (peak_value - edge_low) * position

        if position <= peak_position:
            t = position / peak_position
            return edge_low + (peak_value - edge_low) * t
        else:
            denom = (1.0 - peak_position)
            if denom <= 0.0:
                return peak_value
            t = (position - peak_position) / denom
            return peak_value + (edge_high - peak_value) * t

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
        defects = self.defect_model.detect_defects(image)
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
        """Convert health score to letter grade"""
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

# Utility functions
def create_models(device: str = 'cpu'):
    """Create and initialize all models"""
    device = torch.device(device)
    
    # Initialize models
    cnn = BeanClassifierCNN(num_classes=4, pretrained=True)
    defect_detector = DefectDetectorMaskRCNN(num_classes=6, pretrained=True)
    shelf_life_model = RuleBasedShelfLife()  # Rule-based instead of LSTM
    
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
    load_models(device=device, models=models)
    
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
            'defect_detector': 'best_model.pth'  # Use best_model.pth for defect detection
        }
        
        model_filename = model_file_map.get(name, f'{name}.pth')
        model_path = os.path.join(model_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[OK] Loaded {name} model from {model_path}")
            except RuntimeError as e:
                print(f"[WARNING] Architecture mismatch for {name} model: {str(e)[:100]}...")
                print(f"[WARNING] Using initialized weights for {name} (trained model has different architecture)")
        else:
            print(f"[WARNING] No saved weights found for {name} ({model_filename}), using initialized weights")
    
    return models
