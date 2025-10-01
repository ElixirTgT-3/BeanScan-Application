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
    
    def predict(self, x, threshold: float = 0.5):
        """Predict bean type with confidence"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
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
        
        # Base shelf life by bean type (in days)
        self.base_shelf_life = {
            'Arabica': 30,
            'Robusta': 25,
            'Liberica': 28,
            'Excelsa': 26,
            'Other': 20
        }
        
        # Shelf life categories
        self.shelf_life_categories = ["Expired", "Critical", "Warning", "Good", "Excellent"]
    
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
        base_days = self.base_shelf_life.get(bean_type, 20)
        predicted_days = base_days
        
        # Calculate defect impact
        total_defect_score = 0
        defect_counts = {}
        
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
        
        # Apply defect penalties to shelf life
        if total_defect_score > 0:
            # Exponential decay based on defect score
            penalty_factor = min(0.9, total_defect_score / 50.0)  # Cap at 90% reduction
            predicted_days = base_days * (1 - penalty_factor)
        
        # Apply specific rules for critical defects
        
        if 'insect_damage' in defect_counts and defect_counts['insect_damage'] > 2:
            predicted_days = min(predicted_days, 5)  # Heavy insect damage = critical
        
        # Ensure minimum shelf life
        predicted_days = max(0, predicted_days)
        
        # Categorize shelf life
        if predicted_days <= 0:
            category = "Expired"
            confidence = 0.95
        elif predicted_days <= 3:
            category = "Critical"
            confidence = 0.9
        elif predicted_days <= 7:
            category = "Warning"
            confidence = 0.8
        elif predicted_days <= 14:
            category = "Good"
            confidence = 0.75
        else:
            category = "Excellent"
            confidence = 0.7
        
        # Adjust confidence based on defect diversity and severity
        if len(defect_counts) > 3:  # Multiple defect types
            confidence *= 0.9
        if total_defect_score > 20:  # High severity
            confidence *= 0.85
        
        # Adjust confidence based on threshold
        if confidence < confidence_threshold:
            category = "Uncertain"
            confidence = confidence_threshold - 0.1
        
        return {
            'predicted_days': max(0, int(predicted_days)),
            'category': category,
            'confidence': min(0.95, confidence),
            'raw_prediction': predicted_days,
            'defect_score': total_defect_score,
            'defect_counts': defect_counts,
            'base_shelf_life': base_days
        }

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
            print(f"✅ Rule-based {name} model (no state to save)")
            continue
        if name == 'ensemble':
            print("✅ Skipping saving 'ensemble' (composed of cnn + defect; no standalone weights needed)")
            continue
        
        torch.save(model.state_dict(), os.path.join(save_dir, f'{name}.pth'))
        print(f"✅ Saved {name} model")

def load_models(device: str = 'cpu', models: Dict = None, model_dir: str = './models'):
    """Load all models"""
    import os
    device = torch.device(device)
    
    if models is None:
        # This should not happen in our current usage
        print("⚠️  No models provided to load_models")
        return {}
    
    # Load saved weights if available
    for name, model in models.items():
        # Skip loading rule-based model (no state to load)
        if name == 'shelf_life_model':
            print(f"✅ Rule-based {name} model (no state to load)")
            continue
        # Skip loading ensemble weights to avoid architecture mismatch; it's composed from submodels
        if name == 'ensemble':
            print("✅ Skipping loading weights for 'ensemble' (composed of cnn + defect); using submodels' weights")
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
                print(f"✅ Loaded {name} model from {model_path}")
            except RuntimeError as e:
                print(f"⚠️  Architecture mismatch for {name} model: {str(e)[:100]}...")
                print(f"⚠️  Using initialized weights for {name} (trained model has different architecture)")
        else:
            print(f"⚠️  No saved weights found for {name} ({model_filename}), using initialized weights")
    
    return models
