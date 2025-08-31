import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np

class MobileNetV3Backbone(nn.Module):
    """Custom MobileNetV3 backbone for feature extraction"""
    
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
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Bean type names
        self.class_names = ["Arabica", "Robusta", "Liberica", "Excelsa", "Other"]
        
    def forward(self, x):
        features = self.backbone(x)
        # Use the last feature map for classification
        x = features[-1]
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
    """Mask R-CNN for defect detection using MobileNetV3 backbone"""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        # Create custom backbone with MobileNetV3
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        
        # Create FPN from backbone features
        self.fpn = BackboneWithFPN(
            self.backbone,
            return_layers={'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5'},
            in_channels_list=[16, 24, 40, 48, 96, 576],
            out_channels=256
        )
        
        # Create Mask R-CNN with custom backbone
        self.mask_rcnn = maskrcnn_resnet50_fpn(
            pretrained=False,
            num_classes=num_classes + 1  # +1 for background
        )
        
        # Replace backbone
        self.mask_rcnn.backbone = self.fpn
        
        # Customize box and mask predictors
        in_features = self.mask_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        in_features_mask = self.mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes + 1
        )
        
        # Defect types
        self.defect_types = ["Mold", "Insect_Damage", "Discoloration", "Physical_Damage"]
        
    def forward(self, images, targets=None):
        return self.mask_rcnn(images, targets)
    
    def detect_defects(self, image, confidence_threshold: float = 0.5):
        """Detect defects in bean image"""
        self.eval()
        with torch.no_grad():
            # Prepare image
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # Get predictions
            predictions = self.forward(image)
            
            # Process results
            defects = []
            for pred in predictions:
                boxes = pred['boxes']
                scores = pred['scores']
                masks = pred['masks']
                labels = pred['labels']
                
                for i in range(len(scores)):
                    if scores[i] >= confidence_threshold:
                        defect = {
                            'bbox': boxes[i].tolist(),
                            'confidence': scores[i].item(),
                            'mask': masks[i].squeeze().tolist(),
                            'defect_type': self.defect_types[labels[i].item() - 1],  # -1 for background
                            'area': torch.sum(masks[i]).item(),
                            'coordinates': {
                                'x1': boxes[i][0].item(),
                                'y1': boxes[i][1].item(),
                                'x2': boxes[i][2].item(),
                                'y2': boxes[i][3].item()
                            }
                        }
                        defects.append(defect)
            
            return defects

class ShelfLifeLSTM(nn.Module):
    """LSTM for shelf life prediction based on defect progression"""
    
    def __init__(self, input_size: int = 64, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Predict days until expiration
        )
        
        # Shelf life categories
        self.shelf_life_categories = ["Expired", "Critical", "Warning", "Good", "Excellent"]
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_size*2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_size*2)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Predict shelf life
        shelf_life = self.classifier(pooled)
        
        return shelf_life, hidden
    
    def predict_shelf_life(self, defect_sequence, confidence_threshold: float = 0.7):
        """Predict shelf life based on defect progression sequence"""
        self.eval()
        with torch.no_grad():
            # Prepare input sequence
            if isinstance(defect_sequence, list):
                defect_sequence = torch.tensor(defect_sequence, dtype=torch.float32)
            
            if len(defect_sequence.shape) == 2:
                defect_sequence = defect_sequence.unsqueeze(0)  # Add batch dimension
            
            # Get prediction
            shelf_life_days, _ = self.forward(defect_sequence)
            predicted_days = shelf_life_days.item()
            
            # Categorize shelf life
            if predicted_days <= 0:
                category = "Expired"
                confidence = 1.0
            elif predicted_days <= 3:
                category = "Critical"
                confidence = 0.9
            elif predicted_days <= 7:
                category = "Warning"
                confidence = 0.8
            elif predicted_days <= 14:
                category = "Good"
                confidence = 0.7
            else:
                category = "Excellent"
                confidence = 0.6
            
            # Adjust confidence based on threshold
            if confidence < confidence_threshold:
                category = "Uncertain"
            
            return {
                'predicted_days': max(0, int(predicted_days)),
                'category': category,
                'confidence': confidence,
                'raw_prediction': predicted_days
            }

class BeanScanEnsemble(nn.Module):
    """Ensemble model combining CNN, Mask R-CNN, and LSTM"""
    
    def __init__(self, cnn_model: BeanClassifierCNN, 
                 defect_model: DefectDetectorMaskRCNN,
                 lstm_model: ShelfLifeLSTM):
        super().__init__()
        self.cnn_model = cnn_model
        self.defect_model = defect_model
        self.lstm_model = lstm_model
        
    def forward(self, image, defect_sequence=None):
        """Complete bean analysis pipeline"""
        results = {}
        
        # 1. Bean type classification
        bean_type = self.cnn_model.predict(image)
        results['bean_classification'] = bean_type
        
        # 2. Defect detection
        defects = self.defect_model.detect_defects(image)
        results['defect_detection'] = defects
        
        # 3. Shelf life prediction (if sequence provided)
        if defect_sequence is not None:
            shelf_life = self.lstm_model.predict_shelf_life(defect_sequence)
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
                if defect['defect_type'] == 'Mold':
                    defect_penalty += 0.3
                elif defect['defect_type'] == 'Insect_Damage':
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
    cnn = BeanClassifierCNN(num_classes=5, pretrained=True)
    defect_detector = DefectDetectorMaskRCNN(num_classes=4, pretrained=True)
    lstm = ShelfLifeLSTM(input_size=64, hidden_size=128, num_layers=2)
    
    # Move to device
    cnn.to(device)
    defect_detector.to(device)
    lstm.to(device)
    
    # Create ensemble
    ensemble = BeanScanEnsemble(cnn, defect_detector, lstm)
    ensemble.to(device)
    
    return {
        'cnn': cnn,
        'defect_detector': defect_detector,
        'lstm': lstm,
        'ensemble': ensemble
    }

def save_models(models: Dict, save_dir: str = './models'):
    """Save all models"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        torch.save(model.state_dict(), os.path.join(save_dir, f'{name}.pth'))
        print(f"✅ Saved {name} model")

def load_models(device: str = 'cpu', model_dir: str = './models'):
    """Load all models"""
    device = torch.device(device)
    
    # Create models
    models = create_models(device)
    
    # Load saved weights if available
    for name, model in models.items():
        model_path = os.path.join(model_dir, f'{name}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded {name} model from {model_path}")
        else:
            print(f"⚠️  No saved weights found for {name}, using initialized weights")
    
    return models
