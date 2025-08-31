import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import io

class BeanClassifier(nn.Module):
    """CNN-based bean classifier using PyTorch"""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(BeanClassifier, self).__init__()
        
        # Use ResNet18 as backbone (pretrained on ImageNet)
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'resnet18', pretrained=pretrained)
        
        # Modify the final layer for our number of classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Bean classes (customize based on your needs)
        self.class_names = [
            "Arabica",
            "Robusta", 
            "Liberica",
            "Excelsa",
            "Other"
        ]
        
        # Set device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, image_path: str) -> Dict[str, any]:
        """Predict bean type from image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Set to evaluation mode
            self.eval()
            
            with torch.no_grad():
                outputs = self(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            return {
                "predicted_class": self.class_names[predicted_class],
                "confidence": confidence,
                "all_probabilities": {
                    name: prob.item() for name, prob in zip(self.class_names, probabilities[0])
                },
                "success": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, any]:
        """Predict bean type from image bytes"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Set to evaluation mode
            self.eval()
            
            with torch.no_grad():
                outputs = self(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            return {
                "predicted_class": self.class_names[predicted_class],
                "confidence": confidence,
                "all_probabilities": {
                    name: prob.item() for name, prob in zip(self.class_names, probabilities[0])
                },
                "success": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath, map_location=self.device))
            self.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

# Factory function to create classifier
def create_bean_classifier(model_path: Optional[str] = None) -> BeanClassifier:
    """Create and optionally load a bean classifier"""
    classifier = BeanClassifier()
    
    if model_path and os.path.exists(model_path):
        classifier.load_model(model_path)
    
    return classifier
