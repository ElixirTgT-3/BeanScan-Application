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
    
    def __init__(self, num_classes: Optional[int] = None, class_names: Optional[List[str]] = None, pretrained: bool = True):
        super(BeanClassifier, self).__init__()
        
        # Determine class configuration
        # Priority: class_names > num_classes > try to infer from saved model > default
        model_path = './models/cnn_best.pth'
        inferred_num_classes = None
        
        # Try to infer number of classes from saved model weights
        if os.path.exists(model_path) and (num_classes is None and class_names is None):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                # Find the LAST linear layer (output layer) in classifier
                # Look for classifier layers with weight tensors, sorted to get the last one
                classifier_weights = [(k, v) for k, v in state_dict.items() 
                                    if 'classifier' in k and 'weight' in k and len(v.shape) == 2]
                if classifier_weights:
                    # Sort by layer index (classifier.0.weight, classifier.3.weight, classifier.6.weight, etc.)
                    # Get the one with highest index (last layer = output layer)
                    classifier_weights.sort(key=lambda x: int(x[0].split('.')[1]) if x[0].split('.')[1].isdigit() else -1)
                    last_layer_key, last_layer_weight = classifier_weights[-1]
                    inferred_num_classes = last_layer_weight.shape[0]
                    print(f"[INFO] Inferred {inferred_num_classes} classes from saved model (from {last_layer_key})")
            except Exception as e:
                print(f"[WARN] Could not infer classes from model: {e}")
        
        # Use our trained MobileNetV3 model
        from ml.custom_models import BeanClassifierCNN
        
        # Determine final class configuration
        if class_names is not None:
            # Use provided class_names
            self.model = BeanClassifierCNN(class_names=class_names, pretrained=pretrained)
            self.class_names = class_names
        elif num_classes is not None:
            # Use provided num_classes
            self.model = BeanClassifierCNN(num_classes=num_classes, pretrained=pretrained)
            self.class_names = self.model.class_names
        elif inferred_num_classes is not None:
            # Use inferred num_classes from saved model
            self.model = BeanClassifierCNN(num_classes=inferred_num_classes, pretrained=pretrained)
            self.class_names = self.model.class_names
            print(f"[INFO] Using {inferred_num_classes} classes inferred from saved model")
        else:
            # Default to 4 classes (CoffeeNet order: Liberica, Arabica, Robusta, Excelsa)
            default_class_names = ["Liberica", "Arabica", "Robusta", "Excelsa"]
            self.model = BeanClassifierCNN(class_names=default_class_names, pretrained=pretrained)
            self.class_names = default_class_names
            print(f"[WARN] Using default 4 classes. Make sure this matches your trained model!")
        
        # Load the trained weights if available
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
                print(f"[OK] Loaded trained model from {model_path}")
            except RuntimeError as e:
                print(f"[ERROR] Failed to load model weights: {e}")
                print(f"[WARN] Architecture mismatch! Model expects {len(self.class_names)} classes.")
                print(f"[WARN] Make sure the model was trained with class_names={self.class_names}")
        else:
            print(f"[WARN] No trained model found at {model_path}, using untrained model")
        
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Set device (CPU for now)
        self.device = torch.device("cpu")
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, image_path: str) -> Dict[str, any]:
        """Predict bean type from image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
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
            image_tensor = self.transform(image).unsqueeze(0)
            
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
def create_bean_classifier(
    model_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None
) -> BeanClassifier:
    """
    Create and optionally load a bean classifier
    
    Args:
        model_path: Path to saved model weights (optional)
        class_names: List of class names to use (must match training). 
                    If None, will try to infer from saved model.
        num_classes: Number of classes (alternative to class_names).
                    If None, will try to infer from saved model.
    
    Returns:
        BeanClassifier instance
    """
    classifier = BeanClassifier(class_names=class_names, num_classes=num_classes)
    
    if model_path and os.path.exists(model_path):
        classifier.load_model(model_path)
    
    return classifier
