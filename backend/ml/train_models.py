import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

from custom_models import (
    BeanClassifierCNN, 
    DefectDetectorMaskRCNN, 
    ShelfLifeLSTM,
    BeanScanEnsemble,
    create_models
)

class BeanImageDataset(Dataset):
    """Dataset for bean images with labels"""
    
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Image transformations
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_annotations(self):
        """Load dataset annotations"""
        annotations_file = os.path.join(self.data_dir, f'{self.split}_annotations.json')
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                return json.load(f)
        else:
            # Create dummy annotations for testing
            return self._create_dummy_annotations()
    
    def _create_dummy_annotations(self):
        """Create dummy annotations for testing"""
        annotations = []
        bean_types = ["Arabica", "Robusta", "Liberica", "Excelsa"]
        
        # Create dummy data
        for i in range(100):
            annotation = {
                'image_id': f'bean_{i:04d}.jpg',
                'bean_type': bean_types[i % len(bean_types)],
                'defects': [
                    {
                        'type': 'Mold' if i % 10 == 0 else 'None',
                        'bbox': [10, 10, 100, 100] if i % 10 == 0 else [0, 0, 0, 0],
                        'mask': np.zeros((224, 224)).tolist() if i % 10 == 0 else []
                    }
                ],
                'health_score': max(0.1, 1.0 - (i % 10) * 0.1)
            }
            annotations.append(annotation)
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image (create dummy if not exists)
        image_path = os.path.join(self.data_dir, annotation['image_id'])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            # Create dummy image
            image = Image.new('RGB', (224, 224), color=(139, 69, 19))  # Brown color
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Prepare labels
        bean_type_label = self._get_bean_type_label(annotation['bean_type'])
        defect_labels = self._get_defect_labels(annotation['defects'])
        
        return {
            'image': image,
            'bean_type_label': bean_type_label,
            'defect_labels': defect_labels,
            'health_score': annotation['health_score'],
            'image_id': annotation['image_id']
        }
    
    def _get_bean_type_label(self, bean_type: str):
        """Convert bean type to label index"""
        bean_types = ["Arabica", "Robusta", "Liberica", "Excelsa"]
        return bean_types.index(bean_type) if bean_type in bean_types else 0
    
    def _get_defect_labels(self, defects: List):
        """Convert defects to label format"""
        defect_types = ["Mold", "Insect_Damage", "Discoloration", "Physical_Damage"]
        labels = []
        
        for defect in defects:
            if defect['type'] in defect_types:
                label = {
                    'boxes': torch.tensor([defect['bbox']], dtype=torch.float32),
                    'labels': torch.tensor([defect_types.index(defect['type']) + 1], dtype=torch.long),
                    'masks': torch.tensor([defect['mask']], dtype=torch.uint8)
                }
                labels.append(label)
        
        return labels

class ModelTrainer:
    """Trainer class for all models"""
    
    def __init__(self, device: str = 'cpu', models_dir: str = './models'):
        self.device = torch.device(device)
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize models
        self.models = create_models(device)
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 16
        self.num_epochs = 50
        self.save_interval = 10
        
        # Loss functions
        self.cnn_criterion = nn.CrossEntropyLoss()
        self.defect_criterion = self._get_defect_loss()
        
        # Optimizers
        self.optimizers = self._create_optimizers()
        
        # Training history
        self.training_history = {
            'cnn_loss': [], 'cnn_acc': [],
            'defect_loss': [], 'defect_map': [],
            'lstm_loss': [], 'lstm_mae': []
        }
    
    def _get_defect_loss(self):
        """Get loss function for defect detection"""
        return {
            'classification': nn.CrossEntropyLoss(),
            'bbox_regression': nn.SmoothL1Loss(),
            'mask_loss': nn.BCEWithLogitsLoss()
        }
    
    def _create_optimizers(self):
        """Create optimizers for all models"""
        return {
            'cnn': optim.Adam(self.models['cnn'].parameters(), lr=self.learning_rate),
            'defect_detector': optim.Adam(self.models['defect_detector'].parameters(), lr=self.learning_rate),
            'lstm': optim.Adam(self.models['lstm'].parameters(), lr=self.learning_rate)
        }
    
    def train_cnn(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the CNN classifier"""
        print("🚀 Training CNN Classifier...")
        
        model = self.models['cnn']
        optimizer = self.optimizers['cnn']
        model.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                images = batch['image'].to(self.device)
                labels = batch['bean_type_label'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.cnn_criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            self.training_history['cnn_loss'].append(epoch_loss)
            self.training_history['cnn_acc'].append(epoch_acc)
            
            print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%')
            
            # Save model periodically
            if (epoch + 1) % self.save_interval == 0:
                self._save_model('cnn', epoch + 1)
        
        print("✅ CNN Training Complete!")
        return self.training_history['cnn_loss'], self.training_history['cnn_acc']
    
    def train_defect_detector(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the Mask R-CNN defect detector"""
        print("🚀 Training Defect Detector...")
        
        model = self.models['defect_detector']
        optimizer = self.optimizers['defect_detector']
        model.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            
            # Training loop
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                images = batch['image'].to(self.device)
                targets = batch['defect_labels']
                
                # Prepare targets for Mask R-CNN
                formatted_targets = self._format_targets(targets)
                
                # Forward pass
                optimizer.zero_grad()
                loss_dict = model(images, formatted_targets)
                
                # Calculate total loss
                total_loss = sum(loss_dict.values())
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += total_loss.item()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            self.training_history['defect_loss'].append(epoch_loss)
            
            print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}')
            
            # Save model periodically
            if (epoch + 1) % self.save_interval == 0:
                self._save_model('defect_detector', epoch + 1)
        
        print("✅ Defect Detector Training Complete!")
        return self.training_history['defect_loss']
    
    def train_lstm(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the LSTM for shelf life prediction"""
        print("🚀 Training LSTM...")
        
        model = self.models['lstm']
        optimizer = self.optimizers['lstm']
        model.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_mae = 0.0
            
            # Training loop
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                # Create dummy sequence data for training
                seq_length = 10
                batch_size = batch['image'].size(0)
                
                # Generate dummy defect sequences
                sequences = torch.randn(batch_size, seq_length, 64).to(self.device)
                targets = torch.tensor([batch['health_score'] * 30 for _ in range(batch_size)], 
                                     dtype=torch.float32).to(self.device)  # Convert to days
                
                # Forward pass
                optimizer.zero_grad()
                outputs, _ = model(sequences)
                loss = nn.MSELoss()(outputs.squeeze(), targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                running_mae += torch.mean(torch.abs(outputs.squeeze() - targets)).item()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_mae = running_mae / len(train_loader)
            
            self.training_history['lstm_loss'].append(epoch_loss)
            self.training_history['lstm_mae'].append(epoch_mae)
            
            print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, MAE={epoch_mae:.2f}')
            
            # Save model periodically
            if (epoch + 1) % self.save_interval == 0:
                self._save_model('lstm', epoch + 1)
        
        print("✅ LSTM Training Complete!")
        return self.training_history['lstm_loss'], self.training_history['lstm_mae']
    
    def _format_targets(self, targets: List):
        """Format targets for Mask R-CNN training"""
        formatted = []
        for target in targets:
            if target:  # If defects exist
                formatted.append({
                    'boxes': target[0]['boxes'].to(self.device),
                    'labels': target[0]['labels'].to(self.device),
                    'masks': target[0]['masks'].to(self.device)
                })
            else:  # No defects
                formatted.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32).to(self.device),
                    'labels': torch.empty((0,), dtype=torch.long).to(self.device),
                    'masks': torch.empty((0, 224, 224), dtype=torch.uint8).to(self.device)
                })
        return formatted
    
    def _save_model(self, model_name: str, epoch: int):
        """Save model checkpoint"""
        save_path = os.path.join(self.models_dir, f'{model_name}_epoch_{epoch}.pth')
        torch.save(self.models[model_name].state_dict(), save_path)
        print(f"💾 Saved {model_name} checkpoint: {save_path}")
    
    def save_final_models(self):
        """Save final trained models"""
        for name, model in self.models.items():
            save_path = os.path.join(self.models_dir, f'{name}_final.pth')
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved final {name} model: {save_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # CNN metrics
        axes[0, 0].plot(self.training_history['cnn_loss'])
        axes[0, 0].set_title('CNN Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        axes[0, 1].plot(self.training_history['cnn_acc'])
        axes[0, 1].set_title('CNN Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        
        # Defect detector metrics
        axes[0, 2].plot(self.training_history['defect_loss'])
        axes[0, 2].set_title('Defect Detector Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        
        # LSTM metrics
        axes[1, 0].plot(self.training_history['lstm_loss'])
        axes[1, 0].set_title('LSTM Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        
        axes[1, 1].plot(self.training_history['lstm_mae'])
        axes[1, 1].set_title('LSTM MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        
        # Hide empty subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'training_history.png'))
        plt.show()

def main():
    """Main training function"""
    print("🎯 BeanScan Deep Learning Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ModelTrainer(device='cpu')  # Use 'cuda' if GPU available
    
    # Create dummy datasets (replace with real data)
    train_dataset = BeanImageDataset('./data', split='train')
    val_dataset = BeanImageDataset('./data', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Train all models
    try:
        # Train CNN
        cnn_loss, cnn_acc = trainer.train_cnn(train_loader, val_loader)
        
        # Train Defect Detector
        defect_loss = trainer.train_defect_detector(train_loader, val_loader)
        
        # Train LSTM
        lstm_loss, lstm_mae = trainer.train_lstm(train_loader, val_loader)
        
        # Save final models
        trainer.save_final_models()
        
        # Plot training history
        trainer.plot_training_history()
        
        print("\n🎉 All models trained successfully!")
        print("📁 Models saved in:", trainer.models_dir)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
