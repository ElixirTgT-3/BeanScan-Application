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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import shutil
from datetime import datetime

from ml.custom_models import BeanClassifierCNN

class BeanImageDataset(Dataset):
    """Dataset for bean images with labels"""
    
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Pre-compute class weights if training split (for imbalance handling)
        self.class_weights = None
        if split == 'train' and len(self.annotations) > 0:
            bean_types = ["Arabica", "Robusta", "Liberica", "Excelsa"]
            counts = {bt: 0 for bt in bean_types}
            for ann in self.annotations:
                if ann.get('bean_type') in counts:
                    counts[ann['bean_type']] += 1
            totals = sum(counts.values())
            # Inverse frequency weights normalized
            weights = []
            for bt in bean_types:
                freq = counts[bt] / totals if totals > 0 else 0
                weights.append(0.0 if freq == 0 else 1.0 / freq)
            # Normalize to mean 1.0 for stability
            mean_w = sum(weights) / len(weights) if len(weights) > 0 else 1.0
            self.class_weights = torch.tensor([w / mean_w for w in weights], dtype=torch.float32)
        
        # Image transformations with strong augmentation for training
        if self.transform is None:
            if split == 'train':
                # Strong augmentation for training
                self.transform = transforms.Compose([
                    # Stronger spatial augmentation
                    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
                    transforms.RandomVerticalFlip(p=0.3),  # Vertical flip
                    transforms.RandomRotation(degrees=15),  # Rotation
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
                    transforms.RandomGrayscale(p=0.1),  # Random grayscale
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                # Simple transforms for validation
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
    
    def _load_annotations(self):
        """Load dataset annotations"""
        # Try different possible file names
        possible_files = [
            os.path.join(self.data_dir, f'{self.split}_annotations.json'),
            os.path.join(self.data_dir, f'{self.split}_annotation.json'),
            os.path.join(self.data_dir, f'{self.split}_annotations.json')
        ]
        
        for annotations_file in possible_files:
            if os.path.exists(annotations_file):
                with open(annotations_file, 'r') as f:
                    return json.load(f)
        
        print(f"❌ Annotations file not found in {self.data_dir}")
        print(f"   Tried: {possible_files}")
        return []
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, 'images', annotation['image_id'])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            # Create dummy image if file not found
            print(f"⚠️ Image not found: {image_path}")
            image = Image.new('RGB', (224, 224), color=(139, 69, 19))  # Brown color
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Prepare labels
        bean_type_label = self._get_bean_type_label(annotation['bean_type'])
        
        return {
            'image': image,
            'bean_type_label': bean_type_label,
            'health_score': annotation['health_score'],
            'image_id': annotation['image_id']
        }
    
    def _get_bean_type_label(self, bean_type: str):
        """Convert bean type to label index"""
        bean_types = ["Arabica", "Robusta", "Liberica", "Excelsa"]
        return bean_types.index(bean_type) if bean_type in bean_types else 0

class CNNTrainer:
    """Trainer class for CNN classifier only"""
    
    def __init__(self, device: str = 'cpu', models_dir: str = './models'):
        self.device = torch.device(device)
        self.non_blocking = self.device.type == 'cuda'
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize CNN model
        self.model = BeanClassifierCNN(num_classes=4).to(self.device)
        
        # Training parameters
        self.learning_rate = 0.0005
        self.batch_size = 16
        self.num_epochs = 40
        self.save_interval = 10
        
        # Loss function (optionally weighted if train dataset indicates imbalance)
        self.criterion = nn.CrossEntropyLoss()
        # Weight decay (L2) to reduce overfitting (increased to 1e-3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        
        # Learning rate scheduler (monitor validation loss – lower is better)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'confusion_matrices': [], 'class_reports': []
        }
    
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            images = batch['image'].to(self.device, non_blocking=self.non_blocking)
            labels = batch['bean_type_label'].to(self.device, non_blocking=self.non_blocking)

            # Mixup augmentation to improve robustness
            lam = np.random.beta(0.4, 0.4) if np.random.rand() < 0.5 else 1.0
            if lam < 1.0:
                index = torch.randperm(images.size(0)).to(self.device)
                mixed_images = lam * images + (1 - lam) * images[index, :]
                labels_a, labels_b = labels, labels[index]
            else:
                mixed_images = images
                labels_a, labels_b = labels, labels
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mixed_images)
            if lam < 1.0:
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader):
        """Validate for one epoch with detailed metrics"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device, non_blocking=self.non_blocking)
                labels = batch['bean_type_label'].to(self.device, non_blocking=self.non_blocking)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the CNN classifier"""
        print("🚀 Training CNN Bean Classifier...")
        print(f"📊 Training on {len(train_loader.dataset)} images")
        if val_loader:
            print(f"📊 Validating on {len(val_loader.dataset)} images")
        
        # Early stopping on validation loss
        best_val_loss = float('inf')
        patience = 15  # Early stopping patience (epochs without val loss improvement)
        min_delta = 1e-4  # Minimum change to qualify as improvement
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{self.num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            
            print(f"✅ Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            
            # Validation
            if val_loader:
                val_loss, val_acc, predictions, labels = self.validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                
                # Calculate confusion matrix and class report
                cm = confusion_matrix(labels, predictions)
                class_report = classification_report(labels, predictions, 
                                                   target_names=['Arabica', 'Robusta', 'Liberica', 'Excelsa'],
                                                   output_dict=True)
                
                self.training_history['confusion_matrices'].append(cm)
                self.training_history['class_reports'].append(class_report)
                
                print(f"🎯 Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
                
                # Print per-class accuracy
                bean_types = ['Arabica', 'Robusta', 'Liberica', 'Excelsa']
                print("📊 Per-class accuracy:")
                for i, bean_type in enumerate(bean_types):
                    if i < len(class_report):
                        precision = class_report[bean_type]['precision']
                        recall = class_report[bean_type]['recall']
                        f1 = class_report[bean_type]['f1-score']
                        support = class_report[bean_type]['support']
                        print(f"   {bean_type}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (n={support})")
                
                # Save best model on lowest validation loss
                if (best_val_loss - val_loss) > min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model('cnn_best.pth')
                    print(f"💾 Saved best model with Val Loss {val_loss:.4f} (Val Acc {val_acc:.2f}%)")
                else:
                    patience_counter += 1

                # Update learning rate scheduler with validation loss
                self.scheduler.step(val_loss)
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.save_interval == 0:
                self.save_model(f'cnn_epoch_{epoch+1}.pth')

            # Save/update training history figure each epoch so it is always current
            # Non-blocking save to disk; safe to stop training anytime and keep the latest figure
            try:
                self.plot_training_history(show=False)
            except Exception:
                pass
        
        print("\n🎉 CNN Training Complete!")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        self.save_model('cnn_final.pth')
        
        return self.training_history
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.models_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"💾 Saved model: {save_path}")
    
    def plot_training_history(self, show: bool = True):
        """Plot training history with confusion matrix

        Args:
            show: If True, display the figure window. If False, just save to disk.
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Accuracy')
        if self.training_history['val_acc']:
            axes[0, 1].plot(self.training_history['val_acc'], label='Val Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Confusion matrix (latest)
        if self.training_history['confusion_matrices']:
            cm = self.training_history['confusion_matrices'][-1]
            bean_types = ['Arabica', 'Robusta', 'Liberica', 'Excelsa']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=bean_types, yticklabels=bean_types, ax=axes[1, 0])
            axes[1, 0].set_title('Confusion Matrix (Latest Epoch)')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        
        # Per-class F1 scores over time
        if self.training_history['class_reports']:
            bean_types = ['Arabica', 'Robusta', 'Liberica', 'Excelsa']
            f1_scores = {bean: [] for bean in bean_types}
            
            for report in self.training_history['class_reports']:
                for bean in bean_types:
                    if bean in report:
                        f1_scores[bean].append(report[bean]['f1-score'])
                    else:
                        f1_scores[bean].append(0.0)
            
            for bean in bean_types:
                axes[1, 1].plot(f1_scores[bean], label=bean, marker='o')
            
            axes[1, 1].set_title('Per-Class F1 Score Over Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        # Primary save inside models directory
        primary_path = os.path.join(self.models_dir, 'cnn_training_history.png')
        plt.savefig(primary_path, dpi=300, bbox_inches='tight')

        # Also export a copy as "figure_1" into ./exports for easy access
        try:
            exports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exports'))
            os.makedirs(exports_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            figure1_path = os.path.join(exports_dir, 'figure_1.png')
            figure1_ts_path = os.path.join(exports_dir, f'figure_1_{timestamp}.png')
            shutil.copyfile(primary_path, figure1_path)
            shutil.copyfile(primary_path, figure1_ts_path)
            print(f"📁 Figure saved: {primary_path}")
            print(f"📥 Auto-exported to: {figure1_path}")
            print(f"🕒 Timestamped copy: {figure1_ts_path}")
            # Auto-open on Windows for convenience
            if os.name == 'nt':
                try:
                    os.startfile(figure1_path)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Could not export figure_1: {e}")

        if show:
            plt.show()
        else:
            plt.close(fig)

def analyze_dataset_distribution(dataset, name):
    """Analyze class distribution in dataset"""
    bean_types = ["Arabica", "Robusta", "Liberica", "Excelsa"]
    class_counts = {bean: 0 for bean in bean_types}
    
    for i in range(len(dataset)):
        annotation = dataset.annotations[i]
        bean_type = annotation['bean_type']
        if bean_type in class_counts:
            class_counts[bean_type] += 1
    
    print(f"\n📊 {name} Dataset Distribution:")
    total = sum(class_counts.values())
    for bean_type, count in class_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"   {bean_type}: {count} ({percentage:.1f}%)")
    
    return class_counts

def main():
    """Main training function"""
    print("🎯 BeanScan CNN Bean Classifier Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Create datasets
    train_dataset = BeanImageDataset('./data/train', split='train')
    val_dataset = BeanImageDataset('./data/val', split='val')
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    # Analyze dataset distribution
    train_dist = analyze_dataset_distribution(train_dataset, "Training")
    val_dist = analyze_dataset_distribution(val_dataset, "Validation")
    
    # Check for class imbalance
    train_total = sum(train_dist.values())
    val_total = sum(val_dist.values())
    
    print(f"\n⚠️ Class Balance Analysis:")
    for bean_type in train_dist.keys():
        train_pct = (train_dist[bean_type] / train_total) * 100 if train_total > 0 else 0
        val_pct = (val_dist[bean_type] / val_total) * 100 if val_total > 0 else 0
        imbalance = abs(train_pct - val_pct)
        if imbalance > 10:
            print(f"   ⚠️ {bean_type}: Train={train_pct:.1f}%, Val={val_pct:.1f}% (Δ={imbalance:.1f}%)")
        else:
            print(f"   ✅ {bean_type}: Train={train_pct:.1f}%, Val={val_pct:.1f}%")
    
    # Initialize trainer
    trainer = CNNTrainer(device=device)
    
    # Optionally set class-weighted loss to handle imbalance
    try:
        bean_types_order = ["Arabica", "Robusta", "Liberica", "Excelsa"]
        counts_list = [train_dist.get(bt, 0) for bt in bean_types_order]
        total_count = sum(counts_list)
        inv_freq = []
        for c in counts_list:
            freq = (c / total_count) if total_count > 0 else 0.0
            inv_freq.append(0.0 if freq == 0 else 1.0 / freq)
        mean_w = sum(inv_freq) / len(inv_freq) if len(inv_freq) > 0 else 1.0
        norm_weights = [(w / mean_w) for w in inv_freq]
        class_weights_tensor = torch.tensor(norm_weights, dtype=torch.float32).to(device)
        trainer.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"⚖️ Using class weights: {norm_weights}")
    except Exception as e:
        print(f"⚠️ Could not set class weights: {e}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Train the model
    try:
        history = trainer.train(train_loader, val_loader)
        
        # Plot training history
        trainer.plot_training_history()
        
        print("\n🎉 Training completed successfully!")
        print("📁 Models saved in:", trainer.models_dir)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
