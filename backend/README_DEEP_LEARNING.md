# 🧠 BeanScan Custom Deep Learning System

## 🎯 Overview

This system implements **three custom deep learning models** using **PyTorch** with **MobileNetV3 backbone** for comprehensive bean analysis:

1. **CNN Classifier** - Bean type identification
2. **Mask R-CNN** - Defect detection and segmentation  
3. **LSTM** - Shelf life prediction with attention mechanism

## 🏗️ Architecture

### **MobileNetV3 Backbone**
- **Lightweight**: Optimized for mobile/edge devices
- **Efficient**: Reduced computational complexity
- **Feature Extraction**: Multi-scale feature maps for different tasks

### **Model Components**

#### **1. BeanClassifierCNN**
```python
# Architecture
MobileNetV3 → Adaptive Pooling → Dropout → Linear Layers → Classification
# Output: 5 bean types (Arabica, Robusta, Liberica, Excelsa, Other)
# Features: Confidence scores, probability distributions
```

#### **2. DefectDetectorMaskRCNN**
```python
# Architecture  
MobileNetV3 + FPN → RPN → ROI Heads → Mask Prediction
# Output: Bounding boxes, masks, defect types
# Defect Types: Mold, Insect Damage, Discoloration, Physical Damage
```

#### **3. ShelfLifeLSTM**
```python
# Architecture
Bidirectional LSTM → Multi-Head Attention → Dense Layers → Regression
# Output: Days until expiration, confidence, risk category
# Features: Sequential defect progression analysis
```

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
# Activate virtual environment
venv\Scripts\activate

# Install deep learning requirements
pip install -r requirements_ml.txt
```

### **2. Test Models**
```python
from ml.custom_models import create_models

# Create all models
models = create_models(device='cpu')  # Use 'cuda' for GPU

# Test individual models
cnn = models['cnn']
defect_detector = models['defect_detector']
lstm = models['lstm']
ensemble = models['ensemble']
```

### **3. Run Training**
```bash
# Train all models with dummy data
python ml/train_models.py

# Or train individually
python -c "
from ml.train_models import ModelTrainer
trainer = ModelTrainer()
# trainer.train_cnn(train_loader)
# trainer.train_defect_detector(train_loader)  
# trainer.train_lstm(train_loader)
"
```

## 📊 Model Details

### **CNN Classifier**
- **Input**: 224x224 RGB images
- **Backbone**: MobileNetV3-Small (pretrained)
- **Classification Head**: 576 → 256 → 5 classes
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)

### **Mask R-CNN**
- **Input**: Variable size RGB images
- **Backbone**: MobileNetV3 + Feature Pyramid Network
- **Detection Head**: RPN + Fast R-CNN
- **Segmentation Head**: Mask prediction
- **Loss**: Classification + BBox + Mask losses
- **Optimizer**: Adam (lr=0.001)

### **LSTM**
- **Input**: Sequential defect features (64 dimensions)
- **Architecture**: 2-layer bidirectional LSTM
- **Attention**: Multi-head self-attention (8 heads)
- **Output**: Days until expiration
- **Loss**: MSE Loss
- **Optimizer**: Adam (lr=0.001)

## 🔧 Customization

### **Modify Model Architecture**
```python
# Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = MobileNetV3Backbone()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features[-1])
```

### **Add New Defect Types**
```python
# In DefectDetectorMaskRCNN
self.defect_types = [
    "Mold", "Insect_Damage", "Discoloration", 
    "Physical_Damage", "Your_New_Defect"  # Add here
]
```

### **Custom Training Loop**
```python
# Custom training
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = model(batch['images'])
        
        # Custom loss calculation
        loss = custom_loss_function(outputs, batch['targets'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

## 📈 Training & Evaluation

### **Training Parameters**
```python
# Default settings
learning_rate = 0.001
batch_size = 16
num_epochs = 50
save_interval = 10
```

### **Metrics Tracked**
- **CNN**: Loss, Accuracy
- **Mask R-CNN**: Loss, mAP (mean Average Precision)
- **LSTM**: Loss, MAE (Mean Absolute Error)

### **Model Checkpoints**
```python
# Save checkpoints
trainer._save_model('cnn', epoch)

# Load checkpoints  
model.load_state_dict(torch.load('models/cnn_epoch_50.pth'))
```

## 🎨 Data Pipeline

### **Dataset Structure**
```
data/
├── train/
│   ├── images/
│   └── train_annotations.json
├── val/
│   ├── images/
│   └── val_annotations.json
└── test/
    ├── images/
    └── test_annotations.json
```

### **Annotation Format**
```json
{
  "image_id": "bean_0001.jpg",
  "bean_type": "Arabica",
  "defects": [
    {
      "type": "Mold",
      "bbox": [x1, y1, x2, y2],
      "mask": [[0,1,0], [1,1,1], [0,1,0]]
    }
  ],
  "health_score": 0.85
}
```

## 🚀 Production Deployment

### **GPU Acceleration**
```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to GPU
model = model.to(device)
```

### **Model Optimization**
```python
# TorchScript for production
traced_model = torch.jit.trace(model, example_input)
traced_model.save('production_model.pt')

# ONNX export
torch.onnx.export(model, example_input, 'model.onnx')
```

### **API Integration**
```python
# FastAPI endpoint
@router.post("/scan")
async def scan_bean(image: UploadFile):
    # Load and preprocess image
    image_tensor = preprocess_image(image)
    
    # Run inference
    results = ensemble.forward(image_tensor)
    
    return {"analysis": results}
```

## 🔍 Troubleshooting

### **Common Issues**

#### **1. CUDA Out of Memory**
```python
# Reduce batch size
batch_size = 8  # Instead of 16

# Use gradient checkpointing
model.use_checkpoint = True
```

#### **2. Model Not Learning**
```python
# Check learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR

# Add learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

#### **3. Poor Performance**
```python
# Data augmentation
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## 📚 Advanced Features

### **Ensemble Methods**
```python
# Combine predictions from multiple models
ensemble_prediction = (
    0.4 * cnn_prediction + 
    0.4 * defect_prediction + 
    0.2 * lstm_prediction
)
```

### **Transfer Learning**
```python
# Freeze backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Only train classifier
for param in model.classifier.parameters():
    param.requires_grad = True
```

### **Active Learning**
```python
# Uncertainty sampling
def get_uncertain_samples(model, unlabeled_data, n_samples=10):
    predictions = model(unlabeled_data)
    uncertainties = 1 - torch.max(predictions, dim=1)[0]
    return torch.topk(uncertainties, n_samples)[1]
```

## 🎯 Future Enhancements

### **Planned Features**
- [ ] **YOLO Integration** for real-time detection
- [ ] **Transformer Models** for better feature extraction
- [ ] **Federated Learning** for distributed training
- [ ] **AutoML** for hyperparameter optimization
- [ ] **Edge Deployment** with TensorRT

### **Research Areas**
- **Few-shot Learning** for rare defect types
- **Self-supervised Learning** for unlabeled data
- **Multi-modal Fusion** (image + sensor data)
- **Explainable AI** for interpretable predictions

## 📞 Support

### **Getting Help**
1. Check the troubleshooting section above
2. Review PyTorch documentation
3. Check GitHub issues
4. Contact the development team

### **Contributing**
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**🎉 Congratulations! You now have a state-of-the-art deep learning system for bean analysis!**

This system combines the latest advances in computer vision, object detection, and sequential modeling to provide comprehensive bean quality assessment.
