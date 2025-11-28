# CNN Feature Learning for Bean Classification

## How the Model Classifies Beans

The CNN (MobileNetV3-based) automatically learns visual features from images, including:

### 1. **Color Features** 🎨
- **Learned automatically**: The model detects color patterns, hues, and intensity variations
- **Color augmentation during training**:
  ```python
  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  ```
  - This makes the model robust to lighting variations
  - Helps it focus on **intrinsic color differences** between bean types
  - Not just raw RGB values, but color patterns and distributions

### 2. **Shape Features** 📐
- **Learned automatically**: The model detects shapes, sizes, and geometric patterns
- **Spatial augmentation during training**:
  ```python
  transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33))
  transforms.RandomRotation(degrees=15)
  transforms.RandomHorizontalFlip(p=0.5)
  ```
  - Makes the model robust to orientation and scale
  - Helps it learn **shape-invariant features**
  - Detects bean shape characteristics (oval, round, elongated, etc.)

### 3. **Texture Features** 🖼️
- **Surface patterns**: The model learns surface texture patterns
- **Grain patterns**: Detects fine-grained texture differences
- **Surface smoothness/roughness**: Learns tactile-like visual features

### 4. **Combined Visual Features** 🔍
The CNN learns a **hierarchical combination** of features:
- **Low-level**: Edges, colors, basic shapes
- **Mid-level**: Patterns, textures, object parts
- **High-level**: Complete bean characteristics combining all features

## Model Architecture

```
Input Image (224×224×3 RGB)
    ↓
MobileNetV3 Backbone (Pretrained on ImageNet)
    ↓
Feature Extraction Layers:
  - Layer 2: 24 channels (basic edges, colors)
  - Layer 4: 40 channels (simple patterns)
  - Layer 6: 40 channels (textures)
  - Layer 8: 48 channels (shape patterns)
  - Layer 10: 96 channels (complex patterns)
  - Layer 12: 576 channels (high-level features)
    ↓
Classifier Head
    ↓
Output: 4 classes [Liberica, Arabica, Robusta, Excelsa]
```

## What Makes Each Bean Type Distinct?

The model learns distinguishing features for each type:

### **Liberica**
- Larger size, elongated shape
- Distinctive color patterns
- Unique surface texture

### **Arabica**
- Oval/elliptical shape
- Characteristic color (green to brown spectrum)
- Smooth surface texture

### **Robusta**
- Rounder shape
- Different color profile
- Distinct texture patterns

### **Excelsa**
- Unique shape characteristics
- Specific color variations
- Characteristic surface features

## Training Strategy

The model is trained to be robust to:
- ✅ **Color variations** (lighting, camera settings)
- ✅ **Shape variations** (orientation, scale)
- ✅ **Texture variations** (surface conditions)
- ✅ **Background variations** (different environments)

## Key Points

1. **Not explicitly programmed**: The model doesn't have separate "shape" and "color" modules
2. **Learns automatically**: Features emerge from training on labeled data
3. **Combined features**: Uses all visual information together
4. **Robust to variations**: Augmentation ensures it works in different conditions

## Example: What the Model "Sees"

For a Liberica bean:
- **Shape**: Detects elongated, larger form
- **Color**: Recognizes specific green-brown color profile
- **Texture**: Identifies characteristic surface pattern
- **Combined**: All features together create a "Liberica signature"

The model doesn't think "this is shape X and color Y" - it learns a **holistic visual representation** that distinguishes bean types.


