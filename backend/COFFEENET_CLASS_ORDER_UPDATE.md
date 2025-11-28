# CoffeeNet Class Order Update

## Summary

Updated the class order to match the CoffeeNet model specification: **["Liberica", "Arabica", "Robusta", "Excelsa"]**

## Changes Made

### Files Updated

1. **`backend/ml/bean_classifier.py`**
   - Changed default class order from `["Arabica", "Robusta", "Liberica", "Excelsa"]` 
   - To: `["Liberica", "Arabica", "Robusta", "Excelsa"]` (CoffeeNet order)

2. **`backend/ml/custom_models.py`**
   - Updated `BeanClassifierCNN` default class names to match CoffeeNet order
   - Changed from: `["Arabica", "Robusta", "Liberica", "Excelsa"]`
   - To: `["Liberica", "Arabica", "Robusta", "Excelsa"]`

3. **`backend/predict_bean.py`**
   - Updated class names list to match CoffeeNet order

## CoffeeNet Model Pattern

The implementation now follows the CoffeeNet pattern:

1. ✅ **224×224 normalization transform**
   ```python
   transforms.Resize((224, 224))
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ```

2. ✅ **Run model(image_tensor) to get prediction**
   ```python
   outputs = self(image_tensor)
   ```

3. ✅ **Map output index to class names**
   ```python
   predicted_class = torch.argmax(probabilities, dim=1).item()
   class_name = self.class_names[predicted_class]  # CoffeeNet order
   ```

4. ✅ **Class order matches CoffeeNet**
   - Index 0: "Liberica"
   - Index 1: "Arabica"
   - Index 2: "Robusta"
   - Index 3: "Excelsa"

## Important Notes

⚠️ **Model Compatibility**: If you have a trained model (`cnn_best.pth`), make sure it was trained with this class order. If your model was trained with a different class order, you'll need to either:
- Retrain the model with the new class order, OR
- Remap the class indices when loading predictions

## Verification

To verify the class order is correct:

```python
from ml.bean_classifier import create_bean_classifier

classifier = create_bean_classifier()
print("Class names:", classifier.class_names)
# Should output: ['Liberica', 'Arabica', 'Robusta', 'Excelsa']
```

## CoffeeNet Architecture Reference

The CoffeeNet model uses:
- **Backbone**: MobileNetV3 Small (pretrained on ImageNet)
- **Classifier**: 
  - Linear(576, 512) → ReLU → Dropout(0.5)
  - Linear(512, 128) → ReLU → Dropout(0.4)
  - Linear(128, num_classes)
- **Class order**: ["Liberica", "Arabica", "Robusta", "Excelsa"]

Our current `BeanClassifierCNN` uses a similar but slightly different architecture:
- **Backbone**: MobileNetV3 Small (pretrained)
- **Classifier**:
  - AdaptiveAvgPool2d(1) → Flatten
  - Dropout(0.3) → Linear(576, 256) → ReLU → Dropout(0.3) → Linear(256, num_classes)

Both architectures are compatible as long as the class order matches.


