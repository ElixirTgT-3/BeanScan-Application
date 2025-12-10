# BeanScan Application - Core Algorithms Documentation

This document contains the important code for Bean Classification, Defect Detection, and Shelf Life Prediction algorithms.

---

## 1. BEAN CLASSIFICATION ALGORITHM

### 1.1 CNN Model Architecture (MobileNetV3-based)

```python
class BeanClassifierCNN(nn.Module):
    """CNN for bean type classification using MobileNetV3 backbone"""
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.backbone = MobileNetV3Backbone(pretrained=pretrained)

        default_class_names = ["Arabica", "Robusta", "Liberica", "Excelsa"]
        if class_names is not None:
            if len(class_names) == 0:
                raise ValueError("class_names must contain at least one entry")
            self.class_names = class_names
            num_classes = len(class_names)
        else:
            if num_classes <= len(default_class_names):
                self.class_names = default_class_names[:num_classes]
            else:
                extra = [f"Class_{i}" for i in range(len(default_class_names), num_classes)]
                self.class_names = default_class_names + extra
        
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
        
    def forward(self, x):
        features = self.backbone(x)
        # Use the last feature map for classification
        x = features[-1]  # Last feature layer
        x = self.classifier(x)
        return x
```

### 1.2 Prediction with Test-Time Augmentation

```python
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
```

### 1.3 Client-Side Bean Classification Processing (Flutter/Dart)

```dart
// Validate that the image looks like coffee beans
String predictedClass = (predictionData['predicted_class'] ?? '').toString();
double predictedConfidence = (predictionData['confidence'] ?? 0.0).toDouble();
const List<String> beanTypes = ['Arabica', 'Robusta', 'Liberica', 'Excelsa'];
final List<dynamic> probabilitiesRaw =
    predictionData['all_probabilities'] as List<dynamic>? ?? const <dynamic>[];
final List<double> probabilityValues = [
  for (int i = 0; i < probabilitiesRaw.length && i < beanTypes.length; i++)
    (probabilitiesRaw[i] as num).toDouble(),
];

final _ImageHeuristics heuristics = await _analyzeImageHeuristics(imageFile);
final double brownRatio = heuristics.brownRatio;
final double textureScore = heuristics.textureScore;
final double contrastScore = heuristics.contrast;
final double visualScore = _computeVisualBeanScore(
  brownRatio: brownRatio,
  textureScore: textureScore,
  contrastScore: contrastScore,
);

bool isKnownBean = beanTypes.contains(predictedClass);
if (!isKnownBean && probabilityValues.isNotEmpty) {
  final int topIndex = _indexOfMax(probabilityValues);
  if (topIndex != -1) {
    final double topProbability = probabilityValues[topIndex];
    final bool heuristicsSupport = visualScore >= 0.3;
    if (topProbability >= 0.28 && heuristicsSupport && topIndex < beanTypes.length) {
      predictedClass = beanTypes[topIndex];
      predictedConfidence = topProbability;
      isKnownBean = true;
    }
  }
}

final double bestConfidence =
    predictedConfidence > derivedConfidence ? predictedConfidence : derivedConfidence;
final double normalizedConfidence = math.min(1.0, math.max(0.0, bestConfidence));
```

---

## 2. DEFECT DETECTION ALGORITHM

### 2.1 Faster R-CNN Defect Detector Model

```python
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
```

### 2.2 Defect Detection Service

```python
def detect_defects(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
    """
    Detect defects in a coffee bean image
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score for detections
        
    Returns:
        Dictionary containing detection results
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        detections = []
        if predictions and len(predictions) > 0:
            pred = predictions[0]  # Get first (and only) prediction
            
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            valid_indices = scores >= confidence_threshold
            
            for i, (box, score, label) in enumerate(zip(boxes[valid_indices], 
                                                      scores[valid_indices], 
                                                      labels[valid_indices])):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(score),
                    'defect_type': self.class_names[label],
                    'coordinates': {
                        'x1': float(box[0]),
                        'y1': float(box[1]), 
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    },
                    'area': float((box[2] - box[0]) * (box[3] - box[1])),
                    'center': {
                        'x': float((box[0] + box[2]) / 2),
                        'y': float((box[1] + box[3]) / 2)
                    }
                }
                detections.append(detection)
        
        # Calculate summary statistics
        total_defects = len(detections)
        defect_types = {}
        for detection in detections:
            defect_type = detection['defect_type']
            if defect_type not in defect_types:
                defect_types[defect_type] = 0
            defect_types[defect_type] += 1
        
        # Calculate defect percentage (rough estimate)
        image_area = image.size[0] * image.size[1]
        total_defect_area = sum(d['area'] for d in detections)
        defect_percentage = (total_defect_area / image_area) * 100 if image_area > 0 else 0
        
        # Determine overall quality
        quality_score = self._calculate_quality_score(detections, defect_percentage)
        
        return {
            'success': True,
            'detections': detections,
            'summary': {
                'total_defects': total_defects,
                'defect_types': defect_types,
                'defect_percentage': round(defect_percentage, 2),
                'quality_score': quality_score,
                'quality_grade': self._get_quality_grade(quality_score)
            },
            'image_info': {
                'width': image.size[0],
                'height': image.size[1],
                'format': image.format
            }
        }
```

### 2.3 Quality Score Calculation

```python
def _calculate_quality_score(self, detections: List[Dict], defect_percentage: float) -> float:
    """Calculate overall quality score based on defects"""
    if not detections:
        return 1.0  # Perfect quality if no defects
    
    # Base score starts at 1.0
    score = 1.0
    
    # Penalty for defect percentage
    score -= min(0.5, defect_percentage / 100)  # Max 50% penalty for area
    
    # Penalty for number of defects
    score -= min(0.3, len(detections) * 0.05)  # 5% penalty per defect, max 30%
    
    # Penalty for specific defect types (severity)
    for detection in detections:
        defect_type = detection['defect_type']
        if defect_type in ['insect_damage']:
            score -= 0.1  # High severity
        elif defect_type in ['quaker', 'under_roast']:
            score -= 0.05  # Medium severity
        else:
            score -= 0.02  # Low severity
    
    return max(0.0, min(1.0, score))  # Clamp between 0 and 1

def _get_quality_grade(self, score: float) -> str:
    """Convert quality score to letter grade"""
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
```

---

## 3. SHELF LIFE PREDICTION ALGORITHM

### 3.1 Table-aligned rule-based model

```python
class RuleBasedShelfLife:
    def __init__(self):
        self.grade_baselines_months = {
            'specialty': (24.0, 36.0),   # No defects (specialty grade)
            'commodity': (12.0, 24.0),   # No defects (commodity)
        }
        self.bean_grade_map = {
            'arabica': 'specialty',
            'liberica': 'specialty',
            'excelsa': 'specialty',
            'robusta': 'commodity',
            'other': 'commodity',
        }
        self.defect_weights = {
            'fully_black': 12.0,
            'black_bean': 12.0,
            'insect_damage': 9.0,
            'insect': 9.0,
            'borer': 9.0,
            'physical_damage': 5.0,
            'broken': 5.0,
            'broken_cut': 5.0,
            'roasted-beans': 2.0,
            'unknown': 1.5,
        }
        self.scenario_profiles = {
            'clean_specialty': {'months_range': (24.0, 36.0), 'category': 'No defects (specialty grade)', 'severity': 'normal'},
            'clean_commodity': {'months_range': (12.0, 24.0), 'category': 'No defects (commodity)', 'severity': 'normal', 'scale_with_grade': False},
            'broken_cut': {'months_range': (18.0, 27.0), 'reduction_range': (0.10, 0.25), 'category': 'Broken/cut beans', 'severity': 'mild'},
            'insect': {'months_range': (12.0, 16.0), 'reduction_range': (0.30, 0.50), 'category': 'Insect damage', 'severity': 'moderate'},
            'black': {'months_range': (10.0, 14.0), 'reduction_range': (0.40, 0.60), 'category': 'Fully black beans', 'severity': 'severe'},
            'mixed': {'reduction_range': (0.30, 0.70), 'category': 'Multiple defects mixed', 'severity': 'severe'},
        }
```

### 3.2 Shelf life prediction (core flow)

1. **Score defects**: normalize defect types, ignore benign ones, compute `impact = weight * confidence * count`, and accumulate totals. Defect percentage = `clamp(min(score / 20.0, 1.6) * 100, 0, 100)`.
2. **Map to table buckets**: bin counts into `black`, `insect`, `broken`, `other`. Choose a scenario: if multiple buckets are present ? `mixed`; else black ? insect ? broken ? clean (specialty/commodity from bean type).
3. **Baseline range**: fetch the scenario month range and scale with bean grade (specialty factor 1.0; commodity factor based on grade midpoint vs specialty midpoint). `mixed` falls back to the grade baseline with a 30?70% drop when no explicit range is present.
4. **Place the estimate**: intensity = clamp(score / 18.0 + bonus_for_multiple_buckets, scenario_floor, 1). Position = `0.68 - 0.45 * intensity` clamped to [0.2, 0.85]; `predicted_months = months_min + (months_max - months_min) * position`, then convert to days.
5. **Confidence**: start from profile base confidence, subtract an intensity penalty, scale by detection confidence/count, clamp to 0.25?0.96. If below `confidence_threshold`, category becomes `Uncertain` and confidence is nudged up slightly.
6. **Output**: days/months estimate + range, category/severity, bean-grade quality, confidence, defect score/percentage/counts, defect category breakdown, storage note, reduction applied vs grade midpoint, and the profile key used.

### 3.3 Table summary

| Scenario (category)          | Target range (months) | Reduction guide | Note                    |
|------------------------------|-----------------------|-----------------|-------------------------|
| No defects (specialty grade) | 24-36                 | -              | Best storage stability  |
| No defects (commodity)       | 12-24                 | -              | Faster fade             |
| Broken/cut beans             | 18-27                 | drop 10-25%    | Mild risk               |
| Insect damage                | 12-16                 | drop 30-50%    | Significant oxidation   |
| Fully black beans            | 10-14                 | drop 40-60%    | High microbial load     |
| Multiple defects mixed       | Very variable         | drop 30-70%    | Synergistic degradation |

## Key Formulas

### Defect Impact Calculation
```
Impact = weight * confidence * count
Total_Defect_Score = sum(impact_i) for all defects i
```

### Defect Percentage Normalization
```
normalized_score = min(total_defect_score / 20.0, 1.6)
defect_percentage = clamp(normalized_score * 100, 0, 100)
```

### Range Placement
```
intensity = clamp(score / 18.0 + bonus_for_multiple_buckets, floor, 1.0)
position = clamp(0.68 - 0.45 * intensity, 0.2, 0.85)
predicted_months = months_min + (months_max - months_min) * position
predicted_days = predicted_months * 30
```

### Grade Scale
```
grade_scale = midpoint(grade_baseline) / midpoint(specialty_baseline)
```

### Confidence
```
confidence = base_confidence - 0.22 * intensity
confidence *= (0.85 + 0.15 * avg_detection_confidence)
confidence = clamp(confidence, 0.25, 0.96)
```

---

## File Locations

- **Bean Classification**: `backend/ml/custom_models.py` (BeanClassifierCNN class)
- **Defect Detection**: `backend/ml/defect_detector.py` (DefectDetectionService class)
- **Shelf Life**: `backend/ml/custom_models.py` (RuleBasedShelfLife class)
- **Client Processing**: `lib/pages/scan_page.dart` (Flutter/Dart validation logic)

