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

### 3.1 Rule-Based Shelf Life Model

```python
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
        
        # Base shelf life by bean type (in days) -- modelled as multi-month storage
        self.base_shelf_life = {
            'Arabica': 240,   # ≈8 months
            'Robusta': 210,   # ≈7 months
            'Liberica': 225,  # ≈7.5 months
            'Excelsa': 210,   # ≈7 months
            'Other': 180      # ≈6 months fallback
        }
        
        # Severity bands (percentage ranges with peak values and scaling information)
        self.severity_bands = [
            {
                'name': 'mild',
                'percent_range': (0.0, 22.0),
                'percent_peak': 8.0,
                'month_edges': (0.60, 0.82),   # scale at range edges
                'month_peak': 0.96,            # scale near peak
                'confidence_edges': (0.78, 0.88),
                'confidence_peak': 0.95,
            },
            {
                'name': 'moderate',
                'percent_range': (18.0, 78.0),
                'percent_peak': 45.0,
                'month_edges': (0.28, 0.52),
                'month_peak': 0.62,
                'confidence_edges': (0.42, 0.64),
                'confidence_peak': 0.74,
            },
            {
                'name': 'severe',
                'percent_range': (70.0, 100.0),
                'percent_peak': 90.0,
                'month_edges': (0.08, 0.20),
                'month_peak': 0.18,
                'confidence_edges': (0.18, 0.38),
                'confidence_peak': 0.5,
            },
        ]
```

### 3.2 Shelf Life Prediction Core Algorithm

```python
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
    base_days = self.base_shelf_life.get(bean_type, self.base_shelf_life['Other'])
    base_months = base_days / 30.0
    predicted_days = base_days
    
    # Calculate defect impact
    total_defect_score = 0
    defect_counts = {}
    total_detected = 0
    cumulative_confidence = 0.0
    
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
        total_detected += count
        cumulative_confidence += confidence * count
    
    avg_detection_confidence = (cumulative_confidence / total_detected) if total_detected > 0 else 0.0
    
    # Translate raw defect score into a 0-100% indicator
    normalized_score = min(total_defect_score / 45.0, 1.5)  # allow slight spillover for severe cases
    defect_percentage = max(0.0, min(100.0, normalized_score * 100.0))
    
    band = self._select_severity_band(defect_percentage)
    severity = band['name']
    percent_low, percent_high = band['percent_range']
    span = band['span']
    severity_position = min(1.0, max(0.0, (defect_percentage - percent_low) / span))
    peak_position = band['peak_position']
    
    edge_low, edge_high = band['month_edges']
    peak_scale = band['month_peak']

    severity_scale = self._interpolate_with_peak(
        severity_position,
        peak_position,
        edge_low,
        edge_high,
        peak_scale,
    )
    predicted_months = max(0.1, base_months * severity_scale)
    predicted_days = int(predicted_months * 30)
    
    # Additional guard rails for extreme insect damage
    if defect_counts.get('insect_damage', 0) > 2:
        predicted_months = min(predicted_months, base_months * 0.2)
        predicted_days = int(predicted_months * 30)
    
    predicted_days = max(0, predicted_days)
    
    confidence = self._interpolate_with_peak(
        severity_position,
        peak_position,
        band['confidence_edges'][0],
        band['confidence_edges'][1],
        band['confidence_peak'],
    )
    
    if total_detected:
        confidence -= min(0.18, (total_detected - 1) * 0.02)
        confidence *= (0.85 + 0.15 * avg_detection_confidence)
    
    confidence = max(0.2, min(0.96, confidence))
    
    months_scale_candidates = [edge_low, edge_high, peak_scale, severity_scale]
    valid_scales = [s for s in months_scale_candidates if s is not None]
    months_min = max(0.1, base_months * min(valid_scales))
    months_max = max(months_min, base_months * max(valid_scales))
    
    # Categorise shelf life and quality grade
    if severity == "mild":
        category = "Excellent"
        quality_grade = "Grade A"
    elif severity == "moderate":
        category = "Warning" if defect_percentage > 40 else "Good"
        quality_grade = "Grade B" if defect_percentage <= 30 else "Grade C"
    else:
        category = "Critical"
        quality_grade = "Grade D"
    
    # Ensure confidence thresholding behaviour
    if confidence < confidence_threshold:
        category = "Uncertain"
        confidence = max(0.15, confidence_threshold - 0.05)
    
    return {
        'predicted_days': max(0, int(predicted_days)),
        'category': category,
        'confidence': round(min(0.96, confidence), 4),
        'raw_prediction': predicted_months * 30,
        'defect_score': round(total_defect_score, 3),
        'defect_counts': defect_counts,
        'defect_percentage': round(defect_percentage, 1),
        'average_detection_confidence': round(avg_detection_confidence, 3),
        'total_defects_detected': total_detected,
        'severity': severity,
        'severity_position': round(severity_position, 3),
        'estimated_months': round(predicted_months, 1),
        'estimated_months_range': {
            'min': round(months_min, 1),
            'max': round(months_max, 1)
        },
        'quality_grade': quality_grade,
        'base_shelf_life': base_days
    }
```

### 3.3 Severity Band Selection and Interpolation

```python
def _select_severity_band(self, percentage: float) -> dict:
    # Choose the first band containing the percentage; if none, pick the closest by range distance.
    for band in self.severity_bands:
        low, high = band['percent_range']
        if low <= percentage <= high:
            return self._prepare_band(band)

    # Fallback: choose band whose range midpoint is closest to percentage
    def distance(b):
        low, high = b['percent_range']
        midpoint = (low + high) / 2
        return abs(percentage - midpoint)

    band = min(self.severity_bands, key=distance)
    return self._prepare_band(band)

def _prepare_band(self, band: dict) -> dict:
    prepared = dict(band)
    low, high = prepared['percent_range']
    span = max(1.0, high - low)
    peak = prepared['percent_peak']
    peak_position = (peak - low) / span
    prepared['span'] = span
    prepared['peak_position'] = min(1.0, max(0.0, peak_position))
    return prepared

def _interpolate_with_peak(self, position: float, peak_position: float, edge_low: float, edge_high: float, peak_value: float) -> float:
    position = min(1.0, max(0.0, position))
    peak_position = min(1.0, max(0.0, peak_position))

    if peak_position == 0.0:
        # Avoid division by zero, fall back to edge_high trajectory
        return peak_value + (edge_high - peak_value) * position
    if peak_position == 1.0:
        return edge_low + (peak_value - edge_low) * position

    if position <= peak_position:
        t = position / peak_position
        return edge_low + (peak_value - edge_low) * t
    else:
        denom = (1.0 - peak_position)
        if denom <= 0.0:
            return peak_value
        t = (position - peak_position) / denom
        return peak_value + (edge_high - peak_value) * t
```

---

## Key Formulas

### Defect Impact Calculation
```
Impact = Weight × Confidence × Count
Total_Defect_Score = Σ(Impact_i) for all defects i
```

### Defect Percentage Normalization
```
normalized_score = min(total_defect_score / 45.0, 1.5)
defect_percentage = max(0.0, min(100.0, normalized_score * 100.0))
```

### Shelf Life Prediction
```
predicted_months = base_months × severity_scale
predicted_days = predicted_months × 30
```

### Quality Score Calculation
```
score = 1.0 - (defect_percentage_penalty) - (defect_count_penalty) - (severity_penalty)
```

---

## File Locations

- **Bean Classification**: `backend/ml/custom_models.py` (BeanClassifierCNN class)
- **Defect Detection**: `backend/ml/defect_detector.py` (DefectDetectionService class)
- **Shelf Life**: `backend/ml/custom_models.py` (RuleBasedShelfLife class)
- **Client Processing**: `lib/pages/scan_page.dart` (Flutter/Dart validation logic)

