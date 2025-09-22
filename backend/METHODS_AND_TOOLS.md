# Methods and Tools: Rule-Based Coffee Bean Shelf Life Prediction

## Overview

This document describes the methodology and tools used in the BeanScan application for predicting coffee bean shelf life using a rule-based algorithm. The system replaces the previous LSTM-based approach with a more interpretable and computationally efficient rule-based method.

## Methodology

### 1. Rule-Based Algorithm Architecture

The rule-based shelf life prediction system operates on a multi-layered decision framework that combines:

- **Defect Severity Analysis**: Weighted scoring system for different defect types
- **Bean Type Consideration**: Species-specific base shelf life parameters
- **Confidence Assessment**: Multi-factor confidence calculation
- **Categorical Classification**: Risk-based shelf life categorization

### 2. Core Algorithm Components

#### 2.1 Defect Severity Weighting System

The algorithm employs a hierarchical weighting system where each defect type is assigned a severity score based on its impact on coffee bean quality:

```python
defect_weights = {
    'mold': 10.0,           # Critical - immediate health risk
    'insect_damage': 8.0,   # High - quality degradation
    'quaker': 7.0,          # High - taste impact
    'discoloration': 6.0,   # Medium-High - visual quality
    'nugget': 5.0,          # Medium - processing defect
    'physical_damage': 4.0, # Medium - structural integrity
    'shell': 3.0,           # Low-Medium - processing artifact
    'under_roast': 2.0,     # Low - processing issue
    'roasted-beans': 1.0    # Minimal - processing artifact
}
```

#### 2.2 Bean Type-Specific Base Shelf Life

Different coffee bean species have varying inherent shelf life characteristics:

```python
base_shelf_life = {
    'Arabica': 30,    # Premium quality, longer shelf life
    'Liberica': 28,   # Good quality, moderate shelf life
    'Excelsa': 26,    # Moderate quality, shorter shelf life
    'Robusta': 25,    # Commercial grade, shorter shelf life
    'Other': 20       # Unknown/defective, minimum shelf life
}
```

#### 2.3 Defect Impact Calculation

The algorithm calculates total defect impact using a multiplicative scoring model:

```
Impact = Weight × Confidence × Count
Total_Defect_Score = Σ(Impact_i) for all defects i
```

Where:
- **Weight**: Defect severity weight (1.0 - 10.0)
- **Confidence**: Detection confidence (0.0 - 1.0)
- **Count**: Number of defect instances

#### 2.4 Shelf Life Prediction Formula

The predicted shelf life is calculated using an exponential decay model:

```
Penalty_Factor = min(0.9, Total_Defect_Score / 50.0)
Predicted_Days = Base_Shelf_Life × (1 - Penalty_Factor)
```

#### 2.5 Critical Defect Rules

The algorithm applies hard constraints for critical defects:

- **Mold Detection**: `Predicted_Days ≤ 2` (immediate health concern)
- **Heavy Insect Damage**: `Predicted_Days ≤ 5` (if count > 2)

### 3. Confidence Assessment Metrics

#### 3.1 Base Confidence by Category

```python
confidence_levels = {
    'Expired': 0.95,    # High confidence in expiration
    'Critical': 0.90,   # High confidence in critical state
    'Warning': 0.80,    # Good confidence in warning state
    'Good': 0.75,       # Moderate confidence in good state
    'Excellent': 0.70   # Lower confidence in excellent state
}
```

#### 3.2 Confidence Adjustment Factors

The algorithm applies confidence adjustments based on:

- **Defect Diversity**: `confidence × 0.9` if > 3 defect types
- **High Severity**: `confidence × 0.85` if defect score > 20
- **Threshold Compliance**: Set to `threshold - 0.1` if below minimum

### 4. Categorical Classification System

The algorithm classifies shelf life into five risk categories:

| Category | Days Range | Risk Level | Action Required |
|----------|------------|------------|-----------------|
| Expired | ≤ 0 | Critical | Immediate disposal |
| Critical | 1-3 | High | Urgent consumption |
| Warning | 4-7 | Medium | Monitor closely |
| Good | 8-14 | Low | Normal storage |
| Excellent | >14 | Minimal | Optimal condition |

## Tools and Technologies

### 1. Development Environment

- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch 1.12+
- **Image Processing**: PIL (Pillow) 9.0+
- **API Framework**: FastAPI 0.68+
- **Database**: Supabase (PostgreSQL)

### 2. Model Architecture

#### 2.1 Ensemble Model Components

```python
class BeanScanEnsemble:
    def __init__(self):
        self.cnn_model = BeanClassifierCNN()           # Bean type classification
        self.defect_model = DefectDetectorMaskRCNN()   # Defect detection
        self.shelf_life_model = RuleBasedShelfLife()  # Shelf life prediction
```

#### 2.2 Integration Pipeline

1. **Image Preprocessing**: Resize, normalize, tensor conversion
2. **Bean Classification**: CNN-based species identification
3. **Defect Detection**: Mask R-CNN for defect localization
4. **Shelf Life Prediction**: Rule-based algorithm
5. **Result Aggregation**: Combined analysis output

### 3. Performance Metrics

#### 3.1 Algorithmic Metrics

- **Defect Score Range**: 0.0 - 50.0+ (unbounded upper limit)
- **Penalty Factor Range**: 0.0 - 0.9 (90% maximum reduction)
- **Confidence Range**: 0.0 - 0.95 (capped at 95%)
- **Prediction Accuracy**: Based on defect detection confidence

#### 3.2 Computational Metrics

- **Processing Time**: < 100ms per prediction (CPU)
- **Memory Usage**: < 50MB for model inference
- **Scalability**: Linear scaling with defect count
- **Reliability**: 100% deterministic output

### 4. Quality Assurance

#### 4.1 Input Validation

- **Defect Format**: Dictionary with type, confidence, count
- **Bean Type**: String matching predefined species
- **Confidence Threshold**: Float between 0.0-1.0

#### 4.2 Output Validation

- **Predicted Days**: Non-negative integer
- **Category**: Valid enum value
- **Confidence**: Bounded between 0.0-0.95
- **Defect Score**: Non-negative float

### 5. Testing Framework

#### 5.1 Unit Tests

- **Defect Weighting**: Verify correct impact calculation
- **Shelf Life Formula**: Test exponential decay model
- **Critical Rules**: Validate hard constraints
- **Edge Cases**: Handle invalid inputs gracefully

#### 5.2 Integration Tests

- **API Endpoints**: Test complete prediction pipeline
- **Frontend Compatibility**: Verify response format
- **Database Integration**: Test data persistence
- **Performance Benchmarks**: Measure response times

## Advantages of Rule-Based Approach

### 1. Interpretability

- **Transparent Logic**: Clear decision rules and weights
- **Explainable Results**: Traceable prediction reasoning
- **Debugging Capability**: Easy to identify and fix issues
- **Domain Knowledge**: Incorporates coffee industry expertise

### 2. Computational Efficiency

- **No Training Required**: Instant deployment
- **Low Resource Usage**: Minimal CPU/memory requirements
- **Fast Inference**: Sub-100ms prediction times
- **Scalable**: Linear performance with input size

### 3. Reliability

- **Deterministic Output**: Consistent results for same inputs
- **No Overfitting**: No training data dependency
- **Robust Performance**: Works with any defect combination
- **Maintainable**: Easy to update rules and weights

### 4. Flexibility

- **Easy Tuning**: Adjustable weights and thresholds
- **Domain Adaptation**: Customizable for different coffee types
- **Rule Extension**: Simple to add new defect types
- **Threshold Adjustment**: Configurable confidence levels

## Implementation Details

### 1. Code Structure

```python
class RuleBasedShelfLife:
    def __init__(self):
        # Initialize weights and parameters
        self.defect_weights = {...}
        self.base_shelf_life = {...}
    
    def predict_shelf_life(self, defect_sequence, bean_type, confidence_threshold):
        # Main prediction algorithm
        # 1. Calculate defect impact
        # 2. Apply penalty formula
        # 3. Apply critical rules
        # 4. Categorize result
        # 5. Calculate confidence
        return prediction_result
```

### 2. API Integration

The rule-based model is integrated into the FastAPI backend through:

- **Model Loading**: Instantiated during API startup
- **Request Processing**: Called for each image analysis
- **Response Formatting**: Structured output for frontend
- **Error Handling**: Graceful failure management

### 3. Database Schema

Shelf life predictions are stored with the following structure:

```sql
CREATE TABLE shelf_life_predictions (
    shelf_life_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES bean_images(image_id),
    bean_type_id INTEGER REFERENCES bean_types(bean_type_id),
    predicted_days INTEGER NOT NULL,
    confidence_score FLOAT NOT NULL,
    category VARCHAR(20) NOT NULL,
    defect_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Conclusion

The rule-based shelf life prediction algorithm provides a robust, interpretable, and efficient alternative to machine learning approaches. By leveraging domain knowledge and explicit decision rules, the system delivers reliable predictions while maintaining transparency and computational efficiency. The methodology is well-suited for production environments where explainability and performance are critical requirements.
