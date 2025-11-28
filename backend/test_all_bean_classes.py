"""
Test script to evaluate the bean classifier model on unseen data for all 4 bean classes.
This script processes all images in assets/arabica/, assets/robusta/, assets/liberica/, and assets/excelsa/
and generates a detailed report with per-class metrics and confusion matrix.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from ml.custom_models import BeanClassifierCNN
import json
from datetime import datetime
import numpy as np

# Add parent directory to path to access assets
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / 'assets'
MODEL_PATH = Path(__file__).parent / 'models' / 'cnn_best.pth'

CLASS_NAMES = ['Arabica', 'Robusta', 'Liberica', 'Excelsa']
BEAN_DIRECTORIES = {
    'Arabica': ASSETS_DIR / 'arabica',
    'Robusta': ASSETS_DIR / 'robusta',
    'Liberica': ASSETS_DIR / 'liberica',
    'Excelsa': ASSETS_DIR / 'excelsa'
}


def load_model(model_path):
    """Load the trained bean classifier model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"📦 Loading model from {model_path}...")
    model = BeanClassifierCNN(num_classes=4)
    
    # Try loading with strict=False to handle architecture mismatches
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully (with some mismatched keys ignored)!")
    except Exception as e:
        print(f"⚠️  Warning: Error loading model: {e}")
        print("   Attempting to load only matching keys...")
        # Try loading only matching keys
        model_dict = model.state_dict()
        state_dict = torch.load(model_path, map_location='cpu')
        # Filter to only matching keys
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"✅ Loaded {len(filtered_dict)} matching keys out of {len(state_dict)} total keys")
    
    model.eval()
    return model


def predict_image(model, image_path, transform):
    """Predict bean type for a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        all_probs = {
            name: prob.item() 
            for name, prob in zip(CLASS_NAMES, probabilities[0])
        }
        
        return {
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'predicted_class': None,
            'predicted_class_idx': None,
            'confidence': 0.0,
            'all_probabilities': {},
            'success': False,
            'error': str(e)
        }


def test_all_classes():
    """Test all bean classes and generate a comprehensive report"""
    
    # Check if assets directory exists
    if not ASSETS_DIR.exists():
        print(f"❌ Error: Assets directory not found at {ASSETS_DIR}")
        return
    
    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect all images by class
    all_results = []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    print("\n" + "=" * 80)
    print("🔍 Collecting images from all bean class directories...")
    print("=" * 80)
    
    for expected_class, class_dir in BEAN_DIRECTORIES.items():
        if not class_dir.exists():
            print(f"⚠️  Warning: Directory not found: {class_dir}")
            continue
        
        # Get all image files
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        image_files = sorted(image_files)
        
        print(f"\n📁 {expected_class}: Found {len(image_files)} images in {class_dir.name}/")
        
        if not image_files:
            print(f"   ⚠️  No images found in {class_dir}")
            continue
        
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            if idx % 50 == 0 or idx == len(image_files):
                print(f"   Processing: {idx}/{len(image_files)}...", end='\r')
            
            result = predict_image(model, image_path, transform)
            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            result['expected_class'] = expected_class
            result['expected_class_idx'] = class_to_idx[expected_class]
            result['is_correct'] = result['success'] and result['predicted_class'] == expected_class
            
            all_results.append(result)
        
        print(f"   ✅ Processed {len(image_files)} images")
    
    print("\n" + "=" * 80)
    print("📊 GENERATING TEST RESULTS")
    print("=" * 80)
    
    # Calculate statistics
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    # Per-class statistics
    per_class_stats = {}
    confusion_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    
    for class_name in CLASS_NAMES:
        class_results = [r for r in successful_results if r['expected_class'] == class_name]
        correct = [r for r in class_results if r['is_correct']]
        
        per_class_stats[class_name] = {
            'total': len(class_results),
            'correct': len(correct),
            'accuracy': (len(correct) / len(class_results) * 100) if class_results else 0,
            'avg_confidence': np.mean([r['confidence'] for r in correct]) if correct else 0,
            'predictions': defaultdict(int),
            'confidence_distribution': {
                'high': len([r for r in class_results if r['confidence'] >= 0.8]),
                'medium': len([r for r in class_results if 0.5 <= r['confidence'] < 0.8]),
                'low': len([r for r in class_results if r['confidence'] < 0.5])
            }
        }
        
        # Count predictions for this class
        for result in class_results:
            predicted = result['predicted_class']
            per_class_stats[class_name]['predictions'][predicted] += 1
            
            # Update confusion matrix
            expected_idx = class_to_idx[class_name]
            predicted_idx = result['predicted_class_idx']
            confusion_matrix[expected_idx][predicted_idx] += 1
    
    # Overall statistics
    total_images = len(all_results)
    total_successful = len(successful_results)
    total_correct = len([r for r in successful_results if r['is_correct']])
    overall_accuracy = (total_correct / total_successful * 100) if total_successful > 0 else 0
    avg_confidence = np.mean([r['confidence'] for r in successful_results]) if successful_results else 0
    
    # Print summary
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total images tested: {total_images}")
    print(f"   Successfully processed: {total_successful}")
    print(f"   Failed to process: {len(failed_results)}")
    print(f"   Correct predictions: {total_correct}")
    print(f"   Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"   Average Confidence: {avg_confidence:.2%}")
    
    # Print per-class statistics
    print(f"\n📊 PER-CLASS STATISTICS:")
    print("-" * 80)
    for class_name in CLASS_NAMES:
        stats = per_class_stats[class_name]
        print(f"\n{class_name}:")
        print(f"   Total images: {stats['total']}")
        print(f"   Correct predictions: {stats['correct']}")
        print(f"   Accuracy: {stats['accuracy']:.2f}%")
        print(f"   Average confidence (correct): {stats['avg_confidence']:.2%}")
        print(f"   Prediction distribution:")
        for pred_class, count in sorted(stats['predictions'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            marker = "✅" if pred_class == class_name else "❌"
            print(f"      {marker} {pred_class}: {count} ({percentage:.1f}%)")
        print(f"   Confidence distribution:")
        print(f"      High (≥80%): {stats['confidence_distribution']['high']}")
        print(f"      Medium (50-80%): {stats['confidence_distribution']['medium']}")
        print(f"      Low (<50%): {stats['confidence_distribution']['low']}")
    
    # Print confusion matrix
    print(f"\n📋 CONFUSION MATRIX:")
    print("   (Rows = Expected, Columns = Predicted)")
    print("-" * 80)
    header = " " * 12 + " | " + " | ".join([f"{name:>10}" for name in CLASS_NAMES])
    print(header)
    print("-" * len(header))
    for i, expected_name in enumerate(CLASS_NAMES):
        row = f"{expected_name:>10} | " + " | ".join([f"{confusion_matrix[i][j]:>10}" for j in range(len(CLASS_NAMES))])
        print(row)
    
    # Calculate precision, recall, F1 for each class
    print(f"\n📉 PER-CLASS METRICS (Precision, Recall, F1):")
    print("-" * 80)
    for i, class_name in enumerate(CLASS_NAMES):
        tp = confusion_matrix[i][i]  # True positives
        fp = sum(confusion_matrix[j][i] for j in range(len(CLASS_NAMES)) if j != i)  # False positives
        fn = sum(confusion_matrix[i][j] for j in range(len(CLASS_NAMES)) if j != i)  # False negatives
        
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        print(f"{class_name}:")
        print(f"   Precision: {precision:.2%} (TP: {tp}, FP: {fp})")
        print(f"   Recall:    {recall:.2%} (TP: {tp}, FN: {fn})")
        print(f"   F1-Score:  {f1:.2%}")
    
    # Errors
    if failed_results:
        print(f"\n❌ ERRORS ({len(failed_results)}):")
        error_summary = defaultdict(int)
        for err in failed_results:
            error_summary[err['error']] += 1
        for error_msg, count in error_summary.items():
            print(f"   {error_msg}: {count} images")
        if len(failed_results) <= 10:
            print("\n   Failed images:")
            for err in failed_results:
                print(f"      {err['image_name']}")
    
    # Save detailed results to JSON
    report_path = PROJECT_ROOT / 'backend' / 'test_results_all_classes.json'
    report_data = {
        'test_date': datetime.now().isoformat(),
        'model_path': str(MODEL_PATH),
        'test_directories': {k: str(v) for k, v in BEAN_DIRECTORIES.items()},
        'summary': {
            'total_images': total_images,
            'successful_predictions': total_successful,
            'failed_predictions': len(failed_results),
            'correct_predictions': total_correct,
            'overall_accuracy_percentage': overall_accuracy,
            'average_confidence': float(avg_confidence),
        },
        'per_class_statistics': {
            k: {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['accuracy'],
                'average_confidence': float(v['avg_confidence']),
                'prediction_distribution': dict(v['predictions']),
                'confidence_distribution': v['confidence_distribution']
            }
            for k, v in per_class_stats.items()
        },
        'confusion_matrix': confusion_matrix.tolist(),
        'confusion_matrix_labels': CLASS_NAMES,
        'detailed_results': [
            {
                'image_name': r['image_name'],
                'expected_class': r['expected_class'],
                'predicted_class': r['predicted_class'],
                'confidence': float(r['confidence']) if r['success'] else None,
                'is_correct': r['is_correct'] if r['success'] else False,
                'error': r['error'] if not r['success'] else None
            }
            for r in all_results
        ],
        'errors': [
            {
                'image': r['image_name'],
                'error': r['error']
            }
            for r in failed_results
        ]
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {report_path}")
    print("=" * 80)
    
    return report_data


if __name__ == "__main__":
    print("🧪 Bean Classifier - Unseen Data Test (All 4 Classes)")
    print("Testing model on Arabica, Robusta, Liberica, and Excelsa images")
    print("=" * 80)
    
    test_all_classes()

