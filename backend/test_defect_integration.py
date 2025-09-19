#!/usr/bin/env python3
"""
Test script to verify defect detection integration
"""
import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from ml.defect_detector import create_defect_detector

def test_defect_detector():
    """Test the defect detection service"""
    print("Testing defect detection integration...")
    
    # Check if model exists
    model_path = "models/best_model.pth"
    if not Path(model_path).exists():
        print(f"❌ Model file not found at {model_path}")
        return False
    
    try:
        # Create defect detector
        detector = create_defect_detector(model_path, device="cpu")
        print("✅ Defect detector created successfully")
        
        # Test with a sample image (if available)
        sample_images = [
            "data/val/insect_damage-1-_mp4-0344_jpg.rf.03a031c8c8c8c8c8.jpg",
            "data/val/quaker-1-_mp4-0344_jpg.rf.03a031c8c8c8c8c8.jpg",
        ]
        
        for img_path in sample_images:
            if Path(img_path).exists():
                print(f"Testing with {img_path}...")
                result = detector.detect_defects(img_path)
                print(f"  Success: {result['success']}")
                print(f"  Total defects: {result['summary']['total_defects']}")
                print(f"  Quality grade: {result['summary']['quality_grade']}")
                break
        else:
            print("⚠️  No sample images found for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing defect detector: {e}")
        return False

if __name__ == "__main__":
    success = test_defect_detector()
    sys.exit(0 if success else 1)
