from predict_bean import predict_bean_type
import os

def test_predictions():
    """Test predictions on different bean types"""
    
    # Test images for each bean type
    test_images = {
        'Arabica': './data/train/images/Arabica_100_jpg.rf.4542bf586b48ce1b2a75f0f8229d7508.jpg',
        'Robusta': './data/train/images/Robusta_101_jpg.rf.280de90da7acf3c5e25b5fb80e9a1a2c.jpg'
    }
    
    print("🧪 Testing Bean Type Predictions")
    print("=" * 40)
    
    for expected_type, image_path in test_images.items():
        if os.path.exists(image_path):
            print(f"\n📸 Testing {expected_type} image...")
            result = predict_bean_type(image_path)
            
            print(f"   Expected: {expected_type}")
            print(f"   Predicted: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Correct: {'✅' if result['predicted_class'] == expected_type else '❌'}")
            
            # Show top 2 probabilities
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            print(f"   Top predictions:")
            for i, (bean_type, prob) in enumerate(sorted_probs[:2]):
                print(f"     {i+1}. {bean_type}: {prob:.2%}")
        else:
            print(f"❌ Image not found: {image_path}")

if __name__ == "__main__":
    test_predictions()
