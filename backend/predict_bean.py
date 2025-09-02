import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from ml.custom_models import BeanClassifierCNN
import os

def predict_bean_type(image_path, model_path='./models/cnn_best.pth'):
    """Predict bean type from an image"""
    
    # Load the trained model
    model = BeanClassifierCNN(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Class names
    class_names = ['Arabica', 'Robusta', 'Liberica', 'Excelsa']
    
    # Get all probabilities
    all_probs = {name: prob.item() for name, prob in zip(class_names, probabilities[0])}
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': all_probs
    }

if __name__ == "__main__":
    # Example usage
    print("🔍 Bean Type Classifier")
    print("=" * 30)
    
    # Check if model exists
    if not os.path.exists('./models/cnn_best.pth'):
        print("❌ Model not found! Please train the model first.")
        exit()
    
    # Test with a sample image from your dataset
    sample_image = './data/train/images/Arabica_100_jpg.rf.4542bf586b48ce1b2a75f0f8229d7508.jpg'
    
    if os.path.exists(sample_image):
        print(f"📸 Testing with: {sample_image}")
        result = predict_bean_type(sample_image)
        
        print(f"\n🎯 Prediction: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"\n📈 All probabilities:")
        for bean_type, prob in result['all_probabilities'].items():
            print(f"   {bean_type}: {prob:.2%}")
    else:
        print("❌ Sample image not found. Please provide a valid image path.")
        print("\n💡 Usage:")
        print("   python predict_bean.py")
        print("   # Or import and use:")
        print("   from predict_bean import predict_bean_type")
        print("   result = predict_bean_type('path/to/your/image.jpg')")
