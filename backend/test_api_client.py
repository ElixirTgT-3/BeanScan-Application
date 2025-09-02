import requests
import json

def test_prediction():
    """Test the prediction API endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/api/predict"
    
    # Image file to test
    image_path = "./data/train/images/Arabica_100_jpg.rf.4542bf586b48ce1b2a75f0f8229d7508.jpg"
    
    try:
        # Prepare the file for upload
        with open(image_path, 'rb') as f:
            files = {'image': f}
            
            print(f"📸 Testing prediction with: {image_path}")
            print(f"🌐 Sending request to: {url}")
            
            # Make the request
            response = requests.post(url, files=files)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                print("\n✅ Prediction successful!")
                print(f"🎯 Predicted class: {result['prediction']}")
                print(f"📊 Confidence: {result['confidence']:.2%}")
                print(f"\n📈 All probabilities:")
                for bean_type, prob in result['all_probabilities'].items():
                    print(f"   {bean_type}: {prob:.2%}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the API is running.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_prediction()
