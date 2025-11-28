"""Test script to verify CNN architecture matches training setup"""
import torch
from torchvision.models import mobilenet_v3_small
from ml.custom_models import BeanClassifierCNN

def test_backbone_output():
    """Test if MobileNetV3 backbone outputs the expected feature dimensions"""
    print("=" * 60)
    print("Testing MobileNetV3 Backbone Output")
    print("=" * 60)
    
    # Create a MobileNetV3 small model
    model = mobilenet_v3_small(pretrained=False)
    
    # Count layers
    num_layers = len(list(model.features))
    print(f"Total feature layers in MobileNetV3 small: {num_layers}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    features = []
    
    for i, layer in enumerate(model.features):
        x = layer(x)
        if i in [2, 4, 6, 8, 10, 12]:
            features.append(x)
            print(f"Layer {i}: shape = {x.shape}, channels = {x.shape[1]}")
    
    print(f"\nTotal features extracted: {len(features)}")
    if features:
        last_feature = features[-1]
        print(f"Last feature shape: {last_feature.shape}")
        print(f"Last feature channels: {last_feature.shape[1]}")
        print(f"Expected channels: 576")
        print(f"Match: {'YES' if last_feature.shape[1] == 576 else 'NO'}")
    
    return features[-1].shape[1] if features else None

def test_classifier_input():
    """Test if the classifier expects the correct input size"""
    print("\n" + "=" * 60)
    print("Testing Classifier Input/Output")
    print("=" * 60)
    
    model = BeanClassifierCNN(num_classes=4, pretrained=False)
    
    # Check classifier first layer
    first_linear = None
    for module in model.classifier:
        if isinstance(module, torch.nn.Linear):
            first_linear = module
            break
    
    if first_linear:
        print(f"First Linear layer input features: {first_linear.in_features}")
        print(f"Expected input features: 576")
        print(f"Match: {'YES' if first_linear.in_features == 576 else 'NO'}")
        return first_linear.in_features
    
    return None

def test_full_forward_pass():
    """Test the full forward pass"""
    print("\n" + "=" * 60)
    print("Testing Full Forward Pass")
    print("=" * 60)
    
    model = BeanClassifierCNN(num_classes=4, pretrained=False)
    model.eval()
    
    # Create test input (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)
    
    try:
        with torch.no_grad():
            output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: (2, 4)")
        print(f"Match: {'YES' if output.shape == (2, 4) else 'NO'}")
        
        # Check if output is valid (not NaN or Inf)
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("ERROR: Output contains NaN or Inf values!")
        else:
            print("Output values are valid")
        
        return True
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_compatibility():
    """Test if the model is compatible with training setup"""
    print("\n" + "=" * 60)
    print("Testing Training Compatibility")
    print("=" * 60)
    
    model = BeanClassifierCNN(num_classes=4, pretrained=False)
    model.train()
    
    # Simulate training step
    x = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 4, (2,))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # Forward pass
        outputs = model(x)
        print(f"Forward pass successful: output shape = {outputs.shape}")
        
        # Loss calculation
        loss = criterion(outputs, labels)
        print(f"Loss calculation successful: loss = {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        print("Backward pass successful")
        
        # Optimizer step
        optimizer.step()
        print("Optimizer step successful")
        
        print("Training compatibility: All checks passed!")
        return True
    except Exception as e:
        print(f"ERROR in training compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nCNN Architecture Verification Test\n")
    
    # Run all tests
    backbone_channels = test_backbone_output()
    classifier_input = test_classifier_input()
    forward_ok = test_full_forward_pass()
    training_ok = test_training_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    issues = []
    if backbone_channels != 576:
        issues.append(f"Backbone outputs {backbone_channels} channels, expected 576")
    if classifier_input != 576:
        issues.append(f"Classifier expects {classifier_input} input features, expected 576")
    if not forward_ok:
        issues.append("Forward pass failed")
    if not training_ok:
        issues.append("Training compatibility check failed")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nWARNING: The model architecture may not match the training setup!")
    else:
        print("All checks passed! The model architecture matches the training setup.")
    
    print()


