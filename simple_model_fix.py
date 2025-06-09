# Simple AI Model Fix Script
import os
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.cnn_model import FrequencyHoppingCNN

def main():
    print("AI Model Fix Starting...")
    
    model_path = 'models/frequency_hopping_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    try:
        # Load model
        model = FrequencyHoppingCNN(input_size=30)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
        
        # Test current predictions
        print("\nTesting current model...")
        for i in range(3):
            test_features = np.random.randn(30)
            with torch.no_grad():
                features_tensor = torch.FloatTensor(test_features).unsqueeze(0)
                probs = model(features_tensor)
                pred_band = torch.argmax(probs, dim=1).item() + 1
                print(f"Test {i+1}: Band {pred_band}, Probs: {probs[0].numpy()}")
        
        # Apply bias correction
        print("\nApplying bias correction...")
        with torch.no_grad():
            final_layer = model.fc3
            current_bias = final_layer.bias.clone()
            print(f"Original bias: {current_bias.numpy()}")
            
            # Adjust bias to make predictions more balanced
            new_bias = current_bias.clone()
            new_bias[4] -= 3.0  # Reduce Band 5 preference significantly
            new_bias[0:4] += 0.75  # Boost other bands
            
            final_layer.bias.data = new_bias
            print(f"New bias: {new_bias.numpy()}")
        
        # Test after correction
        print("\nTesting after correction...")
        predictions = []
        for i in range(10):
            test_features = np.random.randn(30)
            with torch.no_grad():
                features_tensor = torch.FloatTensor(test_features).unsqueeze(0)
                probs = model(features_tensor)
                pred_band = torch.argmax(probs, dim=1).item() + 1
                predictions.append(pred_band)
                if i < 5:  # Show first 5 tests
                    print(f"Test {i+1}: Band {pred_band}, Probs: {probs[0].numpy()}")
        
        # Check diversity
        unique_predictions = len(set(predictions))
        print(f"\nDiversity check: {unique_predictions} unique predictions out of 10 tests")
        print(f"Predictions: {predictions}")
        
        if unique_predictions >= 3:
            print("Good diversity achieved! Saving corrected model...")
            
            # Backup original
            backup_path = model_path.replace('.pth', '_original.pth')
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"Original backed up to: {backup_path}")
            
            # Save corrected model
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': 30,
                    'sequence_length': 100,
                    'num_frequencies': 5
                }
            }, model_path)
            
            print(f"Corrected model saved to: {model_path}")
            print("Model fix complete!")
        else:
            print("Correction not sufficient - more work needed")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
