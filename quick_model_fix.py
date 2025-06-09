#!/usr/bin/env python3
"""
Quick AI Model Fix Script
Simple approach to address the Band 5 bias
"""

import os
import numpy as np
import torch
import torch.nn as nn

# Add project path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.cnn_model import FrequencyHoppingCNN

def quick_model_fix():
    """Quick fix to make the model more balanced"""
    print("Quick AI Model Fix")
    print("=" * 30)
    
    model_path = 'models/frequency_hopping_model.pth'
      # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False
    
    # Load existing model
    try:
        model = FrequencyHoppingCNN(input_size=30)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded existing model")
        
        # Test current behavior
        print(f"\nTesting current model behavior:")
        test_features = np.random.randn(30)  # Random features
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(test_features).unsqueeze(0)
            probs = model(features_tensor)
            pred_band = torch.argmax(probs, dim=1).item() + 1
            confidence = torch.max(probs).item()
            
            prob_str = ", ".join([f"B{i+1}:{probs[0][i].item():.3f}" for i in range(5)])
            print(f"   Current prediction: Band {pred_band} ({confidence:.3f})")
            print(f"   Probabilities: [{prob_str}]")
        
        # Simple fix: Adjust the final layer bias to balance predictions
        print(f"\n🔄 Applying bias correction...")
        
        with torch.no_grad():
            # Get current final layer bias
            final_layer = model.fc3
            current_bias = final_layer.bias.clone()
            
            print(f"   Original bias: {current_bias.numpy()}")
            
            # Reduce bias for band 5 (index 4) and boost others
            new_bias = current_bias.clone()
            new_bias[4] -= 2.0  # Reduce Band 5 bias
            new_bias[0:4] += 0.5  # Boost other bands
            
            final_layer.bias.data = new_bias
            print(f"   Adjusted bias: {new_bias.numpy()}")
        
        # Test after fix
        print(f"\n🧪 Testing after bias correction:")
        for test_num in range(5):
            test_features = np.random.randn(30)
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(test_features).unsqueeze(0)
                probs = model(features_tensor)
                pred_band = torch.argmax(probs, dim=1).item() + 1
                confidence = torch.max(probs).item()
                
                prob_str = ", ".join([f"B{i+1}:{probs[0][i].item():.3f}" for i in range(5)])
                print(f"   Test {test_num+1}: Band {pred_band} ({confidence:.3f}) - [{prob_str}]")
        
        # Save the corrected model
        backup_path = model_path.replace('.pth', '_original.pth')
        
        # Backup original
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"\n📦 Original model backed up to: {backup_path}")
        
        # Save corrected model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': 30,
                'sequence_length': 100,
                'num_frequencies': 5
            }
        }, model_path)
        
        print(f"✅ Corrected model saved to: {model_path}")
        print(f"\n🎯 Model bias correction complete!")
        print(f"   The AI should now make more diverse frequency decisions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model fix: {str(e)}")
        return False

if __name__ == "__main__":
    success = quick_model_fix()
    if success:
        print(f"\n🚀 Ready to test the corrected AI model!")
        print(f"   Run the simulation GUI to see improved decision making")
    else:
        print(f"\n🔧 Model correction failed - may need manual intervention")
