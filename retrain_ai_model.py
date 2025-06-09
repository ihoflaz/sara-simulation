#!/usr/bin/env python3
"""
AI Model Retraining Script
Addresses the Band 5 bias issue found in authenticity testing
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.cnn_model import FrequencyHoppingCNN
from ai.data_generator import SyntheticDataGenerator
from ai.training import ModelTrainer
from config import *

class BalancedModelRetrainer:
    """Retrain the model with balanced data to fix Band 5 bias"""
    
    def __init__(self):
        self.model_path = 'models/frequency_hopping_model.pth'
        self.backup_path = 'models/frequency_hopping_model_backup.pth'
        self.new_model_path = 'models/frequency_hopping_model_fixed.pth'
        
    def backup_existing_model(self):
        """Backup the existing model before retraining"""
        if os.path.exists(self.model_path):
            import shutil
            shutil.copy2(self.model_path, self.backup_path)
            print(f"📦 Backed up existing model to {self.backup_path}")
        
    def generate_balanced_training_data(self, num_samples: int = 10000):
        """Generate balanced training data with equal representation of all bands"""
        print(f"🔄 Generating {num_samples} balanced training samples...")
        
        data_generator = SyntheticDataGenerator()
        
        # Ensure equal representation of each band as optimal choice
        samples_per_band = num_samples // 5
        all_features = []
        all_labels = []
        
        for target_band in range(1, 6):  # Bands 1-5
            print(f"   Generating {samples_per_band} samples favoring Band {target_band}...")
            
            for _ in range(samples_per_band):
                # Generate features for all 5 bands
                band_features = []
                
                for band in range(1, 6):
                    if band == target_band:
                        # Make target band optimal
                        snr = np.random.normal(30.0, 3.0)  # High SNR
                        interference = np.random.normal(0.5, 0.2)  # Low interference
                        rss = np.random.normal(-55.0, 3.0)  # Strong signal
                    else:
                        # Make other bands less optimal
                        snr = np.random.normal(15.0, 5.0)  # Lower SNR
                        interference = np.random.normal(5.0, 2.0)  # Higher interference
                        rss = np.random.normal(-75.0, 5.0)  # Weaker signal
                    
                    # Ensure realistic bounds
                    snr = max(0, snr)
                    interference = max(0.1, interference)
                    rss = max(-100, min(-30, rss))
                    
                    features = [
                        snr,           # SNR
                        interference,  # Interference
                        rss,          # RSS
                        20.0,         # Coherence time
                        5.0,          # Doppler spread
                        0.1           # Delay spread
                    ]
                    
                    band_features.extend(features)
                
                # Create one-hot label for target band
                label = np.zeros(5)
                label[target_band - 1] = 1.0
                
                all_features.append(band_features)
                all_labels.append(label)
        
        # Shuffle the data
        indices = np.random.permutation(len(all_features))
        all_features = np.array(all_features)[indices]
        all_labels = np.array(all_labels)[indices]
        
        print(f"✅ Generated {len(all_features)} balanced training samples")
        
        # Verify label distribution
        label_counts = np.sum(all_labels, axis=0)
        print(f"📊 Label distribution: {dict(zip([f'Band_{i+1}' for i in range(5)], label_counts))}")
        
        return all_features, all_labels
    
    def train_balanced_model(self, X_train, y_train, epochs: int = 50):
        """Train a new model with balanced data"""
        print(f"🧠 Training new balanced model...")
        
        # Create model
        model = FrequencyHoppingCNN(input_size=30, sequence_length=1, num_frequencies=5)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add sequence dimension
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Add sequence dimension
        
        # Split into train/validation
        split_idx = int(0.8 * len(X_tensor))
        X_train_split = X_tensor[:split_idx]
        y_train_split = y_tensor[:split_idx]
        X_val_split = X_tensor[split_idx:]
        y_val_split = y_tensor[split_idx:]
        
        print(f"   Training samples: {len(X_train_split)}")
        print(f"   Validation samples: {len(X_val_split)}")
        
        # Create trainer
        trainer = ModelTrainer(model=model)
        trainer.num_epochs = epochs
        trainer.learning_rate = 0.001
        trainer.batch_size = 32
        
        # Train the model
        results = trainer.train(
            X_train_split, y_train_split, 
            X_val_split, y_val_split,
            save_path=self.new_model_path
        )
        
        return model, results
    
    def test_fixed_model(self, model):
        """Test the fixed model to ensure balanced predictions"""
        print(f"🧪 Testing fixed model...")
        
        model.eval()
        
        # Test various scenarios
        test_scenarios = {
            'excellent_band_1': [35.0, 0.5, -55.0],  # Band 1 excellent
            'excellent_band_2': [32.0, 0.3, -58.0],  # Band 2 excellent  
            'excellent_band_3': [34.0, 0.4, -56.0],  # Band 3 excellent
            'excellent_band_4': [33.0, 0.6, -57.0],  # Band 4 excellent
            'excellent_band_5': [36.0, 0.2, -54.0],  # Band 5 excellent
        }
        
        predictions = {}
        
        for scenario, [snr, interference, rss] in test_scenarios.items():
            # Create features for all bands with target band being optimal
            features = []
            
            for band in range(1, 6):
                if f'band_{band}' in scenario:
                    # This is the optimal band
                    band_features = [snr, interference, rss, 20.0, 5.0, 0.1]
                else:
                    # Other bands are less optimal
                    band_features = [
                        snr - 10 + np.random.normal(0, 2),  # Lower SNR
                        interference + 3 + np.random.normal(0, 1),  # Higher interference
                        rss - 10 + np.random.normal(0, 3),  # Weaker signal
                        20.0, 5.0, 0.1
                    ]
                
                features.extend(band_features)
            
            # Make prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                probabilities = model(features_tensor)
                predicted_band = torch.argmax(probabilities, dim=1).item() + 1
                confidence = torch.max(probabilities).item()
                
                predictions[scenario] = {
                    'predicted_band': predicted_band,
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy()
                }
                
                prob_str = ", ".join([f"B{i+1}:{probabilities[0][i].item():.3f}" for i in range(5)])
                print(f"   {scenario}: Band {predicted_band} ({confidence:.3f}) - [{prob_str}]")
        
        # Check if model predictions are diverse
        predicted_bands = [pred['predicted_band'] for pred in predictions.values()]
        unique_predictions = len(set(predicted_bands))
        
        print(f"\n📈 Model diversity:")
        print(f"   Unique predictions: {unique_predictions}/5")
        print(f"   Band distribution: {dict(zip(*np.unique(predicted_bands, return_counts=True)))}")
        
        if unique_predictions >= 4:
            print("✅ Model shows good diversity in predictions")
            return True
        else:
            print("⚠️  Model still shows limited diversity")
            return False
    
    def run_retraining(self):
        """Run the complete retraining process"""
        print("🔄 AI Model Retraining Process")
        print("=" * 40)
        
        # Backup existing model
        self.backup_existing_model()
        
        # Generate balanced training data
        X_train, y_train = self.generate_balanced_training_data(num_samples=15000)
        
        # Train balanced model
        model, results = self.train_balanced_model(X_train, y_train, epochs=100)
        
        # Test the fixed model
        is_fixed = self.test_fixed_model(model)
        
        if is_fixed:
            print(f"\n✅ Model retraining successful!")
            print(f"🔄 Replacing original model with fixed version...")
            
            # Replace the original model
            import shutil
            if os.path.exists(self.new_model_path):
                shutil.copy2(self.new_model_path, self.model_path)
                print(f"✅ Updated {self.model_path} with balanced model")
            
            print(f"\n🎯 RETRAINING COMPLETE:")
            print(f"   Original model backed up to: {self.backup_path}")
            print(f"   New balanced model saved to: {self.model_path}")
            print(f"   Training accuracy: {results.get('val_accuracies', [0])[-1]:.4f}")
            
        else:
            print(f"\n⚠️  Model retraining needs improvement")
            print(f"   Original model preserved at: {self.model_path}")
            print(f"   New model available at: {self.new_model_path}")
            print(f"   Backup available at: {self.backup_path}")
        
        return is_fixed

def main():
    """Main retraining function"""
    retrainer = BalancedModelRetrainer()
    success = retrainer.run_retraining()
    
    if success:
        print(f"\n🚀 Ready to test the updated AI model in the GUI!")
        print(f"   Run the simulation to see improved AI decision making")
    else:
        print(f"\n🔧 Consider adjusting training parameters or data generation")

if __name__ == "__main__":
    main()
