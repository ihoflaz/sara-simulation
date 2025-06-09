# CNN Model Training Module

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.cnn_model import FrequencyHoppingCNN
from ai.data_generator import SyntheticDataGenerator
from config import *

class ModelTrainer:
    """Training pipeline for the CNN frequency hopping model"""
    def __init__(self, model: FrequencyHoppingCNN = None, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Create model with correct input size: 5 bands × 6 features = 30 features
        self.model = model if model else FrequencyHoppingCNN(
            input_size=30, sequence_length=100, num_frequencies=5
        ).to(self.device)
        self.data_generator = SyntheticDataGenerator()
        
        # Training configuration
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 15
        self.weight_decay = 1e-4
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print(f"Training device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def prepare_data(self, num_samples: int = 5000, 
                    test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """Prepare training, validation, and test datasets"""
        
        self.logger.info(f"Generating {num_samples} training samples...")
        
        # Generate synthetic data
        features, labels = self.data_generator.generate_dataset(num_samples)
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, shuffle=True
        )
        
        self.logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Create PyTorch data loaders"""
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
        
    def calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate top-1 accuracy for multi-class probability distributions"""
        # Get predicted class (argmax of model output)
        predicted_classes = torch.argmax(outputs, dim=-1)
        # Get true class (argmax of target distribution)
        true_classes = torch.argmax(targets, dim=-1)
        
        # Calculate accuracy
        correct = (predicted_classes == true_classes).float()
        return correct.mean().item()
        
    def kl_divergence_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Custom KL divergence loss for probability distributions"""
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        outputs = torch.clamp(outputs, min=eps, max=1-eps)
        targets = torch.clamp(targets, min=eps, max=1-eps)
        
        # KL divergence: sum(target * log(target / output))
        kl_div = targets * torch.log(targets / outputs)
        return torch.mean(torch.sum(kl_div, dim=-1))
        
    def train_epoch(self, train_loader, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass - process each sequence
            batch_size, seq_length, feature_dim = data.shape
            total_loss_batch = 0.0
            total_acc_batch = 0.0
            
            for t in range(seq_length):
                # Get features for current time step
                current_features = data[:, t, :]  # Shape: (batch_size, feature_dim)
                current_target = target[:, t, :]  # Shape: (batch_size, num_bands)
                
                # Forward pass
                output = self.model(current_features)
                
                # Calculate loss
                loss = criterion(output, current_target)
                total_loss_batch += loss
                
                # Calculate accuracy
                acc = self.calculate_accuracy(output, current_target)
                total_acc_batch += acc
                
            # Average over sequence length
            avg_loss = total_loss_batch / seq_length
            avg_acc = total_acc_batch / seq_length
            
            # Backward pass
            avg_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += avg_loss.item()
            total_accuracy += avg_acc
            num_batches += 1
            
            if batch_idx % 20 == 0:
                self.logger.info(f'Batch {batch_idx}/{len(train_loader)}, '
                               f'Loss: {avg_loss.item():.4f}, Acc: {avg_acc:.4f}')
                
        return total_loss / num_batches, total_accuracy / num_batches
        
    def validate_epoch(self, val_loader, criterion) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                batch_size, seq_length, feature_dim = data.shape
                total_loss_batch = 0.0
                total_acc_batch = 0.0
                
                for t in range(seq_length):
                    current_features = data[:, t, :]
                    current_target = target[:, t, :]
                    
                    output = self.model(current_features)
                    loss = criterion(output, current_target)
                    
                    total_loss_batch += loss
                    total_acc_batch += self.calculate_accuracy(output, current_target)
                    
                avg_loss = total_loss_batch / seq_length
                avg_acc = total_acc_batch / seq_length
                
                total_loss += avg_loss.item()
                total_accuracy += avg_acc
                num_batches += 1
                
        return total_loss / num_batches, total_accuracy / num_batches
        
    def train(self, num_samples: int = 5000, save_path: str = None) -> Dict:
        """Complete training pipeline"""
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(num_samples)
        train_loader, val_loader, test_loader = self.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Setup training
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            self.logger.info(f'Epoch {epoch+1}/{self.num_epochs}:')
            self.logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            self.logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if save_path:
                    self.save_model(save_path)
                    
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                self.logger.info(f'Early stopping at epoch {epoch+1}')
                break
                
        # Final evaluation on test set
        test_loss, test_acc = self.validate_epoch(test_loader, criterion)
        
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'best_val_loss': best_val_loss,
            'final_epoch': len(self.train_losses)
        }
        
        self.logger.info(f'Training completed!')
        self.logger.info(f'Best validation loss: {best_val_loss:.4f}')
        self.logger.info(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')
        
        return results
        
    def save_model(self, save_path: str):
        """Save model and training state"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'model_config': {
                'input_size': self.model.input_size,
                'sequence_length': self.model.sequence_length,
                'num_frequencies': self.model.num_frequencies
            }
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f'Model saved to {save_path}')
        
    def load_model(self, load_path: str):
        """Load model and training state"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        self.logger.info(f'Model loaded from {load_path}')
        
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def evaluate_model(self, test_loader) -> Dict:
        """Detailed model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                batch_size, seq_length, feature_dim = data.shape
                
                for t in range(seq_length):
                    current_features = data[:, t, :]
                    current_target = target[:, t, :]
                    
                    output = self.model(current_features)
                    
                    # Convert to numpy for analysis
                    pred_np = torch.argmax(output, dim=1).cpu().numpy()
                    target_np = torch.argmax(current_target, dim=1).cpu().numpy()
                    
                    all_predictions.extend(pred_np)
                    all_targets.extend(target_np)
                    
                    # Count correct predictions
                    correct_predictions += (pred_np == target_np).sum()
                    total_samples += len(pred_np)
                    
        # Calculate metrics
        accuracy = correct_predictions / total_samples
        
        # Per-class analysis
        unique_classes = np.unique(all_targets)
        class_accuracies = {}
        
        for class_id in unique_classes:
            class_mask = np.array(all_targets) == class_id
            class_predictions = np.array(all_predictions)[class_mask]
            class_targets = np.array(all_targets)[class_mask]
            
            if len(class_targets) > 0:
                class_acc = (class_predictions == class_targets).mean()
                class_accuracies[f'Band_{class_id+1}'] = class_acc
                
        evaluation_results = {
            'overall_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total_samples,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        self.logger.info(f'Evaluation Results:')
        self.logger.info(f'  Overall Accuracy: {accuracy:.4f}')
        for band, acc in class_accuracies.items():
            self.logger.info(f'  {band} Accuracy: {acc:.4f}')
            
        return evaluation_results

def main():
    """Example training script"""
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Train model
    results = trainer.train(
        num_samples=3000,
        save_path='models/frequency_hopping_model.pth'
    )
    
    # Plot training curves
    fig = trainer.plot_training_curves('models/training_curves.png')
    plt.show()
    
    # Save training log
    training_log = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'config': {
            'learning_rate': trainer.learning_rate,
            'batch_size': trainer.batch_size,
            'num_epochs': trainer.num_epochs,
            'device': trainer.device
        }
    }
    
    with open('models/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
        
    print("Training completed and results saved!")

if __name__ == "__main__":
    main()
