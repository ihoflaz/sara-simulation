"""
CNN Model Architecture for Frequency Selection
Implements the AI model for intelligent frequency hopping
"""

import numpy as np
from keras_compat import keras, layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import os

# Import TensorFlow through keras_compat for consistency
try:
    import tensorflow as tf
except ImportError:
    tf = None

from config import CNN_CONFIG, FREQUENCY_BANDS

class CNNFrequencyModel:
    """CNN model for frequency band selection"""
    
    def __init__(self, input_shape: Tuple = None):
        if input_shape is None:
            input_shape = CNN_CONFIG['input_shape']
        
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = None
        
        # Model architecture parameters
        self.conv_filters = [32, 64, 128, 256]
        self.dense_units = CNN_CONFIG['hidden_layers']
        self.dropout_rate = 0.3
        self.learning_rate = CNN_CONFIG['learning_rate']
    
    def build_advanced_cnn(self) -> keras.Model:
        """Build advanced CNN architecture"""
        inputs = keras.Input(shape=self.input_shape, name='spectrum_input')
        
        # Convolutional backbone
        x = layers.Conv1D(
            filters=self.conv_filters[0],
            kernel_size=7,
            strides=1,
            padding='same',
            activation='relu',
            name='conv1d_1'
        )(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool_1')(x)
        
        x = layers.Conv1D(
            filters=self.conv_filters[1],
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu',
            name='conv1d_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool_2')(x)
        
        x = layers.Conv1D(
            filters=self.conv_filters[2],
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            name='conv1d_3'
        )(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        
        x = layers.Conv1D(
            filters=self.conv_filters[3],
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            name='conv1d_4'
        )(x)
        x = layers.BatchNormalization(name='bn_4')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers with attention mechanism
        x = layers.Dense(
            self.dense_units[0],
            activation='relu',
            name='dense_1'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(
            self.dense_units[1],
            activation='relu',
            name='dense_2'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        x = layers.Dense(
            self.dense_units[2],
            activation='relu',
            name='dense_3'
        )(x)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_3')(x)
        
        # Output layer - softmax for probability distribution
        outputs = layers.Dense(
            5,  # 5 frequency bands
            activation='softmax',
            name='frequency_probabilities'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='FrequencySelectionCNN')
        
        return model
    
    def compile_model(self, model: keras.Model):
        """Compile the model with appropriate loss and metrics"""
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Custom metrics
        def top_2_accuracy(y_true, y_pred):
            return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                'top_k_categorical_accuracy',
                top_2_accuracy
            ]
        )
        
        return model
    
    def create_model(self) -> keras.Model:
        """Create and compile the complete model"""
        self.model = self.build_advanced_cnn()
        self.model = self.compile_model(self.model)
        
        print("CNN Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def prepare_callbacks(self, model_save_path: str = 'best_model.h5') -> List:
        """Prepare training callbacks"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Learning rate scheduler
            callbacks.LearningRateScheduler(
                lambda epoch, lr: lr * 0.95 if epoch > 10 else lr,
                verbose=0
            )
        ]
        
        return callbacks_list
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = None, batch_size: int = None,
                   model_save_path: str = 'frequency_model.h5') -> Dict:
        """Train the CNN model"""
        
        if self.model is None:
            self.create_model()
        
        if epochs is None:
            epochs = CNN_CONFIG['epochs']
        if batch_size is None:
            batch_size = CNN_CONFIG['batch_size']
        
        # Prepare data
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # Prepare callbacks
        callbacks_list = self.prepare_callbacks(model_save_path)
        
        print(f"Training CNN model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1,
            shuffle=True
        )
        
        self.training_history = history.history
        
        # Evaluate final model
        val_loss, val_acc, val_top_k, val_top_2 = self.model.evaluate(
            X_val_scaled, y_val, verbose=0
        )
        
        print(f"\nFinal Validation Results:")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"Top-K Accuracy: {val_top_k:.4f}")
        print(f"Top-2 Accuracy: {val_top_2:.4f}")
        print(f"Loss: {val_loss:.4f}")
        
        return self.training_history
    
    def predict_frequency_band(self, spectrum_data: np.ndarray,
                             return_probabilities: bool = True) -> Tuple:
        """Predict optimal frequency band"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess input
        if len(spectrum_data.shape) == 1:
            spectrum_data = spectrum_data.reshape(1, -1, 1)
        elif len(spectrum_data.shape) == 2:
            spectrum_data = spectrum_data.reshape(spectrum_data.shape[0], -1, 1)
        
        # Scale data
        original_shape = spectrum_data.shape
        scaled_data = self.scaler.transform(
            spectrum_data.reshape(-1, spectrum_data.shape[-1])
        ).reshape(original_shape)
        
        # Predict
        predictions = self.model.predict(scaled_data, verbose=0)
        
        if return_probabilities:
            return predictions
        else:
            predicted_bands = np.argmax(predictions, axis=1) + 1  # Convert to 1-indexed
            return predicted_bands
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Scale test data
        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)
        
        # Get predictions
        predictions = self.model.predict(X_test_scaled, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        accuracy = accuracy_score(true_classes, predicted_classes)
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        class_report = classification_report(
            true_classes, predicted_classes,
            target_names=[f'Band_{i+1}' for i in range(5)],
            output_dict=True
        )
        
        # Model evaluation
        test_loss, test_acc, test_top_k, test_top_2 = self.model.evaluate(
            X_test_scaled, y_test, verbose=0
        )
        
        evaluation_results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'top_k_accuracy': test_top_k,
            'top_2_accuracy': test_top_2,
            'sklearn_accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'per_class_accuracy': {
                f'Band_{i+1}': class_report[f'Band_{i+1}']['f1-score']
                for i in range(5)
            }
        }
        
        return evaluation_results
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if self.training_history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.training_history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.training_history['loss'], label='Training Loss')
        axes[0, 1].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-K Accuracy plot
        if 'top_k_categorical_accuracy' in self.training_history:
            axes[1, 0].plot(self.training_history['top_k_categorical_accuracy'], 
                           label='Training Top-K Accuracy')
            axes[1, 0].plot(self.training_history['val_top_k_categorical_accuracy'], 
                           label='Validation Top-K Accuracy')
            axes[1, 0].set_title('Top-K Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-K Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate plot
        if 'lr' in self.training_history:
            axes[1, 1].plot(self.training_history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(filepath)
        
        # Save scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to: {filepath}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model
        self.model = keras.models.load_model(filepath)
        
        # Load scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print("Warning: Scaler not found, using default scaler")
            self.scaler = StandardScaler()
        
        print(f"Model loaded from: {filepath}")
    
    def get_model_summary(self) -> Dict:
        """Get model summary information"""
        if self.model is None:
            return {"error": "No model available"}
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([
            tf.keras.backend.count_params(w) for w in self.model.trainable_weights
        ])
        
        summary_info = {
            'model_name': self.model.name,
            'input_shape': self.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers_count': len(self.model.layers),
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size in MB
        }
        
        return summary_info

# PyTorch CNN Model for Frequency Hopping
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyHoppingCNN(nn.Module):
    """PyTorch CNN model for frequency selection"""
    
    def __init__(self, input_size=10, sequence_length=100, num_frequencies=5):
        super(FrequencyHoppingCNN, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.num_frequencies = num_frequencies
          # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global pooling and dense layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_frequencies)
    
    def forward(self, x):
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            # Input shape: (batch_size, features) - add sequence dimension
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, features)
        
        # Input shape should now be: (batch_size, sequence_length, features)
        # Transpose for Conv1d: (batch_size, features, sequence_length)
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Set to eval mode for single sample inference
        was_training = self.training
        if x.size(0) == 1:
            self.eval()
        
        # Convolutional layers with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dense layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Restore training mode if needed
        if was_training:
            self.train()
          # Softmax for probability distribution
        return F.softmax(x, dim=1)
    
    def predict_frequency(self, features):
        """Predict best frequency band"""
        self.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            
            if len(features.shape) == 2:
                features = features.unsqueeze(0)  # Add batch dimension
                
            probabilities = self.forward(features)
            return probabilities.argmax(dim=1).cpu().numpy()

# Keep the original TensorFlow model as well for compatibility
