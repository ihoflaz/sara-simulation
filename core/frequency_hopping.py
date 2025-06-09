"""
CNN-based Frequency Hopping Algorithm
AI-driven frequency selection for jammer avoidance
"""

import numpy as np
from keras_compat import keras, layers, models, optimizers, callbacks
from typing import Tuple, List, Dict, Optional
import pickle
import os

# Import TensorFlow through keras_compat for better error handling
try:
    import tensorflow as tf
except ImportError:
    # If TensorFlow is not available, we'll use the fallback
    tf = None

from config import CNN_CONFIG, FREQUENCY_BANDS, COMPETITION_CONFIG

class FrequencyHoppingCNN:
    """CNN model for intelligent frequency selection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.input_shape = CNN_CONFIG['input_shape']
        self.hidden_layers = CNN_CONFIG['hidden_layers']
        self.output_size = CNN_CONFIG['output_size']
        self.learning_rate = CNN_CONFIG['learning_rate']
        
        self.model = None
        self.training_history = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build CNN architecture for frequency selection"""
        self.model = keras.Sequential([
            # Input layer for spectrum analysis
            layers.Input(shape=self.input_shape),
            
            # Convolutional layers for feature extraction
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            # Global average pooling
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(self.hidden_layers[0], activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.hidden_layers[1], activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.hidden_layers[2], activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer - probability distribution over frequency bands
            layers.Dense(self.output_size, activation='softmax')
        ])
          # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("CNN model built successfully")
        print(f"Model summary:")
        self.model.summary()
    
    def preprocess_spectrum_data(self, spectrum_data: np.ndarray) -> np.ndarray:
        """Preprocess spectrum data for CNN input"""
        # Normalize spectrum data
        normalized = (spectrum_data - np.mean(spectrum_data)) / (np.std(spectrum_data) + 1e-8)
        
        # Reshape for CNN input
        if len(normalized.shape) == 1:
            normalized = normalized.reshape(-1, 1)
        
        # Ensure correct input shape
        if normalized.shape[0] != self.input_shape[0]:
            # Interpolate or pad to correct size
            from scipy import interpolate
            x_old = np.linspace(0, 1, len(normalized))
            x_new = np.linspace(0, 1, self.input_shape[0])
            f = interpolate.interp1d(x_old, normalized.flatten(), kind='linear')
            normalized = f(x_new).reshape(-1, 1)
        
        return normalized
    
    def predict_best_frequency(self, spectrum_data: np.ndarray, 
                             jammed_bands: List[int] = []) -> Tuple[int, float, np.ndarray]:
        """
        Predict best frequency band to use
        Returns: (best_band, confidence, all_probabilities)
        """
        # Preprocess input
        processed_input = self.preprocess_spectrum_data(spectrum_data)
        input_batch = np.expand_dims(processed_input, axis=0)
        
        # Get model predictions
        predictions = self.model.predict(input_batch, verbose=0)[0]
        
        # Apply jammer avoidance - zero out jammed bands
        filtered_predictions = predictions.copy()
        for jammed_band in jammed_bands:
            if 1 <= jammed_band <= 5:
                filtered_predictions[jammed_band - 1] = 0
        
        # Renormalize probabilities
        if np.sum(filtered_predictions) > 0:
            filtered_predictions = filtered_predictions / np.sum(filtered_predictions)
        else:
            # If all bands are jammed, use original predictions
            filtered_predictions = predictions
        
        # Select band with highest probability
        best_band_idx = np.argmax(filtered_predictions)
        best_band = best_band_idx + 1  # Convert to 1-based indexing
        confidence = filtered_predictions[best_band_idx]
        
        return best_band, confidence, filtered_predictions
    
    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray,
                   validation_data: Optional[Tuple] = None,
                   epochs: int = None, batch_size: int = None) -> Dict:
        """Train the CNN model"""
        if epochs is None:
            epochs = CNN_CONFIG['epochs']
        if batch_size is None:
            batch_size = CNN_CONFIG['batch_size']
          # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
          # Train model
        history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.training_history = history.history
        
        # Print final metrics
        if validation_data:
            val_loss, val_acc = self.model.evaluate(validation_data[0], validation_data[1], verbose=0)
            print(f"Final validation accuracy: {val_acc:.4f}")
            print(f"Final validation loss: {val_loss:.4f}")
        
        return self.training_history
    
    def save_model(self, filepath: str):
        """Save trained model"""
        self.model.save(filepath)
          # Save training history
        history_path = filepath.replace('.h5', '_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load pre-trained model"""
        self.model = models.load_model(filepath)
        
        # Load training history if available
        history_path = filepath.replace('.h5', '_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
    
    def evaluate_model(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict:
        """Evaluate model performance"""
        results = self.model.evaluate(test_data, test_labels, verbose=0)
        
        evaluation_results = {
            'loss': results[0],
            'accuracy': results[1],
            'top_k_accuracy': results[2] if len(results) > 2 else None
        }
        
        # Get predictions for detailed analysis
        predictions = self.model.predict(test_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Calculate per-class accuracy
        from sklearn.metrics import classification_report, confusion_matrix
        
        evaluation_results['classification_report'] = classification_report(
            true_classes, predicted_classes, target_names=[f'Band_{i+1}' for i in range(5)]
        )
        evaluation_results['confusion_matrix'] = confusion_matrix(true_classes, predicted_classes)
        
        return evaluation_results

class FrequencyHoppingStrategy:
    """High-level frequency hopping strategy using CNN"""
    
    def __init__(self, cnn_model: FrequencyHoppingCNN):
        self.cnn_model = cnn_model
        self.current_band = 1
        self.band_history = []
        self.confidence_history = []
        self.switch_threshold = 0.7  # Minimum confidence to switch
        self.min_dwell_time = 0.5    # Minimum time to stay in a band (seconds)
        self.last_switch_time = 0
        
    def should_switch_band(self, spectrum_data: np.ndarray, 
                          jammed_bands: List[int],
                          current_time: float) -> Tuple[bool, int, float]:
        """
        Decide whether to switch frequency band
        Returns: (should_switch, recommended_band, confidence)
        """
        # Check minimum dwell time
        time_since_switch = current_time - self.last_switch_time
        if time_since_switch < self.min_dwell_time:
            return False, self.current_band, 0.0
        
        # Get CNN recommendation
        recommended_band, confidence, _ = self.cnn_model.predict_best_frequency(
            spectrum_data, jammed_bands
        )
        
        # Decide whether to switch
        should_switch = (
            recommended_band != self.current_band and 
            confidence > self.switch_threshold
        ) or (self.current_band in jammed_bands)
        
        return should_switch, recommended_band, confidence
    
    def execute_frequency_hop(self, spectrum_data: np.ndarray,
                            jammed_bands: List[int],
                            current_time: float) -> Dict:
        """
        Execute frequency hopping decision
        Returns: hop_info dictionary
        """
        should_switch, recommended_band, confidence = self.should_switch_band(
            spectrum_data, jammed_bands, current_time
        )
        
        hop_info = {
            'timestamp': current_time,
            'previous_band': self.current_band,
            'recommended_band': recommended_band,
            'confidence': confidence,
            'jammed_bands': jammed_bands.copy(),
            'switched': False,
            'reason': 'none'
        }
        
        if should_switch:
            # Determine reason for switch
            if self.current_band in jammed_bands:
                hop_info['reason'] = 'jammer_avoidance'
            else:
                hop_info['reason'] = 'quality_improvement'
            
            # Execute switch
            self.current_band = recommended_band
            self.last_switch_time = current_time
            hop_info['switched'] = True
            
            # Update history
            self.band_history.append(recommended_band)
            self.confidence_history.append(confidence)
            
            print(f"Frequency hop: Band {hop_info['previous_band']} -> Band {recommended_band} "
                  f"(Confidence: {confidence:.3f}, Reason: {hop_info['reason']})")
        
        return hop_info
    
    def get_band_usage_statistics(self) -> Dict:
        """Get statistics about band usage"""
        if not self.band_history:
            return {}
        
        unique_bands, counts = np.unique(self.band_history, return_counts=True)
        total_switches = len(self.band_history)
        
        stats = {
            'total_switches': total_switches,
            'band_usage': dict(zip(unique_bands.astype(int), counts)),
            'most_used_band': int(unique_bands[np.argmax(counts)]),
            'average_confidence': np.mean(self.confidence_history),
            'switch_rate': total_switches / max(1, self.last_switch_time)
        }
        
        return stats
    
    def reset_strategy(self):
        """Reset strategy to initial state"""
        self.current_band = 1
        self.band_history = []
        self.confidence_history = []
        self.last_switch_time = 0

class AdaptiveFrequencyManager:
    """Comprehensive frequency management system"""
    
    def __init__(self, cnn_model_path: Optional[str] = None):
        self.cnn_model = FrequencyHoppingCNN(cnn_model_path)
        self.hopping_strategy = FrequencyHoppingStrategy(self.cnn_model)
        
        # Competition configuration
        self.num_bands = COMPETITION_CONFIG['num_subbands']
        
        # Performance tracking
        self.performance_metrics = {
            'successful_transmissions': 0,
            'failed_transmissions': 0,
            'total_data_transmitted': 0,
            'jammer_encounters': 0,
            'successful_avoidances': 0
        }
    
    def initialize_system(self):
        """Initialize the frequency management system"""
        print("Adaptive Frequency Management System Initialized")
        print(f"Current band: {self.hopping_strategy.current_band}")
        print(f"CNN model ready: {self.cnn_model.model is not None}")
    
    def process_transmission_attempt(self, spectrum_data: np.ndarray,
                                   jammed_bands: List[int],
                                   current_time: float,
                                   data_size: int) -> Dict:
        """
        Process a transmission attempt with adaptive frequency management
        """
        # Execute frequency hopping decision
        hop_info = self.hopping_strategy.execute_frequency_hop(
            spectrum_data, jammed_bands, current_time
        )
        
        # Simulate transmission success/failure
        current_band = self.hopping_strategy.current_band
        transmission_success = current_band not in jammed_bands
        
        # Update performance metrics
        if transmission_success:
            self.performance_metrics['successful_transmissions'] += 1
            self.performance_metrics['total_data_transmitted'] += data_size
        else:
            self.performance_metrics['failed_transmissions'] += 1
        
        if jammed_bands:
            self.performance_metrics['jammer_encounters'] += 1
            if transmission_success:
                self.performance_metrics['successful_avoidances'] += 1
        
        # Prepare result
        result = {
            'hop_info': hop_info,
            'transmission_success': transmission_success,
            'current_band': current_band,
            'data_transmitted': data_size if transmission_success else 0,
            'performance_metrics': self.performance_metrics.copy()
        }
        
        return result
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'current_band': self.hopping_strategy.current_band,
            'band_history': self.hopping_strategy.band_history[-10:],  # Last 10
            'performance_metrics': self.performance_metrics,
            'band_usage_stats': self.hopping_strategy.get_band_usage_statistics(),
            'model_ready': self.cnn_model.model is not None
        }
        
        return status
    
    def train_system(self, training_data_path: str):
        """Train the CNN model with provided data"""
        # This would load training data and train the model
        # Implementation depends on data format
        print(f"Training system with data from: {training_data_path}")
        # Implementation to be added based on specific data format
        pass
    
    def save_system(self, save_path: str):
        """Save the entire system state"""
        # Save CNN model
        model_path = os.path.join(save_path, 'frequency_hopping_model.h5')
        self.cnn_model.save_model(model_path)
        
        # Save system state
        state_path = os.path.join(save_path, 'system_state.pkl')
        system_state = {
            'hopping_strategy': {
                'current_band': self.hopping_strategy.current_band,
                'band_history': self.hopping_strategy.band_history,
                'confidence_history': self.hopping_strategy.confidence_history,
                'last_switch_time': self.hopping_strategy.last_switch_time
            },
            'performance_metrics': self.performance_metrics
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(system_state, f)
        
        print(f"System saved to {save_path}")
    
    def load_system(self, load_path: str):
        """Load the entire system state"""
        # Load CNN model
        model_path = os.path.join(load_path, 'frequency_hopping_model.h5')
        if os.path.exists(model_path):
            self.cnn_model.load_model(model_path)
        
        # Load system state
        state_path = os.path.join(load_path, 'system_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                system_state = pickle.load(f)
            
            # Restore hopping strategy state
            hs = system_state['hopping_strategy']
            self.hopping_strategy.current_band = hs['current_band']
            self.hopping_strategy.band_history = hs['band_history']
            self.hopping_strategy.confidence_history = hs['confidence_history']
            self.hopping_strategy.last_switch_time = hs['last_switch_time']
            
            # Restore performance metrics
            self.performance_metrics = system_state['performance_metrics']
        
        print(f"System loaded from {load_path}")
