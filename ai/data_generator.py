# Synthetic Data Generator for CNN Training

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class SyntheticDataGenerator:
    """Generate synthetic training data for CNN frequency hopping model"""
    
    def __init__(self):
        self.num_bands = COMPETITION_CONFIG['num_subbands']
        self.sampling_rate = 1000  # 1 kHz for feature extraction
        self.window_size = 100     # 100ms window
        
    def generate_snr_profile(self, length: int, base_snr: float = 20, 
                           noise_variance: float = 5) -> np.ndarray:
        """Generate realistic SNR profile with variations"""
        # Add temporal correlation and variations
        t = np.linspace(0, length/self.sampling_rate, length)
        
        # Base SNR with slow variations
        slow_variation = 3 * np.sin(2 * np.pi * 0.1 * t)
        
        # Fast fading
        fast_fading = noise_variance * np.random.randn(length) * 0.3
        
        # Occasional deep fades
        deep_fades = np.zeros(length)
        fade_points = np.random.choice(length, size=int(length * 0.05), replace=False)
        deep_fades[fade_points] = -10 * np.random.exponential(1, len(fade_points))
        
        snr_profile = base_snr + slow_variation + fast_fading + deep_fades
        return np.clip(snr_profile, -10, 40)  # Realistic SNR range
        
    def generate_interference_profile(self, length: int, band_id: int, 
                                    jammer_present: bool = True) -> np.ndarray:
        """Generate interference profile for specific band"""
        interference = np.zeros(length)
        
        if not jammer_present:
            # Only thermal noise
            return np.random.randn(length) * 0.1
            
        # Add jammer interference patterns
        t = np.linspace(0, length/self.sampling_rate, length)
        
        # Sweeping jammer
        sweep_freq = 0.5  # Hz
        sweep_interference = 5 * np.sin(2 * np.pi * sweep_freq * t) ** 2
        
        # Pulsed jammer
        pulse_period = int(self.sampling_rate * 0.1)  # 100ms pulses
        pulse_pattern = np.tile([1]*int(pulse_period*0.3) + [0]*int(pulse_period*0.7), 
                               length//pulse_period + 1)[:length]
        pulse_interference = pulse_pattern * 8
        
        # Broadband noise jammer
        noise_interference = np.random.exponential(2, length)
        
        # Combine interference types
        interference = sweep_interference + pulse_interference + noise_interference
        
        # Band-specific interference levels
        band_factors = {1: 1.0, 2: 1.2, 3: 0.8, 4: 1.5, 5: 0.9}
        interference *= band_factors.get(band_id, 1.0)
        
        return interference
        
    def generate_channel_features(self, length: int, band_id: int,
                                scenario: str = 'no_jammer') -> Dict[str, np.ndarray]:
        """Generate comprehensive channel features for a frequency band"""
        
        # SNR profile
        if scenario == 'no_jammer':
            base_snr = np.random.uniform(15, 35)
            snr = self.generate_snr_profile(length, base_snr, 3)
        elif scenario == 'pattern_jammer':
            # Lower SNR when jammer is active in this band
            jammer_active = np.random.random() < 0.6
            base_snr = np.random.uniform(5, 20) if jammer_active else np.random.uniform(20, 35)
            snr = self.generate_snr_profile(length, base_snr, 5)
        else:  # random_jammer
            # Highly variable SNR
            base_snr = np.random.uniform(0, 30)
            snr = self.generate_snr_profile(length, base_snr, 8)
            
        # Interference profile
        jammer_present = scenario != 'no_jammer'
        interference = self.generate_interference_profile(length, band_id, jammer_present)
        
        # Channel quality indicators
        t = np.linspace(0, length/self.sampling_rate, length)
        
        # Received signal strength
        rss = snr + np.random.randn(length) * 2
        
        # Channel coherence time variations
        coherence = 50 + 30 * np.sin(2 * np.pi * 0.05 * t) + np.random.randn(length) * 5
        
        # Doppler spread
        doppler = 10 + 5 * np.abs(np.sin(2 * np.pi * 0.02 * t)) + np.random.randn(length) * 2
        
        # Multipath delay spread
        delay_spread = 0.1 + 0.05 * np.sin(2 * np.pi * 0.03 * t) + np.random.randn(length) * 0.01
        
        return {
            'snr': snr,
            'interference': interference,
            'rss': rss,
            'coherence_time': coherence,
            'doppler_spread': doppler,
            'delay_spread': delay_spread
        }
        
    def create_feature_vector(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Create normalized feature vector for CNN input"""
        # Stack all features
        feature_stack = np.stack([
            features_dict['snr'],
            features_dict['interference'], 
            features_dict['rss'],
            features_dict['coherence_time'],
            features_dict['doppler_spread'],
            features_dict['delay_spread']
        ], axis=0)
        
        # Normalize each feature channel
        normalized_features = np.zeros_like(feature_stack)
        for i in range(feature_stack.shape[0]):
            channel = feature_stack[i]
            normalized_features[i] = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
            
        return normalized_features
        
    def generate_optimal_labels(self, all_band_features: List[Dict], 
                              current_band: int, scenario: str) -> np.ndarray:
        """Generate optimal frequency band selection labels"""
        num_samples = len(all_band_features[0]['snr'])
        labels = np.zeros((num_samples, self.num_bands))
        
        for i in range(num_samples):
            # Calculate quality metric for each band
            band_qualities = []
            
            for band_id in range(1, self.num_bands + 1):
                idx = band_id - 1
                features = all_band_features[idx]
                
                # Quality metric based on SNR and interference
                snr_val = features['snr'][i]
                interference_val = features['interference'][i]
                
                # Penalize high interference and low SNR
                quality = snr_val - 0.5 * interference_val
                
                # Add diversity bonus (avoid staying in same band)
                if band_id != current_band:
                    quality += 2
                    
                # Scenario-specific adjustments
                if scenario == 'pattern_jammer':
                    # Known jammer pattern: bands 1,3,5,4,2
                    jammer_bands = [1, 3, 5, 4, 2]
                    time_step = i // (self.sampling_rate)  # seconds
                    active_jammer_band = jammer_bands[time_step % len(jammer_bands)]
                    if band_id == active_jammer_band:
                        quality -= 15  # Heavy penalty for jammer band
                        
                elif scenario == 'random_jammer':
                    # Random jammer - use ML prediction confidence
                    # Simulate prediction uncertainty
                    if interference_val > 5:
                        quality -= 8
                        
                band_qualities.append(quality)
            
            # Softmax conversion to probabilities
            qualities = np.array(band_qualities)
            exp_qualities = np.exp(qualities - np.max(qualities))
            probabilities = exp_qualities / np.sum(exp_qualities)
            
            labels[i] = probabilities
            
        return labels
        
    def generate_training_batch(self, batch_size: int = 32, 
                              sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a training batch with features and labels"""
        
        scenarios = ['no_jammer', 'pattern_jammer', 'random_jammer']
        
        batch_features = []
        batch_labels = []
        
        for _ in range(batch_size):
            # Random scenario selection
            scenario = np.random.choice(scenarios)
            current_band = np.random.randint(1, self.num_bands + 1)
            
            # Generate features for all bands
            all_band_features = []
            for band_id in range(1, self.num_bands + 1):
                features = self.generate_channel_features(sequence_length, band_id, scenario)
                all_band_features.append(features)
                
            # Create input features (stack all bands)
            combined_features = []
            for i in range(sequence_length):
                sample_features = []
                for band_features in all_band_features:
                    sample_vector = np.array([
                        band_features['snr'][i],
                        band_features['interference'][i],
                        band_features['rss'][i],
                        band_features['coherence_time'][i],
                        band_features['doppler_spread'][i],
                        band_features['delay_spread'][i]
                    ])
                    sample_features.extend(sample_vector)
                combined_features.append(sample_features)
                
            # Normalize features
            features_array = np.array(combined_features)
            features_array = (features_array - np.mean(features_array, axis=0)) / (np.std(features_array, axis=0) + 1e-8)
            
            # Generate optimal labels
            labels = self.generate_optimal_labels(all_band_features, current_band, scenario)
            
            batch_features.append(features_array)
            batch_labels.append(labels)
            
        return np.array(batch_features), np.array(batch_labels)
        
    def generate_dataset(self, num_samples: int = 1000, 
                        save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete training dataset"""
        print(f"Generating {num_samples} training samples...")
        
        all_features = []
        all_labels = []
        
        batch_size = 32
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            actual_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            batch_features, batch_labels = self.generate_training_batch(
                batch_size=actual_batch_size
            )
            
            all_features.extend(batch_features)
            all_labels.extend(batch_labels)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Generated {(batch_idx + 1) * batch_size} samples...")
                
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        print(f"Dataset shape: Features {features_array.shape}, Labels {labels_array.shape}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            np.savez_compressed(save_path, 
                              features=features_array, 
                              labels=labels_array)
            print(f"Dataset saved to {save_path}")
            
        return features_array, labels_array
        
    def visualize_sample(self, features: np.ndarray, labels: np.ndarray, 
                        sample_idx: int = 0):
        """Visualize a training sample"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Training Sample {sample_idx}', fontsize=16)
        
        sample_features = features[sample_idx]  # Shape: (sequence_length, num_features)
        sample_labels = labels[sample_idx]      # Shape: (sequence_length, num_bands)
        
        time_steps = range(len(sample_features))
        
        # Plot SNR for each band
        axes[0, 0].set_title('SNR per Band')
        for band in range(self.num_bands):
            snr_values = sample_features[:, band * 6]  # Every 6th feature is SNR
            axes[0, 0].plot(time_steps, snr_values, label=f'Band {band+1}')
        axes[0, 0].set_ylabel('SNR (dB)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot interference for each band  
        axes[0, 1].set_title('Interference per Band')
        for band in range(self.num_bands):
            interference_values = sample_features[:, band * 6 + 1]  # Interference is 2nd feature
            axes[0, 1].plot(time_steps, interference_values, label=f'Band {band+1}')
        axes[0, 1].set_ylabel('Interference Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot selection probabilities
        axes[1, 0].set_title('Optimal Band Selection Probabilities')
        for band in range(self.num_bands):
            axes[1, 0].plot(time_steps, sample_labels[:, band], label=f'Band {band+1}')
        axes[1, 0].set_ylabel('Selection Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot selected band (argmax of probabilities)
        axes[1, 1].set_title('Selected Band Over Time')
        selected_bands = np.argmax(sample_labels, axis=1) + 1
        axes[1, 1].step(time_steps, selected_bands, where='post')
        axes[1, 1].set_ylabel('Selected Band')
        axes[1, 1].set_ylim(0.5, self.num_bands + 0.5)
        axes[1, 1].grid(True)
        
        # Plot channel quality metrics
        axes[2, 0].set_title('RSS and Coherence Time')
        rss_values = sample_features[:, 2]  # 3rd feature is RSS
        coherence_values = sample_features[:, 3]  # 4th feature is coherence
        
        ax1 = axes[2, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(time_steps, rss_values, 'b-', label='RSS')
        line2 = ax2.plot(time_steps, coherence_values, 'r-', label='Coherence Time')
        
        ax1.set_ylabel('RSS (dBm)', color='b')
        ax2.set_ylabel('Coherence Time (ms)', color='r')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True)
        
        # Plot Doppler and delay spread
        axes[2, 1].set_title('Doppler Spread and Delay Spread')
        doppler_values = sample_features[:, 4]  # 5th feature
        delay_values = sample_features[:, 5]    # 6th feature
        
        ax3 = axes[2, 1]
        ax4 = ax3.twinx()
        
        line3 = ax3.plot(time_steps, doppler_values, 'g-', label='Doppler Spread')
        line4 = ax4.plot(time_steps, delay_values, 'm-', label='Delay Spread')
        
        ax3.set_ylabel('Doppler Spread (Hz)', color='g')
        ax4.set_ylabel('Delay Spread (μs)', color='m')
        
        lines = line3 + line4
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        ax3.grid(True)
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator()
    
    # Generate small dataset for testing
    features, labels = generator.generate_dataset(
        num_samples=100,
        save_path="data/training_data.npz"
    )
    
    # Visualize a sample
    fig = generator.visualize_sample(features, labels, sample_idx=0)
    plt.show()
