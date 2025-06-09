# simplified_frequency_hopping.py - Simplified version without complex ML dependencies

import numpy as np
import time
from typing import Dict, List, Tuple, Optional

class SimplifiedFrequencyManager:
    """Simplified frequency manager without ML dependencies for testing"""
    
    def __init__(self):
        self.current_band = 1
        self.band_history = []
        self.confidence_history = []
        self.last_switch_time = 0
        self.switch_threshold = 0.7
        self.min_dwell_time = 0.5
        
        # Enhanced AI-like decision parameters
        self.band_scores = np.array([0.8, 0.6, 0.9, 0.7, 0.5])  # Initial scores
        self.ai_confidence = 0.8
        self.exploration_rate = 0.3
        
    def ai_frequency_decision(self, channel_state: Dict) -> int:
        """Enhanced AI-like frequency decision with realistic variation"""
        current_time = time.time()
        
        # Prepare enhanced features for all 5 bands with realistic variation
        band_features = []
        time_penalty = min(2.0, current_time % 10)  # Time-based degradation
        
        for band in range(1, 6):
            if band == self.current_band:
                # Current band degrades over time
                base_snr = 25.0 - time_penalty
                base_interference = 0.3 + time_penalty * 0.1
                base_rss = -65.0 - time_penalty
            else:
                # Other bands have band-specific characteristics
                if band in [1, 3, 5]:  # Low frequency bands
                    base_snr = np.random.normal(27.0, 2.0)
                    base_interference = np.random.normal(0.2, 0.05)
                    base_rss = np.random.normal(-63.0, 3.0)
                else:  # High frequency bands
                    base_snr = np.random.normal(24.0, 2.5)
                    base_interference = np.random.normal(0.4, 0.1)
                    base_rss = np.random.normal(-68.0, 4.0)
            
            # Add small random variations
            snr = base_snr + np.random.normal(0, 0.5)
            interference = max(0.1, base_interference + np.random.normal(0, 0.02))
            rss = base_rss + np.random.normal(0, 1.0)
            
            # Calculate quality score
            quality = snr - 0.6 * interference + 0.2 * rss
            band_features.append([snr, interference, rss, quality])
        
        # Enhanced exploration strategy
        ai_confidence = np.random.uniform(0.75, 0.85)
        exploration_factor = 0.4 if ai_confidence < 0.7 else 0.25
        
        # Time-based cycling behavior (15-second cycles)
        time_factor = current_time % 15
        if time_factor < 5:
            exploration_factor *= 1.2
        elif time_factor > 10:
            exploration_factor *= 0.8
        
        # Calculate band scores with exploration
        scores = np.array([features[3] for features in band_features])
        
        # Add exploration noise
        if np.random.random() < exploration_factor:
            noise = np.random.normal(0, 2.0, 5)
            scores += noise
        
        # Weighted selection with softmax-like distribution
        exp_scores = np.exp(scores / 3.0)  # Temperature scaling
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Select band based on probabilities
        selected_band = np.random.choice(range(1, 6), p=probabilities)
        
        # Update internal state
        self.band_scores = probabilities
        self.ai_confidence = ai_confidence
        
        return selected_band
    
    def rule_based_decision(self, channel_state: Dict) -> int:
        """Enhanced rule-based decision with time-varying behavior"""
        current_time = time.time()
        
        # Time-varying thresholds for dynamic behavior
        base_snr_threshold = 20.0 + 3.0 * np.sin(current_time * 0.1)
        base_interference_threshold = 0.5 - 0.2 * np.cos(current_time * 0.15)
        
        # Get current band quality
        current_snr = channel_state.get('snr', 25.0)
        current_interference = channel_state.get('interference', 0.3)
        
        # Check if current band is still good enough
        time_since_switch = current_time - self.last_switch_time
        
        # Enhanced jammer detection
        jammer_detected = channel_state.get('jammer_active', False)
        jammer_type = channel_state.get('jammer_type', 'none')
        
        # Scenario-aware decision making
        if jammer_detected and jammer_type == 'pattern_jammer':
            # For pattern jammer, use predictive avoidance
            avoided_bands = channel_state.get('predicted_jammed_bands', [])
            candidates = [b for b in range(1, 6) if b not in avoided_bands]
        else:
            # For other scenarios, use quality-based selection
            candidates = list(range(1, 6))
        
        # Apply time-based exploration
        exploration_factor = 0.3 + 0.2 * np.sin(current_time * 0.2)
        
        if (current_snr < base_snr_threshold or 
            current_interference > base_interference_threshold or
            time_since_switch > 3.0 or
            np.random.random() < exploration_factor):
            
            # Quality-based selection with noise
            band_qualities = []
            for band in candidates:
                if band == self.current_band:
                    quality = current_snr - current_interference * 10
                else:
                    # Estimate quality for other bands
                    estimated_snr = 25.0 + np.random.normal(0, 3.0)
                    estimated_interference = 0.3 + np.random.normal(0, 0.1)
                    quality = estimated_snr - estimated_interference * 10
                
                # Add exploration noise
                quality += np.random.normal(0, 2.0)
                band_qualities.append(quality)
            
            # Select best candidate
            if band_qualities:
                best_idx = np.argmax(band_qualities)
                selected_band = candidates[best_idx]
            else:
                selected_band = self.current_band
        else:
            selected_band = self.current_band
        
        return selected_band
    
    def update_frequency(self, new_band: int, transmitted_bits: int, step_bits: int = 1000):
        """Update frequency with enhanced signal emission logic"""
        old_band = self.current_band
        
        # Enhanced signal emission for better visualization
        if new_band != self.current_band:
            self.current_band = new_band
            self.last_switch_time = time.time()
            self.band_history.append(new_band)
            return {'switched': True, 'old_band': old_band, 'new_band': new_band}
        else:
            # Even if no change, emit for graph continuity (every 5th iteration)
            if transmitted_bits % (step_bits * 5) == 0:
                return {'switched': False, 'old_band': old_band, 'new_band': new_band, 'continuity_emit': True}
            return {'switched': False, 'old_band': old_band, 'new_band': new_band}
    
    def get_band_usage_statistics(self) -> Dict:
        """Get enhanced statistics about band usage"""
        if not self.band_history:
            return {'total_switches': 0, 'band_coverage': 0, 'switch_rate': 0}
        
        unique_bands = np.unique(self.band_history)
        total_switches = len(self.band_history)
        
        stats = {
            'total_switches': total_switches,
            'band_coverage': len(unique_bands),
            'unique_bands_used': unique_bands.tolist(),
            'switch_rate': total_switches / max(1, self.last_switch_time),
            'average_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.8,
            'current_ai_confidence': self.ai_confidence,
            'current_band_scores': self.band_scores.tolist()
        }
        
        return stats
    
    def reset_strategy(self):
        """Reset strategy to initial state"""
        self.current_band = 1
        self.band_history = []
        self.confidence_history = []
        self.last_switch_time = 0
        self.ai_confidence = 0.8
        self.band_scores = np.array([0.8, 0.6, 0.9, 0.7, 0.5])

# Compatibility wrapper for existing code
class AdaptiveFrequencyManager:
    """Wrapper to maintain compatibility with existing GUI code"""
    
    def __init__(self, cnn_model_path: Optional[str] = None):
        print("Using simplified frequency manager (TensorFlow/Keras not available)")
        self.frequency_manager = SimplifiedFrequencyManager()
        
    def initialize_system(self):
        """Initialize the frequency management system"""
        print("Simplified Frequency Management System Initialized")
        print(f"Current band: {self.frequency_manager.current_band}")
        print("AI-like decision logic ready")
    
    def ai_frequency_decision(self, channel_state: Dict) -> int:
        return self.frequency_manager.ai_frequency_decision(channel_state)
    
    def rule_based_decision(self, channel_state: Dict) -> int:
        return self.frequency_manager.rule_based_decision(channel_state)
    
    def update_frequency(self, new_band: int, transmitted_bits: int, step_bits: int = 1000):
        return self.frequency_manager.update_frequency(new_band, transmitted_bits, step_bits)
    
    def get_band_usage_statistics(self) -> Dict:
        return self.frequency_manager.get_band_usage_statistics()
    
    def reset_strategy(self):
        return self.frequency_manager.reset_strategy()
    
    @property
    def current_band(self):
        return self.frequency_manager.current_band
    
    @property
    def ai_confidence(self):
        return self.frequency_manager.ai_confidence
    
    @property
    def band_scores(self):
        return self.frequency_manager.band_scores
