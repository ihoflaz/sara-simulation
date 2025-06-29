# Main GUI Window for 5G Communication System

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QGridLayout, QWidget, QPushButton, QLabel, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit, 
                            QGroupBox, QTabWidget, QSlider, QCheckBox, QLineEdit,
                            QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor
import threading
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from core.modulation import OFDMSystem
from core.channel import ChannelModel

# Suppress TensorFlow warnings for cleaner output
import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Try to import AI-powered frequency hopping
try:
    from core.frequency_hopping import AdaptiveFrequencyManager
    from ai.cnn_model import FrequencyHoppingCNN
    print("✅ Using full AI frequency hopping with TensorFlow/Keras")
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TensorFlow/AI modules not available ({e}), using simplified frequency hopping")
    from simplified_frequency_hopping import AdaptiveFrequencyManager
    AI_AVAILABLE = False
    # Create a dummy class for compatibility
    class FrequencyHoppingCNN:
        def __init__(self, *args, **kwargs):
            pass

from core.data_processing import TransmissionSimulator
import torch

# Import enhanced modules
try:
    from core.enhanced_data_processing import EnhancedTransmissionSimulator, SecurityMetrics
    from core.encryption import EncryptedPacketProcessor
    from core.recovery import DataRecoveryManager, RecoveryMode
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced security features available")
except ImportError as e:
    print(f"⚠️  Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

class RealTimePlotter(QWidget):
    """Real-time plotting widget for signal analysis"""
    
    def __init__(self, title: str = "Signal Plot", parent=None):
        super().__init__(parent)
        self.title = title
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Data storage
        self.max_points = 1000
        self.time_data = []
        self.signal_data = []
        
        # Initialize plot
        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.ax.set_title(self.title)
        self.ax.grid(True)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-40, 40)
        
    def update_plot(self, new_time: float, new_value: float):
        """Update plot with new data point"""
        self.time_data.append(new_time)
        self.signal_data.append(new_value)
        
        # Keep only recent data
        if len(self.time_data) > self.max_points:
            self.time_data.pop(0)
            self.signal_data.pop(0)
            
        # Update plot
        if len(self.time_data) > 1:
            self.line.set_data(self.time_data, self.signal_data)
            
            # Adjust axes
            if len(self.time_data) > 10:
                self.ax.set_xlim(min(self.time_data), max(self.time_data))
                
            if len(self.signal_data) > 0:
                data_range = max(self.signal_data) - min(self.signal_data)
                if data_range > 0:
                    margin = data_range * 0.1
                    self.ax.set_ylim(min(self.signal_data) - margin, 
                                   max(self.signal_data) + margin)
        
        self.canvas.draw()
        
    def clear_plot(self):
        """Clear all data and reset plot"""
        self.time_data.clear()
        self.signal_data.clear()
        self.line.set_data([], [])
        self.canvas.draw()

class ConstellationPlot(QWidget):
    """Constellation diagram widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Constellation Diagram')
        self.ax.set_xlabel('In-Phase')
        self.ax.set_ylabel('Quadrature')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
    def update_constellation(self, constellation_points: np.ndarray):
        """Update constellation diagram"""
        self.ax.clear()
        self.ax.set_title('Constellation Diagram')
        self.ax.set_xlabel('In-Phase')
        self.ax.set_ylabel('Quadrature')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        if len(constellation_points) > 0:
            real_parts = np.real(constellation_points)
            imag_parts = np.imag(constellation_points)
            self.ax.scatter(real_parts, imag_parts, alpha=0.6, s=10)
            
        self.canvas.draw()

class SimulationWorker(QThread):
    """Background thread for running simulations"""
    
    # Signals for communication with main thread
    snr_updated = pyqtSignal(float)
    ber_updated = pyqtSignal(float)
    throughput_updated = pyqtSignal(float)
    frequency_changed = pyqtSignal(int)
    modulation_changed = pyqtSignal(str)
    constellation_updated = pyqtSignal(np.ndarray)
    log_message = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    # New signals for AI verification and security
    ai_confidence_updated = pyqtSignal(float)
    ai_probabilities_updated = pyqtSignal(np.ndarray)
    jammer_detection_updated = pyqtSignal(bool, str)
    security_status_updated = pyqtSignal(str, float)
    security_event_logged = pyqtSignal(str, str)  # message, level
    subband_snr_updated = pyqtSignal(np.ndarray, int)  # all_band_snrs, active_band
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        
        # Initialize simulation components
        self.ofdm_system = OFDMSystem()
        self.channel_sim = ChannelModel()
        self.freq_hopping = AdaptiveFrequencyManager()
        self.data_processor = TransmissionSimulator()
        
        # Initialize enhanced simulator if available
        if ENHANCED_FEATURES_AVAILABLE:
            self.enhanced_simulator = EnhancedTransmissionSimulator(
                enable_encryption=True,
                enable_recovery=True
            )
            # Set up event callbacks
            self.enhanced_simulator.set_security_event_callback(self._handle_security_event)
            self.enhanced_simulator.set_recovery_event_callback(self._handle_recovery_event)
        else:
            self.enhanced_simulator = None
        
        # Load AI model if available
        try:
            self.ai_model = FrequencyHoppingCNN()
            if os.path.exists('models/frequency_hopping_model.pth'):
                checkpoint = torch.load('models/frequency_hopping_model.pth', 
                                      map_location='cpu')
                self.ai_model.load_state_dict(checkpoint['model_state_dict'])
                self.ai_model.eval()
                self.ai_enabled = True
                self.log_message.emit("AI model loaded successfully")
            else:
                self.ai_enabled = False
                self.log_message.emit("AI model not found, using rule-based hopping")
        except Exception as e:
            self.ai_enabled = False
            self.log_message.emit(f"Failed to load AI model: {str(e)}")
            
        # Simulation parameters
        self.current_scenario = 'no_jammer'
        self.file_size = 100e6  # 100 MB
        self.current_frequency_band = 1
        self.current_modulation = 'QPSK'
        self.simulation_speed = 1.0
        
        # Enhanced features initialization
        if ENHANCED_FEATURES_AVAILABLE:
            self.encryption_processor = EncryptedPacketProcessor()
            self.recovery_manager = DataRecoveryManager()
            self.log_message.emit("Enhanced security features initialized")
        else:
            self.encryption_processor = None
            self.recovery_manager = None
    
    def set_scenario(self, scenario: str):
        """Set simulation scenario"""
        self.current_scenario = scenario
        self.channel_sim.set_scenario(scenario)
        
    def set_file_size(self, size: float):
        """Set file size for transmission"""
        self.file_size = size
        
    def set_simulation_speed(self, speed: float):
        """Set simulation speed multiplier"""
        self.simulation_speed = speed
        
    def start_simulation(self):
        """Start the simulation"""
        self.running = True
        self.paused = False
        
    def pause_simulation(self):
        """Pause/resume simulation"""
        self.paused = not self.paused
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        
    def run(self):
        """Main simulation loop"""
        self.log_message.emit("Starting simulation...")
          # Initialize simulation state
        transmitted_bits = 0
        total_bits = int(self.file_size * 8)  # Convert bytes to bits
        error_bits = 0
        start_time = time.time()
        
        while self.running and transmitted_bits < total_bits:
            if self.paused:
                time.sleep(0.1)
                continue
                
            try:
                # Adaptive step size based on file size for better progress visibility
                if total_bits > 1e9:  # > 1GB
                    step_bits = int(total_bits / 1000)  # 1000 progress updates
                elif total_bits > 1e8:  # > 100MB  
                    step_bits = int(total_bits / 500)   # 500 progress updates
                else:
                    step_bits = max(1000, int(total_bits / 100))  # 100 progress updates
                
                # Ensure minimum progress increment
                step_bits = max(step_bits, int(total_bits / 1000))
                current_time = time.time() - start_time
                
                # Channel simulation
                channel_state = self.channel_sim.simulate_step(
                    current_time, self.current_frequency_band
                )
                
                # SNR calculation
                snr_db = channel_state['snr']
                self.snr_updated.emit(snr_db)
                
                # Adaptive modulation based on SNR
                if snr_db > 30:
                    self.current_modulation = '1024QAM'
                elif snr_db > 25:
                    self.current_modulation = '256QAM'
                elif snr_db > 20:
                    self.current_modulation = '64QAM'
                elif snr_db > 15:
                    self.current_modulation = '16QAM'
                elif snr_db > 10:
                    self.current_modulation = 'QPSK'
                else:
                    self.current_modulation = 'BPSK'
                    
                self.modulation_changed.emit(self.current_modulation)
                
                # Generate constellation points for display
                constellation = self.ofdm_system.generate_constellation_samples(
                    self.current_modulation, 100
                )
                self.constellation_updated.emit(constellation)
                
                # BER calculation based on channel conditions
                ber = self.calculate_ber(snr_db, self.current_modulation)
                self.ber_updated.emit(ber)
                  # Frequency hopping decision
                if self.ai_enabled:
                    new_band = self.ai_frequency_decision(channel_state)
                else:
                    new_band = self.rule_based_frequency_decision(channel_state)
                  # Enhanced signal emission for better visualization  
                if new_band != self.current_frequency_band:
                    self.current_frequency_band = new_band
                    self.frequency_changed.emit(new_band)
                    self.log_message.emit(f"Hopped to frequency band {new_band}")
                else:
                    # Even if no change, emit for graph continuity (every 5th iteration)
                    if transmitted_bits % (step_bits * 5) == 0:
                        self.frequency_changed.emit(self.current_frequency_band)
                
                # **ENHANCED AI TRANSPARENCY** - Emit AI decision data for real-time display
                if self.ai_enabled and hasattr(self, 'ai_confidence'):
                    self.ai_confidence_updated.emit(self.ai_confidence)
                    if hasattr(self, 'band_scores'):
                        self.ai_probabilities_updated.emit(np.array(self.band_scores))
                    
                    # Log AI decision transparency info periodically 
                    if transmitted_bits % (step_bits * 10) == 0:  # Every 10th step
                        if hasattr(self, 'ai_probabilities'):
                            prob_summary = ", ".join([f"B{i+1}:{self.ai_probabilities[i]:.3f}" for i in range(5)])
                            self.log_message.emit(f"AI Model Output: [{prob_summary}]")
                            self.log_message.emit(f"AI Confidence: {self.ai_confidence:.4f} | Current Band: {self.current_frequency_band}")
                              # Show decision reasoning
                            if hasattr(self, 'band_scores'):
                                best_quality_idx = np.argmax(self.band_scores)
                                best_ai_idx = np.argmax(self.ai_probabilities)
                                self.log_message.emit(f"AI prefers Band {best_ai_idx+1}, Quality analysis prefers Band {best_quality_idx+1}")
                  # Emit jammer detection status
                jammer_detected = channel_state.get('interference', 0) > 5
                jammer_type = "Pattern" if self.current_scenario == 'pattern_jammer' else "Random" if self.current_scenario == 'random_jammer' else "None"
                self.jammer_detection_updated.emit(jammer_detected, jammer_type)                # Emit security/coding status
                channel_quality = "excellent" if snr_db > 25 else "good" if snr_db > 15 else "fair" if snr_db > 10 else "poor"
                
                # Simulate data corruption detection and correction
                original_errors = transmitted_bits * (10 ** (-snr_db/10))  # Error estimation
                        
                # Select appropriate coding scheme
                if hasattr(self.data_processor, 'adaptive_coder'):
                    coding_scheme = self.data_processor.adaptive_coder.select_coding_scheme(channel_quality)
                else:
                    # Fallback selection based on channel quality
                    if channel_quality == "excellent":
                        coding_scheme = "none"
                    elif channel_quality == "good":
                        coding_scheme = "ldpc"
                    else:
                        coding_scheme = "turbo"
                
                # Calculate coding effectiveness and correction
                if coding_scheme == "ldpc":
                    correction_rate = 0.95  # LDPC can correct 95% of errors
                    coding_effectiveness = min(100, max(0, (snr_db - 5) * 12))
                elif coding_scheme == "turbo":
                    correction_rate = 0.90  # Turbo can correct 90% of errors  
                    coding_effectiveness = min(100, max(0, (snr_db - 3) * 10))
                else:
                    correction_rate = 0.50  # No coding, basic error handling
                    coding_effectiveness = min(100, max(0, snr_db * 5))
                
                corrected_errors = original_errors * correction_rate
                remaining_errors = original_errors - corrected_errors
                
                # Log security events when significant corrections occur
                if corrected_errors > 10:
                    security_message = f"Security: {coding_scheme.upper()} corrected {int(corrected_errors)} errors (SNR: {snr_db:.1f}dB)"
                    self.security_event_logged.emit(security_message, "INFO")
                
                # Log jammer interference and countermeasures
                jammer_detected = channel_state.get('interference', 0) > 5
                jammer_type = "Pattern" if self.current_scenario == 'pattern_jammer' else "Random" if self.current_scenario == 'random_jammer' else "None"
                
                if jammer_detected:
                    jammer_message = f"Security: {jammer_type} jammer detected, interference: {channel_state.get('interference', 0):.1f}"
                    self.security_event_logged.emit(jammer_message, "WARNING")
                    
                    # Log frequency hopping as countermeasure
                    if new_band != self.current_frequency_band:
                        countermeasure_message = f"Security: Frequency hop {self.current_frequency_band}→{new_band} to avoid jammer"
                        self.security_event_logged.emit(countermeasure_message, "INFO")
                
                # Simulate sub-band SNR measurements for all 5 frequency bands
                all_band_snrs = np.zeros(5)
                for band_idx in range(5):
                    band_num = band_idx + 1
                    if band_num == self.current_frequency_band:
                        # Current band has measured SNR
                        all_band_snrs[band_idx] = snr_db
                    else:
                        # Estimate other bands based on jamming and channel conditions
                        base_snr = snr_db + np.random.normal(0, 2.0)  # Similar to current
                        
                        # Apply jammer effects to specific bands
                        if jammer_detected:
                            if jammer_type == "Pattern" and band_num in [2, 4]:  # Pattern jammer targets even bands
                                base_snr -= 15  # Heavy interference
                            elif jammer_type == "Random" and np.random.random() < 0.3:  # Random jammer
                                base_snr -= np.random.uniform(10, 20)
                        
                        all_band_snrs[band_idx] = max(5, base_snr)  # Minimum 5dB
                
                # Emit sub-band SNR data for visualization
                self.subband_snr_updated.emit(all_band_snrs, self.current_frequency_band)
                
                # Enhanced jammer detection and security status
                self.jammer_detection_updated.emit(jammer_detected, jammer_type)
                self.security_status_updated.emit(f"{coding_scheme.upper()} Active", coding_effectiveness)
                
                # Update transmission progress
                transmitted_bits += step_bits
                progress = int((transmitted_bits / total_bits) * 100)
                self.progress_updated.emit(progress)
                  
                # Calculate throughput
                elapsed_time = current_time
                if elapsed_time > 0:
                    throughput_mbps = (transmitted_bits / 1e6) / elapsed_time
                    self.throughput_updated.emit(throughput_mbps)
                
                # Adaptive simulation timing for better responsiveness
                if total_bits > 1e9:  # Large files (>1GB)
                    time.sleep(0.05 / self.simulation_speed)  # Faster updates
                else:
                    time.sleep(0.1 / self.simulation_speed)   # Standard updates                
            except Exception as e:
                self.log_message.emit(f"Simulation error: {str(e)}")
                break
                
        if transmitted_bits >= total_bits:
            self.log_message.emit("Transmission completed successfully!")
        else:
            self.log_message.emit("Simulation stopped.")
            
        self.running = False
    
    def calculate_ber(self, snr_db: float, modulation: str) -> float:
        """Calculate BER based on SNR and modulation scheme"""
        # Simplified BER calculation
        modulation_penalties = {
            'BPSK': 0,
            'QPSK': 3,
            '16QAM': 6,
            '64QAM': 9,
            '256QAM': 12,
            '1024QAM': 15
        }
        
        effective_snr = snr_db - modulation_penalties.get(modulation, 0)
          # Simplified BER formula
        if effective_snr > 20:
            ber = 1e-8
        elif effective_snr > 15:
            ber = 1e-6
        elif effective_snr > 10:
            ber = 1e-4
        elif effective_snr > 5:
            ber = 1e-3
        else:
            ber = 1e-2
            
        return ber
        
    def ai_frequency_decision(self, channel_state: Dict) -> int:
        """Enhanced AI-based frequency band decision with dynamic variation"""
        try:
            # Prepare base features for current channel state
            base_features = np.array([
                channel_state['snr'],
                channel_state['interference'],
                channel_state['rss'],
                20.0,  # Coherence time placeholder
                5.0,   # Doppler spread placeholder
                0.1    # Delay spread placeholder
            ])
            
            # Enhanced feature preparation for all bands with realistic variation
            all_band_features = []
            band_scores = []
            
            # **AI TRANSPARENCY LOGGING** - Log input features
            self.log_message.emit(f"🧠 AI Input: SNR={channel_state['snr']:.1f}dB, INT={channel_state['interference']:.1f}, RSS={channel_state['rss']:.1f}dBm")
            
            for band in range(5):  # 5 bands
                if band + 1 == self.current_frequency_band:
                    # Current band - use actual features with added uncertainty
                    band_features = base_features.copy()
                    band_features[0] += np.random.normal(0, 1.5)  # SNR uncertainty
                    band_features[1] += np.random.exponential(0.8)  # Interference variation
                    # Quality degrades over time to encourage switching
                    time_penalty = min(2.0, time.time() % 10)  # Increases over 10 seconds
                    band_features[0] -= time_penalty
                    band_features[1] += time_penalty * 0.5
                else:
                    # Other bands - simulate potentially better conditions
                    band_features = base_features.copy()
                    # Simulate better opportunities in other bands
                    band_features[0] += np.random.normal(3, 2)  # Better SNR potential
                    band_features[1] = np.random.exponential(1.5)  # Lower interference
                    band_features[2] += np.random.normal(2, 3)  # Varied RSS
                    
                    # Add band-specific characteristics
                    if band == 0:  # Band 1: Low frequency, stable
                        band_features[0] += 2
                        band_features[1] *= 0.8
                    elif band == 2:  # Band 3: Mid frequency, balanced  
                        band_features[0] += 1
                    elif band == 4:  # Band 5: High frequency, fast
                        band_features[0] += np.random.normal(0, 2)
                        band_features[1] += np.random.normal(0, 1)
                
                # Calculate band quality score for decision logic
                quality_score = band_features[0] - 0.6 * band_features[1] + 0.2 * band_features[2]
                band_scores.append(quality_score)
                all_band_features.extend(band_features)
            
            full_features = np.array(all_band_features)
            
            # **REAL AI MODEL PREDICTION**
            with torch.no_grad():
                features_tensor = torch.FloatTensor(full_features).unsqueeze(0)
                probabilities = self.ai_model(features_tensor)
                ai_confidence = torch.max(probabilities).item()
                predicted_band = torch.argmax(probabilities, dim=1).item() + 1
                
                # **AI TRANSPARENCY LOGGING** - Log model outputs
                prob_str = ", ".join([f"B{i+1}:{probabilities[0][i].item():.3f}" for i in range(5)])
                self.log_message.emit(f"🤖 AI Probabilities: [{prob_str}]")
                self.log_message.emit(f"🎯 AI Raw Prediction: Band {predicted_band} (confidence: {ai_confidence:.4f})")
                
                # Enhanced exploration strategy based on confidence and time
                exploration_rate = 0.4 if ai_confidence < 0.7 else 0.25
                
                # Time-based switching to create dynamic behavior
                time_factor = time.time() % 15  # 15-second cycle
                if time_factor > 12:  # Last 3 seconds of cycle
                    exploration_rate += 0.2
                
                # **DECISION LOGIC WITH TRANSPARENCY**
                # Quality-based switching: prefer better scoring bands
                best_quality_band = np.argmax(band_scores) + 1
                
                # Log band quality scores for transparency
                quality_str = ", ".join([f"B{i+1}:{band_scores[i]:.1f}" for i in range(5)])
                self.log_message.emit(f"📊 Band Quality Scores: [{quality_str}]")
                
                if (best_quality_band != self.current_frequency_band and 
                    band_scores[best_quality_band-1] > band_scores[self.current_frequency_band-1] + 2):
                    predicted_band = best_quality_band
                    self.log_message.emit(f"🔄 AI: Quality-based override → Band {predicted_band} (score: {band_scores[best_quality_band-1]:.2f})")
                elif np.random.random() < exploration_rate:
                    # Weighted exploration - prefer bands with good scores
                    available_bands = [b for b in range(1, 6) if b != self.current_frequency_band]
                    available_scores = [band_scores[b-1] for b in available_bands]
                    
                    # Softmax selection based on scores
                    exp_scores = np.exp(np.array(available_scores) / 3.0)  # Temperature scaling
                    probabilities_weighted = exp_scores / np.sum(exp_scores)
                    predicted_band = np.random.choice(available_bands, p=probabilities_weighted)
                    self.log_message.emit(f"🔍 AI: Exploration switch → Band {predicted_band} (exploration rate: {exploration_rate:.2f})")
                else:
                    self.log_message.emit(f"✅ AI: Final decision → Band {predicted_band} (confidence: {ai_confidence:.4f})")
                  # **TRANSPARENCY FEATURES** - Store AI decision data for GUI display
                self.ai_confidence = ai_confidence
                self.band_scores = band_scores
                self.ai_probabilities = probabilities[0].cpu().numpy()  # Store raw model probabilities
                
            return predicted_band
            
        except Exception as e:
            self.log_message.emit(f"❌ AI decision error: {str(e)}")
            self.log_message.emit(f"🔄 Falling back to rule-based decision")
            return self.rule_based_frequency_decision(channel_state)
            
    def rule_based_frequency_decision(self, channel_state: Dict) -> int:
        """Enhanced rule-based frequency band decision"""
        current_quality = channel_state['snr'] - 0.5 * channel_state['interference']
        
        # Enhanced rule-based logic with more dynamic behavior
        # Time-based variation to ensure frequent updates
        time_factor = time.time() % 20  # 20-second cycle
        
        # Quality thresholds that vary over time
        switch_threshold = 10 + 3 * np.sin(time_factor / 3)  # Oscillating threshold
        exploration_factor = 0.2 if time_factor > 15 else 0.1  # Higher exploration in last 5 seconds
        
        # Add some randomness based on time to encourage switching
        noise_factor = np.random.normal(0, 2) if time_factor > 10 else 0
        adjusted_quality = current_quality + noise_factor
        
        # Enhanced switching logic
        if adjusted_quality < switch_threshold or np.random.random() < exploration_factor:
            # Try different band based on scenario-aware logic
            if self.current_scenario == 'pattern_jammer':
                # Pattern jammer: bands 1->3->5->4->2, avoid predicted next jammer band
                jammer_pattern = [1, 3, 5, 4, 2]
                time_index = int(time.time()) % len(jammer_pattern)
                predicted_jammer_band = jammer_pattern[(time_index + 1) % len(jammer_pattern)]
                
                # Choose a band that's not the predicted jammer band
                available_bands = [b for b in range(1, 6) if b != predicted_jammer_band and b != self.current_frequency_band]
                if available_bands:
                    new_band = np.random.choice(available_bands)
                    self.log_message.emit(f"Rule: Pattern avoidance switch to band {new_band} (avoiding {predicted_jammer_band})")
                    return new_band
            
            elif self.current_scenario == 'random_jammer':
                # Random jammer: switch to band with potentially better conditions
                # Simulate checking other bands
                best_band = self.current_frequency_band
                best_estimated_quality = adjusted_quality
                
                for test_band in range(1, 6):
                    if test_band != self.current_frequency_band:
                        # Estimate quality of other bands with some uncertainty
                        estimated_snr = channel_state['snr'] + np.random.normal(2, 3)
                        estimated_interference = np.random.exponential(1.5)
                        estimated_quality = estimated_snr - 0.5 * estimated_interference
                        
                        if estimated_quality > best_estimated_quality + 1:  # Small hysteresis
                            best_band = test_band
                            best_estimated_quality = estimated_quality
                
                if best_band != self.current_frequency_band:
                    self.log_message.emit(f"Rule: Quality-based switch to band {best_band} (estimated quality: {best_estimated_quality:.2f})")
                    return best_band
            
            # Default switching logic - try next band with some randomness
            if np.random.random() < 0.5:
                new_band = (self.current_frequency_band % 5) + 1  # Next band
            else:
                available_bands = [b for b in range(1, 6) if b != self.current_frequency_band]
                new_band = np.random.choice(available_bands)  # Random other band
            
            self.log_message.emit(f"Rule: Threshold-based switch to band {new_band} (quality: {adjusted_quality:.2f} < {switch_threshold:.2f})")
            return new_band
        else:
            # Stay in current band but log the decision
            if time_factor % 5 < 1:  # Log every 5 seconds
                self.log_message.emit(f"Rule: Staying in band {self.current_frequency_band} (quality: {adjusted_quality:.2f})")
            return self.current_frequency_band
    
    def _handle_security_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Handle security events from the enhanced simulator"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if event_type == "info":
            emoji = "🔐"
        elif event_type == "warning":
            emoji = "⚠️"
        elif event_type == "critical":
            emoji = "🚨"
        else:
            emoji = "ℹ️"
        
        log_message = f"[{timestamp}] {emoji} Security: {message}"
        self.log_message.emit(log_message)
    
    def _handle_recovery_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Handle recovery events from the enhanced simulator"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if event_type == "info":
            emoji = "📦"
        elif event_type == "warning":
            emoji = "⚠️"
        elif event_type == "retry":
            emoji = "🔄"
        else:
            emoji = "ℹ️"
        
        log_message = f"[{timestamp}] {emoji} Recovery: {message}"
        self.log_message.emit(log_message)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("5G Adaptive Communication System - TEKNOFEST")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize simulation worker
        self.simulation_worker = SimulationWorker()
        self.setup_worker_connections()
        
        # Setup UI
        self.setup_ui()
        self.setup_status_bar()
        
        # Simulation state
        self.simulation_running = False
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Visualizations
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Scenario selection
        scenario_group = QGroupBox("Simulation Scenario")
        scenario_layout = QVBoxLayout(scenario_group)
        
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems([
            "Phase 1: No Jammer",
            "Phase 2: Pattern-based Jammer", 
            "Phase 3: Random Jammer"
        ])
        self.scenario_combo.currentTextChanged.connect(self.on_scenario_changed)
        scenario_layout.addWidget(self.scenario_combo)
        
        layout.addWidget(scenario_group)
        
        # File size selection
        file_group = QGroupBox("File Transmission")
        file_layout = QGridLayout(file_group)
        
        file_layout.addWidget(QLabel("File Size:"), 0, 0)
        self.file_size_combo = QComboBox()
        self.file_size_combo.addItems(["100 MB", "1 GB", "10 GB"])
        self.file_size_combo.currentTextChanged.connect(self.on_file_size_changed)
        file_layout.addWidget(self.file_size_combo, 0, 1)
        
        # Progress bar
        file_layout.addWidget(QLabel("Progress:"), 1, 0)
        self.progress_bar = QProgressBar()
        file_layout.addWidget(self.progress_bar, 1, 1)
        
        layout.addWidget(file_group)
        
        # Control buttons
        button_group = QGroupBox("Simulation Control")
        button_layout = QVBoxLayout(button_group)
        
        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        button_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_simulation)
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addWidget(button_group)
        
        # Current status
        status_group = QGroupBox("Current Status")
        status_layout = QGridLayout(status_group)
        
        # SNR
        status_layout.addWidget(QLabel("SNR:"), 0, 0)
        self.snr_label = QLabel("-- dB")
        status_layout.addWidget(self.snr_label, 0, 1)
        
        # BER
        status_layout.addWidget(QLabel("BER:"), 1, 0)
        self.ber_label = QLabel("--")
        status_layout.addWidget(self.ber_label, 1, 1)
        
        # Throughput
        status_layout.addWidget(QLabel("Throughput:"), 2, 0)
        self.throughput_label = QLabel("-- Mbps")
        status_layout.addWidget(self.throughput_label, 2, 1)
        
        # Current frequency band
        status_layout.addWidget(QLabel("Frequency Band:"), 3, 0)
        self.frequency_label = QLabel("Band 1")
        status_layout.addWidget(self.frequency_label, 3, 1)
        
        # Current modulation
        status_layout.addWidget(QLabel("Modulation:"), 4, 0)
        self.modulation_label = QLabel("QPSK")
        status_layout.addWidget(self.modulation_label, 4, 1)
        
        layout.addWidget(status_group)
        
        # Speed control
        speed_group = QGroupBox("Simulation Speed")
        speed_layout = QVBoxLayout(speed_group)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("5x")
        speed_layout.addWidget(self.speed_label)
        
        layout.addWidget(speed_group)
        
        # Log window
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # Security and Recovery Controls
        security_group = QGroupBox("🔐 Security & Recovery")
        security_layout = QVBoxLayout(security_group)
        
        # Encryption Controls Group
        encryption_group = QGroupBox("🔒 AES-128 Encryption")
        encryption_layout = QGridLayout(encryption_group)
        
        # Encryption Enable/Disable
        self.encryption_enabled_cb = QCheckBox("Enable AES-128 Encryption")
        self.encryption_enabled_cb.setChecked(False)
        self.encryption_enabled_cb.stateChanged.connect(self.toggle_encryption)
        encryption_layout.addWidget(self.encryption_enabled_cb, 0, 0, 1, 2)
        
        # Encryption Mode Selection
        encryption_layout.addWidget(QLabel("Encryption Mode:"), 1, 0)
        self.encryption_mode_combo = QComboBox()
        self.encryption_mode_combo.addItems(["CTR", "GCM"])
        self.encryption_mode_combo.setCurrentText("CTR")
        self.encryption_mode_combo.currentTextChanged.connect(self.change_encryption_mode)
        encryption_layout.addWidget(self.encryption_mode_combo, 1, 1)
        
        # Key Management
        encryption_layout.addWidget(QLabel("Encryption Key:"), 2, 0)
        self.encryption_key_display = QLineEdit()
        self.encryption_key_display.setPlaceholderText("Key will be generated automatically")
        self.encryption_key_display.setReadOnly(True)
        encryption_layout.addWidget(self.encryption_key_display, 2, 1)
        
        # Key Export/Import Buttons
        key_buttons_layout = QHBoxLayout()
        self.export_key_btn = QPushButton("Export Key")
        self.export_key_btn.clicked.connect(self.export_encryption_key)
        self.import_key_btn = QPushButton("Import Key")
        self.import_key_btn.clicked.connect(self.import_encryption_key)
        key_buttons_layout.addWidget(self.export_key_btn)
        key_buttons_layout.addWidget(self.import_key_btn)
        encryption_layout.addLayout(key_buttons_layout, 3, 0, 1, 2)
        
        # Encryption Statistics
        self.encryption_stats_label = QLabel("Encryption Statistics:\nPackets Encrypted: 0\nPackets Decrypted: 0\nEncryption Success Rate: 0%")
        encryption_layout.addWidget(self.encryption_stats_label, 4, 0, 1, 2)
        
        security_layout.addWidget(encryption_group)
        
        # Data Recovery Controls Group
        recovery_group = QGroupBox("📡 Data Recovery Algorithm")
        recovery_layout = QGridLayout(recovery_group)
        
        # Recovery Enable/Disable
        self.recovery_enabled_cb = QCheckBox("Enable Data Recovery")
        self.recovery_enabled_cb.setChecked(True)
        self.recovery_enabled_cb.stateChanged.connect(self.toggle_recovery)
        recovery_layout.addWidget(self.recovery_enabled_cb, 0, 0, 1, 2)
        
        # Recovery Mode Selection
        recovery_layout.addWidget(QLabel("Recovery Mode:"), 1, 0)
        self.recovery_mode_combo = QComboBox()
        self.recovery_mode_combo.addItems([
            "Adaptive", "Retry Only", "Redundant Packets", 
            "Erasure Coding", "Disabled"
        ])
        self.recovery_mode_combo.setCurrentText("Adaptive")
        self.recovery_mode_combo.currentTextChanged.connect(self.change_recovery_mode)
        recovery_layout.addWidget(self.recovery_mode_combo, 1, 1)
        
        # Recovery Parameters
        recovery_layout.addWidget(QLabel("Max Retry Attempts:"), 2, 0)
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(1, 10)
        self.max_retries_spin.setValue(5)
        self.max_retries_spin.valueChanged.connect(self.update_recovery_params)
        recovery_layout.addWidget(self.max_retries_spin, 2, 1)
        
        recovery_layout.addWidget(QLabel("Redundancy Factor:"), 3, 0)
        self.redundancy_factor_spin = QSpinBox()
        self.redundancy_factor_spin.setRange(1, 5)
        self.redundancy_factor_spin.setValue(2)
        self.redundancy_factor_spin.valueChanged.connect(self.update_recovery_params)
        recovery_layout.addWidget(self.redundancy_factor_spin, 3, 1)
        
        # Recovery Status Display
        self.recovery_status_label = QLabel("Recovery Status: Standby")
        self.recovery_status_label.setStyleSheet("QLabel { background-color: #e6f3ff; padding: 5px; border-radius: 3px; }")
        recovery_layout.addWidget(self.recovery_status_label, 4, 0, 1, 2)
        
        # Recovery Statistics
        self.recovery_stats_label = QLabel("Recovery Statistics:\nPackets Failed: 0\nPackets Recovered: 0\nRecovery Success Rate: 0%")
        recovery_layout.addWidget(self.recovery_stats_label, 5, 0, 1, 2)
        
        # Recovery Queue Status
        self.recovery_queue_label = QLabel("Recovery Queue: 0 packets pending")
        recovery_layout.addWidget(self.recovery_queue_label, 6, 0, 1, 2)
        
        security_layout.addWidget(recovery_group)
        
        # Security Events Log Group
        events_group = QGroupBox("🛡️ Security Events Log")
        events_layout = QVBoxLayout(events_group)
        
        self.security_events_text = QTextEdit()
        self.security_events_text.setMaximumHeight(150)
        self.security_events_text.setReadOnly(True)
        self.security_events_text.setPlaceholderText("Security events will appear here...")
        events_layout.addWidget(self.security_events_text)
        
        # Clear Events Button
        clear_events_btn = QPushButton("Clear Events Log")
        clear_events_btn.clicked.connect(self.clear_security_events)
        events_layout.addWidget(clear_events_btn)
        
        security_layout.addWidget(events_group)
        
        # Security Recommendations Group
        recommendations_group = QGroupBox("💡 Security Recommendations")
        recommendations_layout = QVBoxLayout(recommendations_group)
        
        self.recommendations_label = QLabel("No recommendations at this time.")
        self.recommendations_label.setWordWrap(True)
        self.recommendations_label.setStyleSheet("QLabel { background-color: #f0f8ff; padding: 8px; border-radius: 5px; }")
        recommendations_layout.addWidget(self.recommendations_label)
        
        security_layout.addWidget(recommendations_group)
        
        layout.addWidget(security_group)
        
        return panel
        
    def create_visualization_panel(self) -> QWidget:
        """Create the visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different visualizations
        tab_widget = QTabWidget()
        
        # SNR Plot Tab
        snr_tab = QWidget()
        snr_layout = QVBoxLayout(snr_tab)
        self.snr_plotter = RealTimePlotter("SNR over Time (dB)")
        snr_layout.addWidget(self.snr_plotter)
        tab_widget.addTab(snr_tab, "SNR")
        
        # BER Plot Tab
        ber_tab = QWidget()
        ber_layout = QVBoxLayout(ber_tab)
        self.ber_plotter = RealTimePlotter("BER over Time")
        ber_layout.addWidget(self.ber_plotter)
        tab_widget.addTab(ber_tab, "BER")
        
        # Throughput Plot Tab
        throughput_tab = QWidget()
        throughput_layout = QVBoxLayout(throughput_tab)
        self.throughput_plotter = RealTimePlotter("Throughput (Mbps)")
        throughput_layout.addWidget(self.throughput_plotter)
        tab_widget.addTab(throughput_tab, "Throughput")
        
        # Constellation Tab
        constellation_tab = QWidget()
        constellation_layout = QVBoxLayout(constellation_tab)
        self.constellation_plot = ConstellationPlot()
        constellation_layout.addWidget(self.constellation_plot)
        tab_widget.addTab(constellation_tab, "Constellation")
        
        # Frequency hopping visualization
        freq_tab = QWidget()
        freq_layout = QVBoxLayout(freq_tab)
        self.freq_plotter = RealTimePlotter("Frequency Band Selection")
        freq_layout.addWidget(self.freq_plotter)
        tab_widget.addTab(freq_tab, "Frequency Hopping")
        
        # Sub-band SNR Tab
        subband_tab = QWidget()
        subband_layout = QVBoxLayout(subband_tab)
        self.subband_snr_plot = SubBandSNRPlot()
        subband_layout.addWidget(self.subband_snr_plot)
        tab_widget.addTab(subband_tab, "Sub-band SNR")
        
        # Enhanced Security and Recovery Tab
        security_tab = QWidget()
        tabs.addTab(security_tab, "🔐 Security & Recovery")
        
        security_layout = QVBoxLayout(security_tab)
        
        # Encryption Controls Group
        encryption_group = QGroupBox("🔒 AES-128 Encryption")
        encryption_layout = QGridLayout(encryption_group)
        
        # Encryption Enable/Disable
        self.encryption_enabled_cb = QCheckBox("Enable AES-128 Encryption")
        self.encryption_enabled_cb.setChecked(False)
        self.encryption_enabled_cb.stateChanged.connect(self.toggle_encryption)
        encryption_layout.addWidget(self.encryption_enabled_cb, 0, 0, 1, 2)
        
        # Encryption Mode Selection
        encryption_layout.addWidget(QLabel("Encryption Mode:"), 1, 0)
        self.encryption_mode_combo = QComboBox()
        self.encryption_mode_combo.addItems(["CTR", "GCM"])
        self.encryption_mode_combo.setCurrentText("CTR")
        self.encryption_mode_combo.currentTextChanged.connect(self.change_encryption_mode)
        encryption_layout.addWidget(self.encryption_mode_combo, 1, 1)
        
        # Key Management
        encryption_layout.addWidget(QLabel("Encryption Key:"), 2, 0)
        self.encryption_key_display = QLineEdit()
        self.encryption_key_display.setPlaceholderText("Key will be generated automatically")
        self.encryption_key_display.setReadOnly(True)
        encryption_layout.addWidget(self.encryption_key_display, 2, 1)
        
        # Key Export/Import Buttons
        key_buttons_layout = QHBoxLayout()
        self.export_key_btn = QPushButton("Export Key")
        self.export_key_btn.clicked.connect(self.export_encryption_key)
        self.import_key_btn = QPushButton("Import Key")
        self.import_key_btn.clicked.connect(self.import_encryption_key)
        key_buttons_layout.addWidget(self.export_key_btn)
        key_buttons_layout.addWidget(self.import_key_btn)
        encryption_layout.addLayout(key_buttons_layout, 3, 0, 1, 2)
        
        # Encryption Statistics
        self.encryption_stats_label = QLabel("Encryption Statistics:\nPackets Encrypted: 0\nPackets Decrypted: 0\nEncryption Success Rate: 0%")
        encryption_layout.addWidget(self.encryption_stats_label, 4, 0, 1, 2)
        
        security_layout.addWidget(encryption_group)
        
        # Data Recovery Controls Group
        recovery_group = QGroupBox("📡 Data Recovery Algorithm")
        recovery_layout = QGridLayout(recovery_group)
        
        # Recovery Enable/Disable
        self.recovery_enabled_cb = QCheckBox("Enable Data Recovery")
        self.recovery_enabled_cb.setChecked(True)
        self.recovery_enabled_cb.stateChanged.connect(self.toggle_recovery)
        recovery_layout.addWidget(self.recovery_enabled_cb, 0, 0, 1, 2)
        
        # Recovery Mode Selection
        recovery_layout.addWidget(QLabel("Recovery Mode:"), 1, 0)
        self.recovery_mode_combo = QComboBox()
        self.recovery_mode_combo.addItems([
            "Adaptive", "Retry Only", "Redundant Packets", 
            "Erasure Coding", "Disabled"
        ])
        self.recovery_mode_combo.setCurrentText("Adaptive")
        self.recovery_mode_combo.currentTextChanged.connect(self.change_recovery_mode)
        recovery_layout.addWidget(self.recovery_mode_combo, 1, 1)
        
        # Recovery Parameters
        recovery_layout.addWidget(QLabel("Max Retry Attempts:"), 2, 0)
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(1, 10)
        self.max_retries_spin.setValue(5)
        self.max_retries_spin.valueChanged.connect(self.update_recovery_params)
        recovery_layout.addWidget(self.max_retries_spin, 2, 1)
        
        recovery_layout.addWidget(QLabel("Redundancy Factor:"), 3, 0)
        self.redundancy_factor_spin = QSpinBox()
        self.redundancy_factor_spin.setRange(1, 5)
        self.redundancy_factor_spin.setValue(2)
        self.redundancy_factor_spin.valueChanged.connect(self.update_recovery_params)
        recovery_layout.addWidget(self.redundancy_factor_spin, 3, 1)
        
        # Recovery Status Display
        self.recovery_status_label = QLabel("Recovery Status: Standby")
        self.recovery_status_label.setStyleSheet("QLabel { background-color: #e6f3ff; padding: 5px; border-radius: 3px; }")
        recovery_layout.addWidget(self.recovery_status_label, 4, 0, 1, 2)
        
        # Recovery Statistics
        self.recovery_stats_label = QLabel("Recovery Statistics:\nPackets Failed: 0\nPackets Recovered: 0\nRecovery Success Rate: 0%")
        recovery_layout.addWidget(self.recovery_stats_label, 5, 0, 1, 2)
        
        # Recovery Queue Status
        self.recovery_queue_label = QLabel("Recovery Queue: 0 packets pending")
        recovery_layout.addWidget(self.recovery_queue_label, 6, 0, 1, 2)
        
        security_layout.addWidget(recovery_group)
        
        # Security Events Log Group
        events_group = QGroupBox("🛡️ Security Events Log")
        events_layout = QVBoxLayout(events_group)
        
        self.security_events_text = QTextEdit()
        self.security_events_text.setMaximumHeight(150)
        self.security_events_text.setReadOnly(True)
        self.security_events_text.setPlaceholderText("Security events will appear here...")
        events_layout.addWidget(self.security_events_text)
        
        # Clear Events Button
        clear_events_btn = QPushButton("Clear Events Log")
        clear_events_btn.clicked.connect(self.clear_security_events)
        events_layout.addWidget(clear_events_btn)
        
        security_layout.addWidget(events_group)
        
        # Security Recommendations Group
        recommendations_group = QGroupBox("💡 Security Recommendations")
        recommendations_layout = QVBoxLayout(recommendations_group)
        
        self.recommendations_label = QLabel("No recommendations at this time.")
        self.recommendations_label.setWordWrap(True)
        self.recommendations_label.setStyleSheet("QLabel { background-color: #f0f8ff; padding: 8px; border-radius: 5px; }")
        recommendations_layout.addWidget(self.recommendations_label)
        
        security_layout.addWidget(recommendations_group)
        
        layout.addWidget(security_group)
        
        return panel
        
    def setup_worker_connections(self):
        """Setup connections to simulation worker signals"""
        self.simulation_worker.snr_updated.connect(self.update_snr)
        self.simulation_worker.ber_updated.connect(self.update_ber)
        self.simulation_worker.throughput_updated.connect(self.update_throughput)
        self.simulation_worker.frequency_changed.connect(self.update_frequency)
        self.simulation_worker.modulation_changed.connect(self.update_modulation)
        self.simulation_worker.constellation_updated.connect(self.update_constellation)
        self.simulation_worker.log_message.connect(self.add_log_message)
        self.simulation_worker.progress_updated.connect(self.update_progress)
        # Connect new security and sub-band SNR signals
        self.simulation_worker.security_event_logged.connect(self.add_security_log)
        self.simulation_worker.subband_snr_updated.connect(self.update_subband_snr)
        
    def setup_status_bar(self):
        """Setup status bar"""
        self.statusBar().showMessage("Ready")
        
    # Slot methods for UI updates
    def update_snr(self, snr: float):
        current_time = time.time()
        self.snr_label.setText(f"{snr:.1f} dB")
        self.snr_plotter.update_plot(current_time, snr)
        
    def update_ber(self, ber: float):
        current_time = time.time()
        self.ber_label.setText(f"{ber:.2e}")
        self.ber_plotter.update_plot(current_time, np.log10(ber))
        
    def update_throughput(self, throughput: float):
        current_time = time.time()
        self.throughput_label.setText(f"{throughput:.1f} Mbps")
        self.throughput_plotter.update_plot(current_time, throughput)
        
    def update_frequency(self, band: int):
        current_time = time.time()
        self.frequency_label.setText(f"Band {band}")
        self.freq_plotter.update_plot(current_time, band)
        
    def update_modulation(self, modulation: str):
        self.modulation_label.setText(modulation)
        
    def update_constellation(self, constellation: np.ndarray):
        self.constellation_plot.update_constellation(constellation)
        
    def add_log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def update_progress(self, progress: int):
        self.progress_bar.setValue(progress)
        
    def add_security_log(self, message: str, level: str):
        """Add security event to log with appropriate formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "WARNING":
            formatted_message = f"[{timestamp}] ⚠️  {message}"
        elif level == "INFO":
            formatted_message = f"[{timestamp}] 🔒 {message}"
        else:
            formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        
    def update_subband_snr(self, all_band_snrs: np.ndarray, active_band: int):
        """Update sub-band SNR visualization"""
        if hasattr(self, 'subband_snr_plot'):
            self.subband_snr_plot.update_subband_data(all_band_snrs, active_band)
        
    # Event handlers
    def on_scenario_changed(self, scenario_text: str):
        scenario_map = {
            "Phase 1: No Jammer": "no_jammer",
            "Phase 2: Pattern-based Jammer": "pattern_jammer",
            "Phase 3: Random Jammer": "random_jammer"
        }
        scenario = scenario_map.get(scenario_text, "no_jammer")
        self.simulation_worker.set_scenario(scenario)
        self.add_log_message(f"Scenario changed to: {scenario_text}")
        
    def on_file_size_changed(self, size_text: str):
        size_map = {
            "100 MB": 100e6,
            "1 GB": 1e9,
            "10 GB": 10e9
        }
        size = size_map.get(size_text, 100e6)
        self.simulation_worker.set_file_size(size)
        self.add_log_message(f"File size set to: {size_text}")
        
    def on_speed_changed(self, speed: int):
        self.simulation_worker.set_simulation_speed(speed)
        self.speed_label.setText(f"{speed}x")
        
    def start_simulation(self):
        if not self.simulation_running:
            self.simulation_running = True
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            
            # Clear previous data
            self.snr_plotter.clear_plot()
            self.ber_plotter.clear_plot()
            self.throughput_plotter.clear_plot()
            self.freq_plotter.clear_plot()
            self.subband_snr_plot.clear_plot()
            self.progress_bar.setValue(0)
            
            # Start simulation thread
            self.simulation_worker.start_simulation()
            self.simulation_worker.start()
            
            self.statusBar().showMessage("Simulation running...")
            
    def pause_simulation(self):
        self.simulation_worker.pause_simulation()
        
    def stop_simulation(self):
        if self.simulation_running:
            self.simulation_worker.stop_simulation()
            self.simulation_worker.wait()
            
            self.simulation_running = False
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            
            self.statusBar().showMessage("Simulation stopped")
    
    def toggle_encryption(self):
        """Toggle encryption on/off"""
        enabled = self.encryption_enabled_cb.isChecked()
        mode = self.encryption_mode_combo.currentText()
        
        if hasattr(self.simulation_worker, 'enhanced_simulator') and self.simulation_worker.enhanced_simulator:
            self.simulation_worker.enhanced_simulator.set_encryption_enabled(enabled, mode)
            
            # Update key display
            if enabled:
                key = self.simulation_worker.enhanced_simulator.export_security_key()
                if key:
                    self.encryption_key_display.setText(key[:16] + "..." if len(key) > 16 else key)
            else:
                self.encryption_key_display.setText("")
        else:
            self.log_security_event("⚠️ Enhanced simulator not available")
            
        self.log_security_event(
            f"🔒 Encryption {'enabled' if enabled else 'disabled'} (Mode: {mode})"
        )
    
    def change_encryption_mode(self, mode: str):
        """Change encryption mode"""
        if self.encryption_enabled_cb.isChecked():
            self.toggle_encryption()  # Re-initialize with new mode
    
    def export_encryption_key(self):
        """Export encryption key to file"""
        if hasattr(self.simulation_worker, 'enhanced_simulator'):
            key = self.simulation_worker.enhanced_simulator.export_security_key()
            if key:
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Export Encryption Key", "encryption_key.txt", "Text Files (*.txt)"
                )
                if filename:
                    try:
                        with open(filename, 'w') as f:
                            f.write(key)
                        QMessageBox.information(self, "Success", "Encryption key exported successfully!")
                        self.log_security_event(f"🔑 Encryption key exported to {filename}")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to export key: {e}")
            else:
                QMessageBox.warning(self, "Warning", "No encryption key available to export!")
    
    def import_encryption_key(self):
        """Import encryption key from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Encryption Key", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    key = f.read().strip()
                
                if hasattr(self.simulation_worker, 'enhanced_simulator'):
                    self.simulation_worker.enhanced_simulator.import_security_key(key)
                    self.encryption_key_display.setText(key[:16] + "..." if len(key) > 16 else key)
                    QMessageBox.information(self, "Success", "Encryption key imported successfully!")
                    self.log_security_event(f"🔑 Encryption key imported from {filename}")
                else:
                    QMessageBox.warning(self, "Warning", "Enhanced simulator not available!")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import key: {e}")
    
    def toggle_recovery(self):
        """Toggle data recovery on/off"""
        enabled = self.recovery_enabled_cb.isChecked()
        mode_str = self.recovery_mode_combo.currentText()
        
        # Convert mode string to enum
        mode_map = {
            "Disabled": "disabled",
            "Retry Only": "retry_only", 
            "Redundant Packets": "redundant_packets",
            "Erasure Coding": "erasure_coding",
            "Adaptive": "adaptive"
        }
        
        if hasattr(self.simulation_worker, 'enhanced_simulator') and self.simulation_worker.enhanced_simulator:
            if ENHANCED_FEATURES_AVAILABLE:
                # Convert string to RecoveryMode enum
                mode_value = mode_map.get(mode_str, "adaptive")
                try:
                    recovery_mode = RecoveryMode(mode_value)
                    self.simulation_worker.enhanced_simulator.set_recovery_enabled(enabled, recovery_mode)
                except:
                    # Fallback if enum not available
                    self.simulation_worker.enhanced_simulator.set_recovery_enabled(enabled)
            else:
                self.simulation_worker.enhanced_simulator.set_recovery_enabled(enabled)
        else:
            self.log_security_event("⚠️ Enhanced simulator not available")
            
        status = "Active" if enabled else "Disabled"
        self.recovery_status_label.setText(f"Recovery Status: {status}")
        self.recovery_status_label.setStyleSheet(
            f"QLabel {{ background-color: {'#d4edda' if enabled else '#f8d7da'}; "
            f"color: {'#155724' if enabled else '#721c24'}; padding: 5px; border-radius: 3px; }}"
        )
        
        self.log_security_event(
            f"📡 Data recovery {'enabled' if enabled else 'disabled'} (Mode: {mode_str})"
        )
    
    def change_recovery_mode(self, mode: str):
        """Change recovery mode"""
        if self.recovery_enabled_cb.isChecked():
            self.toggle_recovery()  # Re-initialize with new mode
    
    def update_recovery_params(self):
        """Update recovery parameters"""
        max_retries = self.max_retries_spin.value()
        redundancy = self.redundancy_factor_spin.value()
        
        if hasattr(self.simulation_worker, 'enhanced_simulator') and \
           hasattr(self.simulation_worker.enhanced_simulator, 'recovery_manager') and \
           self.simulation_worker.enhanced_simulator.recovery_manager:
            
            recovery_mgr = self.simulation_worker.enhanced_simulator.recovery_manager
            recovery_mgr.max_retry_attempts = max_retries
            recovery_mgr.redundancy_factor = redundancy
            
        self.log_security_event(
            f"⚙️ Recovery parameters updated: Max retries={max_retries}, Redundancy={redundancy}"
        )
    
    def log_security_event(self, message: str):
        """Log security events to the events display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        event_text = f"[{timestamp}] {message}"
        
        self.security_events_text.append(event_text)
        
        # Auto-scroll to bottom
        scrollbar = self.security_events_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_security_events(self):
        """Clear security events log"""
        self.security_events_text.clear()
        self.log_security_event("🗑️ Security events log cleared")
    
    def update_security_displays(self):
        """Update security-related displays with current statistics"""
        if not hasattr(self.simulation_worker, 'enhanced_simulator'):
            return
        
        simulator = self.simulation_worker.enhanced_simulator
        
        # Update encryption statistics
        if simulator.encryption_processor:
            enc_stats = simulator.encryption_processor.get_encryption_stats()
            if enc_stats.get('encryption_enabled', False):
                enc_text = (
                    f"Encryption Statistics:\n"
                    f"Packets Encrypted: {enc_stats.get('packets_encrypted', 0)}\n"
                    f"Packets Decrypted: {enc_stats.get('packets_decrypted', 0)}\n"
                    f"Encryption Success Rate: {enc_stats.get('encryption_success_rate', 0)*100:.1f}%\n"
                    f"Authentication Success Rate: {enc_stats.get('authentication_success_rate', 0)*100:.1f}%"
                )
            else:
                enc_text = "Encryption Statistics:\nEncryption is disabled"
            self.encryption_stats_label.setText(enc_text)
        
        # Update recovery statistics
        if simulator.recovery_manager:
            rec_stats = simulator.recovery_manager.get_recovery_stats()
            rec_text = (
                f"Recovery Statistics:\n"
                f"Packets Failed: {rec_stats.get('packets_failed', 0)}\n"
                f"Packets Recovered: {rec_stats.get('packets_recovered', 0)}\n"
                f"Recovery Success Rate: {rec_stats.get('recovery_success_rate', 0)*100:.1f}%\n"
                f"Retry Success Rate: {rec_stats.get('retry_success_rate', 0)*100:.1f}%"
            )
            self.recovery_stats_label.setText(rec_text)
            
            # Update recovery queue status
            queue_size = rec_stats.get('pending_recoveries', 0)
            retry_queue = rec_stats.get('retry_queue_size', 0)
            queue_text = f"Recovery Queue: {queue_size} failed packets, {retry_queue} pending retries"
            self.recovery_queue_label.setText(queue_text)
            
            # Update recommendations
            recommendations = simulator.recovery_manager.get_recovery_recommendations()
            if recommendations.get('recommendations'):
                rec_text = "Current Recommendations:\n" + "\n".join(recommendations['recommendations'])
            else:
                rec_text = "No specific recommendations at this time.\nSystem operating normally."
            self.recommendations_label.setText(rec_text)
    
    def handle_security_events(self):
        """Handle security events from the simulation worker"""
        # This would be called by signals from the simulation worker
        # For now, we'll update displays periodically
        self.update_security_displays()
        
        # Check for specific security events
        if hasattr(self.simulation_worker, 'enhanced_simulator'):
            simulator = self.simulation_worker.enhanced_simulator
            
            # Check for high failure rates
            if simulator.recovery_manager:
                stats = simulator.recovery_manager.get_recovery_stats()
                failure_rate = stats.get('packets_failed', 0) / max(1, stats.get('packets_failed', 0) + stats.get('packets_recovered', 0))
                
                if failure_rate > 0.3:  # 30% failure rate threshold
                    self.log_security_event("⚠️ High packet failure rate detected!")
                    
                if stats.get('pending_recoveries', 0) > 10:
                    self.log_security_event("📊 Recovery queue is getting large - consider adjusting parameters")
                    
            # Check encryption status
            if simulator.encryption_processor:
                enc_stats = simulator.encryption_processor.get_encryption_stats()
                if enc_stats.get('authentication_failures', 0) > 0:
                    self.log_security_event("🚨 Authentication failures detected - possible tampering!")
    
    def export_security_report(self):
        """Export comprehensive security report"""
        if not hasattr(self.simulation_worker, 'enhanced_simulator'):
            QMessageBox.warning(self, "Warning", "Enhanced simulator not available!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Security Report", "security_report.txt", "Text Files (*.txt)"
        )
        
        if filename:
            try:
                simulator = self.simulation_worker.enhanced_simulator
                
                with open(filename, 'w') as f:
                    f.write("TEKNOFEST 2025 - Security and Recovery Report\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Encryption Report
                    f.write("ENCRYPTION STATUS\n")
                    f.write("-" * 20 + "\n")
                    if simulator.encryption_processor:
                        enc_stats = simulator.encryption_processor.get_encryption_stats()
                        f.write(f"Encryption Enabled: {enc_stats.get('encryption_enabled', False)}\n")
                        f.write(f"Encryption Mode: {enc_stats.get('encryption_mode', 'N/A')}\n")
                        f.write(f"Packets Encrypted: {enc_stats.get('packets_encrypted', 0)}\n")
                        f.write(f"Packets Decrypted: {enc_stats.get('packets_decrypted', 0)}\n")
                        f.write(f"Encryption Success Rate: {enc_stats.get('encryption_success_rate', 0)*100:.2f}%\n")
                        f.write(f"Authentication Success Rate: {enc_stats.get('authentication_success_rate', 0)*100:.2f}%\n")
                    else:
                        f.write("Encryption not available\n")
                    
                    f.write("\n")
                    
                    # Recovery Report
                    f.write("RECOVERY STATUS\n")
                    f.write("-" * 15 + "\n")
                    if simulator.recovery_manager:
                        rec_stats = simulator.recovery_manager.get_recovery_stats()
                        f.write(f"Recovery Enabled: {rec_stats.get('recovery_enabled', False)}\n")
                        f.write(f"Recovery Mode: {rec_stats.get('recovery_mode', 'N/A')}\n")
                        f.write(f"Packets Failed: {rec_stats.get('packets_failed', 0)}\n")
                        f.write(f"Packets Recovered: {rec_stats.get('packets_recovered', 0)}\n")
                        f.write(f"Recovery Success Rate: {rec_stats.get('recovery_success_rate', 0)*100:.2f}%\n")
                        f.write(f"Retry Attempts: {rec_stats.get('retry_attempts', 0)}\n")
                        f.write(f"Retry Success Rate: {rec_stats.get('retry_success_rate', 0)*100:.2f}%\n")
                        f.write(f"Pending Recoveries: {rec_stats.get('pending_recoveries', 0)}\n")
                        
                        # Recommendations
                        recommendations = simulator.recovery_manager.get_recovery_recommendations()
                        f.write(f"\nRecommendations:\n")
                        for rec in recommendations.get('recommendations', []):
                            f.write(f"  - {rec}\n")
                    else:
                        f.write("Recovery not available\n")
                    
                    f.write("\n")
                    
                    # Overall Statistics
                    overall_stats = simulator.get_comprehensive_stats()
                    f.write("OVERALL TRANSMISSION STATISTICS\n")
                    f.write("-" * 32 + "\n")
                    f.write(f"Total Packets Sent: {overall_stats.get('total_packets_sent', 0)}\n")
                    f.write(f"Total Packets Received: {overall_stats.get('total_packets_received', 0)}\n")
                    f.write(f"Recovered Packets: {overall_stats.get('recovered_packets', 0)}\n")
                    f.write(f"Failed Recoveries: {overall_stats.get('failed_recoveries', 0)}\n")
                
                QMessageBox.information(self, "Success", f"Security report exported to {filename}")
                self.log_security_event(f"📋 Security report exported to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report: {e}")
                
class SubBandSNRPlot(QWidget):
    """Sub-band SNR visualization widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Sub-band SNR Monitoring')
        self.ax.set_xlabel('Frequency Band')
        self.ax.set_ylabel('SNR (dB)')
        self.ax.grid(True)
        
        # Initialize with 5 frequency bands
        self.band_labels = ['Band 1\n(2.4 GHz)', 'Band 2\n(2.45 GHz)', 'Band 3\n(2.5 GHz)', 
                           'Band 4\n(2.55 GHz)', 'Band 5\n(2.6 GHz)']
        self.band_positions = np.arange(len(self.band_labels))
        
        # Initialize bars
        self.bars = self.ax.bar(self.band_positions, [0]*len(self.band_labels), 
                               color='lightblue', alpha=0.7)
        self.ax.set_xticks(self.band_positions)
        self.ax.set_xticklabels(self.band_labels)
        self.ax.set_ylim(0, 30)
        
        # Add legend
        self.ax.legend(['Inactive Band', 'Active Band', 'Jammed/Poor SNR'], 
                      loc='upper right')
        
    def update_subband_data(self, all_band_snrs: np.ndarray, active_band: int):
        """Update sub-band SNR data"""
        # Clear previous plot
        self.ax.clear()
        self.ax.set_title('Sub-band SNR Monitoring')
        self.ax.set_xlabel('Frequency Band')
        self.ax.set_ylabel('SNR (dB)')
        self.ax.grid(True)
        
        # Color code bars: green for active, red for jammed, blue for normal
        colors = []
        for i, snr in enumerate(all_band_snrs):
            if i == active_band - 1:  # Active band (1-indexed to 0-indexed)
                colors.append('green')
            elif snr < 10:  # Jammed/poor SNR
                colors.append('red')
            else:
                colors.append('lightblue')
        
        # Create bars
        bars = self.ax.bar(self.band_positions, all_band_snrs, color=colors, alpha=0.7)
          # Add value labels on bars
        for i, (bar, snr) in enumerate(zip(bars, all_band_snrs)):
            self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{snr:.1f} dB', ha='center', va='bottom', fontsize=9)
        
        self.ax.set_xticks(self.band_positions)
        self.ax.set_xticklabels(self.band_labels)
        self.ax.set_ylim(0, max(30, max(all_band_snrs) + 5))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Active Band'),
                          Patch(facecolor='lightblue', alpha=0.7, label='Available Band'),
                          Patch(facecolor='red', alpha=0.7, label='Jammed/Poor SNR')]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        self.canvas.draw()
        
    def clear_plot(self):
        """Clear the sub-band SNR plot"""
        self.ax.clear()
        self.ax.set_title('Sub-band SNR Monitoring')
        self.ax.set_xlabel('Frequency Band')
        self.ax.set_ylabel('SNR (dB)')
        self.ax.grid(True)
        
        # Reset to initial state
        self.bars = self.ax.bar(self.band_positions, [0]*len(self.band_labels), 
                               color='lightblue', alpha=0.7)
        self.ax.set_xticks(self.band_positions)
        self.ax.set_xticklabels(self.band_labels)
        self.ax.set_ylim(0, 30)
        
        # Add legend
        self.ax.legend(['Inactive Band', 'Active Band', 'Jammed/Poor SNR'], 
                      loc='upper right')
        
        self.canvas.draw()

    def _handle_security_event(self, event_type: str, message: str, data: dict = None):
        """Handle security events from the enhanced simulator"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if event_type == "info":
            emoji = "🔐"
        elif event_type == "warning":
            emoji = "⚠️"
        elif event_type == "critical":
            emoji = "🚨"
        else:
            emoji = "ℹ️"
        
        log_message = f"[{timestamp}] {emoji} Security: {message}"
        self.log_message.emit(log_message)
        
        # Update security statistics if data provided
        if data:
            self.security_stats.update(data)
    
    def _handle_recovery_event(self, event_type: str, message: str, data: dict = None):
        """Handle recovery events from the enhanced simulator"""

        timestamp = datetime.now().strftime("%H:%M:%S")
        if event_type == "info":
            emoji = "📦"
        elif event_type == "warning":
            emoji = "⚠️"
        elif event_type == "retry":
            emoji = "🔄"
        else:
            emoji = "ℹ️"
        
        log_message = f"[{timestamp}] {emoji} Recovery: {message}"
        self.log_message.emit(log_message)
        
        # Update recovery statistics if data provided
        if data:
            self.recovery_stats.update(data)
