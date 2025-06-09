# Competition Simulation Environment

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from core.modulation import OFDMSystem
from core.channel import ChannelModel
from core.frequency_hopping import AdaptiveFrequencyManager
from core.coding import AdaptiveCoding
from core.data_processing import TransmissionSimulator
from ai.cnn_model import FrequencyHoppingCNN
from ai.training import ModelTrainer
import torch

class CompetitionSimulator:
    """
    Complete competition simulation environment
    Simulates all three phases of the TEKNOFEST competition
    """
    
    def __init__(self, enable_ai: bool = True, enable_gui: bool = False):
        self.enable_ai = enable_ai
        self.enable_gui = enable_gui
        
        # Logging setup - initialize first
        self.logger = logging.getLogger(__name__)
        
        # Initialize all system components
        self.ofdm_system = OFDMSystem()
        self.channel_sim = ChannelModel()
        self.freq_hopping = AdaptiveFrequencyManager()
        self.channel_coder = AdaptiveCoding()
        self.data_processor = TransmissionSimulator()
        
        # AI model setup
        self.ai_model = None
        if enable_ai:
            self.setup_ai_model()
            
        # Competition phases
        self.phases = {
            1: {'name': 'No Jammer', 'scenario': 'no_jammer'},
            2: {'name': 'Pattern-based Jammer', 'scenario': 'pattern_jammer'},
            3: {'name': 'Random Jammer', 'scenario': 'random_jammer'}
        }
        
        # Results storage
        self.results = {}
        
    def setup_ai_model(self):
        """Setup and load AI model"""
        try:
            # Try to load pre-trained model first to get config
            model_path = 'models/frequency_hopping_model.pth'
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Get model config from checkpoint
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    self.ai_model = FrequencyHoppingCNN(
                        input_size=config['input_size'],
                        sequence_length=config['sequence_length'],
                        num_frequencies=config['num_frequencies']
                    )
                else:
                    # Fallback to default with saved model's dimensions
                    self.ai_model = FrequencyHoppingCNN(input_size=30)
                
                self.ai_model.load_state_dict(checkpoint['model_state_dict'])
                self.ai_model.eval()
                self.logger.info("Pre-trained AI model loaded successfully")
            else:
                self.logger.info("No pre-trained model found, will train new model")
                self.ai_model = FrequencyHoppingCNN(input_size=30)
                self.train_ai_model()
                
        except Exception as e:
            self.logger.error(f"Failed to setup AI model: {str(e)}")
            self.enable_ai = False
            
    def train_ai_model(self):
        """Train AI model if not available"""
        self.logger.info("Training new AI model...")
        
        trainer = ModelTrainer(self.ai_model)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train with reasonable dataset size
        results = trainer.train(
            num_samples=2000,
            save_path='models/frequency_hopping_model.pth'
        )
        
        # Plot training curves
        fig = trainer.plot_training_curves('models/training_curves.png')
        plt.close(fig)  # Close to prevent display in headless mode
        
        self.logger.info(f"AI model training completed. Final accuracy: {results['test_accuracy']:.4f}")
        
    def simulate_phase(self, phase_number: int, file_size: float = 100e6, 
                      duration: float = 60.0) -> Dict:
        """
        Simulate a single competition phase
        
        Args:
            phase_number: Competition phase (1, 2, or 3)
            file_size: File size in bytes
            duration: Simulation duration in seconds
            
        Returns:
            Dictionary with simulation results
        """
        
        phase_info = self.phases[phase_number]
        scenario = phase_info['scenario']
        
        self.logger.info(f"Starting Phase {phase_number}: {phase_info['name']}")
        self.logger.info(f"File size: {file_size/1e6:.1f} MB, Duration: {duration}s")
        
        # Configure channel simulator for this phase
        self.channel_sim.set_scenario(scenario)
        
        # Initialize metrics
        metrics = {
            'phase': phase_number,
            'scenario': scenario,
            'file_size_mb': file_size / 1e6,
            'duration': duration,
            'start_time': datetime.now().isoformat(),
            'snr_history': [],
            'ber_history': [],
            'throughput_history': [],
            'frequency_history': [],
            'modulation_history': [],
            'total_bits_transmitted': 0,
            'total_bits_received': 0,
            'total_errors': 0,
            'frequency_switches': 0,
            'modulation_switches': 0,
            'success': False
        }
        
        # Simulation parameters
        total_bits = int(file_size * 8)
        bits_per_step = 10000  # Bits transmitted per time step
        time_step = 0.1  # 100ms time steps
        
        current_time = 0.0
        transmitted_bits = 0
        current_frequency = 1
        current_modulation = 'QPSK'
        
        # Simulation loop
        while current_time < duration and transmitted_bits < total_bits:
            # Channel simulation step
            channel_state = self.channel_sim.simulate_step(current_time, current_frequency)
            
            # Record SNR
            snr_db = channel_state['snr']
            metrics['snr_history'].append(snr_db)
            
            # Adaptive modulation based on channel quality
            new_modulation = self.select_modulation(snr_db)
            if new_modulation != current_modulation:
                current_modulation = new_modulation
                metrics['modulation_switches'] += 1
            metrics['modulation_history'].append(current_modulation)
            
            # Frequency hopping decision
            if self.enable_ai and self.ai_model is not None:
                new_frequency = self.ai_frequency_decision(channel_state, current_frequency)
            else:
                new_frequency = self.rule_based_frequency_decision(
                    channel_state, current_frequency, scenario, current_time
                )
                
            if new_frequency != current_frequency:
                current_frequency = new_frequency
                metrics['frequency_switches'] += 1
            metrics['frequency_history'].append(current_frequency)
            
            # Calculate BER and transmission performance
            ber = self.calculate_ber(snr_db, current_modulation)
            metrics['ber_history'].append(ber)
            
            # Simulate transmission
            step_bits = min(bits_per_step, total_bits - transmitted_bits)
            error_bits = int(step_bits * ber)
            received_bits = step_bits - error_bits
            
            transmitted_bits += step_bits
            metrics['total_bits_transmitted'] = transmitted_bits
            metrics['total_bits_received'] += received_bits
            metrics['total_errors'] += error_bits
            
            # Calculate throughput
            if current_time > 0:
                throughput_mbps = (metrics['total_bits_received'] / 1e6) / current_time
                metrics['throughput_history'].append(throughput_mbps)
            else:
                metrics['throughput_history'].append(0)
                
            current_time += time_step
            
            # Progress logging
            if int(current_time) % 10 == 0 and int(current_time * 10) % 10 == 0:
                progress = (transmitted_bits / total_bits) * 100
                self.logger.info(f"Phase {phase_number} progress: {progress:.1f}%, "
                               f"SNR: {snr_db:.1f}dB, Band: {current_frequency}, "
                               f"Mod: {current_modulation}")
                
        # Calculate final results
        metrics['end_time'] = datetime.now().isoformat()
        metrics['actual_duration'] = current_time
        metrics['success'] = transmitted_bits >= total_bits * 0.95  # 95% threshold
        
        if len(metrics['snr_history']) > 0:
            metrics['avg_snr'] = np.mean(metrics['snr_history'])
            metrics['min_snr'] = np.min(metrics['snr_history'])
            metrics['max_snr'] = np.max(metrics['snr_history'])
            
        if len(metrics['ber_history']) > 0:
            metrics['avg_ber'] = np.mean(metrics['ber_history'])
            
        if len(metrics['throughput_history']) > 0:
            metrics['avg_throughput'] = np.mean(metrics['throughput_history'])
            metrics['final_throughput'] = metrics['throughput_history'][-1]
            
        metrics['transmission_efficiency'] = (metrics['total_bits_received'] / 
                                            max(metrics['total_bits_transmitted'], 1))
        
        self.logger.info(f"Phase {phase_number} completed!")
        self.logger.info(f"Success: {metrics['success']}")
        self.logger.info(f"Transmission efficiency: {metrics['transmission_efficiency']:.3f}")
        self.logger.info(f"Average throughput: {metrics.get('avg_throughput', 0):.2f} Mbps")
        
        return metrics
        
    def select_modulation(self, snr_db: float) -> str:
        """Select appropriate modulation scheme based on SNR"""
        if snr_db > 30:
            return '1024QAM'
        elif snr_db > 25:
            return '256QAM'
        elif snr_db > 20:
            return '64QAM'
        elif snr_db > 15:
            return '16QAM'
        elif snr_db > 10:
            return 'QPSK'
        else:
            return 'BPSK'
            
    def calculate_ber(self, snr_db: float, modulation: str) -> float:
        """Calculate BER based on SNR and modulation"""
        # SNR penalties for different modulation schemes
        modulation_penalties = {
            'BPSK': 0,
            'QPSK': 3,
            '16QAM': 6,
            '64QAM': 9,
            '256QAM': 12,
            '1024QAM': 15
        }
        
        effective_snr = snr_db - modulation_penalties.get(modulation, 0)
        
        # Theoretical BER calculation (simplified)
        if effective_snr > 25:
            return 1e-9
        elif effective_snr > 20:
            return 1e-7
        elif effective_snr > 15:
            return 1e-5
        elif effective_snr > 10:
            return 1e-4
        elif effective_snr > 5:
            return 1e-3
        elif effective_snr > 0:
            return 1e-2
        else:
            return 1e-1
            
    def ai_frequency_decision(self, channel_state: Dict, current_band: int) -> int:
        """AI-based frequency band selection"""
        try:
            # Prepare features for all bands
            features = []
            for band in range(1, 6):  # 5 bands
                band_features = [
                    channel_state['snr'] if band == current_band else channel_state['snr'] + np.random.randn() * 2,
                    channel_state['interference'] if band == current_band else np.random.exponential(2),
                    channel_state['rss'] if band == current_band else channel_state['rss'] + np.random.randn() * 3,
                    20.0,  # Coherence time
                    5.0,   # Doppler spread  
                    0.1    # Delay spread
                ]
                features.extend(band_features)
                
            # AI prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                probabilities = self.ai_model(features_tensor)
                predicted_band = torch.argmax(probabilities, dim=1).item() + 1
                
            return predicted_band
            
        except Exception as e:
            self.logger.error(f"AI frequency decision error: {str(e)}")
            return self.rule_based_frequency_decision(channel_state, current_band, 'no_jammer', 0)
            
    def rule_based_frequency_decision(self, channel_state: Dict, current_band: int,
                                    scenario: str, current_time: float) -> int:
        """Rule-based frequency band selection"""
        
        if scenario == 'pattern_jammer':
            # Known jammer pattern: 1 -> 3 -> 5 -> 4 -> 2
            jammer_pattern = [1, 3, 5, 4, 2]
            time_index = int(current_time) % len(jammer_pattern)
            jammer_band = jammer_pattern[time_index]
            
            # Avoid jammer band
            available_bands = [b for b in range(1, 6) if b != jammer_band]
            if current_band in available_bands and channel_state['snr'] > 10:
                return current_band  # Stay if good quality
            else:
                return np.random.choice(available_bands)
                
        elif scenario == 'random_jammer':
            # Random jammer - use quality-based decision
            current_quality = channel_state['snr'] - 0.5 * channel_state['interference']
            
            if current_quality < 8:  # Switch if quality is poor
                # Try different band
                return (current_band % 5) + 1
            else:
                return current_band
                
        else:  # no_jammer
            # Stay in current band unless quality is very poor
            if channel_state['snr'] < 5:
                return (current_band % 5) + 1
            else:
                return current_band
                
    def run_full_competition(self, file_sizes: List[float] = None) -> Dict:
        """
        Run complete competition simulation (all phases)
        
        Args:
            file_sizes: List of file sizes for each phase [Phase1, Phase2, Phase3]
                       Default: [100MB, 1GB, 10GB]
        """
        
        if file_sizes is None:
            file_sizes = [100e6, 1e9, 10e9]  # 100MB, 1GB, 10GB
            
        self.logger.info("Starting full TEKNOFEST competition simulation...")
        
        competition_results = {
            'start_time': datetime.now().isoformat(),
            'ai_enabled': self.enable_ai,
            'phases': {},
            'overall_score': 0,
            'success_phases': 0
        }
        
        for phase in [1, 2, 3]:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"PHASE {phase}: {self.phases[phase]['name']}")
            self.logger.info(f"{'='*50}")
            
            # Run phase simulation
            phase_results = self.simulate_phase(
                phase_number=phase,
                file_size=file_sizes[phase-1],
                duration=120.0  # 2 minutes per phase
            )
            
            competition_results['phases'][phase] = phase_results
            
            # Calculate phase score
            phase_score = self.calculate_phase_score(phase_results)
            phase_results['score'] = phase_score
            competition_results['overall_score'] += phase_score
            
            if phase_results['success']:
                competition_results['success_phases'] += 1
                
            self.logger.info(f"Phase {phase} score: {phase_score:.2f}/100")
            
        competition_results['end_time'] = datetime.now().isoformat()
        competition_results['final_score'] = competition_results['overall_score'] / 3
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("COMPETITION RESULTS")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Overall Score: {competition_results['final_score']:.2f}/100")
        self.logger.info(f"Successful Phases: {competition_results['success_phases']}/3")
        self.logger.info(f"AI Model Used: {self.enable_ai}")
        
        # Save results
        self.save_results(competition_results)
        
        return competition_results
        
    def calculate_phase_score(self, phase_results: Dict) -> float:
        """Calculate score for a phase (0-100)"""
        score = 0.0
        
        # Base score for completion
        if phase_results['success']:
            score += 40.0
            
        # Transmission efficiency score (0-30)
        efficiency = phase_results['transmission_efficiency']
        score += efficiency * 30.0
        
        # Throughput score (0-20)
        if 'avg_throughput' in phase_results and phase_results['avg_throughput'] > 0:
            # Normalize throughput (assume max ~50 Mbps)
            throughput_score = min(phase_results['avg_throughput'] / 50.0, 1.0) * 20.0
            score += throughput_score
            
        # BER score (0-10)
        if 'avg_ber' in phase_results:
            # Lower BER is better
            ber_score = max(0, 10.0 - np.log10(phase_results['avg_ber']))
            score += min(ber_score, 10.0)
            
        return min(score, 100.0)
        
    def save_results(self, results: Dict, filename: str = None):
        """Save competition results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"competition_results_{timestamp}.json"
            
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
                
        results_json = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
            
        self.logger.info(f"Results saved to: {filepath}")
        
    def plot_results(self, results: Dict, save_plots: bool = True):
        """Generate plots for competition results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TEKNOFEST Competition Results', fontsize=16)
        
        for phase in [1, 2, 3]:
            if phase in results['phases']:
                phase_data = results['phases'][phase]
                col = phase - 1
                
                # SNR plot
                if phase_data['snr_history']:
                    time_axis = np.arange(len(phase_data['snr_history'])) * 0.1
                    axes[0, col].plot(time_axis, phase_data['snr_history'])
                    axes[0, col].set_title(f'Phase {phase}: SNR')
                    axes[0, col].set_ylabel('SNR (dB)')
                    axes[0, col].set_xlabel('Time (s)')
                    axes[0, col].grid(True)
                    
                # Throughput plot
                if phase_data['throughput_history']:
                    time_axis = np.arange(len(phase_data['throughput_history'])) * 0.1
                    axes[1, col].plot(time_axis, phase_data['throughput_history'])
                    axes[1, col].set_title(f'Phase {phase}: Throughput')
                    axes[1, col].set_ylabel('Throughput (Mbps)')
                    axes[1, col].set_xlabel('Time (s)')
                    axes[1, col].grid(True)
                    
        plt.tight_layout()
        
        if save_plots:
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'results/competition_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
            
        return fig

def main():
    """Main simulation runner"""
    print("TEKNOFEST 5G Communication Competition Simulator")
    print("=" * 50)
    
    # Create simulator
    simulator = CompetitionSimulator(enable_ai=True, enable_gui=False)
    
    # Run full competition
    results = simulator.run_full_competition()
    
    # Generate plots
    fig = simulator.plot_results(results, save_plots=True)
    plt.show()
    
    print(f"\nSimulation completed! Check 'results' folder for detailed output.")

if __name__ == "__main__":
    main()
