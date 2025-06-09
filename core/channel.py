"""
Channel Modeling and Interference Simulation
Includes jammer simulation for competition scenarios
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Dict, Optional
import time

from config import COMPETITION_CONFIG, JAMMER_PATTERNS, FREQUENCY_BANDS, SIMULATION_CONFIG

class ChannelModel:
    """Simulate wireless channel with fading, noise, and interference"""
    
    def __init__(self, snr_db: float = 20):
        self.snr_db = snr_db
        self.noise_variance = self._calculate_noise_variance()
        self.multipath_enabled = True
        self.fading_enabled = True
        
    def _calculate_noise_variance(self) -> float:
        """Calculate noise variance from SNR"""
        snr_linear = 10 ** (self.snr_db / 10)
        return 1.0 / snr_linear
    
    def add_awgn(self, signal: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
        """Add Additive White Gaussian Noise"""
        if snr_db is not None:
            snr_linear = 10 ** (snr_db / 10)
            noise_var = 1.0 / snr_linear
        else:
            noise_var = self.noise_variance
        
        noise_real = np.random.normal(0, np.sqrt(noise_var/2), len(signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_var/2), len(signal))
        noise = noise_real + 1j * noise_imag
        
        return signal + noise
    
    def rayleigh_fading(self, signal: np.ndarray, fd_norm: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Rayleigh fading to signal
        fd_norm: Normalized Doppler frequency (fd * Ts)
        """
        if not self.fading_enabled:
            return signal, np.ones(len(signal))
        
        # Generate complex Gaussian samples
        h_real = np.random.normal(0, 1/np.sqrt(2), len(signal))
        h_imag = np.random.normal(0, 1/np.sqrt(2), len(signal))
        h_complex = h_real + 1j * h_imag
        
        # Apply Doppler filtering if fd_norm > 0
        if fd_norm > 0:
            # Design Doppler filter
            fc = fd_norm  # Cutoff frequency
            b, a = signal.butter(2, fc, 'lowpass')
            h_complex = signal.filtfilt(b, a, h_complex)
        
        # Apply fading to signal
        faded_signal = signal * h_complex
        
        return faded_signal, h_complex
    
    def multipath_channel(self, signal: np.ndarray, 
                         delays: List[float] = [0, 1e-6, 2e-6],
                         gains: List[float] = [1.0, 0.5, 0.25]) -> np.ndarray:
        """
        Apply multipath channel with specified delays and gains
        """
        if not self.multipath_enabled:
            return signal
        
        # Convert delays to sample delays
        fs = SIMULATION_CONFIG['sampling_rate']
        delay_samples = [int(delay * fs) for delay in delays]
        
        # Create output signal
        max_delay = max(delay_samples)
        output = np.zeros(len(signal) + max_delay, dtype=complex)
        
        # Add each path
        for delay_samp, gain in zip(delay_samples, gains):
            start_idx = delay_samp
            end_idx = start_idx + len(signal)
            output[start_idx:end_idx] += gain * signal
        
        return output[:len(signal)]
    
    def apply_channel(self, signal: np.ndarray, snr_db: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """Apply complete channel model"""
        channel_info = {}
        
        # Apply multipath
        signal_mp = self.multipath_channel(signal)
        
        # Apply fading
        signal_faded, fading_coeff = self.rayleigh_fading(signal_mp)
        channel_info['fading_coefficients'] = fading_coeff
        
        # Add noise
        received_signal = self.add_awgn(signal_faded, snr_db)
        
        channel_info['snr_db'] = snr_db if snr_db is not None else self.snr_db
        
        return received_signal, channel_info
    
    def set_scenario(self, scenario: str):
        """Set simulation scenario"""
        self.scenario = scenario
        # Initialize jammer simulator if needed
        if not hasattr(self, 'jammer'):
            self.jammer = JammerSimulator()
        
        if scenario == 'no_jammer':
            self.jammer.set_mode('off')
        elif scenario == 'pattern_jammer':
            self.jammer.set_mode('pattern')
        elif scenario == 'random_jammer':
            self.jammer.set_mode('random')
    
    def simulate_step(self, current_time: float, current_band: int) -> Dict:
        """Simulate channel state for one time step"""
        # Base SNR
        base_snr = self.snr_db + np.random.normal(0, 2)  # Add some variation
        
        # Initialize jammer if not exists
        if not hasattr(self, 'jammer'):
            self.jammer = JammerSimulator()
            
        # Check for jamming
        jammed_band = self.jammer.get_current_jammed_band()
        interference = 1.0  # Base interference
        
        if jammed_band == current_band:
            # High interference in jammed band
            interference = 15.0 + np.random.exponential(5.0)
            base_snr -= 10  # Reduce SNR due to jamming
        else:
            # Random interference in other bands
            interference = np.random.exponential(1.5)
        
        # RSS (Received Signal Strength)
        rss = -70 + np.random.normal(0, 5)  # dBm
        
        return {
            'snr': max(base_snr, 0),
            'interference': interference,
            'rss': rss,
            'jammed_band': jammed_band,
            'current_band': current_band
        }

class JammerSimulator:
    """Simulate jammer interference for competition scenarios"""
    
    def __init__(self):
        self.jammer_power_dbm = SIMULATION_CONFIG['jammer_power']
        self.switch_time = COMPETITION_CONFIG['jammer_switch_time']
        self.pattern_sequence = JAMMER_PATTERNS['pattern']
        self.current_band = 1
        self.pattern_index = 0
        self.last_switch_time = time.time()
        self.jammer_active = False
        
        # Jammer modes
        self.mode = 'off'  # 'off', 'pattern', 'random'
        
    def set_mode(self, mode: str):
        """Set jammer mode: 'off', 'pattern', 'random'"""
        if mode in ['off', 'pattern', 'random']:
            self.mode = mode
            if mode == 'off':
                self.jammer_active = False
            else:
                self.jammer_active = True
            print(f"Jammer mode set to: {mode}")
    
    def get_current_jammed_band(self) -> Optional[int]:
        """Get currently jammed band based on mode and timing"""
        if not self.jammer_active:
            return None
        
        current_time = time.time()
        time_since_switch = current_time - self.last_switch_time
        
        # Check if it's time to switch
        if time_since_switch >= self.switch_time:
            self._switch_jammer_band()
            self.last_switch_time = current_time
        
        return self.current_band
    
    def _switch_jammer_band(self):
        """Switch jammer to next band based on current mode"""
        if self.mode == 'pattern':
            # Follow the competition pattern: 1 → 3 → 5 → 4 → 2
            self.current_band = self.pattern_sequence[self.pattern_index]
            self.pattern_index = (self.pattern_index + 1) % len(self.pattern_sequence)
            
        elif self.mode == 'random':
            # Random band selection
            available_bands = list(range(1, 6))  # Bands 1-5
            self.current_band = np.random.choice(available_bands)
    
    def generate_jammer_signal(self, signal_length: int, target_band: int) -> np.ndarray:
        """Generate jammer signal for specific band"""
        if not self.jammer_active or target_band != self.current_band:
            return np.zeros(signal_length, dtype=complex)
        
        # Generate high-power interference signal
        # Jammer uses maximum power across the entire band
        fs = SIMULATION_CONFIG['sampling_rate']
        t = np.arange(signal_length) / fs
        
        # Generate wideband interference (covers entire sub-band)
        bandwidth = COMPETITION_CONFIG['total_bandwidth'] / COMPETITION_CONFIG['num_subbands']
        
        # Create multi-tone jammer
        num_tones = 10
        jammer_signal = np.zeros(signal_length, dtype=complex)
        
        for i in range(num_tones):
            freq_offset = (i - num_tones/2) * bandwidth / num_tones
            phase = np.random.uniform(0, 2*np.pi)
            tone = np.exp(1j * (2 * np.pi * freq_offset * t + phase))
            jammer_signal += tone
        
        # Normalize and apply jammer power
        jammer_signal = jammer_signal / np.sqrt(num_tones)
        power_linear = 10 ** (self.jammer_power_dbm / 10) / 1000  # Convert dBm to watts
        jammer_signal *= np.sqrt(power_linear)
        
        return jammer_signal
    
    def apply_jamming(self, signal: np.ndarray, target_band: int) -> np.ndarray:
        """Apply jamming to signal if target band is jammed"""
        jammer_signal = self.generate_jammer_signal(len(signal), target_band)
        return signal + jammer_signal
    
    def get_jammer_status(self) -> Dict:
        """Get current jammer status information"""
        return {
            'mode': self.mode,
            'active': self.jammer_active,
            'current_band': self.current_band if self.jammer_active else None,
            'pattern_index': self.pattern_index,
            'power_dbm': self.jammer_power_dbm,
            'switch_time': self.switch_time
        }
    
    def reset_jammer(self):
        """Reset jammer to initial state"""
        self.current_band = 1
        self.pattern_index = 0
        self.last_switch_time = time.time()

class SpectrumAnalyzer:
    """Analyze spectrum for jammer detection and channel quality assessment"""
    
    def __init__(self, nfft: int = 1024):
        self.nfft = nfft
        self.detection_threshold = -80  # dBm
        
    def analyze_spectrum(self, signal: np.ndarray, fs: float) -> Dict:
        """
        Analyze signal spectrum and detect interference
        """
        # Compute power spectral density
        freqs, psd = signal.welch(signal, fs=fs, nperseg=self.nfft, 
                                 return_onesided=False, scaling='density')
        
        # Convert to dBm
        psd_dbm = 10 * np.log10(psd * 1000)  # Convert to dBm
        
        # Detect interference peaks
        interference_detected = np.any(psd_dbm > self.detection_threshold)
        
        # Find peak frequencies
        peak_indices = signal.find_peaks(psd_dbm, height=self.detection_threshold)[0]
        peak_frequencies = freqs[peak_indices]
        peak_powers = psd_dbm[peak_indices]
        
        return {
            'frequencies': freqs,
            'psd_dbm': psd_dbm,
            'interference_detected': interference_detected,
            'peak_frequencies': peak_frequencies,
            'peak_powers': peak_powers,
            'noise_floor': np.median(psd_dbm),
            'peak_power': np.max(psd_dbm)
        }
    
    def estimate_snr(self, signal: np.ndarray, noise_bw_hz: float = 1e6) -> float:
        """
        Estimate SNR from received signal
        """
        fs = SIMULATION_CONFIG['sampling_rate']
        spectrum_info = self.analyze_spectrum(signal, fs)
        
        noise_floor = spectrum_info['noise_floor']
        peak_power = spectrum_info['peak_power']
        
        # Simple SNR estimation
        snr_db = peak_power - noise_floor
        return max(snr_db, 0)  # Ensure non-negative SNR
    
    def detect_jammed_bands(self, signal: np.ndarray) -> List[int]:
        """
        Detect which frequency bands are currently jammed
        """
        fs = SIMULATION_CONFIG['sampling_rate']
        spectrum_info = self.analyze_spectrum(signal, fs)
        
        jammed_bands = []
        
        # Check each frequency band for interference
        for band_num in range(1, 6):
            band_info = FREQUENCY_BANDS[band_num]
            center_freq = band_info['center']
            bandwidth = band_info['end'] - band_info['start']
            
            # Find frequency indices corresponding to this band
            freq_mask = (np.abs(spectrum_info['frequencies'] - center_freq) <= bandwidth/2)
            
            if np.any(freq_mask):
                band_power = np.max(spectrum_info['psd_dbm'][freq_mask])
                
                # Check if power exceeds threshold (indicating jamming)
                if band_power > self.detection_threshold + 10:  # 10 dB margin
                    jammed_bands.append(band_num)
        
        return jammed_bands

class ChannelQualityEstimator:
    """Estimate channel quality metrics for adaptive transmission"""
    
    def __init__(self):
        self.history_length = 100
        self.snr_history = []
        self.ber_history = []
        
    def update_metrics(self, snr_db: float, ber: float):
        """Update channel quality metrics"""
        self.snr_history.append(snr_db)
        self.ber_history.append(ber)
        
        # Maintain history length
        if len(self.snr_history) > self.history_length:
            self.snr_history.pop(0)
            self.ber_history.pop(0)
    
    def get_average_snr(self) -> float:
        """Get average SNR over recent history"""
        if not self.snr_history:
            return 0
        return np.mean(self.snr_history)
    
    def get_average_ber(self) -> float:
        """Get average BER over recent history"""
        if not self.ber_history:
            return 1
        return np.mean(self.ber_history)
    
    def predict_channel_quality(self) -> str:
        """Predict channel quality: 'excellent', 'good', 'fair', 'poor'"""
        avg_snr = self.get_average_snr()
        avg_ber = self.get_average_ber()
        
        if avg_snr > 25 and avg_ber < 1e-6:
            return 'excellent'
        elif avg_snr > 15 and avg_ber < 1e-4:
            return 'good'
        elif avg_snr > 10 and avg_ber < 1e-2:
            return 'fair'
        else:
            return 'poor'
    
    def recommend_adaptation(self) -> Dict:
        """Recommend system adaptations based on channel quality"""
        quality = self.predict_channel_quality()
        avg_snr = self.get_average_snr()
        
        recommendations = {
            'modulation': 'QPSK',
            'coding_rate': 0.5,
            'transmit_power': 'normal',
            'reason': quality
        }
        
        if quality == 'excellent':
            recommendations['modulation'] = '256QAM' if avg_snr > 28 else '64QAM'
            recommendations['coding_rate'] = 0.8
        elif quality == 'good':
            recommendations['modulation'] = '64QAM' if avg_snr > 20 else '16QAM'
            recommendations['coding_rate'] = 0.75
        elif quality == 'fair':
            recommendations['modulation'] = '16QAM' if avg_snr > 12 else 'QPSK'
            recommendations['coding_rate'] = 0.5
        else:  # poor
            recommendations['modulation'] = 'BPSK'
            recommendations['coding_rate'] = 0.25
            recommendations['transmit_power'] = 'high'
        
        return recommendations
