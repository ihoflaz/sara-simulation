"""
OFDM and Modulation Schemes Implementation
5G-based adaptive modulation system for wireless communication
"""

import numpy as np
import scipy.signal as signal
from scipy import fft
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from config import OFDM_CONFIG, MODULATION_SCHEMES, COMPETITION_CONFIG

class ModulationScheme:
    """Base class for modulation schemes"""
    
    def __init__(self, scheme_name: str):
        self.scheme_name = scheme_name
        self.order = MODULATION_SCHEMES[scheme_name]['order']
        self.bits_per_symbol = MODULATION_SCHEMES[scheme_name]['bits_per_symbol']
        self.constellation = self._generate_constellation()
    
    def _generate_constellation(self) -> np.ndarray:
        """Generate constellation points for the modulation scheme"""
        if self.scheme_name == 'BPSK':
            return np.array([-1, 1])
        elif self.scheme_name == 'QPSK':
            return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        else:
            # QAM constellations
            m = int(np.sqrt(self.order))
            real_levels = np.linspace(-(m-1), m-1, m)
            imag_levels = np.linspace(-(m-1), m-1, m)
            constellation = []
            for i in real_levels:
                for q in imag_levels:
                    constellation.append(complex(i, q))
            return np.array(constellation) / np.sqrt(np.mean(np.abs(constellation)**2)) * np.sqrt(2)
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate binary data to complex symbols"""
        # Reshape bits into symbols
        if len(bits) % self.bits_per_symbol != 0:
            padding = self.bits_per_symbol - (len(bits) % self.bits_per_symbol)
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        symbols_bits = bits.reshape(-1, self.bits_per_symbol)
        
        # Convert to decimal indices
        indices = np.sum(symbols_bits * (2 ** np.arange(self.bits_per_symbol)[::-1]), axis=1)
        
        # Map to constellation
        return self.constellation[indices]
    
    def demodulate(self, symbols: np.ndarray, noise_var: float = 0.1) -> np.ndarray:
        """Demodulate complex symbols to binary data"""
        # Find closest constellation points
        distances = np.abs(symbols[:, np.newaxis] - self.constellation[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        
        # Convert indices to bits
        bits_list = []
        for idx in indices:
            bits = np.array([int(b) for b in format(idx, f'0{self.bits_per_symbol}b')])
            bits_list.append(bits)
        
        return np.concatenate(bits_list)

class OFDMSystem:
    """OFDM Implementation for 5G-like waveform"""
    
    def __init__(self, modulation_scheme: str = 'QPSK'):
        self.num_subcarriers = OFDM_CONFIG['num_subcarriers']
        self.cp_length = OFDM_CONFIG['cyclic_prefix_length']
        self.pilot_spacing = OFDM_CONFIG['pilot_spacing']
        self.guard_carriers = OFDM_CONFIG['guard_carriers']
        
        self.modulator = ModulationScheme(modulation_scheme)
        self.current_modulation = modulation_scheme
        
        # Calculate usable subcarriers
        self.data_subcarriers = self._get_data_subcarrier_indices()
        self.pilot_subcarriers = self._get_pilot_subcarrier_indices()
        
        # Generate pilot symbols
        self.pilot_symbols = self._generate_pilot_symbols()
    
    def _get_data_subcarrier_indices(self) -> np.ndarray:
        """Get indices of data subcarriers (excluding pilots and guards)"""
        total_indices = np.arange(self.num_subcarriers)
        
        # Remove guard carriers
        guard_start = self.num_subcarriers - self.guard_carriers // 2
        guard_end = self.guard_carriers // 2
        usable_indices = total_indices[guard_end:guard_start]
        
        # Remove pilot locations
        pilot_indices = usable_indices[::self.pilot_spacing]
        data_indices = np.setdiff1d(usable_indices, pilot_indices)
        
        return data_indices
    
    def _get_pilot_subcarrier_indices(self) -> np.ndarray:
        """Get indices of pilot subcarriers"""
        guard_start = self.num_subcarriers - self.guard_carriers // 2
        guard_end = self.guard_carriers // 2
        usable_indices = np.arange(guard_end, guard_start)
        
        return usable_indices[::self.pilot_spacing]
    
    def _generate_pilot_symbols(self) -> np.ndarray:
        """Generate known pilot symbols for channel estimation"""
        np.random.seed(42)  # Fixed seed for reproducible pilots
        return np.exp(1j * np.random.uniform(0, 2*np.pi, len(self.pilot_subcarriers)))
    
    def change_modulation(self, new_scheme: str):
        """Adaptively change modulation scheme"""
        if new_scheme in MODULATION_SCHEMES:
            self.modulator = ModulationScheme(new_scheme)
            self.current_modulation = new_scheme
            print(f"Modulation changed to: {new_scheme}")
    
    def ofdm_modulate(self, data_bits: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Complete OFDM modulation process
        Returns: (time_domain_signal, metadata)
        """
        # Modulate data bits to symbols
        data_symbols = self.modulator.modulate(data_bits)
        
        # Calculate symbols per OFDM frame
        symbols_per_frame = len(self.data_subcarriers)
        
        # Pad data if necessary
        if len(data_symbols) % symbols_per_frame != 0:
            padding_needed = symbols_per_frame - (len(data_symbols) % symbols_per_frame)
            data_symbols = np.concatenate([data_symbols, np.zeros(padding_needed)])
        
        # Reshape into OFDM frames
        num_frames = len(data_symbols) // symbols_per_frame
        data_frames = data_symbols.reshape(num_frames, symbols_per_frame)
        
        ofdm_symbols = []
        
        for frame_data in data_frames:
            # Create frequency domain symbol
            freq_symbol = np.zeros(self.num_subcarriers, dtype=complex)
            
            # Insert data symbols
            freq_symbol[self.data_subcarriers] = frame_data
            
            # Insert pilot symbols
            freq_symbol[self.pilot_subcarriers] = self.pilot_symbols
            
            # IFFT to time domain
            time_symbol = fft.ifft(freq_symbol, norm='ortho')
            
            # Add cyclic prefix
            cp_symbol = np.concatenate([time_symbol[-self.cp_length:], time_symbol])
            
            ofdm_symbols.append(cp_symbol)
        
        # Concatenate all OFDM symbols
        signal_out = np.concatenate(ofdm_symbols)
        
        metadata = {
            'num_frames': num_frames,
            'modulation': self.current_modulation,
            'data_symbols_per_frame': symbols_per_frame,
            'pilot_positions': self.pilot_subcarriers,
            'data_positions': self.data_subcarriers
        }
        
        return signal_out, metadata
    
    def ofdm_demodulate(self, received_signal: np.ndarray, metadata: dict, 
                       channel_estimate: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Complete OFDM demodulation process
        """
        num_frames = metadata['num_frames']
        symbol_length = self.num_subcarriers + self.cp_length
        
        # Split signal into OFDM symbols
        received_symbols = received_signal[:num_frames * symbol_length].reshape(num_frames, symbol_length)
        
        demodulated_data = []
        
        for i, symbol in enumerate(received_symbols):
            # Remove cyclic prefix
            symbol_no_cp = symbol[self.cp_length:]
            
            # FFT to frequency domain
            freq_symbol = fft.fft(symbol_no_cp, norm='ortho')
            
            # Channel equalization if estimate provided
            if channel_estimate is not None:
                freq_symbol = freq_symbol / (channel_estimate + 1e-10)
            
            # Extract data symbols
            data_symbols = freq_symbol[self.data_subcarriers]
            
            # Demodulate to bits
            data_bits = self.modulator.demodulate(data_symbols)
            demodulated_data.append(data_bits)
        
        return np.concatenate(demodulated_data)
    
    def estimate_channel(self, received_signal: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Estimate channel using pilot symbols
        """
        num_frames = metadata['num_frames']
        symbol_length = self.num_subcarriers + self.cp_length
        
        received_symbols = received_signal[:num_frames * symbol_length].reshape(num_frames, symbol_length)
        
        channel_estimates = []
        
        for symbol in received_symbols:
            # Remove CP and FFT
            symbol_no_cp = symbol[self.cp_length:]
            freq_symbol = fft.fft(symbol_no_cp, norm='ortho')
            
            # Extract received pilots
            received_pilots = freq_symbol[self.pilot_subcarriers]
            
            # Estimate channel at pilot positions
            pilot_channel = received_pilots / self.pilot_symbols
            
            # Interpolate channel across all subcarriers
            channel_full = np.zeros(self.num_subcarriers, dtype=complex)
            channel_full[self.pilot_subcarriers] = pilot_channel
            
            # Simple linear interpolation for data subcarriers
            for i in range(len(self.pilot_subcarriers) - 1):
                start_idx = self.pilot_subcarriers[i]
                end_idx = self.pilot_subcarriers[i + 1]
                
                if end_idx - start_idx > 1:
                    interp_indices = np.arange(start_idx + 1, end_idx)
                    interp_values = np.linspace(
                        pilot_channel[i], 
                        pilot_channel[i + 1], 
                        len(interp_indices) + 2
                    )[1:-1]
                    channel_full[interp_indices] = interp_values
            
            channel_estimates.append(channel_full)
        
        return np.mean(channel_estimates, axis=0)
    
    def calculate_snr(self, transmitted: np.ndarray, received: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(np.abs(transmitted) ** 2)
        noise_power = np.mean(np.abs(received - transmitted) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        return 10 * np.log10(snr_linear)
    
    def adaptive_modulation_selection(self, snr_db: float) -> str:
        """
        Select modulation scheme based on SNR
        """
        # SNR thresholds for different modulation schemes
        snr_thresholds = {
            'BPSK': 0,
            'QPSK': 6,
            '16QAM': 12,
            '64QAM': 18,
            '256QAM': 24,
            '1024QAM': 30
        }
        
        # Select highest order modulation that meets SNR requirement
        selected_mod = 'BPSK'
        for mod, threshold in snr_thresholds.items():
            if snr_db >= threshold:
                selected_mod = mod
        
        return selected_mod

    def modulate_data(self, data_bits: np.ndarray, modulation_scheme: str) -> np.ndarray:
        """
        Modulate data bits using specified modulation scheme
        This is a compatibility method for testing
        """
        # Change modulation if different
        if modulation_scheme != self.current_modulation:
            self.change_modulation(modulation_scheme)
        
        # Perform OFDM modulation
        signal, metadata = self.ofdm_modulate(data_bits)
        return signal

    def select_modulation(self, snr_db: float) -> str:
        """
        Select modulation scheme based on SNR (compatibility method)
        """
        return self.adaptive_modulation_selection(snr_db)

    def generate_constellation_samples(self, modulation_scheme: str, num_samples: int = 100) -> np.ndarray:
        """
        Generate constellation samples for visualization
        Used by GUI to display constellation diagrams
        """
        # Create a temporary modulation object for the requested scheme
        if modulation_scheme in MODULATION_SCHEMES:
            temp_modulator = ModulationScheme(modulation_scheme)
            
            # Generate random data bits
            num_bits = num_samples * temp_modulator.bits_per_symbol
            random_bits = np.random.randint(0, 2, num_bits)
            
            # Modulate to get constellation points
            constellation_symbols = temp_modulator.modulate(random_bits)
            
            # Add some noise for realistic visualization
            noise_power = 0.05
            noise = np.sqrt(noise_power) * (np.random.randn(len(constellation_symbols)) + 
                                          1j * np.random.randn(len(constellation_symbols)))
            
            return constellation_symbols + noise
        else:
            # Fallback to QPSK if scheme not found
            temp_modulator = ModulationScheme('QPSK')
            num_bits = num_samples * temp_modulator.bits_per_symbol
            random_bits = np.random.randint(0, 2, num_bits)
            return temp_modulator.modulate(random_bits)
            
class FrequencyManager:
    """Manage frequency band selection and switching"""
    
    def __init__(self):
        self.frequency_bands = COMPETITION_CONFIG['num_subbands']
        self.current_band = 1
        self.band_history = []
        
    def get_band_frequency(self, band_number: int) -> float:
        """Get center frequency for specific band"""
        from config import FREQUENCY_BANDS
        return FREQUENCY_BANDS[band_number]['center']
    
    def switch_band(self, new_band: int) -> bool:
        """Switch to new frequency band"""
        if 1 <= new_band <= self.frequency_bands:
            self.current_band = new_band
            self.band_history.append(new_band)
            return True
        return False
    
    def get_available_bands(self, jammed_bands: List[int]) -> List[int]:
        """Get list of available (non-jammed) bands"""
        all_bands = list(range(1, self.frequency_bands + 1))
        return [band for band in all_bands if band not in jammed_bands]
