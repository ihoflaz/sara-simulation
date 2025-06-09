# Comprehensive System Test Suite

import unittest
import numpy as np
import torch
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from core.modulation import OFDMSystem
from core.channel import ChannelModel
from core.frequency_hopping import AdaptiveFrequencyManager
from core.coding import AdaptiveCoding
from core.data_processing import TransmissionSimulator
from ai.cnn_model import FrequencyHoppingCNN
from ai.data_generator import SyntheticDataGenerator
from ai.training import ModelTrainer

class TestOFDMSystem(unittest.TestCase):
    """Test OFDM modulation system"""
    
    def setUp(self):
        self.ofdm = OFDMSystem()
        
    def test_modulation_schemes(self):
        """Test all modulation schemes"""
        test_data = np.random.randint(0, 2, 1000)
        
        for scheme in MODULATION_SCHEMES.keys():
            with self.subTest(scheme=scheme):
                modulated = self.ofdm.modulate(test_data, scheme)
                self.assertIsInstance(modulated, np.ndarray)
                self.assertTrue(len(modulated) > 0)
                
    def test_ofdm_transmission(self):
        """Test complete OFDM transmission"""
        test_bits = np.random.randint(0, 2, 640)  # Multiple of subcarriers
        
        # Modulate
        ofdm_symbols = self.ofdm.ofdm_modulate(test_bits, 'QPSK')
        self.assertIsInstance(ofdm_symbols, np.ndarray)
        
        # Add noise
        noisy_symbols = ofdm_symbols + 0.1 * np.random.randn(*ofdm_symbols.shape)
        
        # Demodulate
        received_bits = self.ofdm.ofdm_demodulate(noisy_symbols, 'QPSK')
        self.assertEqual(len(received_bits), len(test_bits))
        
    def test_channel_estimation(self):
        """Test channel estimation functionality"""
        channel_response = self.ofdm.estimate_channel()
        self.assertIsInstance(channel_response, np.ndarray)
        self.assertEqual(len(channel_response), self.ofdm.num_subcarriers)
        
    def test_adaptive_modulation(self):
        """Test adaptive modulation selection"""
        # Test different SNR levels
        snr_levels = [5, 15, 25, 35]
        for snr in snr_levels:
            with self.subTest(snr=snr):
                modulation = self.ofdm.select_modulation(snr)
                self.assertIn(modulation, MODULATION_SCHEMES.keys())

class TestChannelModel(unittest.TestCase):
    """Test channel simulation"""
    
    def setUp(self):
        self.channel = ChannelModel()
        
    def test_awgn_channel(self):
        """Test AWGN channel"""
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        snr_db = 20
        
        noisy_signal = self.channel.awgn_channel(signal, snr_db)
        self.assertEqual(len(noisy_signal), len(signal))
        self.assertIsInstance(noisy_signal, np.ndarray)
        
    def test_rayleigh_fading(self):
        """Test Rayleigh fading channel"""
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        
        faded_signal = self.channel.rayleigh_fading(signal)
        self.assertEqual(len(faded_signal), len(signal))
        
    def test_jammer_simulation(self):
        """Test jammer interference"""
        scenarios = ['no_jammer', 'pattern_jammer', 'random_jammer']
        
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                self.channel.set_scenario(scenario)
                
                for band in range(1, 6):
                    state = self.channel.simulate_step(0.0, band)
                    
                    self.assertIn('snr', state)
                    self.assertIn('interference', state)
                    self.assertIn('rss', state)
                    self.assertIsInstance(state['snr'], float)

class TestFrequencyHopping(unittest.TestCase):
    """Test frequency hopping system"""
    
    def setUp(self):
        self.freq_hopping = AdaptiveFrequencyManager()
        
    def test_band_selection(self):
        """Test frequency band selection"""
        channel_state = {
            'snr': 20.0,
            'interference': 2.0,
            'rss': -70.0
        }
        
        band = self.freq_hopping.select_frequency_band(channel_state, current_band=1)
        self.assertIn(band, range(1, 6))
        
    def test_jammer_avoidance(self):
        """Test jammer avoidance strategy"""
        # Simulate high interference
        channel_state = {
            'snr': 5.0,
            'interference': 15.0,
            'rss': -90.0
        }
        
        new_band = self.freq_hopping.select_frequency_band(channel_state, current_band=1)
        self.assertIsInstance(new_band, int)
        self.assertIn(new_band, range(1, 6))

class TestChannelCoding(unittest.TestCase):
    """Test channel coding functionality"""
    
    def setUp(self):
        self.coder = AdaptiveCoding()
        
    def test_ldpc_encoding_decoding(self):
        """Test LDPC encoding and decoding"""
        test_data = np.random.randint(0, 2, 100)
        
        # Encode
        encoded = self.coder.ldpc_encode(test_data)
        self.assertTrue(len(encoded) > len(test_data))
        
        # Add some errors
        noisy_encoded = encoded.copy()
        error_positions = np.random.choice(len(noisy_encoded), 5, replace=False)
        noisy_encoded[error_positions] = 1 - noisy_encoded[error_positions]
        
        # Decode
        decoded = self.coder.ldpc_decode(noisy_encoded)
        
        # Check if most bits are correct
        bit_errors = np.sum(test_data != decoded[:len(test_data)])
        error_rate = bit_errors / len(test_data)
        self.assertLess(error_rate, 0.1)  # Less than 10% error rate
        
    def test_turbo_encoding_decoding(self):
        """Test Turbo encoding and decoding"""
        test_data = np.random.randint(0, 2, 100)
        
        # Encode
        encoded = self.coder.turbo_encode(test_data)
        self.assertTrue(len(encoded) > len(test_data))
        
        # Add noise (simulated as bit flips)
        noisy_encoded = encoded.copy()
        error_positions = np.random.choice(len(noisy_encoded), 3, replace=False)
        noisy_encoded[error_positions] = 1 - noisy_encoded[error_positions]
        
        # Decode
        decoded = self.coder.turbo_decode(noisy_encoded)
        
        # Check decoding quality
        self.assertEqual(len(decoded), len(test_data))

class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality"""
    
    def setUp(self):
        self.processor = TransmissionSimulator()
        
    def test_file_packetization(self):
        """Test file packetization"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            test_data = b"Hello, TEKNOFEST!" * 1000
            temp_file.write(test_data)
            temp_file_path = temp_file.name
            
        try:
            packets = self.processor.packetize_file(temp_file_path)
            self.assertIsInstance(packets, list)
            self.assertTrue(len(packets) > 0)
            
            # Test packet structure
            for packet in packets[:5]:  # Check first 5 packets
                self.assertIn('header', packet)
                self.assertIn('data', packet)
                self.assertIn('crc', packet)
                
        finally:
            os.unlink(temp_file_path)
            
    def test_crc_calculation(self):
        """Test CRC calculation and verification"""
        test_data = b"Test data for CRC"
        
        crc = self.processor.calculate_crc(test_data)
        self.assertIsInstance(crc, int)
        
        # Verify CRC
        is_valid = self.processor.verify_crc(test_data, crc)
        self.assertTrue(is_valid)
        
        # Test with corrupted data
        corrupted_data = test_data + b"x"
        is_valid_corrupted = self.processor.verify_crc(corrupted_data, crc)
        self.assertFalse(is_valid_corrupted)

class TestCNNModel(unittest.TestCase):
    """Test CNN model functionality"""
    
    def setUp(self):
        self.model = FrequencyHoppingCNN()
        
    def test_model_forward_pass(self):
        """Test model forward pass"""
        # Create dummy input (batch_size=2, input_size=30)
        test_input = torch.randn(2, 30)
        
        with torch.no_grad():
            output = self.model(test_input)
            
        self.assertEqual(output.shape, (2, 5))  # 5 frequency bands
        
        # Check if output is valid probability distribution
        probabilities = torch.softmax(output, dim=1)
        for i in range(2):
            self.assertAlmostEqual(torch.sum(probabilities[i]).item(), 1.0, places=5)
            
    def test_model_parameters(self):
        """Test model parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)

class TestDataGenerator(unittest.TestCase):
    """Test synthetic data generator"""
    
    def setUp(self):
        self.generator = SyntheticDataGenerator()
        
    def test_snr_profile_generation(self):
        """Test SNR profile generation"""
        length = 1000
        snr_profile = self.generator.generate_snr_profile(length)
        
        self.assertEqual(len(snr_profile), length)
        self.assertTrue(np.all(snr_profile >= -10))  # Within realistic range
        self.assertTrue(np.all(snr_profile <= 40))
        
    def test_interference_profile_generation(self):
        """Test interference profile generation"""
        length = 1000
        band_id = 3
        
        # Test with jammer
        interference_with_jammer = self.generator.generate_interference_profile(
            length, band_id, jammer_present=True
        )
        
        # Test without jammer
        interference_no_jammer = self.generator.generate_interference_profile(
            length, band_id, jammer_present=False
        )
        
        self.assertEqual(len(interference_with_jammer), length)
        self.assertEqual(len(interference_no_jammer), length)
        
        # Interference should generally be higher with jammer
        self.assertGreater(
            np.mean(interference_with_jammer), 
            np.mean(interference_no_jammer)
        )
        
    def test_training_batch_generation(self):
        """Test training batch generation"""
        batch_size = 4
        sequence_length = 50
        
        features, labels = self.generator.generate_training_batch(
            batch_size, sequence_length
        )
        
        self.assertEqual(features.shape[0], batch_size)
        self.assertEqual(features.shape[1], sequence_length)
        self.assertEqual(labels.shape[0], batch_size)
        self.assertEqual(labels.shape[1], sequence_length)
        self.assertEqual(labels.shape[2], 5)  # 5 frequency bands

class TestSystemIntegration(unittest.TestCase):
    """Test system integration"""
    
    def test_end_to_end_transmission(self):
        """Test complete transmission chain"""
        # Initialize components
        ofdm = OFDMSystem()
        channel = ChannelModel()
        
        # Generate test data
        test_bits = np.random.randint(0, 2, 640)
        
        # Transmit
        modulated = ofdm.ofdm_modulate(test_bits, 'QPSK')
        
        # Channel effects
        channel_output = channel.awgn_channel(modulated, snr_db=15)
        
        # Receive
        received_bits = ofdm.ofdm_demodulate(channel_output, 'QPSK')
        
        # Check transmission quality
        bit_errors = np.sum(test_bits != received_bits)
        ber = bit_errors / len(test_bits)
        
        # With SNR=15dB, BER should be reasonably low
        self.assertLess(ber, 0.1)
        
    @patch('matplotlib.pyplot.show')
    def test_simulation_runner(self, mock_show):
        """Test simulation runner (without GUI)"""
        try:
            from simulation.competition_simulator import CompetitionSimulator
            
            # Create simulator with minimal settings
            simulator = CompetitionSimulator(enable_ai=False, enable_gui=False)
            
            # Run a short simulation for phase 1
            results = simulator.simulate_phase(
                phase_number=1,
                file_size=1e6,  # 1 MB
                duration=5.0    # 5 seconds
            )
            
            # Check results structure
            self.assertIn('phase', results)
            self.assertIn('success', results)
            self.assertIn('snr_history', results)
            self.assertIn('throughput_history', results)
            
        except ImportError:
            self.skipTest("Simulation module not available")

def run_performance_test():
    """Run performance benchmarks"""
    print("\nRunning performance tests...")
    
    import time
    
    # Test OFDM performance
    ofdm = OFDMSystem()
    test_bits = np.random.randint(0, 2, 6400)  # Larger dataset
    
    start_time = time.time()
    for _ in range(100):
        modulated = ofdm.ofdm_modulate(test_bits, 'QPSK')
    ofdm_time = time.time() - start_time
    
    print(f"OFDM modulation (100 iterations): {ofdm_time:.3f}s")
    
    # Test CNN inference performance
    model = FrequencyHoppingCNN()
    model.eval()
    test_input = torch.randn(32, 30)  # Batch of 32
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = model(test_input)
    cnn_time = time.time() - start_time
    
    print(f"CNN inference (100 iterations, batch=32): {cnn_time:.3f}s")
    
    # Test data generation performance
    generator = SyntheticDataGenerator()
    
    start_time = time.time()
    features, labels = generator.generate_training_batch(batch_size=16, sequence_length=100)
    data_gen_time = time.time() - start_time
    
    print(f"Data generation (batch=16, seq=100): {data_gen_time:.3f}s")

def main():
    """Run all tests"""
    print("TEKNOFEST 5G Communication System - Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestOFDMSystem,
        TestChannelModel,
        TestFrequencyHopping,
        TestChannelCoding,
        TestDataProcessor,
        TestCNNModel,
        TestDataGenerator,
        TestSystemIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance tests
    run_performance_test()
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
        
    print(f"Total tests run: {result.testsRun}")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
