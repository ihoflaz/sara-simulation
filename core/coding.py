"""
Channel Coding Implementation
LDPC and Turbo coding for error correction
"""

import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import CODING_CONFIG

class LDPCEncoder:
    """LDPC Encoder Implementation"""
    
    def __init__(self, code_rate: float = 0.5, block_length: int = 1944):
        self.code_rate = code_rate
        self.block_length = block_length
        self.info_length = int(block_length * code_rate)
        self.parity_length = block_length - self.info_length
        
        # Generate parity check matrix (simplified)
        self.H = self._generate_parity_check_matrix()
        self.G = self._generate_generator_matrix()
    
    def _generate_parity_check_matrix(self) -> np.ndarray:
        """Generate sparse parity check matrix"""
        # Simplified LDPC matrix generation
        # In practice, use structured codes like QC-LDPC
        
        np.random.seed(42)  # For reproducible matrices
        H = np.zeros((self.parity_length, self.block_length), dtype=int)
        
        # Create regular LDPC code with column weight 3 and row weight varies
        col_weight = 3
        
        for col in range(self.block_length):
            # Randomly place 1s in this column
            row_positions = np.random.choice(self.parity_length, col_weight, replace=False)
            H[row_positions, col] = 1
        
        return H
    
    def _generate_generator_matrix(self) -> np.ndarray:
        """Generate generator matrix from parity check matrix"""
        # Simplified generator matrix (not optimal)
        # In practice, use systematic form: G = [I | P]
        
        G = np.zeros((self.info_length, self.block_length), dtype=int)
        
        # Identity part
        G[:, :self.info_length] = np.eye(self.info_length, dtype=int)
        
        # Parity part (simplified)
        if self.parity_length > 0:
            parity_part = np.random.randint(0, 2, (self.info_length, self.parity_length))
            G[:, self.info_length:] = parity_part
        
        return G
    
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode information bits using LDPC code"""
        # Pad information bits if necessary
        if len(info_bits) % self.info_length != 0:
            padding = self.info_length - (len(info_bits) % self.info_length)
            info_bits = np.concatenate([info_bits, np.zeros(padding, dtype=int)])
        
        # Encode in blocks
        encoded_blocks = []
        for i in range(0, len(info_bits), self.info_length):
            block = info_bits[i:i + self.info_length]
            
            # Systematic encoding: [info_bits | parity_bits]
            parity_bits = self._calculate_parity(block)
            encoded_block = np.concatenate([block, parity_bits])
            encoded_blocks.append(encoded_block)
        
        return np.concatenate(encoded_blocks)
    
    def _calculate_parity(self, info_bits: np.ndarray) -> np.ndarray:
        """Calculate parity bits for information block"""
        # Simplified parity calculation
        # Should use proper LDPC encoding algorithm
        
        parity = np.zeros(self.parity_length, dtype=int)
        
        # Simple parity calculation based on generator matrix
        for i in range(self.parity_length):
            parity[i] = np.sum(info_bits * self.G[:, self.info_length + i]) % 2
        
        return parity
    
    def syndrome_check(self, received_bits: np.ndarray) -> bool:
        """Check if received codeword is valid"""
        syndrome = np.dot(self.H, received_bits) % 2
        return np.all(syndrome == 0)

class LDPCDecoder:
    """LDPC Decoder using Sum-Product Algorithm"""
    
    def __init__(self, encoder: LDPCEncoder, max_iterations: int = 50):
        self.encoder = encoder
        self.max_iterations = max_iterations
        self.H = encoder.H
        
    def decode(self, received_llr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Decode using belief propagation (sum-product algorithm)
        received_llr: Log-likelihood ratios of received bits
        """
        n = len(received_llr)
        m = self.H.shape[0]
        
        # Initialize messages
        var_to_check = np.zeros((n, m))  # Variable to check node messages
        check_to_var = np.zeros((m, n))  # Check to variable node messages
        
        # Initialize variable to check messages with channel LLR
        for i in range(n):
            for j in range(m):
                if self.H[j, i] == 1:
                    var_to_check[i, j] = received_llr[i]
        
        for iteration in range(self.max_iterations):
            # Check node update
            for j in range(m):
                for i in range(n):
                    if self.H[j, i] == 1:
                        # Product of all other variable messages
                        others = [var_to_check[k, j] for k in range(n) 
                                if k != i and self.H[j, k] == 1]
                        
                        if others:
                            # Sign and magnitude processing
                            signs = [1 if llr >= 0 else -1 for llr in others]
                            magnitudes = [abs(llr) for llr in others]
                            
                            check_to_var[j, i] = (np.prod(signs) * 
                                                 min(magnitudes) if magnitudes else 0)
                        else:
                            check_to_var[j, i] = 0
            
            # Variable node update
            for i in range(n):
                total_llr = received_llr[i]
                for j in range(m):
                    if self.H[j, i] == 1:
                        total_llr += check_to_var[j, i]
                
                # Update messages to check nodes
                for j in range(m):
                    if self.H[j, i] == 1:
                        var_to_check[i, j] = total_llr - check_to_var[j, i]
            
            # Make hard decision
            posterior_llr = np.copy(received_llr)
            for i in range(n):
                for j in range(m):
                    if self.H[j, i] == 1:
                        posterior_llr[i] += check_to_var[j, i]
            
            decoded_bits = (posterior_llr < 0).astype(int)
            
            # Check if valid codeword
            if self.encoder.syndrome_check(decoded_bits):
                return decoded_bits, True
        
        # Return hard decision even if not converged
        return decoded_bits, False

class TurboEncoder:
    """Turbo Encoder Implementation"""
    
    def __init__(self, constraint_length: int = 7, code_rate: float = 1/3):
        self.constraint_length = constraint_length
        self.code_rate = code_rate
        self.memory = constraint_length - 1
        
        # Generator polynomials for rate 1/3 turbo code
        self.generator_polynomials = [0o171, 0o133]  # Octal representation
        
        # Interleaver (simplified random interleaver)
        self.interleaver_size = 1024
        np.random.seed(42)
        self.interleaver = np.random.permutation(self.interleaver_size)
        self.deinterleaver = np.argsort(self.interleaver)
    
    def _convolutional_encode(self, info_bits: np.ndarray, 
                            generator: int) -> np.ndarray:
        """Encode using single convolutional encoder"""
        # Convert generator to binary
        gen_binary = [(generator >> i) & 1 for i in range(self.constraint_length)]
        
        # Initialize shift register
        shift_register = np.zeros(self.constraint_length, dtype=int)
        encoded_bits = []
        
        for bit in info_bits:
            # Shift in new bit
            shift_register = np.roll(shift_register, 1)
            shift_register[0] = bit
            
            # Calculate output
            output = np.sum(shift_register * gen_binary) % 2
            encoded_bits.append(output)
        
        return np.array(encoded_bits)
    
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode using turbo code"""
        # Pad to interleaver size
        if len(info_bits) % self.interleaver_size != 0:
            padding = self.interleaver_size - (len(info_bits) % self.interleaver_size)
            info_bits = np.concatenate([info_bits, np.zeros(padding, dtype=int)])
        
        encoded_blocks = []
        
        for i in range(0, len(info_bits), self.interleaver_size):
            block = info_bits[i:i + self.interleaver_size]
            
            # First encoder (systematic)
            systematic = block
            
            # Second encoder (with interleaving)
            interleaved = block[self.interleaver]
            
            # Encode with both generators
            encoded1 = self._convolutional_encode(block, self.generator_polynomials[0])
            encoded2 = self._convolutional_encode(interleaved, self.generator_polynomials[1])
            
            # Puncture and combine (simplified rate 1/3)
            # Output: [systematic, parity1, parity2]
            turbo_block = []
            for j in range(len(systematic)):
                turbo_block.extend([systematic[j], encoded1[j], encoded2[j]])
            
            encoded_blocks.append(np.array(turbo_block))
        
        return np.concatenate(encoded_blocks)

class TurboDecoder:
    """Turbo Decoder using BCJR Algorithm"""
    
    def __init__(self, encoder: TurboEncoder, max_iterations: int = 8):
        self.encoder = encoder
        self.max_iterations = max_iterations
        
    def decode(self, received_llr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Decode using iterative BCJR algorithm
        """
        # Simplified turbo decoding
        block_length = self.encoder.interleaver_size
        num_blocks = len(received_llr) // (3 * block_length)
        
        decoded_blocks = []
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * 3 * block_length
            
            # Extract systematic, parity1, parity2
            systematic_llr = received_llr[start_idx::3][:block_length]
            parity1_llr = received_llr[start_idx + 1::3][:block_length]
            parity2_llr = received_llr[start_idx + 2::3][:block_length]
            
            # Initialize extrinsic information
            extrinsic1 = np.zeros(block_length)
            extrinsic2 = np.zeros(block_length)
            
            for iteration in range(self.max_iterations):
                # Decoder 1
                decoder1_input = systematic_llr + extrinsic2
                extrinsic1 = self._siso_decode(decoder1_input, parity1_llr)
                
                # Interleave extrinsic information
                extrinsic1_interleaved = extrinsic1[self.encoder.interleaver]
                
                # Decoder 2
                systematic_interleaved = systematic_llr[self.encoder.interleaver]
                decoder2_input = systematic_interleaved + extrinsic1_interleaved
                extrinsic2_interleaved = self._siso_decode(decoder2_input, parity2_llr)
                
                # Deinterleave extrinsic information
                extrinsic2 = extrinsic2_interleaved[self.encoder.deinterleaver]
            
            # Final decision
            final_llr = systematic_llr + extrinsic1 + extrinsic2
            decoded_bits = (final_llr < 0).astype(int)
            decoded_blocks.append(decoded_bits)
        
        return np.concatenate(decoded_blocks), True
    
    def _siso_decode(self, systematic_llr: np.ndarray, 
                    parity_llr: np.ndarray) -> np.ndarray:
        """
        Soft-Input Soft-Output decoder (simplified BCJR)
        """
        # Simplified SISO decoder
        # In practice, implement full BCJR algorithm
        
        # For now, return simplified extrinsic information
        extrinsic = np.zeros_like(systematic_llr)
        
        for i in range(len(systematic_llr)):
            # Simple approximation of extrinsic information
            if i > 0:
                extrinsic[i] = 0.5 * (systematic_llr[i-1] + parity_llr[i])
            else:
                extrinsic[i] = 0.5 * parity_llr[i]
        
        return extrinsic

class ErrorDetection:
    """CRC and Checksum for error detection"""
    
    def __init__(self):
        # CRC-32 polynomial
        self.crc_poly = 0x04C11DB7
        self.crc_table = self._generate_crc_table()
    
    def _generate_crc_table(self) -> np.ndarray:
        """Generate CRC lookup table"""
        table = np.zeros(256, dtype=np.uint32)
        
        for i in range(256):
            crc = i << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ self.crc_poly
                else:
                    crc = crc << 1
                crc = crc & 0xFFFFFFFF
            table[i] = crc
        
        return table
    
    def calculate_crc32(self, data: np.ndarray) -> int:
        """Calculate CRC-32 checksum"""
        # Convert bits to bytes
        if len(data) % 8 != 0:
            padding = 8 - (len(data) % 8)
            data = np.concatenate([data, np.zeros(padding, dtype=int)])
        
        bytes_data = []
        for i in range(0, len(data), 8):
            byte_val = 0
            for j in range(8):
                byte_val += data[i + j] * (2 ** (7 - j))
            bytes_data.append(int(byte_val))
        
        crc = 0xFFFFFFFF
        for byte in bytes_data:
            tbl_idx = ((crc >> 24) ^ byte) & 0xFF
            crc = ((crc << 8) ^ self.crc_table[tbl_idx]) & 0xFFFFFFFF
        
        return crc ^ 0xFFFFFFFF
    
    def verify_crc32(self, data_with_crc: np.ndarray, 
                    expected_crc: int) -> bool:
        """Verify CRC-32 checksum"""
        # Extract data (excluding CRC bits)
        data_bits = data_with_crc[:-32]  # Assuming 32-bit CRC
        calculated_crc = self.calculate_crc32(data_bits)
        return calculated_crc == expected_crc
    
    def add_crc32(self, data: np.ndarray) -> np.ndarray:
        """Add CRC-32 to data"""
        crc = self.calculate_crc32(data)
        
        # Convert CRC to 32 bits
        crc_bits = []
        for i in range(32):
            crc_bits.append((crc >> (31 - i)) & 1)
        
        return np.concatenate([data, np.array(crc_bits)])

class AdaptiveCoding:
    """Adaptive coding scheme selection"""
    
    def __init__(self):
        self.ldpc_encoder = LDPCEncoder()
        self.ldpc_decoder = LDPCDecoder(self.ldpc_encoder)
        self.turbo_encoder = TurboEncoder()
        self.turbo_decoder = TurboDecoder(self.turbo_encoder)
        self.error_detector = ErrorDetection()
        
        self.current_scheme = 'none'
        
    def select_coding_scheme(self, channel_quality: str) -> str:
        """Select appropriate coding scheme based on channel quality"""
        if channel_quality in ['poor', 'fair']:
            return 'turbo'
        elif channel_quality == 'good':
            return 'ldpc'
        else:  # excellent
            return 'none'
    
    def encode_data(self, data: np.ndarray, scheme: str) -> np.ndarray:
        """Encode data with selected scheme"""
        # Add CRC for error detection
        data_with_crc = self.error_detector.add_crc32(data)
        
        if scheme == 'ldpc':
            return self.ldpc_encoder.encode(data_with_crc)
        elif scheme == 'turbo':
            return self.turbo_encoder.encode(data_with_crc)
        else:  # no coding
            return data_with_crc
    
    def decode_data(self, received_data: np.ndarray, 
                   scheme: str) -> Tuple[np.ndarray, bool]:
        """Decode data with selected scheme"""
        if scheme == 'ldpc':
            # Convert to LLR (simplified)
            llr = 2 * received_data.real - 1  # Assuming BPSK
            decoded, success = self.ldpc_decoder.decode(llr)
        elif scheme == 'turbo':
            # Convert to LLR (simplified)
            llr = 2 * received_data.real - 1  # Assuming BPSK
            decoded, success = self.turbo_decoder.decode(llr)
        else:  # no coding
            decoded = (received_data.real > 0.5).astype(int)
            success = True
        
        # Verify CRC
        if len(decoded) >= 32:
            data_bits = decoded[:-32]
            crc_bits = decoded[-32:]
            expected_crc = 0
            for i, bit in enumerate(crc_bits):
                expected_crc += bit * (2 ** (31 - i))
            
            crc_valid = self.error_detector.verify_crc32(decoded, expected_crc)
            return data_bits, success and crc_valid
        
        return decoded, success
