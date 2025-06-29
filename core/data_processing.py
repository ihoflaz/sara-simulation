"""
Data Processing and File Handling
CRC, packetization, data integrity management, encryption, and recovery
"""

import numpy as np
import hashlib
import os
import time
from typing import Tuple, List, Dict, Optional, Union
import struct

from config import DATA_SIZES, SIMULATION_CONFIG

# Import new security and recovery modules
try:
    from core.encryption import EncryptedPacketProcessor
    ENCRYPTION_AVAILABLE = True
except ImportError:
    print("⚠️  Encryption module not available")
    ENCRYPTION_AVAILABLE = False

try:
    from core.recovery import DataRecoveryManager, RecoveryMode
    RECOVERY_AVAILABLE = True
except ImportError:
    print("⚠️  Recovery module not available")
    RECOVERY_AVAILABLE = False

class DataPacketizer:
    """Handle data packetization and reassembly"""
    
    def __init__(self, packet_size: int = 1024):
        self.packet_size = packet_size  # bytes
        self.header_size = 16  # bytes for packet header
        self.payload_size = packet_size - self.header_size
        
    def create_packet_header(self, sequence_num: int, total_packets: int,
                           payload_length: int, packet_type: str = 'data') -> bytes:
        """Create packet header with metadata"""
        # Header format: [sync(4), seq_num(4), total(4), length(2), type(1), reserved(1)]
        sync_word = b'SYNC'
        
        header = struct.pack(
            '>4sIIHBB',
            sync_word,
            sequence_num,
            total_packets,
            payload_length,
            ord(packet_type[0]),  # 'd' for data, 'c' for control
            0  # reserved
        )
        
        return header
    
    def parse_packet_header(self, packet: bytes) -> Dict:
        """Parse packet header to extract metadata"""
        if len(packet) < self.header_size:
            return None
        
        try:
            sync, seq_num, total_packets, payload_length, packet_type, reserved = struct.unpack(
                '>4sIIHBB', packet[:self.header_size]
            )
            
            if sync != b'SYNC':
                return None
            
            return {
                'sequence_number': seq_num,
                'total_packets': total_packets,
                'payload_length': payload_length,
                'packet_type': chr(packet_type),
                'header_valid': True
            }
        except:
            return None
    
    def packetize_data(self, data: bytes) -> List[bytes]:
        """Split data into packets with headers"""
        if not data:
            return []
        
        packets = []
        total_packets = (len(data) + self.payload_size - 1) // self.payload_size
        
        for i in range(total_packets):
            start_idx = i * self.payload_size
            end_idx = min(start_idx + self.payload_size, len(data))
            payload = data[start_idx:end_idx]
            
            # Create header
            header = self.create_packet_header(i, total_packets, len(payload))
            
            # Combine header and payload
            packet = header + payload
            
            # Pad packet to fixed size if necessary
            if len(packet) < self.packet_size:
                packet += b'\x00' * (self.packet_size - len(packet))
            
            packets.append(packet)
        
        return packets
    
    def reassemble_data(self, packets: List[bytes]) -> Tuple[bytes, bool]:
        """Reassemble data from packets"""
        if not packets:
            return b'', False
        
        # Parse headers and sort packets
        packet_info = []
        for packet in packets:
            header = self.parse_packet_header(packet)
            if header and header['header_valid']:
                payload = packet[self.header_size:self.header_size + header['payload_length']]
                packet_info.append((header['sequence_number'], payload, header))
        
        if not packet_info:
            return b'', False
        
        # Sort by sequence number
        packet_info.sort(key=lambda x: x[0])
        
        # Check for completeness
        expected_total = packet_info[0][2]['total_packets']
        received_sequences = [info[0] for info in packet_info]
        expected_sequences = list(range(expected_total))
        
        if received_sequences != expected_sequences:
            print(f"Missing packets: expected {expected_sequences}, got {received_sequences}")
            return b'', False
        
        # Reassemble data
        reassembled_data = b''.join([info[1] for info in packet_info])
        
        return reassembled_data, True

class CRCProcessor:
    """Enhanced CRC processing for data integrity"""
    
    def __init__(self):
        # CRC-32 IEEE 802.3 polynomial
        self.crc32_table = self._generate_crc32_table()
        
    def _generate_crc32_table(self) -> List[int]:
        """Generate CRC-32 lookup table"""
        polynomial = 0xEDB88320
        table = []
        
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ polynomial
                else:
                    crc >>= 1
            table.append(crc)
        
        return table
    
    def calculate_crc32(self, data: bytes) -> int:
        """Calculate CRC-32 for byte data"""
        crc = 0xFFFFFFFF
        
        for byte in data:
            tbl_idx = (crc ^ byte) & 0xFF
            crc = (crc >> 8) ^ self.crc32_table[tbl_idx]
        
        return crc ^ 0xFFFFFFFF
    
    def add_crc_to_data(self, data: bytes) -> bytes:
        """Add CRC-32 to data"""
        crc = self.calculate_crc32(data)
        crc_bytes = struct.pack('<I', crc)  # Little-endian 4-byte CRC
        return data + crc_bytes
    
    def verify_and_remove_crc(self, data_with_crc: bytes) -> Tuple[bytes, bool]:
        """Verify CRC and return original data"""
        if len(data_with_crc) < 4:
            return data_with_crc, False
        
        data = data_with_crc[:-4]
        received_crc = struct.unpack('<I', data_with_crc[-4:])[0]
        calculated_crc = self.calculate_crc32(data)
        
        return data, received_crc == calculated_crc

class FileProcessor:
    """Process files for transmission simulation"""
    
    def __init__(self):
        self.crc_processor = CRCProcessor()
        self.supported_extensions = ['.txt', '.pdf', '.jpg', '.png', '.mp4', '.zip', '.bin']
    
    def create_test_file(self, size_bytes: int, filename: str = None) -> str:
        """Create a test file of specified size"""
        if filename is None:
            filename = f"test_file_{size_bytes//1024//1024}MB.bin"
        
        # Generate pseudo-random data for realistic simulation
        np.random.seed(42)  # Reproducible data
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(filename, 'wb') as f:
            remaining = size_bytes
            while remaining > 0:
                chunk_bytes = min(chunk_size, remaining)
                # Create mixed content (text + binary)
                text_portion = np.random.bytes(chunk_bytes // 2)
                binary_portion = np.random.randint(0, 256, chunk_bytes - len(text_portion), dtype=np.uint8).tobytes()
                chunk = text_portion + binary_portion
                f.write(chunk)
                remaining -= chunk_bytes
        
        print(f"Created test file: {filename} ({size_bytes / 1024 / 1024:.1f} MB)")
        return filename
    
    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of file"""
        hasher = hashlib.sha256()
        
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def prepare_file_for_transmission(self, filepath: str) -> Tuple[bytes, Dict]:
        """Prepare file for transmission with metadata"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read file
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        # Calculate metadata
        file_size = len(file_data)
        file_hash = self.calculate_file_hash(filepath)
        filename = os.path.basename(filepath)
        
        # Create file header
        header_info = {
            'filename': filename,
            'file_size': file_size,
            'file_hash': file_hash,
            'timestamp': int(time.time()),
            'compression': 'none'
        }
        
        # Serialize header
        header_str = str(header_info).encode('utf-8')
        header_length = len(header_str)
        
        # Combine header length + header + file data
        transmission_data = struct.pack('<I', header_length) + header_str + file_data
        
        # Add overall CRC
        transmission_data_with_crc = self.crc_processor.add_crc_to_data(transmission_data)
        
        metadata = {
            'original_size': file_size,
            'transmission_size': len(transmission_data_with_crc),
            'header_info': header_info,
            'overhead_ratio': len(transmission_data_with_crc) / file_size
        }
        
        return transmission_data_with_crc, metadata
    
    def reconstruct_file_from_transmission(self, transmission_data: bytes,
                                         output_path: str = None) -> Tuple[bool, Dict]:
        """Reconstruct file from transmitted data"""
        # Verify and remove overall CRC
        data_without_crc, crc_valid = self.crc_processor.verify_and_remove_crc(transmission_data)
        
        if not crc_valid:
            return False, {'error': 'CRC verification failed'}
        
        # Extract header
        if len(data_without_crc) < 4:
            return False, {'error': 'Insufficient data'}
        
        header_length = struct.unpack('<I', data_without_crc[:4])[0]
        
        if len(data_without_crc) < 4 + header_length:
            return False, {'error': 'Header corrupted'}
        
        header_str = data_without_crc[4:4 + header_length].decode('utf-8')
        file_data = data_without_crc[4 + header_length:]
        
        # Parse header
        try:
            header_info = eval(header_str)  # In production, use JSON
        except:
            return False, {'error': 'Header parsing failed'}
        
        # Verify file size
        if len(file_data) != header_info['file_size']:
            return False, {'error': 'File size mismatch'}
        
        # Calculate and verify hash
        received_hash = hashlib.sha256(file_data).hexdigest()
        if received_hash != header_info['file_hash']:
            return False, {'error': 'File hash verification failed'}
        
        # Write reconstructed file
        if output_path is None:
            output_path = f"received_{header_info['filename']}"
        
        with open(output_path, 'wb') as f:
            f.write(file_data)
        
        reconstruction_info = {
            'success': True,
            'output_path': output_path,
            'original_filename': header_info['filename'],
            'file_size': len(file_data),
            'hash_verified': True,
            'timestamp': header_info['timestamp']
        }
        
        return True, reconstruction_info

class TransmissionSimulator:
    """Simulate data transmission with realistic parameters"""
    
    def __init__(self):
        self.packetizer = DataPacketizer()
        self.file_processor = FileProcessor()
        
        # Transmission parameters
        self.bit_rate = 10e6  # 10 Mbps base rate
        self.current_modulation = 'QPSK'
        self.packet_loss_rate = 0.01  # 1% packet loss
        self.transmission_overhead = 1.2  # 20% overhead
        
    def estimate_transmission_time(self, data_size: int, modulation: str = None) -> float:
        """Estimate transmission time based on data size and modulation"""
        if modulation is None:
            modulation = self.current_modulation
        
        # Modulation efficiency factors
        efficiency_factors = {
            'BPSK': 0.5,
            'QPSK': 1.0,
            '16QAM': 1.8,
            '64QAM': 2.5,
            '256QAM': 3.0,
            '1024QAM': 3.2
        }
        
        efficiency = efficiency_factors.get(modulation, 1.0)
        effective_bit_rate = self.bit_rate * efficiency
        
        # Account for overhead
        effective_data_size = data_size * self.transmission_overhead
        
        transmission_time = effective_data_size * 8 / effective_bit_rate  # Convert bytes to bits
        
        return transmission_time
    
    def simulate_packet_transmission(self, packets: List[bytes],
                                   channel_quality: str = 'good') -> Tuple[List[bytes], Dict]:
        """Simulate packet transmission with losses"""
        # Quality-based loss rates
        loss_rates = {
            'excellent': 0.001,
            'good': 0.01,
            'fair': 0.05,
            'poor': 0.15
        }
        
        packet_loss_rate = loss_rates.get(channel_quality, 0.01)
        
        received_packets = []
        transmission_stats = {
            'total_packets': len(packets),
            'transmitted_packets': 0,
            'lost_packets': 0,
            'corrupted_packets': 0,
            'success_rate': 0
        }
        
        for i, packet in enumerate(packets):
            # Simulate packet loss
            if np.random.random() < packet_loss_rate:
                transmission_stats['lost_packets'] += 1
                continue
            
            # Simulate packet corruption
            corruption_rate = packet_loss_rate / 2
            if np.random.random() < corruption_rate:
                # Corrupt random bytes in packet
                corrupted_packet = bytearray(packet)
                num_corruptions = np.random.randint(1, 5)
                for _ in range(num_corruptions):
                    pos = np.random.randint(0, len(corrupted_packet))
                    corrupted_packet[pos] = np.random.randint(0, 256)
                
                received_packets.append(bytes(corrupted_packet))
                transmission_stats['corrupted_packets'] += 1
            else:
                received_packets.append(packet)
            
            transmission_stats['transmitted_packets'] += 1
        
        transmission_stats['success_rate'] = (
            transmission_stats['transmitted_packets'] - transmission_stats['corrupted_packets']
        ) / max(1, transmission_stats['total_packets'])
        
        return received_packets, transmission_stats
    
    def simulate_file_transmission(self, filepath: str, 
                                 channel_quality: str = 'good',
                                 modulation: str = 'QPSK') -> Dict:
        """Complete file transmission simulation"""
        start_time = time.time()
        
        # Prepare file for transmission
        transmission_data, metadata = self.file_processor.prepare_file_for_transmission(filepath)
        
        # Packetize data
        packets = self.packetizer.packetize_data(transmission_data)
        
        # Estimate transmission time
        estimated_time = self.estimate_transmission_time(len(transmission_data), modulation)
        
        # Simulate transmission
        received_packets, transmission_stats = self.simulate_packet_transmission(
            packets, channel_quality
        )
        
        # Attempt reassembly
        reassembled_data, reassembly_success = self.packetizer.reassemble_data(received_packets)
        
        # Attempt file reconstruction if reassembly successful
        reconstruction_success = False
        reconstruction_info = {}
        
        if reassembly_success:
            output_path = f"received_{os.path.basename(filepath)}"
            reconstruction_success, reconstruction_info = (
                self.file_processor.reconstruct_file_from_transmission(
                    reassembled_data, output_path
                )
            )
        
        end_time = time.time()
        actual_time = end_time - start_time
        
        # Compile results
        simulation_results = {
            'file_info': {
                'original_path': filepath,
                'original_size': metadata['original_size'],
                'transmission_size': metadata['transmission_size']
            },
            'transmission_stats': transmission_stats,
            'timing': {
                'estimated_time': estimated_time,
                'actual_simulation_time': actual_time,
                'throughput_mbps': (metadata['original_size'] * 8) / (estimated_time * 1e6)
            },
            'reassembly': {
                'success': reassembly_success,
                'received_packets': len(received_packets),
                'total_packets': len(packets)
            },
            'reconstruction': reconstruction_info,
            'overall_success': reconstruction_success,
            'modulation_used': modulation,
            'channel_quality': channel_quality
        }
        
        return simulation_results

class DataIntegrityManager:
    """Manage data integrity throughout transmission"""
    
    def __init__(self):
        self.crc_processor = CRCProcessor()
        self.integrity_checks = []
        
    def add_integrity_check(self, check_type: str, data: bytes, 
                          timestamp: float = None) -> str:
        """Add integrity checkpoint"""
        if timestamp is None:
            timestamp = time.time()
        
        check_id = f"{check_type}_{int(timestamp*1000)}"
        
        integrity_info = {
            'check_id': check_id,
            'check_type': check_type,
            'timestamp': timestamp,
            'data_size': len(data),
            'crc32': self.crc_processor.calculate_crc32(data),
            'sha256': hashlib.sha256(data).hexdigest()
        }
        
        self.integrity_checks.append(integrity_info)
        return check_id
    
    def verify_integrity(self, check_id: str, data: bytes) -> bool:
        """Verify data integrity against checkpoint"""
        checkpoint = None
        for check in self.integrity_checks:
            if check['check_id'] == check_id:
                checkpoint = check
                break
        
        if not checkpoint:
            return False
        
        # Verify size
        if len(data) != checkpoint['data_size']:
            return False
        
        # Verify CRC32
        current_crc = self.crc_processor.calculate_crc32(data)
        if current_crc != checkpoint['crc32']:
            return False
        
        # Verify SHA256
        current_sha256 = hashlib.sha256(data).hexdigest()
        if current_sha256 != checkpoint['sha256']:
            return False
        
        return True
    
    def get_integrity_report(self) -> Dict:
        """Get comprehensive integrity report"""
        return {
            'total_checks': len(self.integrity_checks),
            'checks': self.integrity_checks.copy()
        }

class EnhancedTransmissionSimulator:
    """Enhanced transmission simulator with encryption and recovery capabilities"""
    
    def __init__(self, encryption_enabled: bool = False, recovery_enabled: bool = True):
        """
        Initialize enhanced transmission simulator
        
        Args:
            encryption_enabled: Enable AES-128 encryption
            recovery_enabled: Enable data recovery algorithms
        """
        # Base components
        self.packetizer = DataPacketizer()
        self.file_processor = FileProcessor()
        
        # Enhanced security and recovery components
        self.encryption_enabled = encryption_enabled
        self.recovery_enabled = recovery_enabled
        
        # Initialize encryption processor
        if ENCRYPTION_AVAILABLE and encryption_enabled:
            self.encryption_processor = EncryptedPacketProcessor(
                encryption_enabled=True, 
                mode='CTR'  # CTR mode for compatibility
            )
        else:
            self.encryption_processor = None
            
        # Initialize recovery manager
        if RECOVERY_AVAILABLE and recovery_enabled:
            self.recovery_manager = DataRecoveryManager(mode=RecoveryMode.ADAPTIVE)
            self.recovery_manager.set_recovery_callback(self._handle_recovery_event)
        else:
            self.recovery_manager = None
        
        # Transmission parameters
        self.bit_rate = 10e6  # 10 Mbps base rate
        self.current_modulation = 'QPSK'
        self.packet_loss_rate = 0.01  # 1% packet loss
        self.transmission_overhead = 1.2  # 20% overhead
        
        # Enhanced statistics
        self.transmission_stats = {
            'total_packets_sent': 0,
            'total_packets_received': 0,
            'encrypted_packets': 0,
            'decrypted_packets': 0,
            'recovered_packets': 0,
            'failed_recoveries': 0,
            'encryption_failures': 0,
            'decryption_failures': 0,
        }
        
    def set_encryption_enabled(self, enabled: bool, mode: str = 'CTR'):
        """Enable or disable encryption"""
        self.encryption_enabled = enabled
        
        if ENCRYPTION_AVAILABLE and enabled:
            self.encryption_processor = EncryptedPacketProcessor(
                encryption_enabled=True, 
                mode=mode
            )
        else:
            self.encryption_processor = None
    
    def set_recovery_enabled(self, enabled: bool, mode: RecoveryMode = RecoveryMode.ADAPTIVE):
        """Enable or disable data recovery"""
        self.recovery_enabled = enabled
        
        if RECOVERY_AVAILABLE and enabled:
            if self.recovery_manager is None:
                self.recovery_manager = DataRecoveryManager(mode=mode)
                self.recovery_manager.set_recovery_callback(self._handle_recovery_event)
            else:
                self.recovery_manager.set_recovery_mode(mode)
        else:
            if self.recovery_manager:
                self.recovery_manager.stop_recovery_thread()
            self.recovery_manager = None
    
    def simulate_enhanced_packet_transmission(self, packets: List[bytes], 
                                            channel_quality: str = 'good',
                                            jammed_bands: List[int] = []) -> Tuple[List[bytes], Dict]:
        """
        Enhanced packet transmission simulation with encryption and recovery
        
        Args:
            packets: List of packets to transmit
            channel_quality: Channel quality ('excellent', 'good', 'fair', 'poor')
            jammed_bands: List of jammed frequency bands
            
        Returns:
            (received_packets, transmission_stats)
        """
        start_time = time.time()
        
        # Prepare packets for transmission
        processed_packets = []
        encryption_info = []
        
        for i, packet in enumerate(packets):
            if self.encryption_processor:
                # Encrypt packet
                try:
                    encrypted_packet, enc_info = self.encryption_processor.process_outgoing_packet(packet, i)
                    processed_packets.append(encrypted_packet)
                    encryption_info.append(enc_info)
                    
                    if enc_info.get('encrypted', False):
                        self.transmission_stats['encrypted_packets'] += 1
                    else:
                        self.transmission_stats['encryption_failures'] += 1
                        
                except Exception as e:
                    # Encryption failed, use original packet
                    processed_packets.append(packet)
                    encryption_info.append({'encrypted': False, 'error': str(e)})
                    self.transmission_stats['encryption_failures'] += 1
            else:
                # No encryption
                processed_packets.append(packet)
                encryption_info.append({'encrypted': False})
        
        # Apply recovery redundancy if enabled
        if self.recovery_manager:
            # Add redundant packets or erasure coding
            if self.recovery_manager.mode in [RecoveryMode.REDUNDANT_PACKETS, RecoveryMode.ADAPTIVE]:
                processed_packets = self.recovery_manager.generate_redundant_packets(processed_packets)
            elif self.recovery_manager.mode in [RecoveryMode.ERASURE_CODING, RecoveryMode.ADAPTIVE]:
                processed_packets = self.recovery_manager.generate_erasure_coded_packets(processed_packets)
        
        # Simulate transmission with losses and errors
        received_packets, base_stats = self._simulate_transmission_with_jamming(
            processed_packets, channel_quality, jammed_bands
        )
        
        # Process received packets (decrypt if needed)
        final_packets = []
        decryption_info = []
        
        for i, packet in enumerate(received_packets):
            if packet is None:
                final_packets.append(None)
                decryption_info.append({'decrypted': False, 'reason': 'packet_lost'})
                continue
                
            if self.encryption_processor:
                # Decrypt packet
                try:
                    decrypted_packet, dec_info = self.encryption_processor.process_incoming_packet(packet, i)
                    final_packets.append(decrypted_packet)
                    decryption_info.append(dec_info)
                    
                    if dec_info.get('encrypted', False):
                        self.transmission_stats['decrypted_packets'] += 1
                    else:
                        self.transmission_stats['decryption_failures'] += 1
                        
                except Exception as e:
                    # Decryption failed, report as corrupted
                    final_packets.append(None)
                    decryption_info.append({'decrypted': False, 'error': str(e)})
                    self.transmission_stats['decryption_failures'] += 1
                    
                    # Report to recovery manager if available
                    if self.recovery_manager:
                        self.recovery_manager.report_transmission_failure(
                            i, packet, 'decryption_failed', 
                            self._get_channel_quality_value(channel_quality), 1
                        )
            else:
                # No decryption needed
                final_packets.append(packet)
                decryption_info.append({'encrypted': False})
        
        # Update transmission statistics
        self.transmission_stats['total_packets_sent'] += len(packets)
        self.transmission_stats['total_packets_received'] += len([p for p in final_packets if p is not None])
        
        # Compile enhanced transmission statistics
        enhanced_stats = {
            'base_transmission': base_stats,
            'encryption_enabled': self.encryption_enabled,
            'recovery_enabled': self.recovery_enabled,
            'processed_packets': len(processed_packets),
            'original_packets': len(packets),
            'received_packets': len([p for p in final_packets if p is not None]),
            'encryption_info': encryption_info,
            'decryption_info': decryption_info,
            'transmission_time': time.time() - start_time,
            'jammed_bands': jammed_bands,
            'channel_quality': channel_quality
        }
        
        # Add encryption statistics if available
        if self.encryption_processor:
            enhanced_stats['encryption_stats'] = self.encryption_processor.get_encryption_stats()
        
        # Add recovery statistics if available
        if self.recovery_manager:
            enhanced_stats['recovery_stats'] = self.recovery_manager.get_recovery_stats()
            enhanced_stats['recovery_recommendations'] = self.recovery_manager.get_recovery_recommendations()
        
        return final_packets, enhanced_stats
    
    def simulate_enhanced_file_transmission(self, filepath: str, 
                                         channel_quality: str = 'good',
                                         modulation: str = 'QPSK',
                                         jammed_bands: List[int] = [],
                                         enable_recovery: bool = True) -> Dict:
        """
        Enhanced file transmission simulation with encryption and recovery
        
        Args:
            filepath: Path to file to transmit
            channel_quality: Channel quality setting
            modulation: Modulation scheme to use
            jammed_bands: List of jammed frequency bands
            enable_recovery: Enable recovery mechanisms for this transmission
            
        Returns:
            Comprehensive transmission results
        """
        start_time = time.time()
        
        # Prepare file for transmission
        transmission_data, metadata = self.file_processor.prepare_file_for_transmission(filepath)
        
        # Packetize data
        packets = self.packetizer.packetize_data(transmission_data)
        
        # Estimate transmission time
        estimated_time = self.estimate_transmission_time(len(transmission_data), modulation)
        
        # Enhanced transmission with encryption and recovery
        received_packets, transmission_stats = self.simulate_enhanced_packet_transmission(
            packets, channel_quality, jammed_bands
        )
        
        # Attempt reassembly
        reassembled_data, reassembly_success = self.packetizer.reassemble_data(
            [p for p in received_packets if p is not None]
        )
        
        # Attempt file reconstruction if reassembly successful
        reconstruction_success = False
        reconstruction_info = {}
        
        if reassembly_success:
            output_path = f"received_{os.path.basename(filepath)}"
            reconstruction_success, reconstruction_info = (
                self.file_processor.reconstruct_file_from_transmission(
                    reassembled_data, output_path
                )
            )
        
        # If reconstruction failed and recovery is enabled, attempt recovery
        recovery_attempts = 0
        if not reconstruction_success and enable_recovery and self.recovery_manager:
            recovery_attempts = self._attempt_file_recovery(
                packets, received_packets, filepath, channel_quality, jammed_bands
            )
        
        end_time = time.time()
        actual_time = end_time - start_time
        
        # Compile comprehensive results
        enhanced_results = {
            'file_info': {
                'original_path': filepath,
                'original_size': metadata['original_size'],
                'transmission_size': metadata['transmission_size']
            },
            'transmission_stats': transmission_stats,
            'timing': {
                'estimated_time': estimated_time,
                'actual_simulation_time': actual_time,
                'throughput_mbps': (metadata['original_size'] * 8) / (estimated_time * 1e6)
            },
            'reassembly': {
                'success': reassembly_success,
                'received_packets': len([p for p in received_packets if p is not None]),
                'total_packets': len(packets)
            },
            'reconstruction': reconstruction_info,
            'recovery_attempts': recovery_attempts,
            'overall_success': reconstruction_success,
            'modulation_used': modulation,
            'channel_quality': channel_quality,
            'jammed_bands': jammed_bands,
            'security_features': {
                'encryption_enabled': self.encryption_enabled,
                'recovery_enabled': self.recovery_enabled,
                'data_integrity_verified': reconstruction_success
            }
        }
        
        return enhanced_results
    
    def _simulate_transmission_with_jamming(self, packets: List[bytes], 
                                          channel_quality: str, 
                                          jammed_bands: List[int]) -> Tuple[List[Optional[bytes]], Dict]:
        """Simulate transmission with jamming effects"""
        # Quality-based loss rates
        base_loss_rates = {
            'excellent': 0.001,
            'good': 0.01,
            'fair': 0.05,
            'poor': 0.15
        }
        
        # Increase loss rate if bands are jammed
        packet_loss_rate = base_loss_rates.get(channel_quality, 0.01)
        if jammed_bands:
            # Simulate jamming effect (increase loss rate)
            jamming_factor = min(len(jammed_bands) / 5.0, 0.8)  # Max 80% additional loss
            packet_loss_rate += jamming_factor * 0.3
        
        received_packets = []
        transmission_stats = {
            'total_packets': len(packets),
            'transmitted_packets': 0,
            'lost_packets': 0,
            'corrupted_packets': 0,
            'jammed_packets': 0,
            'success_rate': 0
        }
        
        for i, packet in enumerate(packets):
            # Simulate packet loss due to jamming
            if jammed_bands and np.random.random() < jamming_factor * 0.5:
                received_packets.append(None)
                transmission_stats['jammed_packets'] += 1
                transmission_stats['lost_packets'] += 1
                
                # Report jamming to recovery manager
                if self.recovery_manager:
                    self.recovery_manager.report_transmission_failure(
                        i, packet, 'jammed', 
                        self._get_channel_quality_value(channel_quality), 
                        jammed_bands[0] if jammed_bands else 1
                    )
                continue
            
            # Simulate general packet loss
            if np.random.random() < packet_loss_rate:
                received_packets.append(None)
                transmission_stats['lost_packets'] += 1
                
                # Report timeout to recovery manager
                if self.recovery_manager:
                    self.recovery_manager.report_transmission_failure(
                        i, packet, 'timeout', 
                        self._get_channel_quality_value(channel_quality), 1
                    )
                continue
            
            # Simulate packet corruption
            corruption_rate = packet_loss_rate / 2
            if np.random.random() < corruption_rate:
                # Corrupt random bytes in packet
                corrupted_packet = bytearray(packet)
                num_corruptions = np.random.randint(1, 5)
                for _ in range(num_corruptions):
                    pos = np.random.randint(0, len(corrupted_packet))
                    corrupted_packet[pos] = np.random.randint(0, 256)
                
                received_packets.append(bytes(corrupted_packet))
                transmission_stats['corrupted_packets'] += 1
                
                # Report corruption to recovery manager
                if self.recovery_manager:
                    self.recovery_manager.report_transmission_failure(
                        i, packet, 'crc_failed', 
                        self._get_channel_quality_value(channel_quality), 1
                    )
            else:
                received_packets.append(packet)
                
                # Report success to recovery manager
                if self.recovery_manager:
                    self.recovery_manager.report_transmission_success(i)
            
            transmission_stats['transmitted_packets'] += 1
        
        transmission_stats['success_rate'] = (
            (transmission_stats['transmitted_packets'] - transmission_stats['corrupted_packets']) /
            max(1, transmission_stats['total_packets'])
        )
        
        return received_packets, transmission_stats
    
    def _attempt_file_recovery(self, original_packets: List[bytes], 
                             received_packets: List[Optional[bytes]], 
                             filepath: str, channel_quality: str, 
                             jammed_bands: List[int]) -> int:
        """Attempt to recover failed file transmission"""
        recovery_attempts = 0
        max_recovery_attempts = 3
        
        while recovery_attempts < max_recovery_attempts:
            recovery_attempts += 1
            
            # Get retry packets from recovery manager
            retry_info = self.recovery_manager.get_next_retry_packet()
            if retry_info is None:
                break
                
            packet_id, packet_data, recovery_method = retry_info
            
            # Simulate retry transmission
            if recovery_method == "redundant":
                # Send multiple copies
                for _ in range(self.recovery_manager.redundancy_factor):
                    success = np.random.random() > 0.3  # 70% success rate for retries
                    if success:
                        if packet_id < len(received_packets):
                            received_packets[packet_id] = packet_data
                        self.recovery_manager.report_transmission_success(packet_id)
                        self.transmission_stats['recovered_packets'] += 1
                        break
            
            elif recovery_method == "erasure_coding":
                # Use erasure coding recovery
                erasure_positions = [i for i, p in enumerate(received_packets) if p is None]
                if len(erasure_positions) <= self.recovery_manager.reed_solomon.recovery_blocks:
                    try:
                        recovered = self.recovery_manager.recover_from_erasure_coded_packets(
                            received_packets, erasure_positions
                        )
                        received_packets[:len(recovered)] = recovered
                        self.transmission_stats['recovered_packets'] += len(erasure_positions)
                        
                        for pos in erasure_positions:
                            self.recovery_manager.report_transmission_success(pos)
                    except Exception:
                        self.transmission_stats['failed_recoveries'] += 1
            
            else:  # Simple retry
                success = np.random.random() > 0.2  # 80% success rate for retries
                if success:
                    if packet_id < len(received_packets):
                        received_packets[packet_id] = packet_data
                    self.recovery_manager.report_transmission_success(packet_id)
                    self.transmission_stats['recovered_packets'] += 1
        
        return recovery_attempts
    
    def _get_channel_quality_value(self, quality_str: str) -> float:
        """Convert channel quality string to numeric value"""
        quality_map = {
            'excellent': 0.95,
            'good': 0.75,
            'fair': 0.50,
            'poor': 0.25
        }
        return quality_map.get(quality_str, 0.5)
    
    def _handle_recovery_event(self, event_type: str, event_data: Dict):
        """Handle recovery events for logging and statistics"""
        if event_type == "packet_recovered":
            self.transmission_stats['recovered_packets'] += 1
        elif event_type == "retry_limit_reached":
            self.transmission_stats['failed_recoveries'] += 1
        
        # Log event (could be extended to write to file or GUI)
        print(f"Recovery Event: {event_type} - {event_data}")
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive transmission statistics"""
        stats = self.transmission_stats.copy()
        
        # Add encryption stats if available
        if self.encryption_processor:
            stats['encryption'] = self.encryption_processor.get_encryption_stats()
        
        # Add recovery stats if available
        if self.recovery_manager:
            stats['recovery'] = self.recovery_manager.get_recovery_stats()
        
        return stats
    
    def export_security_key(self) -> Optional[str]:
        """Export encryption key for sharing between transmitter and receiver"""
        if self.encryption_processor:
            return self.encryption_processor.export_encryption_key()
        return None
    
    def import_security_key(self, key_hex: str):
        """Import encryption key"""
        if self.encryption_processor:
            self.encryption_processor.import_encryption_key(key_hex)
