"""
Data Processing and File Handling
CRC, packetization, and data integrity management
"""

import numpy as np
import hashlib
import os
import time
from typing import Tuple, List, Dict, Optional, Union
import struct

from config import DATA_SIZES, SIMULATION_CONFIG

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
