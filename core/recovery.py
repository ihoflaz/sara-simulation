"""
Data Recovery Algorithm for TEKNOFEST 2025
Advanced data recovery mechanisms for wireless communication under jamming
"""

import time
import threading
import queue
import hashlib
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import numpy as np

from config import COMPETITION_CONFIG, DATA_SIZES

class RecoveryMode(Enum):
    """Data recovery modes"""
    DISABLED = "disabled"
    RETRY_ONLY = "retry_only"
    REDUNDANT_PACKETS = "redundant_packets"
    ERASURE_CODING = "erasure_coding"
    ADAPTIVE = "adaptive"

@dataclass
class RecoveryPacket:
    """Data structure for recovery packet tracking"""
    packet_id: int
    data: bytes
    timestamp: float
    retry_count: int
    priority: int
    original_size: int
    recovery_method: str

@dataclass
class TransmissionAttempt:
    """Track transmission attempts"""
    packet_id: int
    timestamp: float
    success: bool
    error_type: Optional[str]
    channel_quality: float
    frequency_band: int

class ReedSolomonErasure:
    """Simplified Reed-Solomon erasure coding implementation"""
    
    def __init__(self, data_blocks: int = 6, recovery_blocks: int = 3):
        """
        Initialize Reed-Solomon encoder/decoder
        
        Args:
            data_blocks: Number of original data blocks
            recovery_blocks: Number of redundant recovery blocks
        """
        self.data_blocks = data_blocks
        self.recovery_blocks = recovery_blocks
        self.total_blocks = data_blocks + recovery_blocks
        
        # Generate primitive polynomial and generator matrix
        self._initialize_galois_field()
        
    def _initialize_galois_field(self):
        """Initialize Galois Field operations for Reed-Solomon"""
        # Simplified GF(256) implementation
        self.gf_exp = [0] * 512
        self.gf_log = [0] * 256
        
        # Generate exp and log tables
        x = 1
        for i in range(255):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x = (x << 1) ^ (0x11d if x & 0x80 else 0)
            x &= 0xff
        
        # Extend exp table for convenience
        for i in range(255, 512):
            self.gf_exp[i] = self.gf_exp[i - 255]
    
    def _gf_multiply(self, a: int, b: int) -> int:
        """Galois Field multiplication"""
        if a == 0 or b == 0:
            return 0
        return self.gf_exp[self.gf_log[a] + self.gf_log[b]]
    
    def _gf_divide(self, a: int, b: int) -> int:
        """Galois Field division"""
        if a == 0:
            return 0
        if b == 0:
            raise ValueError("Division by zero in GF")
        return self.gf_exp[self.gf_log[a] - self.gf_log[b] + 255]
    
    def encode_blocks(self, data_blocks: List[bytes]) -> List[bytes]:
        """
        Encode data blocks with Reed-Solomon error correction
        
        Args:
            data_blocks: List of data blocks to encode
            
        Returns:
            List of encoded blocks (data + recovery blocks)
        """
        if len(data_blocks) != self.data_blocks:
            raise ValueError(f"Expected {self.data_blocks} data blocks, got {len(data_blocks)}")
        
        # Ensure all blocks are same size
        max_size = max(len(block) for block in data_blocks)
        padded_blocks = []
        for block in data_blocks:
            if len(block) < max_size:
                padded_block = block + b'\x00' * (max_size - len(block))
            else:
                padded_block = block
            padded_blocks.append(padded_block)
        
        # Generate recovery blocks using simplified RS
        recovery_blocks = []
        for i in range(self.recovery_blocks):
            recovery_block = bytearray(max_size)
            for j in range(max_size):
                # Simple XOR-based recovery (simplified for demo)
                value = 0
                for k, block in enumerate(padded_blocks):
                    if j < len(block):
                        value ^= self._gf_multiply(block[j], (i + k + 1) % 255 + 1)
                recovery_block[j] = value & 0xff
            recovery_blocks.append(bytes(recovery_block))
        
        return padded_blocks + recovery_blocks
    
    def decode_blocks(self, received_blocks: List[Optional[bytes]], block_indices: List[int]) -> List[bytes]:
        """
        Decode data blocks from received blocks (with erasures)
        
        Args:
            received_blocks: List of received blocks (None for missing blocks)
            block_indices: Indices of received blocks
            
        Returns:
            List of recovered data blocks
        """
        if len(received_blocks) < self.data_blocks:
            raise ValueError(f"Need at least {self.data_blocks} blocks for recovery")
        
        # For simplified implementation, use available data blocks
        # In a full implementation, this would use Gaussian elimination
        recovered_blocks = []
        
        # Try to recover using available data blocks first
        data_blocks_available = [
            (i, block) for i, block in enumerate(received_blocks[:self.data_blocks])
            if block is not None
        ]
        
        if len(data_blocks_available) >= self.data_blocks:
            # Sufficient data blocks available
            return [block for _, block in data_blocks_available[:self.data_blocks]]
        
        # Need to use recovery blocks (simplified recovery)
        available_blocks = [
            (i, block) for i, block in enumerate(received_blocks)
            if block is not None and i < len(block_indices)
        ]
        
        if len(available_blocks) < self.data_blocks:
            raise ValueError("Insufficient blocks for recovery")
        
        # Simple recovery algorithm (in practice, would use proper RS decoding)
        block_size = len(available_blocks[0][1]) if available_blocks else 0
        for block_idx in range(self.data_blocks):
            if block_idx < len(received_blocks) and received_blocks[block_idx] is not None:
                recovered_blocks.append(received_blocks[block_idx])
            else:
                # Attempt recovery using XOR combination
                recovered_block = bytearray(block_size)
                for j in range(block_size):
                    value = 0
                    for _, block in available_blocks[:self.data_blocks]:
                        if j < len(block):
                            value ^= block[j]
                    recovered_block[j] = value & 0xff
                recovered_blocks.append(bytes(recovered_block))
        
        return recovered_blocks[:self.data_blocks]


class DataRecoveryManager:
    """Advanced data recovery manager for wireless communication"""
    
    def __init__(self, mode: RecoveryMode = RecoveryMode.ADAPTIVE):
        """
        Initialize data recovery manager
        
        Args:
            mode: Recovery mode to use
        """
        self.mode = mode
        self.active = True
        
        # Packet tracking
        self.pending_packets = {}  # packet_id -> RecoveryPacket
        self.transmission_history = deque(maxlen=1000)
        self.failed_transmissions = defaultdict(list)
        
        # Recovery mechanisms
        self.reed_solomon = ReedSolomonErasure(data_blocks=6, recovery_blocks=3)
        self.retry_queue = queue.PriorityQueue()
        self.buffer_queue = deque(maxlen=10000)
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'successful_transmissions': 0,
            'failed_transmissions': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'packets_lost': 0,
            'retransmissions': 0,
            'erasure_corrections': 0,
            'avg_retry_count': 0.0,
            'recovery_efficiency': 0.0
        }
        
        # Configuration
        self.max_retries = 5
        self.retry_delay = 0.5  # seconds
        self.redundancy_factor = 2  # number of redundant copies
        self.buffer_timeout = 30.0  # seconds
        
        # State management
        self.paused = False
        self.escape_failed = False
        self.last_escape_attempt = 0
        self.escape_retry_interval = 5.0  # seconds
        
        # Threading
        self._recovery_thread = None
        self._stop_event = threading.Event()
        
    def start_recovery_thread(self):
        """Start background recovery thread"""
        if self._recovery_thread is None or not self._recovery_thread.is_alive():
            self._stop_event.clear()
            self._recovery_thread = threading.Thread(target=self._recovery_worker, daemon=True)
            self._recovery_thread.start()
    
    def stop_recovery_thread(self):
        """Stop background recovery thread"""
        self._stop_event.set()
        if self._recovery_thread and self._recovery_thread.is_alive():
            self._recovery_thread.join(timeout=1.0)
    
    def _recovery_worker(self):
        """Background worker for recovery operations"""
        while not self._stop_event.is_set():
            try:
                # Process retry queue
                if not self.retry_queue.empty():
                    priority, timestamp, packet = self.retry_queue.get_nowait()
                    if time.time() - timestamp >= self.retry_delay:
                        self._attempt_packet_recovery(packet)
                
                # Clean up old packets
                self._cleanup_old_packets()
                
                # Check for escape retry
                if self.escape_failed and time.time() - self.last_escape_attempt >= self.escape_retry_interval:
                    self._attempt_escape_retry()
                
                time.sleep(0.1)  # Small delay to prevent high CPU usage
                
            except Exception as e:
                print(f"Recovery worker error: {e}")
                time.sleep(1.0)
    
    def submit_packet(self, packet_id: int, data: bytes, priority: int = 0) -> bool:
        """
        Submit packet for transmission with recovery support
        
        Args:
            packet_id: Unique packet identifier
            data: Packet data
            priority: Transmission priority (0 = normal, 1 = high)
            
        Returns:
            True if submitted successfully
        """
        if not self.active:
            return False
        
        # Create recovery packet
        recovery_packet = RecoveryPacket(
            packet_id=packet_id,
            data=data,
            timestamp=time.time(),
            retry_count=0,
            priority=priority,
            original_size=len(data),
            recovery_method='none'
        )
        
        # Store for tracking
        self.pending_packets[packet_id] = recovery_packet
        self.stats['total_packets'] += 1
        
        # Apply recovery method based on mode
        if self.mode == RecoveryMode.REDUNDANT_PACKETS:
            return self._submit_with_redundancy(recovery_packet)
        elif self.mode == RecoveryMode.ERASURE_CODING:
            return self._submit_with_erasure_coding(recovery_packet)
        elif self.mode == RecoveryMode.ADAPTIVE:
            return self._submit_adaptive(recovery_packet)
        else:
            return self._submit_normal(recovery_packet)
    
    def _submit_normal(self, packet: RecoveryPacket) -> bool:
        """Submit packet without recovery mechanisms"""
        # In real implementation, this would interface with the transmission system
        packet.recovery_method = 'none'
        return True
    
    def _submit_with_redundancy(self, packet: RecoveryPacket) -> bool:
        """Submit packet with redundant copies"""
        packet.recovery_method = 'redundant'
        
        # Create redundant copies
        for i in range(self.redundancy_factor):
            redundant_packet = RecoveryPacket(
                packet_id=packet.packet_id + i * 100000,  # Offset to avoid collision
                data=packet.data,
                timestamp=packet.timestamp,
                retry_count=0,
                priority=packet.priority,
                original_size=packet.original_size,
                recovery_method='redundant_copy'
            )
            # Submit each copy (in real implementation)
            
        return True
    
    def _submit_with_erasure_coding(self, packet: RecoveryPacket) -> bool:
        """Submit packet with erasure coding"""
        packet.recovery_method = 'erasure_coding'
        
        try:
            # Split data into blocks
            block_size = (len(packet.data) + 5) // 6  # 6 data blocks
            data_blocks = []
            
            for i in range(6):
                start_idx = i * block_size
                end_idx = min(start_idx + block_size, len(packet.data))
                if start_idx < len(packet.data):
                    block = packet.data[start_idx:end_idx]
                    if len(block) < block_size:
                        block += b'\x00' * (block_size - len(block))
                    data_blocks.append(block)
                else:
                    data_blocks.append(b'\x00' * block_size)
            
            # Encode with Reed-Solomon
            encoded_blocks = self.reed_solomon.encode_blocks(data_blocks)
            
            # Submit encoded blocks as separate packets
            for i, block in enumerate(encoded_blocks):
                block_packet = RecoveryPacket(
                    packet_id=packet.packet_id * 1000 + i,  # Block identifier
                    data=block,
                    timestamp=packet.timestamp,
                    retry_count=0,
                    priority=packet.priority,
                    original_size=len(block),
                    recovery_method='erasure_block'
                )
                # Submit block (in real implementation)
            
            return True
            
        except Exception as e:
            print(f"Erasure coding failed: {e}")
            return self._submit_normal(packet)
    
    def _submit_adaptive(self, packet: RecoveryPacket) -> bool:
        """Submit packet with adaptive recovery based on channel conditions"""
        # Analyze recent transmission success rate
        recent_attempts = list(self.transmission_history)[-50:]  # Last 50 attempts
        if recent_attempts:
            success_rate = sum(1 for attempt in recent_attempts if attempt.success) / len(recent_attempts)
        else:
            success_rate = 1.0
        
        # Choose recovery method based on conditions
        if success_rate > 0.9:
            return self._submit_normal(packet)
        elif success_rate > 0.7:
            return self._submit_with_redundancy(packet)
        else:
            return self._submit_with_erasure_coding(packet)
    
    def report_transmission_result(self, packet_id: int, success: bool, error_type: Optional[str] = None, 
                                 channel_quality: float = 1.0, frequency_band: int = 1):
        """
        Report transmission result for recovery tracking
        
        Args:
            packet_id: Packet identifier
            success: Whether transmission was successful
            error_type: Type of error if failed
            channel_quality: Channel quality (0.0 to 1.0)
            frequency_band: Frequency band used
        """
        # Record transmission attempt
        attempt = TransmissionAttempt(
            packet_id=packet_id,
            timestamp=time.time(),
            success=success,
            error_type=error_type,
            channel_quality=channel_quality,
            frequency_band=frequency_band
        )
        self.transmission_history.append(attempt)
        
        if success:
            self.stats['successful_transmissions'] += 1
            # Remove from pending if successful
            if packet_id in self.pending_packets:
                del self.pending_packets[packet_id]
        else:
            self.stats['failed_transmissions'] += 1
            self.failed_transmissions[packet_id].append(attempt)
            
            # Schedule for retry if appropriate
            if packet_id in self.pending_packets:
                packet = self.pending_packets[packet_id]
                if packet.retry_count < self.max_retries:
                    self._schedule_retry(packet)
                else:
                    self.stats['packets_lost'] += 1
                    del self.pending_packets[packet_id]
    
    def _schedule_retry(self, packet: RecoveryPacket):
        """Schedule packet for retry"""
        packet.retry_count += 1
        self.stats['retransmissions'] += 1
        
        # Add to retry queue with priority
        priority = -packet.priority * 1000 + packet.retry_count  # Higher priority = lower number
        timestamp = time.time()
        self.retry_queue.put((priority, timestamp, packet))
    
    def _attempt_packet_recovery(self, packet: RecoveryPacket):
        """Attempt to recover/retransmit a packet"""
        self.stats['recovery_attempts'] += 1
        
        # In real implementation, this would retransmit the packet
        # For now, simulate recovery attempt
        recovery_success = np.random.random() > 0.3  # 70% success rate for demo
        
        if recovery_success:
            self.stats['successful_recoveries'] += 1
            if packet.packet_id in self.pending_packets:
                del self.pending_packets[packet.packet_id]
        else:
            # Schedule another retry if under limit
            if packet.retry_count < self.max_retries:
                self._schedule_retry(packet)
            else:
                self.stats['packets_lost'] += 1
                if packet.packet_id in self.pending_packets:
                    del self.pending_packets[packet.packet_id]
    
    def _cleanup_old_packets(self):
        """Clean up old pending packets"""
        current_time = time.time()
        expired_packets = []
        
        for packet_id, packet in self.pending_packets.items():
            if current_time - packet.timestamp > self.buffer_timeout:
                expired_packets.append(packet_id)
        
        for packet_id in expired_packets:
            self.stats['packets_lost'] += 1
            del self.pending_packets[packet_id]
    
    def handle_frequency_escape_failure(self, jammed_bands: List[int]):
        """
        Handle scenario where frequency escape fails (all bands jammed)
        
        Args:
            jammed_bands: List of jammed frequency bands
        """
        print(f"⚠️  Frequency escape failed - bands {jammed_bands} are jammed")
        
        self.escape_failed = True
        self.last_escape_attempt = time.time()
        
        # Pause transmission temporarily
        self.paused = True
        
        # Buffer new packets instead of transmitting
        print("📦 Entering buffer mode - packets will be queued for retransmission")
        
        # Start periodic escape attempts
        self._attempt_escape_retry()
    
    def _attempt_escape_retry(self):
        """Attempt to escape jamming periodically"""
        self.stats['recovery_attempts'] += 1
        
        # In real implementation, this would:
        # 1. Check if any frequency bands are now available
        # 2. Attempt transmission on available bands
        # 3. Resume normal operation if successful
        
        # Simulate escape attempt
        escape_success = np.random.random() > 0.6  # 40% success rate
        
        if escape_success:
            print("✅ Frequency escape successful - resuming normal transmission")
            self.escape_failed = False
            self.paused = False
            
            # Process buffered packets
            self._process_buffered_packets()
        else:
            print("🔄 Frequency escape failed - will retry in {:.1f}s".format(self.escape_retry_interval))
            self.last_escape_attempt = time.time()
    
    def _process_buffered_packets(self):
        """Process packets that were buffered during jamming"""
        processed_count = 0
        
        while self.buffer_queue and processed_count < 50:  # Process in batches
            packet_data = self.buffer_queue.popleft()
            # Re-submit packet for transmission
            # In real implementation, would call submit_packet again
            processed_count += 1
        
        if processed_count > 0:
            print(f"📤 Processed {processed_count} buffered packets")
    
    def get_recovery_stats(self) -> Dict:
        """Get comprehensive recovery statistics"""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        total_attempts = stats['successful_transmissions'] + stats['failed_transmissions']
        if total_attempts > 0:
            stats['success_rate'] = stats['successful_transmissions'] / total_attempts
            stats['failure_rate'] = stats['failed_transmissions'] / total_attempts
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        if stats['recovery_attempts'] > 0:
            stats['recovery_efficiency'] = stats['successful_recoveries'] / stats['recovery_attempts']
        else:
            stats['recovery_efficiency'] = 0.0
        
        if stats['retransmissions'] > 0:
            retry_counts = [packet.retry_count for packet in self.pending_packets.values()]
            if retry_counts:
                stats['avg_retry_count'] = sum(retry_counts) / len(retry_counts)
        
        # Current state
        stats['pending_packets'] = len(self.pending_packets)
        stats['buffered_packets'] = len(self.buffer_queue)
        stats['paused'] = self.paused
        stats['escape_failed'] = self.escape_failed
        stats['recovery_mode'] = self.mode.value
        
        return stats
    
    def set_mode(self, mode: RecoveryMode):
        """Set recovery mode"""
        self.mode = mode
        print(f"🔧 Recovery mode set to: {mode.value}")
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_packets': 0,
            'successful_transmissions': 0,
            'failed_transmissions': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'packets_lost': 0,
            'retransmissions': 0,
            'erasure_corrections': 0,
            'avg_retry_count': 0.0,
            'recovery_efficiency': 0.0
        }


# Integration helper functions
def create_integrated_recovery_system(enable_encryption: bool = True, recovery_mode: RecoveryMode = RecoveryMode.ADAPTIVE):
    """
    Create integrated encryption + recovery system
    
    Args:
        enable_encryption: Enable AES encryption
        recovery_mode: Data recovery mode
        
    Returns:
        (encryption_processor, recovery_manager)
    """
    encryption_processor = EncryptedPacketProcessor(
        encryption_enabled=enable_encryption,
        mode='CTR'
    )
    
    recovery_manager = DataRecoveryManager(mode=recovery_mode)
    recovery_manager.start_recovery_thread()
    
    return encryption_processor, recovery_manager


def demo_recovery_system():
    """Demonstrate the data recovery system"""
    print("🔄 Data Recovery System Demo")
    print("=" * 50)
    
    # Create recovery manager
    recovery_manager = DataRecoveryManager(mode=RecoveryMode.ADAPTIVE)
    recovery_manager.start_recovery_thread()
    
    # Simulate packet transmissions
    print("📡 Simulating packet transmissions...")
    
    for i in range(20):
        packet_data = f"Test packet {i} for TEKNOFEST 2025".encode() * 10
        recovery_manager.submit_packet(i, packet_data, priority=i % 2)
        
        # Simulate random transmission results
        success = np.random.random() > 0.3  # 70% success rate
        error_type = None if success else np.random.choice(['crc_error', 'timeout', 'jamming'])
        channel_quality = np.random.uniform(0.3, 1.0)
        frequency_band = np.random.randint(1, 6)
        
        recovery_manager.report_transmission_result(
            i, success, error_type, channel_quality, frequency_band
        )
    
    # Simulate frequency escape failure
    print("\n⚠️  Simulating frequency escape failure...")
    recovery_manager.handle_frequency_escape_failure([1, 2, 3, 4, 5])
    
    # Wait a bit for recovery attempts
    time.sleep(2)
    
    # Show statistics
    stats = recovery_manager.get_recovery_stats()
    print(f"\n📊 Recovery Statistics:")
    print(f"- Total packets: {stats['total_packets']}")
    print(f"- Success rate: {stats['success_rate']:.2%}")
    print(f"- Recovery efficiency: {stats['recovery_efficiency']:.2%}")
    print(f"- Pending packets: {stats['pending_packets']}")
    print(f"- Retransmissions: {stats['retransmissions']}")
    print(f"- Packets lost: {stats['packets_lost']}")
    print(f"- Current mode: {stats['recovery_mode']}")
    print(f"- Paused: {stats['paused']}")
    
    recovery_manager.stop_recovery_thread()
    return recovery_manager


if __name__ == "__main__":
    print("🚀 Running Enhanced Security & Recovery Demo")
    print("=" * 60)
    
    # Demo encryption
    print("\n" + "="*20 + " ENCRYPTION DEMO " + "="*20)
    encryption_demo = demo_encryption_system()
    
    # Demo recovery
    print("\n" + "="*20 + " RECOVERY DEMO " + "="*20)
    recovery_demo = demo_recovery_system()
    
    # Demo integrated system
    print("\n" + "="*20 + " INTEGRATED DEMO " + "="*20)
    enc_proc, rec_mgr = create_integrated_recovery_system(
        enable_encryption=True,
        recovery_mode=RecoveryMode.ADAPTIVE
    )
    
    print("✅ Integrated encryption + recovery system created")
    print(f"- Encryption enabled: {enc_proc.encryption_enabled}")
    print(f"- Recovery mode: {rec_mgr.mode.value}")
    
    rec_mgr.stop_recovery_thread()
