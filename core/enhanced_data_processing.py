"""
Enhanced Data Processing Module for TEKNOFEST 2025
Integrates AES-128 encryption and data recovery algorithms
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Try to import our enhanced modules
try:
    from core.encryption import EncryptedPacketProcessor, create_packet_with_encryption, extract_packet_with_decryption
    ENCRYPTION_AVAILABLE = True
except ImportError:
    print("⚠️  Encryption module not available")
    ENCRYPTION_AVAILABLE = False

try:
    from core.recovery import DataRecoveryManager, RecoveryMode, create_integrated_recovery_system
    RECOVERY_AVAILABLE = True
except ImportError:
    print("⚠️  Recovery module not available")
    RECOVERY_AVAILABLE = False

from core.data_processing import TransmissionSimulator

@dataclass
class SecurityMetrics:
    """Security and recovery metrics"""
    packets_encrypted: int = 0
    packets_decrypted: int = 0
    encryption_success_rate: float = 0.0
    authentication_failures: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    packets_lost: int = 0
    frequency_escape_failures: int = 0

class EnhancedTransmissionSimulator:
    """Enhanced transmission simulator with encryption and recovery capabilities"""
    
    def __init__(self, enable_encryption: bool = True, enable_recovery: bool = True):
        """
        Initialize enhanced transmission simulator
        
        Args:
            enable_encryption: Enable AES-128 encryption
            enable_recovery: Enable data recovery algorithms
        """
        # Base simulator
        self.base_simulator = TransmissionSimulator()
        
        # Enhanced features
        self.encryption_enabled = enable_encryption
        self.recovery_enabled = enable_recovery
        
        # Initialize encryption processor
        if ENCRYPTION_AVAILABLE and enable_encryption:
            self.encryption_processor = EncryptedPacketProcessor(
                encryption_enabled=True, 
                mode='CTR'
            )
        else:
            self.encryption_processor = None
            
        # Initialize recovery manager
        if RECOVERY_AVAILABLE and enable_recovery:
            self.recovery_manager = DataRecoveryManager(mode=RecoveryMode.ADAPTIVE)
            self.recovery_manager.start_recovery_thread()
        else:
            self.recovery_manager = None
            
        # Security metrics
        self.security_metrics = SecurityMetrics()
        
        # Event callbacks
        self.security_event_callback = None
        self.recovery_event_callback = None
        
        # Current state
        self.current_frequency_band = 1
        self.jammed_bands = set()
        self.last_escape_attempt = 0
        
        logging.info("Enhanced transmission simulator initialized")
        
    def set_security_event_callback(self, callback):
        """Set callback for security events"""
        self.security_event_callback = callback
        
    def set_recovery_event_callback(self, callback):
        """Set callback for recovery events"""
        self.recovery_event_callback = callback
        
    def _emit_security_event(self, event_type: str, message: str, severity: str = "info"):
        """Emit security event"""
        if self.security_event_callback:
            self.security_event_callback(event_type, message, severity)
        logging.info(f"Security Event [{severity.upper()}]: {message}")
    
    def _emit_recovery_event(self, event_type: str, message: str, data: Dict = None):
        """Emit recovery event"""
        if self.recovery_event_callback:
            self.recovery_event_callback(event_type, message, data or {})
        logging.info(f"Recovery Event: {message}")
    
    def set_encryption_enabled(self, enabled: bool, mode: str = 'CTR'):
        """Enable or disable encryption"""
        self.encryption_enabled = enabled
        
        if ENCRYPTION_AVAILABLE and enabled:
            self.encryption_processor = EncryptedPacketProcessor(
                encryption_enabled=True, 
                mode=mode
            )
            self._emit_security_event(
                "encryption_enabled", 
                f"🔒 AES-128 encryption enabled (Mode: {mode})", 
                "info"
            )
        else:
            self.encryption_processor = None
            self._emit_security_event(
                "encryption_disabled", 
                "🔓 AES-128 encryption disabled", 
                "warning"
            )
    
    def set_recovery_enabled(self, enabled: bool, mode: RecoveryMode = RecoveryMode.ADAPTIVE):
        """Enable or disable data recovery"""
        self.recovery_enabled = enabled
        
        if RECOVERY_AVAILABLE and enabled:
            if self.recovery_manager is None:
                self.recovery_manager = DataRecoveryManager(mode=mode)
                self.recovery_manager.start_recovery_thread()
            else:
                self.recovery_manager.set_mode(mode)
            
            self._emit_recovery_event(
                "recovery_enabled",
                f"📡 Data recovery enabled (Mode: {mode.value})"
            )
        else:
            if self.recovery_manager:
                self.recovery_manager.stop_recovery_thread()
            self.recovery_manager = None
            
            self._emit_recovery_event(
                "recovery_disabled",
                "📴 Data recovery disabled"
            )
    
    def transmit_packet(self, packet_data: bytes, packet_id: int, priority: int = 0) -> Tuple[bool, Dict]:
        """
        Transmit packet with encryption and recovery support
        
        Args:
            packet_data: Raw packet data
            packet_id: Packet identifier
            priority: Transmission priority (0=normal, 1=high)
            
        Returns:
            (success, transmission_info)
        """
        transmission_info = {
            'packet_id': packet_id,
            'original_size': len(packet_data),
            'encrypted': False,
            'recovery_enabled': False,
            'timestamp': time.time()
        }
        
        try:
            # Step 1: Apply encryption if enabled
            processed_data = packet_data
            if self.encryption_processor and self.encryption_enabled:
                processed_data, enc_info = self.encryption_processor.process_outgoing_packet(
                    packet_data, packet_id
                )
                transmission_info.update({
                    'encrypted': enc_info.get('encrypted', False),
                    'encrypted_size': enc_info.get('encrypted_size', len(processed_data)),
                    'encryption_overhead': enc_info.get('overhead_bytes', 0)
                })
                
                if enc_info.get('encrypted', False):
                    self.security_metrics.packets_encrypted += 1
                    self._emit_security_event(
                        "packet_encrypted",
                        f"🔐 Packet {packet_id} encrypted ({len(packet_data)} → {len(processed_data)} bytes)"
                    )
            
            # Step 2: Submit to recovery system if enabled
            if self.recovery_manager and self.recovery_enabled:
                recovery_success = self.recovery_manager.submit_packet(
                    packet_id, processed_data, priority
                )
                transmission_info['recovery_enabled'] = recovery_success
                
                if recovery_success:
                    self._emit_recovery_event(
                        "packet_submitted",
                        f"📦 Packet {packet_id} submitted to recovery system"
                    )
            
            # Step 3: Simulate transmission (using base simulator)
            success = self._simulate_transmission(processed_data, packet_id)
            
            # Step 4: Report transmission result
            if self.recovery_manager:
                # Simulate channel conditions
                channel_quality = np.random.uniform(0.5, 1.0)
                error_type = None if success else np.random.choice(['crc_error', 'timeout', 'jamming'])
                
                self.recovery_manager.report_transmission_result(
                    packet_id=packet_id,
                    success=success,
                    error_type=error_type,
                    channel_quality=channel_quality,
                    frequency_band=self.current_frequency_band
                )
                
                if not success:
                    self._emit_recovery_event(
                        "transmission_failed",
                        f"❌ Packet {packet_id} transmission failed: {error_type}",
                        {'packet_id': packet_id, 'error_type': error_type}
                    )
            
            transmission_info['success'] = success
            return success, transmission_info
            
        except Exception as e:
            transmission_info['error'] = str(e)
            self._emit_security_event(
                "transmission_error",
                f"🚨 Transmission error for packet {packet_id}: {e}",
                "error"
            )
            return False, transmission_info
    
    def receive_packet(self, received_data: bytes, expected_packet_id: int) -> Tuple[Optional[bytes], Dict]:
        """
        Receive and process packet with decryption support
        
        Args:
            received_data: Received packet data
            expected_packet_id: Expected packet ID
            
        Returns:
            (decrypted_data, reception_info)
        """
        reception_info = {
            'packet_id': expected_packet_id,
            'received_size': len(received_data),
            'decrypted': False,
            'authenticated': False,
            'timestamp': time.time()
        }
        
        try:
            # Step 1: Apply decryption if enabled
            processed_data = received_data
            if self.encryption_processor and self.encryption_enabled:
                processed_data, dec_info = self.encryption_processor.process_incoming_packet(
                    received_data, expected_packet_id
                )
                reception_info.update({
                    'decrypted': dec_info.get('encrypted', False),
                    'authenticated': dec_info.get('processing_info', {}).get('authenticated', False),
                    'decrypted_size': dec_info.get('decrypted_size', len(processed_data))
                })
                
                if dec_info.get('encrypted', False):
                    self.security_metrics.packets_decrypted += 1
                    
                    if reception_info['authenticated']:
                        self._emit_security_event(
                            "packet_decrypted",
                            f"🔓 Packet {expected_packet_id} decrypted and authenticated"
                        )
                    else:
                        self.security_metrics.authentication_failures += 1
                        self._emit_security_event(
                            "authentication_failed",
                            f"🚨 Authentication failed for packet {expected_packet_id}",
                            "warning"
                        )
            
            reception_info['success'] = True
            return processed_data, reception_info
            
        except Exception as e:
            reception_info['error'] = str(e)
            self._emit_security_event(
                "reception_error",
                f"🚨 Reception error for packet {expected_packet_id}: {e}",
                "error"
            )
            return None, reception_info
    
    def _simulate_transmission(self, data: bytes, packet_id: int) -> bool:
        """Simulate packet transmission"""
        # Use base simulator or implement custom logic
        if hasattr(self.base_simulator, 'transmit_packet'):
            return self.base_simulator.transmit_packet(data, packet_id)
        else:
            # Simple simulation: 80% success rate, affected by jamming
            base_success_rate = 0.8
            jamming_penalty = len(self.jammed_bands) * 0.15
            success_rate = max(0.1, base_success_rate - jamming_penalty)
            return np.random.random() < success_rate
    
    def handle_frequency_escape_failure(self, jammed_bands: List[int]):
        """Handle frequency escape failure scenario"""
        self.jammed_bands.update(jammed_bands)
        self.security_metrics.frequency_escape_failures += 1
        
        # Notify recovery manager
        if self.recovery_manager:
            self.recovery_manager.handle_frequency_escape_failure(jammed_bands)
            
        self._emit_recovery_event(
            "frequency_escape_failed",
            f"⚠️ Frequency escape failed - bands {jammed_bands} jammed",
            {'jammed_bands': jammed_bands}
        )
        
        # Emit security alert
        self._emit_security_event(
            "jamming_detected",
            f"🚨 All frequency bands jammed - entering recovery mode",
            "critical"
        )
    
    def attempt_frequency_escape(self, target_band: int) -> bool:
        """Attempt to escape to a different frequency band"""
        if target_band in self.jammed_bands:
            return False
            
        self.current_frequency_band = target_band
        self.last_escape_attempt = time.time()
        
        # Simulate escape success
        escape_success = np.random.random() > 0.3  # 70% success rate
        
        if escape_success:
            self._emit_security_event(
                "frequency_escape_success",
                f"✅ Successfully escaped to frequency band {target_band}"
            )
            # Remove band from jammed list if escape was successful
            self.jammed_bands.discard(target_band)
        else:
            self._emit_security_event(
                "frequency_escape_failed",
                f"❌ Failed to escape to frequency band {target_band}",
                "warning"
            )
            self.jammed_bands.add(target_band)
            
        return escape_success
    
    def get_encryption_stats(self) -> Dict:
        """Get encryption statistics"""
        if self.encryption_processor:
            return self.encryption_processor.get_stats()
        return {'encryption_enabled': False}
    
    def get_recovery_stats(self) -> Dict:
        """Get recovery statistics"""
        if self.recovery_manager:
            return self.recovery_manager.get_recovery_stats()
        return {'recovery_enabled': False}
    
    def get_security_metrics(self) -> Dict:
        """Get comprehensive security metrics"""
        metrics = {
            'packets_encrypted': self.security_metrics.packets_encrypted,
            'packets_decrypted': self.security_metrics.packets_decrypted,
            'authentication_failures': self.security_metrics.authentication_failures,
            'recovery_attempts': self.security_metrics.recovery_attempts,
            'successful_recoveries': self.security_metrics.successful_recoveries,
            'packets_lost': self.security_metrics.packets_lost,
            'frequency_escape_failures': self.security_metrics.frequency_escape_failures,
            'current_frequency_band': self.current_frequency_band,
            'jammed_bands': list(self.jammed_bands),
            'encryption_enabled': self.encryption_enabled,
            'recovery_enabled': self.recovery_enabled
        }
        
        # Calculate derived metrics
        total_encrypted = self.security_metrics.packets_encrypted
        total_decrypted = self.security_metrics.packets_decrypted
        
        if total_encrypted > 0:
            metrics['encryption_success_rate'] = total_encrypted / (total_encrypted + self.security_metrics.authentication_failures)
        else:
            metrics['encryption_success_rate'] = 0.0
            
        if total_decrypted > 0:
            metrics['authentication_success_rate'] = 1.0 - (self.security_metrics.authentication_failures / total_decrypted)
        else:
            metrics['authentication_success_rate'] = 0.0
            
        return metrics
    
    def export_security_key(self) -> Optional[str]:
        """Export current encryption key"""
        if self.encryption_processor and self.encryption_processor.aes_handler:
            return self.encryption_processor.aes_handler.export_key()
        return None
    
    def import_security_key(self, key_hex: str) -> bool:
        """Import encryption key"""
        try:
            if self.encryption_processor and self.encryption_processor.aes_handler:
                self.encryption_processor.aes_handler.import_key(key_hex)
                self._emit_security_event(
                    "key_imported",
                    "🔑 Encryption key imported successfully"
                )
                return True
        except Exception as e:
            self._emit_security_event(
                "key_import_failed",
                f"🚨 Failed to import key: {e}",
                "error"
            )
        return False
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        stats = {
            'security_metrics': self.get_security_metrics(),
            'encryption_stats': self.get_encryption_stats(),
            'recovery_stats': self.get_recovery_stats(),
            'system_status': {
                'encryption_available': ENCRYPTION_AVAILABLE,
                'recovery_available': RECOVERY_AVAILABLE,
                'encryption_enabled': self.encryption_enabled,
                'recovery_enabled': self.recovery_enabled,
                'current_frequency_band': self.current_frequency_band,
                'jammed_bands_count': len(self.jammed_bands)
            }
        }
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.security_metrics = SecurityMetrics()
        
        if self.encryption_processor:
            self.encryption_processor.aes_handler.reset_stats()
            
        if self.recovery_manager:
            self.recovery_manager.reset_stats()
            
        self._emit_security_event(
            "stats_reset",
            "📊 All statistics have been reset"
        )
    
    def shutdown(self):
        """Shutdown the enhanced simulator"""
        if self.recovery_manager:
            self.recovery_manager.stop_recovery_thread()
            
        logging.info("Enhanced transmission simulator shutdown")


def create_enhanced_simulator(encryption_enabled: bool = True, recovery_enabled: bool = True) -> EnhancedTransmissionSimulator:
    """
    Factory function to create enhanced transmission simulator
    
    Args:
        encryption_enabled: Enable AES-128 encryption
        recovery_enabled: Enable data recovery algorithms
        
    Returns:
        Enhanced transmission simulator instance
    """
    return EnhancedTransmissionSimulator(
        enable_encryption=encryption_enabled,
        enable_recovery=recovery_enabled
    )


# Demo and testing functions
def demo_enhanced_system():
    """Demonstrate the enhanced transmission system"""
    print("🚀 Enhanced Transmission System Demo")
    print("=" * 50)
    
    # Create enhanced simulator
    simulator = create_enhanced_simulator(
        encryption_enabled=True,
        recovery_enabled=True
    )
    
    # Set up event callbacks
    def security_event_handler(event_type, message, severity):
        print(f"[SECURITY {severity.upper()}] {message}")
    
    def recovery_event_handler(event_type, message, data):
        print(f"[RECOVERY] {message}")
    
    simulator.set_security_event_callback(security_event_handler)
    simulator.set_recovery_event_callback(recovery_event_handler)
    
    # Simulate packet transmissions
    print("\n📡 Transmitting test packets...")
    for i in range(10):
        packet_data = f"Test packet {i} for TEKNOFEST 2025 competition".encode() * 5
        success, info = simulator.transmit_packet(packet_data, packet_id=i, priority=i % 2)
        
        if success:
            # Simulate reception
            received_data, recv_info = simulator.receive_packet(packet_data, expected_packet_id=i)
            if received_data:
                print(f"✅ Packet {i}: TX/RX successful")
            else:
                print(f"❌ Packet {i}: RX failed")
        else:
            print(f"❌ Packet {i}: TX failed")
    
    # Simulate frequency jamming
    print("\n⚠️ Simulating frequency jamming...")
    simulator.handle_frequency_escape_failure([1, 2, 3, 4, 5])
    
    # Attempt frequency escape
    print("\n🔄 Attempting frequency escape...")
    for band in [3, 4, 5]:
        success = simulator.attempt_frequency_escape(band)
        if success:
            print(f"✅ Escaped to band {band}")
            break
    
    # Show comprehensive statistics
    stats = simulator.get_comprehensive_stats()
    print(f"\n📊 System Statistics:")
    print(f"- Packets encrypted: {stats['security_metrics']['packets_encrypted']}")
    print(f"- Packets decrypted: {stats['security_metrics']['packets_decrypted']}")
    print(f"- Authentication success rate: {stats['security_metrics']['authentication_success_rate']:.2%}")
    print(f"- Recovery attempts: {stats['security_metrics']['recovery_attempts']}")
    print(f"- Frequency escape failures: {stats['security_metrics']['frequency_escape_failures']}")
    
    # Export security key
    key = simulator.export_security_key()
    if key:
        print(f"- Encryption key: {key[:16]}...")
    
    # Cleanup
    simulator.shutdown()
    return simulator


if __name__ == "__main__":
    demo_enhanced_system()
