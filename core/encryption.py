"""
AES-128 Encryption Module for TEKNOFEST 2025
End-to-end security with AES-CTR/GCM modes for wireless communication
"""

import os
import time
import hashlib
import hmac
import secrets
import struct
from typing import Tuple, Dict, Optional, Union, Any
import numpy as np

# Try to import cryptographic libraries (fallback to built-in if not available)
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    print("⚠️  Cryptography library not available, using fallback AES implementation")
    CRYPTO_AVAILABLE = False

from config import COMPETITION_CONFIG

class SimpleAES:
    """Simplified AES implementation for educational/demo purposes"""
    
    def __init__(self, key: bytes):
        """Initialize with 128-bit key"""
        if len(key) != 16:
            raise ValueError("Key must be 16 bytes for AES-128")
        self.key = key
        
    def encrypt_ctr(self, plaintext: bytes, nonce: bytes) -> bytes:
        """Simple CTR mode encryption using XOR with key-derived stream"""
        if len(nonce) != 16:
            raise ValueError("Nonce must be 16 bytes")
            
        # Generate key stream using HMAC-based approach
        keystream = self._generate_keystream(len(plaintext), nonce)
        
        # XOR plaintext with keystream
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream))
        return ciphertext
    
    def decrypt_ctr(self, ciphertext: bytes, nonce: bytes) -> bytes:
        """Simple CTR mode decryption (same as encryption in CTR)"""
        return self.encrypt_ctr(ciphertext, nonce)
    
    def _generate_keystream(self, length: int, nonce: bytes) -> bytes:
        """Generate keystream using HMAC-SHA256"""
        keystream = b''
        counter = 0
        
        while len(keystream) < length:
            # Create counter block
            counter_block = nonce + struct.pack('>I', counter)
            
            # Generate block using HMAC
            block = hashlib.pbkdf2_hmac('sha256', self.key, counter_block, 1, 16)
            keystream += block
            counter += 1
            
        return keystream[:length]

class AESEncryption:
    """AES-128 encryption handler with CTR and simulated GCM modes"""
    
    def __init__(self, mode: str = 'CTR', key: Optional[bytes] = None):
        """
        Initialize AES encryption
        
        Args:
            mode: 'CTR' or 'GCM' (GCM provides authentication)
            key: 16-byte key or None for auto-generation
        """
        self.mode = mode.upper()
        self.key_size = 16  # AES-128
        
        if key is None:
            self.key = self._generate_key()
        else:
            if len(key) != self.key_size:
                raise ValueError(f"Key must be {self.key_size} bytes for AES-128")
            self.key = key
            
        # Initialize cipher based on available libraries
        if CRYPTO_AVAILABLE:
            self._use_cryptography = True
        else:
            self._use_cryptography = False
            self._simple_aes = SimpleAES(self.key)
            
        # Security metrics
        self.encryption_stats = {
            'packets_encrypted': 0,
            'packets_decrypted': 0,
            'encryption_failures': 0,
            'decryption_failures': 0,
            'total_bytes_encrypted': 0,
            'total_bytes_decrypted': 0,
            'authentication_failures': 0,
        }
        
    def _generate_key(self) -> bytes:
        """Generate a cryptographically secure 128-bit key"""
        return secrets.token_bytes(self.key_size)
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive AES key from password using PBKDF2
        
        Args:
            password: User password
            salt: Random salt (generated if None)
            
        Returns:
            (derived_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(16)
            
        if CRYPTO_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.key_size,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            derived_key = kdf.derive(password.encode('utf-8'))
        else:
            # Fallback using hashlib
            derived_key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000,
                self.key_size
            )
            
        self.key = derived_key
        if not self._use_cryptography:
            self._simple_aes = SimpleAES(self.key)
        return derived_key, salt
    
    def encrypt_packet(self, packet_data: bytes, packet_id: int = 0) -> Tuple[bytes, Dict]:
        """
        Encrypt a data packet with AES
        
        Args:
            packet_data: Raw packet data to encrypt
            packet_id: Packet sequence number for nonce generation
            
        Returns:
            (encrypted_packet, encryption_info)
        """
        try:
            if self.mode == 'GCM':
                return self._encrypt_gcm(packet_data, packet_id)
            elif self.mode == 'CTR':
                return self._encrypt_ctr(packet_data, packet_id)
            else:
                raise ValueError(f"Unsupported encryption mode: {self.mode}")
                
        except Exception as e:
            self.encryption_stats['encryption_failures'] += 1
            raise Exception(f"Encryption failed: {e}")
    
    def _encrypt_gcm(self, data: bytes, packet_id: int) -> Tuple[bytes, Dict]:
        """Encrypt using AES-GCM (or simulated GCM with authentication)"""
        # Generate nonce
        nonce = self._generate_nonce(12)  # 96-bit for GCM
        
        if CRYPTO_AVAILABLE:
            # Use real GCM mode
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.GCM(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Add additional authenticated data
            aad = struct.pack('>IQ', packet_id, int(time.time() * 1000))
            encryptor.authenticate_additional_data(aad)
            
            ciphertext = encryptor.update(data) + encryptor.finalize()
            auth_tag = encryptor.tag
            
            # Create encrypted packet structure
            encrypted_packet = nonce + aad + auth_tag + ciphertext
        else:
            # Simulate GCM with CTR + HMAC
            # Extend nonce to 16 bytes for CTR
            extended_nonce = nonce + b'\x00\x00\x00\x01'
            
            # Encrypt with CTR
            ciphertext = self._simple_aes.encrypt_ctr(data, extended_nonce)
            
            # Generate authentication tag using HMAC
            aad = struct.pack('>IQ', packet_id, int(time.time() * 1000))
            auth_data = nonce + aad + ciphertext
            auth_tag = hmac.new(self.key, auth_data, hashlib.sha256).digest()[:16]
            
            # Create encrypted packet structure
            encrypted_packet = nonce + aad + auth_tag + ciphertext
        
        # Update statistics
        self.encryption_stats['packets_encrypted'] += 1
        self.encryption_stats['total_bytes_encrypted'] += len(data)
        
        encryption_info = {
            'mode': 'GCM',
            'nonce': nonce.hex(),
            'auth_tag': auth_tag.hex(),
            'packet_id': packet_id,
            'original_size': len(data),
            'encrypted_size': len(encrypted_packet),
            'timestamp': time.time(),
            'crypto_library': 'cryptography' if CRYPTO_AVAILABLE else 'fallback'
        }
        
        return encrypted_packet, encryption_info
    
    def _encrypt_ctr(self, data: bytes, packet_id: int) -> Tuple[bytes, Dict]:
        """Encrypt using AES-CTR mode"""
        # Generate nonce
        nonce = self._generate_nonce(16)  # 128-bit for CTR
        
        if CRYPTO_AVAILABLE:
            # Use real CTR mode
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CTR(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
        else:
            # Use simple CTR
            ciphertext = self._simple_aes.encrypt_ctr(data, nonce)
        
        # Add integrity check (HMAC-SHA256)
        hmac_key = hashlib.sha256(self.key + b'hmac').digest()[:16]
        hmac_data = nonce + ciphertext
        hmac_digest = hmac.new(hmac_key, hmac_data, hashlib.sha256).digest()[:16]
        
        # Create encrypted packet structure
        encrypted_packet = nonce + hmac_digest + ciphertext
        
        # Update statistics
        self.encryption_stats['packets_encrypted'] += 1
        self.encryption_stats['total_bytes_encrypted'] += len(data)
        
        encryption_info = {
            'mode': 'CTR',
            'nonce': nonce.hex(),
            'hmac': hmac_digest.hex(),
            'packet_id': packet_id,
            'original_size': len(data),
            'encrypted_size': len(encrypted_packet),
            'timestamp': time.time(),
            'crypto_library': 'cryptography' if CRYPTO_AVAILABLE else 'fallback'
        }
        
        return encrypted_packet, encryption_info
    
    def decrypt_packet(self, encrypted_packet: bytes, expected_packet_id: int = 0) -> Tuple[bytes, Dict]:
        """
        Decrypt an encrypted packet
        
        Args:
            encrypted_packet: Encrypted data to decrypt
            expected_packet_id: Expected packet ID for verification
            
        Returns:
            (decrypted_data, decryption_info)
        """
        try:
            if self.mode == 'GCM':
                return self._decrypt_gcm(encrypted_packet, expected_packet_id)
            elif self.mode == 'CTR':
                return self._decrypt_ctr(encrypted_packet, expected_packet_id)
            else:
                raise ValueError(f"Unsupported decryption mode: {self.mode}")
                
        except Exception as e:
            self.encryption_stats['decryption_failures'] += 1
            raise Exception(f"Decryption failed: {e}")
    
    def _decrypt_gcm(self, encrypted_packet: bytes, expected_packet_id: int) -> Tuple[bytes, Dict]:
        """Decrypt using AES-GCM (or simulated GCM)"""
        if len(encrypted_packet) < 40:  # Min size: nonce(12) + aad(12) + tag(16)
            raise ValueError("Encrypted packet too short for GCM")
        
        # Extract components
        nonce = encrypted_packet[:12]
        aad = encrypted_packet[12:24]
        auth_tag = encrypted_packet[24:40]
        ciphertext = encrypted_packet[40:]
        
        # Verify packet ID from AAD
        packet_id, timestamp = struct.unpack('>IQ', aad)
        if packet_id != expected_packet_id:
            self.encryption_stats['authentication_failures'] += 1
            raise ValueError(f"Packet ID mismatch: expected {expected_packet_id}, got {packet_id}")
        
        if CRYPTO_AVAILABLE:
            # Use real GCM mode
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.GCM(nonce, auth_tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decryptor.authenticate_additional_data(aad)
            
            try:
                decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
            except Exception as e:
                self.encryption_stats['authentication_failures'] += 1
                raise ValueError(f"Authentication failed: {e}")
        else:
            # Simulate GCM verification
            auth_data = nonce + aad + ciphertext
            expected_tag = hmac.new(self.key, auth_data, hashlib.sha256).digest()[:16]
            
            if auth_tag != expected_tag:
                self.encryption_stats['authentication_failures'] += 1
                raise ValueError("Authentication tag verification failed")
            
            # Decrypt with CTR
            extended_nonce = nonce + b'\x00\x00\x00\x01'
            decrypted_data = self._simple_aes.decrypt_ctr(ciphertext, extended_nonce)
        
        # Update statistics
        self.encryption_stats['packets_decrypted'] += 1
        self.encryption_stats['total_bytes_decrypted'] += len(decrypted_data)
        
        decryption_info = {
            'mode': 'GCM',
            'packet_id': packet_id,
            'authenticated': True,
            'original_size': len(decrypted_data),
            'encrypted_size': len(encrypted_packet),
            'timestamp': timestamp / 1000.0,
            'crypto_library': 'cryptography' if CRYPTO_AVAILABLE else 'fallback'
        }
        
        return decrypted_data, decryption_info
    
    def _decrypt_ctr(self, encrypted_packet: bytes, expected_packet_id: int) -> Tuple[bytes, Dict]:
        """Decrypt using AES-CTR mode"""
        if len(encrypted_packet) < 32:  # Min size: nonce(16) + hmac(16)
            raise ValueError("Encrypted packet too short for CTR")
        
        # Extract components
        nonce = encrypted_packet[:16]
        hmac_digest = encrypted_packet[16:32]
        ciphertext = encrypted_packet[32:]
        
        # Verify HMAC
        hmac_key = hashlib.sha256(self.key + b'hmac').digest()[:16]
        hmac_data = nonce + ciphertext
        expected_hmac = hmac.new(hmac_key, hmac_data, hashlib.sha256).digest()[:16]
        
        if hmac_digest != expected_hmac:
            self.encryption_stats['authentication_failures'] += 1
            raise ValueError("HMAC verification failed")
        
        if CRYPTO_AVAILABLE:
            # Use real CTR mode
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CTR(nonce),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
        else:
            # Use simple CTR
            decrypted_data = self._simple_aes.decrypt_ctr(ciphertext, nonce)
        
        # Update statistics
        self.encryption_stats['packets_decrypted'] += 1
        self.encryption_stats['total_bytes_decrypted'] += len(decrypted_data)
        
        decryption_info = {
            'mode': 'CTR',
            'authenticated': True,
            'original_size': len(decrypted_data),
            'encrypted_size': len(encrypted_packet),
            'timestamp': time.time(),
            'crypto_library': 'cryptography' if CRYPTO_AVAILABLE else 'fallback'
        }
        
        return decrypted_data, decryption_info
    
    def _decrypt_ctr(self, encrypted_packet: bytes, expected_packet_id: int) -> Tuple[bytes, Dict]:
        """Decrypt using AES-CTR"""
        if len(encrypted_packet) < 32:  # Min size: nonce(16) + hmac(16)
            raise ValueError("Encrypted packet too short for CTR")
        
        # Extract components
        nonce = encrypted_packet[:16]
        hmac_received = encrypted_packet[16:32]
        ciphertext = encrypted_packet[32:]
        
        # Verify HMAC
        hmac_key = hashlib.sha256(self.key + b'hmac').digest()[:16]
        hmac_data = nonce + ciphertext
        hmac_calculated = hashlib.hmac.new(hmac_key, hmac_data, hashlib.sha256).digest()[:16]
        
        if hmac_received != hmac_calculated:
            self.encryption_stats['authentication_failures'] += 1
            raise ValueError("HMAC verification failed")
        
        if CRYPTO_AVAILABLE:
            # Use real CTR mode
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CTR(nonce),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
        else:
            # Use simple CTR
            decrypted_data = self._simple_aes.decrypt_ctr(ciphertext, nonce)
        
        # Update statistics
        self.encryption_stats['packets_decrypted'] += 1
        self.encryption_stats['total_bytes_decrypted'] += len(decrypted_data)
        
        decryption_info = {
            'mode': 'CTR',
            'packet_id': expected_packet_id,
            'authenticated': True,
            'original_size': len(decrypted_data),
            'encrypted_size': len(encrypted_packet),
            'timestamp': time.time(),
            'crypto_library': 'cryptography' if CRYPTO_AVAILABLE else 'fallback'
        }
        
        return decrypted_data, decryption_info
    
    def _generate_nonce(self, size: int) -> bytes:
        """Generate nonce of specified size"""
        return secrets.token_bytes(size)
    
    def get_encryption_stats(self) -> Dict:
        """Get encryption/decryption statistics"""
        stats = self.encryption_stats.copy()
        
        # Calculate additional metrics
        if stats['packets_encrypted'] > 0:
            stats['avg_encrypted_size'] = stats['total_bytes_encrypted'] / stats['packets_encrypted']
            stats['encryption_success_rate'] = (
                stats['packets_encrypted'] / 
                (stats['packets_encrypted'] + stats['encryption_failures'])
            )
        else:
            stats['avg_encrypted_size'] = 0
            stats['encryption_success_rate'] = 0
            
        if stats['packets_decrypted'] > 0:
            stats['avg_decrypted_size'] = stats['total_bytes_decrypted'] / stats['packets_decrypted']
            stats['decryption_success_rate'] = (
                stats['packets_decrypted'] / 
                (stats['packets_decrypted'] + stats['decryption_failures'])
            )
            stats['authentication_success_rate'] = (
                (stats['packets_decrypted'] - stats['authentication_failures']) /
                stats['packets_decrypted']
            )
        else:
            stats['avg_decrypted_size'] = 0
            stats['decryption_success_rate'] = 0
            stats['authentication_success_rate'] = 0
            
        return stats
    
    def reset_stats(self):
        """Reset encryption statistics"""
        self.encryption_stats = {
            'packets_encrypted': 0,
            'packets_decrypted': 0,
            'encryption_failures': 0,
            'decryption_failures': 0,
            'total_bytes_encrypted': 0,
            'total_bytes_decrypted': 0,
            'authentication_failures': 0,
        }
    
    def export_key(self) -> str:
        """Export encryption key as hex string"""
        return self.key.hex()
    
    def import_key(self, key_hex: str):
        """Import encryption key from hex string"""
        key_bytes = bytes.fromhex(key_hex)
        if len(key_bytes) != self.key_size:
            raise ValueError(f"Key must be {self.key_size} bytes")
        self.key = key_bytes
        if not self._use_cryptography:
            self._simple_aes = SimpleAES(self.key)


class EncryptedPacketProcessor:
    """High-level encrypted packet handling compatible with existing data_processing"""
    
    def __init__(self, encryption_enabled: bool = True, mode: str = 'CTR'):
        """
        Initialize encrypted packet processor
        
        Args:
            encryption_enabled: Enable/disable encryption
            mode: AES mode ('GCM' or 'CTR')
        """
        self.encryption_enabled = encryption_enabled
        self.aes_handler = AESEncryption(mode=mode) if encryption_enabled else None
        
        # Compatibility with existing packet structure
        self.packet_overhead = 40 if mode == 'GCM' else 32  # Encryption overhead
        
    def process_outgoing_packet(self, packet_data: bytes, packet_id: int) -> Tuple[bytes, Dict]:
        """
        Process outgoing packet (encrypt if enabled)
        
        Args:
            packet_data: Original packet data
            packet_id: Packet sequence number
            
        Returns:
            (processed_packet, processing_info)
        """
        if not self.encryption_enabled or self.aes_handler is None:
            return packet_data, {'encrypted': False, 'original_size': len(packet_data)}
        
        try:
            encrypted_packet, encryption_info = self.aes_handler.encrypt_packet(packet_data, packet_id)
            
            processing_info = {
                'encrypted': True,
                'encryption_mode': self.aes_handler.mode,
                'original_size': len(packet_data),
                'encrypted_size': len(encrypted_packet),
                'overhead_bytes': len(encrypted_packet) - len(packet_data),
                'encryption_info': encryption_info
            }
            
            return encrypted_packet, processing_info
            
        except Exception as e:
            # Fall back to unencrypted if encryption fails
            processing_info = {
                'encrypted': False,
                'encryption_failed': True,
                'error': str(e),
                'original_size': len(packet_data)
            }
            return packet_data, processing_info
    
    def process_incoming_packet(self, packet_data: bytes, packet_id: int) -> Tuple[bytes, Dict]:
        """
        Process incoming packet (decrypt if encrypted)
        
        Args:
            packet_data: Received packet data
            packet_id: Expected packet sequence number
            
        Returns:
            (processed_packet, processing_info)
        """
        if not self.encryption_enabled or self.aes_handler is None:
            return packet_data, {'encrypted': False, 'original_size': len(packet_data)}
        
        # Check if packet looks encrypted (has encryption overhead)
        if len(packet_data) < self.packet_overhead:
            return packet_data, {'encrypted': False, 'too_short_for_encryption': True}
        
        try:
            decrypted_packet, decryption_info = self.aes_handler.decrypt_packet(packet_data, packet_id)
            
            processing_info = {
                'encrypted': True,
                'decryption_mode': self.aes_handler.mode,
                'encrypted_size': len(packet_data),
                'decrypted_size': len(decrypted_packet),
                'decryption_info': decryption_info
            }
            
            return decrypted_packet, processing_info
            
        except Exception as e:
            # Return original packet if decryption fails
            processing_info = {
                'encrypted': True,
                'decryption_failed': True,
                'error': str(e),
                'encrypted_size': len(packet_data)
            }
            return packet_data, processing_info
    
    def toggle_encryption(self, enabled: bool):
        """Toggle encryption on/off"""
        if enabled and self.aes_handler is None:
            self.aes_handler = AESEncryption(mode='CTR')
        self.encryption_enabled = enabled
    
    def get_stats(self) -> Dict:
        """Get encryption statistics"""
        if self.aes_handler:
            return self.aes_handler.get_encryption_stats()
        return {'encryption_enabled': False}
    
    def set_key(self, key: bytes):
        """Set encryption key"""
        if self.aes_handler:
            self.aes_handler.key = key
            if not self.aes_handler._use_cryptography:
                self.aes_handler._simple_aes = SimpleAES(key)


def create_packet_with_encryption(header: bytes, payload: bytes, encryption_processor: EncryptedPacketProcessor, packet_id: int) -> Tuple[bytes, Dict]:
    """
    Create packet with optional encryption, maintaining compatibility with existing packet structure
    
    Packet Structure (compatible with existing system):
    - Header (16 bytes): SYNC + sequence + total_packets + payload_length + type + reserved
    - Encrypted Payload: AES-encrypted payload data
    - CRC (4 bytes): CRC of header + encrypted_payload
    
    Args:
        header: 16-byte packet header
        payload: Payload data to encrypt
        encryption_processor: Encryption handler
        packet_id: Packet sequence number
        
    Returns:
        (complete_packet, packet_info)
    """
    import struct
    import zlib
    
    # Process payload (encrypt if enabled)
    processed_payload, processing_info = encryption_processor.process_outgoing_packet(payload, packet_id)
    
    # Create complete packet data (header + processed_payload)
    packet_data = header + processed_payload
    
    # Calculate CRC for entire packet
    crc = struct.pack('<I', zlib.crc32(packet_data) & 0xffffffff)
    complete_packet = packet_data + crc
    
    packet_info = {
        'header_size': len(header),
        'payload_size': len(payload),
        'processed_payload_size': len(processed_payload),
        'crc_size': len(crc),
        'total_size': len(complete_packet),
        'processing_info': processing_info
    }
    
    return complete_packet, packet_info


def extract_packet_with_decryption(packet: bytes, encryption_processor: EncryptedPacketProcessor, expected_packet_id: int) -> Tuple[bytes, bytes, Dict]:
    """
    Extract and decrypt packet, maintaining compatibility with existing packet structure
    
    Args:
        packet: Complete packet with header + encrypted_payload + CRC
        encryption_processor: Encryption handler
        expected_packet_id: Expected packet sequence number
        
    Returns:
        (header, decrypted_payload, packet_info)
    """
    import struct
    import zlib
    
    if len(packet) < 20:  # Minimum: header(16) + crc(4)
        raise ValueError("Packet too short")
    
    # Extract header and verify packet structure
    header = packet[:16]
    crc_received = packet[-4:]
    encrypted_payload = packet[16:-4]
    
    # Verify CRC
    packet_data = header + encrypted_payload
    crc_calculated = struct.pack('<I', zlib.crc32(packet_data) & 0xffffffff)
    
    crc_valid = crc_received == crc_calculated
    
    # Process payload (decrypt if encrypted)
    decrypted_payload, processing_info = encryption_processor.process_incoming_packet(
        encrypted_payload, expected_packet_id
    )
    
    packet_info = {
        'header_size': len(header),
        'encrypted_payload_size': len(encrypted_payload),
        'decrypted_payload_size': len(decrypted_payload),
        'crc_valid': crc_valid,
        'total_size': len(packet),
        'processing_info': processing_info
    }
    
    if not crc_valid:
        packet_info['crc_error'] = True
    
    return header, decrypted_payload, packet_info


# Example usage and testing functions
def demo_encryption_system():
    """Demonstrate the encryption system"""
    print("🔐 AES-128 Encryption System Demo")
    print("=" * 50)
    
    # Create encryption processor
    processor = EncryptedPacketProcessor(encryption_enabled=True, mode='CTR')
    
    # Sample data
    test_payload = b"This is a test payload for TEKNOFEST 2025 competition. " * 10
    packet_id = 12345
    
    # Create sample header (compatible with existing format)
    header = b'SYNC' + struct.pack('>III', packet_id, 1, len(test_payload)) + b'\x01\x00\x00\x00'
    
    print(f"Original payload size: {len(test_payload)} bytes")
    
    # Create encrypted packet
    complete_packet, packet_info = create_packet_with_encryption(
        header, test_payload, processor, packet_id
    )
    
    print(f"Encrypted packet size: {packet_info['total_size']} bytes")
    print(f"Encryption overhead: {packet_info['total_size'] - len(test_payload) - 20} bytes")
    print(f"Encrypted: {packet_info['processing_info']['encrypted']}")
    
    # Extract and decrypt
    extracted_header, decrypted_payload, extract_info = extract_packet_with_decryption(
        complete_packet, processor, packet_id
    )
    
    print(f"Decrypted payload size: {extract_info['decrypted_payload_size']} bytes")
    print(f"CRC valid: {extract_info['crc_valid']}")
    print(f"Data integrity: {'✅ PASS' if decrypted_payload == test_payload else '❌ FAIL'}")
    
    # Show statistics
    stats = processor.get_stats()
    print(f"\nEncryption Statistics:")
    print(f"- Packets encrypted: {stats.get('packets_encrypted', 0)}")
    print(f"- Packets decrypted: {stats.get('packets_decrypted', 0)}")
    print(f"- Success rate: {stats.get('encryption_success_rate', 0):.2%}")
    
    return processor


if __name__ == "__main__":
    demo_encryption_system()
