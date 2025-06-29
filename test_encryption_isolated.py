#!/usr/bin/env python3
"""
Quick encryption test to isolate the issue
"""

import os
import sys
import struct

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.encryption import AESEncryption, EncryptedPacketProcessor, create_packet_with_encryption, extract_packet_with_decryption

def test_encryption_isolated():
    """Test encryption functionality in isolation"""
    print("🔒 Testing AES-128 Encryption in isolation...")
    
    # Create test data
    test_data = b"""
TEKNOFEST 2025 Competition Data Packet
=====================================
This is a test payload containing both text and binary data.
Frequency Band: 2.41 GHz
Modulation: 1024QAM
Coding: LDPC (Rate 5/6)
SNR: 28.3 dB
Channel Quality: Excellent
Timestamp: 2025-06-29 12:34:56
""".strip()
    
    print(f"Test data: {len(test_data)} bytes")
    print(f"Preview: {test_data[:50]}...")
    
    # Test packet processor
    processor = EncryptedPacketProcessor(encryption_enabled=True, mode='CTR')
    processed_packet, proc_info = processor.process_outgoing_packet(test_data, packet_id=2)
    recovered_packet, rec_info = processor.process_incoming_packet(processed_packet, packet_id=2)
    
    print(f"Processed: {len(processed_packet)} bytes")
    print(f"Recovered: {len(recovered_packet)} bytes")
    print(f"Match: {recovered_packet == test_data}")
    
    if recovered_packet != test_data:
        print(f"Original: {test_data}")
        print(f"Recovered: {recovered_packet}")
        return False
    
    # Test complete packet creation/extraction  
    # Standard packet header format: SYNC(4) + seq(4) + total(4) + len(2) + type(1) + reserved(1) = 16 bytes
    header = b'SYNC' + struct.pack('>II H BB', 123, 1, len(test_data), 1, 0)
    print(f"Header: {len(header)} bytes - {header}")
    
    complete_packet, packet_info = create_packet_with_encryption(header, test_data, processor, 123)
    print(f"Complete packet: {len(complete_packet)} bytes")
    
    extracted_header, extracted_payload, extract_info = extract_packet_with_decryption(
        complete_packet, processor, 123
    )
    
    print(f"Extracted header: {len(extracted_header)} bytes")
    print(f"Extracted payload: {len(extracted_payload)} bytes")
    print(f"CRC valid: {extract_info['crc_valid']}")
    print(f"Payload match: {extracted_payload == test_data}")
    
    if extracted_payload == test_data:
        print("✅ All encryption tests passed!")
        return True
    else:
        print("❌ Payload mismatch!")
        print(f"Original ({len(test_data)}): {test_data[:100]}...")
        print(f"Extracted ({len(extracted_payload)}): {extracted_payload[:100]}...")
        return False

if __name__ == "__main__":
    success = test_encryption_isolated()
    sys.exit(0 if success else 1)
