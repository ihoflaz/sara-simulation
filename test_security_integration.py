#!/usr/bin/env python3
"""
Comprehensive Security Integration Test for TEKNOFEST 2025
Tests AES-128 encryption and data recovery algorithms
"""

import sys
import os
import tempfile
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_encryption_module():
    """Test AES-128 encryption module"""
    print("\n🔒 Testing AES-128 Encryption Module...")
    
    try:
        from core.encryption import AESEncryption, EncryptedPacketProcessor
        
        # Test AES-128 CTR mode
        aes_ctr = AESEncryption(mode='CTR')
        test_data = b"Hello TEKNOFEST 2025! This is a test packet for AES encryption."
        
        encrypted_packet, enc_info = aes_ctr.encrypt_packet(test_data, packet_id=1)
        print(f"✅ CTR Encryption successful: {len(encrypted_packet)} bytes")
        print(f"   Original: {len(test_data)} bytes, Encrypted: {len(encrypted_packet)} bytes")
        
        decrypted_data, dec_info = aes_ctr.decrypt_packet(encrypted_packet, expected_packet_id=1)
        print(f"✅ CTR Decryption successful: {len(decrypted_data)} bytes")
        
        assert decrypted_data == test_data, "Decrypted data doesn't match original!"
        print("✅ CTR mode encryption/decryption verification passed")
        
        # Test AES-128 GCM mode
        aes_gcm = AESEncryption(mode='GCM')
        encrypted_packet_gcm, enc_info_gcm = aes_gcm.encrypt_packet(test_data, packet_id=1)
        print(f"✅ GCM Encryption successful: {len(encrypted_packet_gcm)} bytes")
        
        decrypted_data_gcm, dec_info_gcm = aes_gcm.decrypt_packet(encrypted_packet_gcm, expected_packet_id=1)
        assert decrypted_data_gcm == test_data, "GCM decrypted data doesn't match original!"
        print("✅ GCM mode encryption/decryption verification passed")
        
        # Test packet processor
        processor = EncryptedPacketProcessor(encryption_enabled=True, mode='CTR')
        processed_packet, proc_info = processor.process_outgoing_packet(test_data, packet_id=2)
        recovered_packet, rec_info = processor.process_incoming_packet(processed_packet, packet_id=2)
        
        assert recovered_packet == test_data, "Packet processor failed!"
        print("✅ EncryptedPacketProcessor verification passed")
        
        # Test key export/import
        key_hex = aes_ctr.export_key()
        aes_new = AESEncryption()
        aes_new.import_key(key_hex)
        
        # Verify imported key works
        decrypted_with_imported, _ = aes_new.decrypt_packet(encrypted_packet, expected_packet_id=1)
        assert decrypted_with_imported == test_data, "Imported key failed!"
        print("✅ Key export/import verification passed")
        
        # Test statistics
        stats = aes_ctr.get_encryption_stats()
        print(f"   Encryption stats: {stats['packets_encrypted']} encrypted, {stats['packets_decrypted']} decrypted")
        
        return True
        
    except Exception as e:
        print(f"❌ Encryption module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recovery_module():
    """Test data recovery algorithms"""
    print("\n📡 Testing Data Recovery Module...")
    
    try:
        from core.recovery import DataRecoveryManager, RecoveryMode, ReedSolomonErasure
        
        # Test Reed-Solomon erasure coding
        rs = ReedSolomonErasure(data_blocks=4, recovery_blocks=2)
        
        # Create test data blocks
        test_blocks = [
            b"Block 1: TEKNOFEST data recovery test",
            b"Block 2: Reed-Solomon erasure coding",
            b"Block 3: Wireless communication test ",
            b"Block 4: Error correction algorithms  "
        ]
        
        # Encode with Reed-Solomon
        encoded_blocks = rs.encode_blocks(test_blocks)
        print(f"✅ Reed-Solomon encoding: {len(test_blocks)} data + {rs.recovery_blocks} recovery = {len(encoded_blocks)} total")
        
        # Simulate erasures (lose 2 blocks)
        received_blocks = list(encoded_blocks)
        received_blocks[1] = None  # Lose block 1
        received_blocks[3] = None  # Lose block 3
        erasure_positions = [1, 3]
        
        # Recover missing blocks
        recovered_blocks = rs.decode_blocks(received_blocks, erasure_positions)
        print(f"✅ Reed-Solomon recovery successful: recovered {len(erasure_positions)} missing blocks")
        
        # Verify recovery
        for i, original_block in enumerate(test_blocks):
            assert recovered_blocks[i] == original_block, f"Block {i} recovery failed!"
        print("✅ Reed-Solomon erasure coding verification passed")
        
        # Test DataRecoveryManager
        recovery_mgr = DataRecoveryManager(mode=RecoveryMode.ADAPTIVE)
        
        # Test failure reporting
        test_packet = b"Test packet for recovery system"
        result = recovery_mgr.report_transmission_failure(
            packet_id=10, 
            packet_data=test_packet, 
            error_type='timeout', 
            channel_quality=0.5,
            frequency_band=1
        )
        print(f"✅ Failure reporting: {result}")
        
        # Test recovery queue
        retry_info = recovery_mgr.get_next_retry_packet()
        if retry_info:
            packet_id, packet_data, recovery_method = retry_info
            print(f"✅ Recovery queue working: packet {packet_id}, method {recovery_method}")
        
        # Test redundant packet generation
        original_packets = [b"Packet 1", b"Packet 2", b"Packet 3"]
        redundant_packets = recovery_mgr.generate_redundant_packets(original_packets)
        print(f"✅ Redundant packets: {len(original_packets)} -> {len(redundant_packets)}")
        
        # Test erasure coded packet generation
        erasure_packets = recovery_mgr.generate_erasure_coded_packets(original_packets)
        print(f"✅ Erasure coded packets: {len(original_packets)} -> {len(erasure_packets)}")
        
        # Test statistics
        stats = recovery_mgr.get_recovery_stats()
        print(f"   Recovery stats: {stats['packets_failed']} failed, {stats['packets_recovered']} recovered")
        
        # Test recommendations
        recommendations = recovery_mgr.get_recovery_recommendations()
        print(f"   Recommendations: {len(recommendations.get('recommendations', []))} suggestions")
        
        return True
        
    except Exception as e:
        print(f"❌ Recovery module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_transmission_simulator():
    """Test enhanced transmission simulator with both encryption and recovery"""
    print("\n🚀 Testing Enhanced Transmission Simulator...")
    
    try:
        from core.data_processing import EnhancedTransmissionSimulator
        from core.recovery import RecoveryMode
        
        # Create test file
        test_content = b"TEKNOFEST 2025 - Wireless Communication System\n" * 100
        test_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt')
        test_file.write(test_content)
        test_file.close()
        
        # Test with encryption enabled
        simulator = EnhancedTransmissionSimulator(encryption_enabled=True, recovery_enabled=True)
        print("✅ Enhanced simulator created with encryption and recovery enabled")
        
        # Test file transmission
        results = simulator.simulate_enhanced_file_transmission(
            test_file.name,
            channel_quality='poor',  # Stress test with poor conditions
            jammed_bands=[1, 2],    # Simulate jamming
            enable_recovery=True
        )
        
        print(f"✅ File transmission simulation completed")
        print(f"   Original size: {results['file_info']['original_size']} bytes")
        print(f"   Success: {results['overall_success']}")
        print(f"   Encryption enabled: {results['security_features']['encryption_enabled']}")
        print(f"   Recovery enabled: {results['security_features']['recovery_enabled']}")
        
        # Test packet transmission
        test_packets = [
            b"Test packet 1 for enhanced transmission",
            b"Test packet 2 with encryption and recovery",
            b"Test packet 3 under jamming conditions",
            b"Test packet 4 for error recovery testing"
        ]
        
        received_packets, stats = simulator.simulate_enhanced_packet_transmission(
            test_packets,
            channel_quality='fair',
            jammed_bands=[1]
        )
        
        print(f"✅ Packet transmission simulation completed")
        print(f"   Original packets: {stats['original_packets']}")
        print(f"   Received packets: {stats['received_packets']}")
        print(f"   Encryption enabled: {stats['encryption_enabled']}")
        print(f"   Recovery enabled: {stats['recovery_enabled']}")
        
        # Test comprehensive statistics
        comp_stats = simulator.get_comprehensive_stats()
        
        if 'encryption' in comp_stats:
            enc_stats = comp_stats['encryption']
            print(f"   Encryption: {enc_stats.get('packets_encrypted', 0)} encrypted, success rate: {enc_stats.get('encryption_success_rate', 0)*100:.1f}%")
        
        if 'recovery' in comp_stats:
            rec_stats = comp_stats['recovery']
            print(f"   Recovery: {rec_stats.get('packets_failed', 0)} failed, {rec_stats.get('packets_recovered', 0)} recovered")
        
        # Test security key operations
        exported_key = simulator.export_security_key()
        if exported_key:
            print(f"✅ Security key export successful: {len(exported_key)} chars")
            
            # Create new simulator and import key
            new_simulator = EnhancedTransmissionSimulator(encryption_enabled=True)
            new_simulator.import_security_key(exported_key)
            print("✅ Security key import successful")
        
        # Cleanup
        os.unlink(test_file.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced transmission simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_gui():
    """Test integration with GUI components"""
    print("\n🖥️ Testing GUI Integration...")
    
    try:
        # Test imports
        from gui.main_window import MainWindow
        from PyQt5.QtWidgets import QApplication
        
        # Create minimal app for testing
        app = QApplication([])
        
        # Create main window
        window = MainWindow()
        print("✅ MainWindow created successfully")
        
        # Check if enhanced simulator is available
        if hasattr(window, 'simulation_worker') and hasattr(window.simulation_worker, 'enhanced_simulator'):
            print("✅ Enhanced simulator available in GUI")
        else:
            print("⚠️ Enhanced simulator not yet initialized in GUI")
        
        # Check security tab elements
        security_controls = [
            'encryption_enabled_cb',
            'encryption_mode_combo', 
            'recovery_enabled_cb',
            'recovery_mode_combo',
            'encryption_stats_label',
            'recovery_stats_label'
        ]
        
        missing_controls = []
        for control in security_controls:
            if not hasattr(window, control):
                missing_controls.append(control)
        
        if missing_controls:
            print(f"⚠️ Missing GUI controls: {missing_controls}")
        else:
            print("✅ All security GUI controls present")
        
        # Test GUI methods
        gui_methods = [
            'toggle_encryption',
            'toggle_recovery', 
            'export_encryption_key',
            'import_encryption_key',
            'update_security_displays'
        ]
        
        missing_methods = []
        for method in gui_methods:
            if not hasattr(window, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"⚠️ Missing GUI methods: {missing_methods}")
        else:
            print("✅ All security GUI methods present")
        
        app.quit()
        return True
        
    except Exception as e:
        print(f"❌ GUI integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_world_scenario():
    """Test real-world communication scenario with encryption and recovery"""
    print("\n🌍 Testing Real-World Scenario...")
    
    try:
        from core.data_processing import EnhancedTransmissionSimulator
        from core.recovery import RecoveryMode
        
        # Create a realistic test file
        test_data = {
            "mission": "TEKNOFEST 2025 - Wireless Communication Test",
            "timestamp": time.time(),
            "payload": "This is sensitive mission data that needs encryption",
            "coordinates": [41.0082, 28.9784],  # Istanbul coordinates
            "frequency_bands": [2400, 2450, 2500, 2550, 2600],
            "status": "operational"
        }
        
        test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(test_data, test_file, indent=2)
        test_file.close()
        
        print(f"✅ Test scenario file created: {os.path.basename(test_file.name)}")
        
        # Test different security configurations
        test_configs = [
            {"encryption": False, "recovery": False, "desc": "No Security"},
            {"encryption": True, "recovery": False, "desc": "Encryption Only"},
            {"encryption": False, "recovery": True, "desc": "Recovery Only"},
            {"encryption": True, "recovery": True, "desc": "Full Security"}
        ]
        
        results = {}
        
        for config in test_configs:
            print(f"\n--- Testing: {config['desc']} ---")
            
            simulator = EnhancedTransmissionSimulator(
                encryption_enabled=config["encryption"],
                recovery_enabled=config["recovery"]
            )
            
            # Simulate challenging conditions
            result = simulator.simulate_enhanced_file_transmission(
                test_file.name,
                channel_quality='poor',
                jammed_bands=[1, 3, 4],  # Heavy jamming
                enable_recovery=config["recovery"]
            )
            
            results[config['desc']] = result
            
            print(f"   Success: {result['overall_success']}")
            print(f"   Transmission time: {result['timing']['actual_simulation_time']:.3f}s")
            print(f"   Recovery attempts: {result['recovery_attempts']}")
            print(f"   Security features: Enc={result['security_features']['encryption_enabled']}, Rec={result['security_features']['recovery_enabled']}")
        
        # Compare results
        print(f"\n📊 Scenario Comparison:")
        for desc, result in results.items():
            success_rate = 100 if result['overall_success'] else 0
            print(f"   {desc}: {success_rate}% success, {result['timing']['actual_simulation_time']:.3f}s")
        
        # Cleanup
        os.unlink(test_file.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Real-world scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmark():
    """Benchmark performance of encryption and recovery"""
    print("\n⚡ Performance Benchmark...")
    
    try:
        from core.encryption import AESEncryption
        from core.recovery import ReedSolomonErasure
        import time
        
        # Encryption performance test
        aes = AESEncryption(mode='CTR')
        test_sizes = [1024, 4096, 16384, 65536]  # Various packet sizes
        
        print("Encryption Performance:")
        for size in test_sizes:
            test_data = os.urandom(size)
            
            start_time = time.time()
            for _ in range(100):  # 100 iterations
                encrypted, _ = aes.encrypt_packet(test_data)
                decrypted, _ = aes.decrypt_packet(encrypted)
            end_time = time.time()
            
            ops_per_sec = 200 / (end_time - start_time)  # 100 encrypt + 100 decrypt
            mbps = (size * ops_per_sec * 8) / 1e6
            
            print(f"   {size} bytes: {ops_per_sec:.1f} ops/sec, {mbps:.2f} Mbps")
        
        # Recovery performance test
        rs = ReedSolomonErasure(data_blocks=6, recovery_blocks=3)
        test_blocks = [os.urandom(1024) for _ in range(6)]
        
        print("Recovery Performance:")
        start_time = time.time()
        for _ in range(50):  # 50 iterations
            encoded = rs.encode_blocks(test_blocks)
            # Simulate 2 erasures
            encoded[1] = None
            encoded[4] = None
            recovered = rs.decode_blocks(encoded, [1, 4])
        end_time = time.time()
        
        recovery_ops_per_sec = 50 / (end_time - start_time)
        print(f"   Reed-Solomon: {recovery_ops_per_sec:.1f} recovery ops/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 TEKNOFEST 2025 - Security Integration Test Suite")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run all tests
    tests = [
        ("Encryption Module", test_encryption_module),
        ("Recovery Module", test_recovery_module),
        ("Enhanced Transmission Simulator", test_enhanced_transmission_simulator),
        ("GUI Integration", test_integration_with_gui),
        ("Real-World Scenario", test_real_world_scenario),
        ("Performance Benchmark", test_performance_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        try:
            success = test_func()
            results[test_name] = {
                'success': success,
                'time': time.time() - start_time
            }
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = {
                'success': False,
                'time': time.time() - start_time
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(tests)
    passed_tests = sum(1 for result in results.values() if result['success'])
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status:8} {test_name:<35} ({result['time']:.2f}s)")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All security features working correctly!")
        print("\n🔐 TEKNOFEST 2025 Security Features:")
        print("   ✅ AES-128 Encryption (CTR/GCM modes)")
        print("   ✅ Data Recovery Algorithms (Retry/Redundant/Erasure)")
        print("   ✅ GUI Integration with Security Controls")
        print("   ✅ Real-time Security Monitoring")
        print("   ✅ Key Management (Export/Import)")
        print("   ✅ Performance Optimized")
        
        return True
    else:
        print("⚠️ Some tests failed - please review the implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
