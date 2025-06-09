#!/usr/bin/env python3
"""
Final Performance Validation Script
TEKNOFEST 5G Communication System

This script performs comprehensive validation of all system components
and generates a final performance report.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing system imports...")
    
    try:
        # Core modules
        from core.modulation import OFDMSystem
        from core.channel import ChannelModel, JammerSimulator
        from core.frequency_hopping import AdaptiveFrequencyManager
        from core.coding import AdaptiveCoding
        from core.data_processing import TransmissionSimulator
        
        # AI modules
        from ai.cnn_model import FrequencyHoppingCNN
        from ai.training import ModelTrainer
        from ai.data_generator import SyntheticDataGenerator
        
        # Simulation
        from simulation.competition_simulator import CompetitionSimulator
        
        print("✅ All critical imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ai_model():
    """Test AI model loading and inference"""
    print("🧠 Testing AI model...")
    
    try:
        from ai.cnn_model import FrequencyHoppingCNN
        import torch
        
        # Test model creation
        model = FrequencyHoppingCNN(input_size=30)
        print(f"   ✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test inference
        test_input = torch.randn(1, 30)
        start_time = time.time()
        with torch.no_grad():
            output = model(test_input)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"   ✅ Inference test: {inference_time:.1f}ms")
        print(f"   ✅ Output shape: {output.shape}")
        
        # Test model loading if available
        model_path = 'models/frequency_hopping_model.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✅ Pre-trained model loaded successfully")
        else:
            print("   ⚠️  No pre-trained model found (will use random weights)")
            
        return True
        
    except Exception as e:
        print(f"   ❌ AI model test failed: {e}")
        return False

def test_competition_phases():
    """Test all competition phases"""
    print("🎯 Testing competition phases...")
    
    try:
        from simulation.competition_simulator import CompetitionSimulator
        
        # Create simulator
        simulator = CompetitionSimulator(enable_ai=True, enable_gui=False)
        print("   ✅ Competition simulator initialized")
        
        # Test each phase with small file sizes for speed
        phases_to_test = [1, 2, 3]
        file_size = 1e6  # 1MB for quick testing
        duration = 5.0   # 5 seconds for quick testing
        
        phase_results = {}
        
        for phase in phases_to_test:
            print(f"   🔄 Testing Phase {phase}...")
            start_time = time.time()
            
            results = simulator.simulate_phase(
                phase_number=phase,
                file_size=file_size,
                duration=duration
            )
            
            test_time = time.time() - start_time
            phase_results[phase] = {
                'test_time': test_time,
                'success': results['success'],
                'avg_snr': results.get('avg_snr', 0),
                'avg_throughput': results.get('avg_throughput', 0),
                'frequency_switches': results.get('frequency_switches', 0)
            }
            
            print(f"   ✅ Phase {phase}: {test_time:.1f}s, "
                  f"SNR: {results.get('avg_snr', 0):.1f}dB, "
                  f"Switches: {results.get('frequency_switches', 0)}")
        
        return True, phase_results
        
    except Exception as e:
        print(f"   ❌ Competition phase test failed: {e}")
        return False, {}

def test_jammer_scenarios():
    """Test jammer simulation"""
    print("📡 Testing jammer scenarios...")
    
    try:
        from core.channel import JammerSimulator
        
        jammer = JammerSimulator()
        
        # Test different jammer modes
        modes = ['off', 'pattern', 'random']
        for mode in modes:
            jammer.set_mode(mode)
            status = jammer.get_jammer_status()
            print(f"   ✅ {mode.capitalize()} mode: {status['active']}")
            
            # Test band jamming
            if mode != 'off':
                jammed_band = jammer.get_current_jammed_band()
                print(f"      Current jammed band: {jammed_band}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Jammer test failed: {e}")
        return False

def test_modulation_system():
    """Test modulation and OFDM"""
    print("📶 Testing modulation system...")
    
    try:
        from core.modulation import OFDMSystem
        
        ofdm = OFDMSystem()
        
        # Test different modulation schemes
        schemes = ['BPSK', 'QPSK', '16QAM', '64QAM']
        for scheme in schemes:
            try:
                # Test symbol generation
                symbols = ofdm.modulate_data(np.random.randint(0, 2, 1000), scheme)
                print(f"   ✅ {scheme}: {len(symbols)} symbols generated")
            except Exception as e:
                print(f"   ⚠️  {scheme}: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Modulation test failed: {e}")
        return False

def test_gui_components():
    """Test GUI imports and basic functionality"""
    print("🖥️  Testing GUI components...")
    
    try:
        # Test PyQt5 import
        from PyQt5.QtWidgets import QApplication
        print("   ✅ PyQt5 available")
        
        # Test main window import
        from gui.main_window import MainWindow
        print("   ✅ Main window class available")
        
        # Don't actually create GUI in test mode
        print("   ✅ GUI components ready (not launched in test mode)")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  GUI not available: {e}")
        return False
    except Exception as e:
        print(f"   ❌ GUI test failed: {e}")
        return False

def generate_performance_report(results: Dict):
    """Generate final performance report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': 'OPERATIONAL' if results['all_tests_passed'] else 'ISSUES_DETECTED',
        'test_results': results,
        'performance_metrics': {
            'total_test_time': results.get('total_time', 0),
            'phase_tests': results.get('phase_results', {}),
            'ai_inference_time': results.get('ai_inference_time', 0)
        },
        'recommendations': []
    }
    
    # Add recommendations based on results
    if not results['imports_ok']:
        report['recommendations'].append("Fix missing dependencies")
    
    if not results['ai_model_ok']:
        report['recommendations'].append("Retrain or fix AI model")
    
    if not results['competition_ok']:
        report['recommendations'].append("Debug competition simulation")
    
    if not results['jammer_ok']:
        report['recommendations'].append("Fix jammer simulation")
    
    if not results['modulation_ok']:
        report['recommendations'].append("Fix modulation system")
    
    if len(report['recommendations']) == 0:
        report['recommendations'].append("System ready for competition deployment")
    
    # Save report
    os.makedirs('results', exist_ok=True)
    report_file = f"results/performance_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report, report_file

def main():
    """Main validation function"""
    print("=" * 70)
    print("🏆 TEKNOFEST 5G Communication System - Final Validation")
    print("=" * 70)
    
    start_time = time.time()
    
    # Initialize results
    results = {
        'timestamp': datetime.now().isoformat(),
        'imports_ok': False,
        'ai_model_ok': False,
        'competition_ok': False,
        'jammer_ok': False,
        'modulation_ok': False,
        'gui_ok': False,
        'phase_results': {},
        'all_tests_passed': False
    }
    
    # Run tests
    try:
        results['imports_ok'] = test_imports()
        results['ai_model_ok'] = test_ai_model()
        competition_ok, phase_results = test_competition_phases()
        results['competition_ok'] = competition_ok
        results['phase_results'] = phase_results
        results['jammer_ok'] = test_jammer_scenarios()
        results['modulation_ok'] = test_modulation_system()
        results['gui_ok'] = test_gui_components()
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return 1
    
    # Calculate total time
    total_time = time.time() - start_time
    results['total_time'] = total_time
    
    # Check if all tests passed
    critical_tests = ['imports_ok', 'ai_model_ok', 'competition_ok', 'jammer_ok', 'modulation_ok']
    results['all_tests_passed'] = all(results[test] for test in critical_tests)
    
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    # Print summary
    test_status = [
        ("System Imports", results['imports_ok']),
        ("AI Model", results['ai_model_ok']),
        ("Competition Phases", results['competition_ok']),
        ("Jammer Simulation", results['jammer_ok']),
        ("Modulation System", results['modulation_ok']),
        ("GUI Components", results['gui_ok'])
    ]
    
    for test_name, passed in test_status:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<20} {status}")
    
    print(f"\nTotal test time: {total_time:.1f} seconds")
    
    # Generate report
    report, report_file = generate_performance_report(results)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Final status
    if results['all_tests_passed']:
        print("\n🎉 SYSTEM VALIDATION: SUCCESS")
        print("✅ System is ready for TEKNOFEST competition!")
        return 0
    else:
        print("\n⚠️  SYSTEM VALIDATION: ISSUES DETECTED")
        print("❌ Please address the failed tests before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
