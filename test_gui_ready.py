#!/usr/bin/env python3
"""
Quick GUI Test - Enhanced Frequency Hopping

Tests that the enhanced GUI can start without errors.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gui_components():
    """Test that GUI components can be imported and initialized"""
    
    print("🧪 Testing Enhanced GUI Components...")
    
    try:
        # Test PyQt5 imports
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QThread, pyqtSignal
        print("   ✅ PyQt5 imports successful")
        
        # Test project imports  
        from gui.main_window import MainWindow, SimulationWorker
        print("   ✅ GUI module imports successful")
        
        # Test core simulation imports
        from core.channel import ChannelModel
        from core.frequency_hopping import AdaptiveFrequencyManager
        from ai.cnn_model import FrequencyHoppingCNN
        print("   ✅ Core simulation imports successful")
        
        # Test that we can create simulation worker
        worker = SimulationWorker()
        print("   ✅ SimulationWorker creation successful")
        
        # Test enhanced methods exist
        assert hasattr(worker, 'ai_frequency_decision'), "ai_frequency_decision method missing"
        assert hasattr(worker, 'rule_based_frequency_decision'), "rule_based_frequency_decision method missing"
        print("   ✅ Enhanced frequency decision methods available")
        
        # Test enhanced signals exist
        assert hasattr(worker, 'ai_confidence_updated'), "ai_confidence_updated signal missing"
        assert hasattr(worker, 'security_status_updated'), "security_status_updated signal missing"
        assert hasattr(worker, 'jammer_detection_updated'), "jammer_detection_updated signal missing"
        print("   ✅ Enhanced transparency signals available")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_frequency_decision_methods():
    """Test the enhanced frequency decision methods"""
    
    print("\n🎯 Testing Enhanced Frequency Decision Methods...")
    
    try:
        from gui.main_window import SimulationWorker
        import numpy as np
        
        worker = SimulationWorker()
        
        # Test channel state
        test_channel_state = {
            'snr': 15.0,
            'interference': 3.0,
            'rss': -70.0
        }
        
        # Test AI frequency decision (if available)
        if worker.ai_enabled:
            ai_decision = worker.ai_frequency_decision(test_channel_state)
            assert 1 <= ai_decision <= 5, f"AI decision {ai_decision} not in valid range 1-5"
            print(f"   ✅ AI frequency decision: Band {ai_decision}")
        else:
            print("   ℹ️  AI model not available, using rule-based")
        
        # Test rule-based frequency decision
        rule_decision = worker.rule_based_frequency_decision(test_channel_state)
        assert 1 <= rule_decision <= 5, f"Rule decision {rule_decision} not in valid range 1-5"
        print(f"   ✅ Rule-based frequency decision: Band {rule_decision}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Frequency decision test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Enhanced GUI Quick Test")
    print("=" * 40)
    
    # Run component tests
    components_ok = test_gui_components()
    frequency_ok = test_frequency_decision_methods()
    
    print("\n" + "=" * 40)
    print("🏆 Test Results:")
    print(f"   GUI Components:      {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"   Frequency Decisions: {'✅ PASS' if frequency_ok else '❌ FAIL'}")
    
    if components_ok and frequency_ok:
        print("\n🎉 All tests passed!")
        print("🎮 The enhanced GUI is ready to run!")
        print("\n📋 To start the GUI:")
        print("   python gui/main_window.py")
        print("\n📈 Expected enhancements:")
        print("   • Dynamic frequency switching visualization")
        print("   • AI decision confidence transparency") 
        print("   • Real-time security status updates")
        print("   • Enhanced jammer detection display")
    else:
        print("\n⚠️  Some tests failed.")
        print("Please check the implementation before running the GUI.")
    
    return components_ok and frequency_ok

if __name__ == "__main__":
    main()
