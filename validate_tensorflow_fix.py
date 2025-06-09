#!/usr/bin/env python3
"""
FINAL TENSORFLOW/KERAS RESOLUTION VALIDATION
============================================

This script validates that the TensorFlow/Keras issues have been resolved
and provides comprehensive reporting.
"""

import sys
import os
import traceback
from datetime import datetime

print("="*80)
print("TEKNOFEST 5G Communication System - TensorFlow Resolution Validation")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python Version: {sys.version}")
print(f"Working Directory: {os.getcwd()}")
print("="*80)

def test_individual_imports():
    """Test each import individually"""
    print("\n1. INDIVIDUAL IMPORT TESTS")
    print("-" * 40)
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - SUCCESS")
        tf_available = True
    except Exception as e:
        print(f"❌ TensorFlow - FAILED: {e}")
        tf_available = False
    
    # Test Keras
    try:
        import keras
        print(f"✅ Keras {keras.__version__} - SUCCESS")
        keras_available = True
    except Exception as e:
        print(f"❌ Keras - FAILED: {e}")
        keras_available = False
    
    # Test keras_compat
    try:
        from keras_compat import keras as compat_keras, layers, models, optimizers, callbacks
        print("✅ Keras Compatibility Layer - SUCCESS")
        compat_available = True
    except Exception as e:
        print(f"❌ Keras Compatibility Layer - FAILED: {e}")
        compat_available = False
    
    return tf_available, keras_available, compat_available

def test_core_imports():
    """Test core module imports"""
    print("\n2. CORE MODULE IMPORT TESTS")
    print("-" * 40)
    
    # Test frequency hopping core
    try:
        from core.frequency_hopping import FrequencyHoppingCNN, AdaptiveFrequencyManager
        print("✅ Core Frequency Hopping - SUCCESS")
        core_available = True
    except Exception as e:
        print(f"❌ Core Frequency Hopping - FAILED: {e}")
        traceback.print_exc()
        core_available = False
    
    # Test AI model
    try:
        from ai.cnn_model import FrequencyHoppingCNN as AICNNModel
        print("✅ AI CNN Model - SUCCESS")
        ai_available = True
    except Exception as e:
        print(f"❌ AI CNN Model - FAILED: {e}")
        traceback.print_exc()
        ai_available = False
    
    return core_available, ai_available

def test_cnn_model_creation():
    """Test actual CNN model creation"""
    print("\n3. CNN MODEL CREATION TESTS")
    print("-" * 40)
    
    try:
        from core.frequency_hopping import FrequencyHoppingCNN
        model = FrequencyHoppingCNN()
        print("✅ Core CNN Model Creation - SUCCESS")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output size: {model.output_size}")
        core_model_ok = True
    except Exception as e:
        print(f"❌ Core CNN Model Creation - FAILED: {e}")
        core_model_ok = False
    
    try:
        from ai.cnn_model import FrequencyHoppingCNN as PyTorchCNN
        model = PyTorchCNN()
        print("✅ PyTorch CNN Model Creation - SUCCESS")
        pytorch_model_ok = True
    except Exception as e:
        print(f"❌ PyTorch CNN Model Creation - FAILED: {e}")
        pytorch_model_ok = False
    
    return core_model_ok, pytorch_model_ok

def test_gui_imports():
    """Test GUI-specific imports"""
    print("\n4. GUI IMPORT TESTS")
    print("-" * 40)
    
    # Simulate GUI environment variables
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    
    try:
        # Simulate the exact GUI import sequence
        from core.frequency_hopping import AdaptiveFrequencyManager
        from ai.cnn_model import FrequencyHoppingCNN
        print("✅ GUI Import Sequence - SUCCESS")
        gui_imports_ok = True
    except ImportError as e:
        print(f"❌ GUI Import Sequence - FAILED: {e}")
        gui_imports_ok = False
    
    return gui_imports_ok

def test_frequency_manager():
    """Test frequency manager functionality"""
    print("\n5. FREQUENCY MANAGER FUNCTIONALITY TESTS")
    print("-" * 40)
    
    try:
        from core.frequency_hopping import AdaptiveFrequencyManager
        manager = AdaptiveFrequencyManager()
        manager.initialize_system()
        print("✅ AI Frequency Manager - SUCCESS")
        ai_manager_ok = True
    except Exception as e:
        print(f"❌ AI Frequency Manager - FAILED: {e}")
        try:
            from simplified_frequency_hopping import AdaptiveFrequencyManager
            manager = AdaptiveFrequencyManager()
            manager.initialize_system()
            print("✅ Simplified Frequency Manager (fallback) - SUCCESS")
            ai_manager_ok = False
        except Exception as e2:
            print(f"❌ Simplified Frequency Manager - FAILED: {e2}")
            ai_manager_ok = False
    
    return ai_manager_ok

def main():
    """Run all tests and provide final status report"""
    tf_ok, keras_ok, compat_ok = test_individual_imports()
    core_ok, ai_ok = test_core_imports()
    core_model_ok, pytorch_model_ok = test_cnn_model_creation()
    gui_ok = test_gui_imports()
    manager_ok = test_frequency_manager()
    
    print("\n" + "="*80)
    print("FINAL STATUS REPORT")
    print("="*80)
    
    all_tests = [
        ("TensorFlow Available", tf_ok),
        ("Keras Available", keras_ok), 
        ("Keras Compatibility Layer", compat_ok),
        ("Core Frequency Hopping", core_ok),
        ("AI CNN Model", ai_ok),
        ("Core CNN Model Creation", core_model_ok),
        ("PyTorch CNN Model Creation", pytorch_model_ok),
        ("GUI Import Sequence", gui_ok),
        ("AI Frequency Manager", manager_ok)
    ]
    
    passed = sum(1 for _, ok in all_tests if ok)
    total = len(all_tests)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    print("-" * 40)
    
    for test_name, result in all_tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    if passed == total:
        print(f"\n🎉 SUCCESS: All TensorFlow/Keras issues have been resolved!")
        print("   The TEKNOFEST 5G Communication System is ready with full AI functionality.")
    elif passed >= 6:
        print(f"\n⚠️  PARTIAL SUCCESS: Core functionality working with {passed}/{total} tests passing.")
        print("   The system will use simplified frequency hopping where needed.")
    else:
        print(f"\n❌ ISSUES REMAIN: Only {passed}/{total} tests passing.")
        print("   Further investigation needed.")
    
    print("\n" + "="*80)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
