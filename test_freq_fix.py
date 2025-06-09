#!/usr/bin/env python3
"""
Test GUI Frequency Manager Fix
Tests the num_bands attribute that was causing GUI errors
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_frequency_manager_fix():
    """Test the num_bands attribute that was failing in GUI"""
    print("🔧 Testing GUI Frequency Manager Fix")
    print("=" * 50)
    
    try:
        # Import the frequency manager
        from core.frequency_hopping import AdaptiveFrequencyManager
        print("✅ AdaptiveFrequencyManager import successful")
        
        # Create frequency manager instance
        freq_manager = AdaptiveFrequencyManager()
        print("✅ AdaptiveFrequencyManager instance created")
        
        # Test the exact attribute that was missing in GUI
        print(f"✅ num_bands attribute: {freq_manager.num_bands}")
        
        # Test the GUI usage patterns
        current_band = 3
        
        # This is the exact calculation from GUI line 358 that was failing
        next_band = (current_band % freq_manager.num_bands) + 1
        print(f"✅ Band calculation works: {current_band} -> {next_band}")
        
        # Test the feature tiling from GUI line 337 that was failing
        features = np.array([1.0, 2.0, 3.0])
        full_features = np.tile(features, freq_manager.num_bands)
        print(f"✅ Feature tiling works: {len(features)} -> {len(full_features)} features")
        
        print("\n🎉 All frequency manager tests passed!")
        print("🚀 GUI should now work without 'num_bands' error")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_ai_frequency_decision():
    """Test the AI frequency decision that uses num_bands"""
    print("\n🧠 Testing GUI AI Frequency Decision Components")
    print("=" * 50)
    
    try:
        # Test imports that GUI AI worker uses
        from core.frequency_hopping import AdaptiveFrequencyManager
        from core.modulation import OFDMSystem
        import torch
        
        print("✅ All AI decision components imported successfully")
        
        # Create instances as GUI would
        freq_manager = AdaptiveFrequencyManager()
        ofdm_system = OFDMSystem()
        
        print("✅ All AI decision components created successfully")
        
        # Test the exact AI frequency decision logic from GUI
        print(f"✅ Number of bands available: {freq_manager.num_bands}")
        
        # Simulate the GUI's feature preparation
        features = np.array([
            20.0,  # SNR
            5.0,   # Interference
            -70.0, # RSS
            20.0,  # Coherence time
            5.0,   # Doppler spread
            0.1    # Delay spread
        ])
        
        # This is the exact tiling operation from GUI that was failing
        full_features = np.tile(features, freq_manager.num_bands)
        print(f"✅ AI features prepared: {len(full_features)} total features")
        
        # Test tensor operations as GUI would do
        features_tensor = torch.FloatTensor(full_features).unsqueeze(0)
        print(f"✅ Tensor shape: {features_tensor.shape}")
        
        print("\n🎉 GUI AI frequency decision components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ GUI AI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏆 TEKNOFEST GUI Frequency Manager Fix Validation")
    print("=" * 60)
    
    success1 = test_frequency_manager_fix()
    success2 = test_gui_ai_frequency_decision()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ GUI frequency manager fix successful")
        print("✅ num_bands attribute available and working")
        print("✅ Phase 3 Random Jammer AI decisions should work")
        print("🚀 Ready to run: python main.py --mode gui")
    else:
        print("❌ Some tests failed - check errors above")
    
    print("=" * 60)
