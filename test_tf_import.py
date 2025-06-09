#!/usr/bin/env python3
"""Test TensorFlow import in frequency hopping module"""

import traceback

try:
    from core.frequency_hopping import AdaptiveFrequencyManager
    print("✅ Success: Core frequency hopping imported!")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    traceback.print_exc()
    
    # Try individual imports to debug
    print("\n--- Testing individual imports ---")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
    except Exception as tf_e:
        print(f"❌ TensorFlow error: {tf_e}")
    
    try:
        from keras_compat import keras
        print(f"✅ Keras compatibility layer working")
    except Exception as keras_e:
        print(f"❌ Keras compat error: {keras_e}")
        
    try:
        from config import CNN_CONFIG, FREQUENCY_BANDS, COMPETITION_CONFIG
        print("✅ Config imports working")
    except Exception as config_e:
        print(f"❌ Config error: {config_e}")
        
    try:
        import numpy as np
        print("✅ NumPy working")
    except Exception as np_e:
        print(f"❌ NumPy error: {np_e}")
