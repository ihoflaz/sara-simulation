#!/usr/bin/env python3
"""Test GUI imports exactly as the main_window.py does"""

import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

print("Testing GUI imports...")

# Try to import AI-powered frequency hopping
try:
    from core.frequency_hopping import AdaptiveFrequencyManager
    from ai.cnn_model import FrequencyHoppingCNN
    print("✅ Using full AI frequency hopping with TensorFlow/Keras")
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TensorFlow/AI modules not available ({e}), using simplified frequency hopping")
    from simplified_frequency_hopping import AdaptiveFrequencyManager
    AI_AVAILABLE = False

print(f"AI_AVAILABLE: {AI_AVAILABLE}")

# Test the actual frequency manager
print("\nTesting frequency manager...")
freq_manager = AdaptiveFrequencyManager()
freq_manager.initialize_system()
print("✅ Frequency manager initialized successfully")
