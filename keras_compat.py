# keras_compat.py - Compatibility layer for keras imports

# Suppress TensorFlow warnings first
import os
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    # Import TensorFlow 2.16.1 with standalone Keras 3.10.0
    import tensorflow as tf
    tf_version = getattr(tf, '__version__', 'unknown')
    print(f"✅ TensorFlow version: {tf_version}")
    
    # For TensorFlow 2.16+, use standalone keras
    import keras
    from keras import layers, models, optimizers, callbacks
    print(f"✅ Using standalone Keras version: {keras.__version__}")
        
except ImportError as e:
    print(f"❌ TensorFlow/Keras import failed: {e}")
    # Create dummy modules for compatibility
    class DummyModule:
        def __getattr__(self, name):
            raise ImportError("TensorFlow/Keras not available")
    
    keras = DummyModule()
    layers = DummyModule()
    models = DummyModule()
    optimizers = DummyModule()
    callbacks = DummyModule()

__all__ = ['keras', 'layers', 'models', 'optimizers', 'callbacks']
