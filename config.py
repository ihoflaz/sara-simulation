# Configuration Settings for 5G Communication System

# Competition Specifications
COMPETITION_CONFIG = {
    'center_frequency': 2.41e9,  # 2.41 GHz
    'total_bandwidth': 20e6,     # 20 MHz
    'num_subbands': 5,           # 5 sub-bands
    'subband_bandwidth': 4e6,    # 4 MHz each
    'max_power_dbm': 10,         # 10 dBm
    'jammer_switch_time': 1.0,   # 1 second
}

# Frequency Bands (according to competition specs)
FREQUENCY_BANDS = {
    1: {'start': 2.400e9, 'end': 2.404e9, 'center': 2.402e9},
    2: {'start': 2.404e9, 'end': 2.408e9, 'center': 2.406e9},
    3: {'start': 2.408e9, 'end': 2.412e9, 'center': 2.410e9},
    4: {'start': 2.412e9, 'end': 2.416e9, 'center': 2.414e9},
    5: {'start': 2.416e9, 'end': 2.420e9, 'center': 2.418e9},
}

# Jammer Patterns
JAMMER_PATTERNS = {
    'pattern': [1, 3, 5, 4, 2],  # Pattern-based sequence
    'random': True,              # Random hopping for phase 3
}

# OFDM Parameters
OFDM_CONFIG = {
    'num_subcarriers': 64,
    'cyclic_prefix_length': 16,
    'pilot_spacing': 4,
    'guard_carriers': 8,
}

# Modulation Schemes
MODULATION_SCHEMES = {
    'BPSK': {'order': 2, 'bits_per_symbol': 1},
    'QPSK': {'order': 4, 'bits_per_symbol': 2},
    '16QAM': {'order': 16, 'bits_per_symbol': 4},
    '64QAM': {'order': 64, 'bits_per_symbol': 6},
    '256QAM': {'order': 256, 'bits_per_symbol': 8},
    '1024QAM': {'order': 1024, 'bits_per_symbol': 10},
}

# Channel Coding
CODING_CONFIG = {
    'ldpc': {
        'code_rate': 0.5,
        'block_length': 1944,
    },
    'turbo': {
        'code_rate': 1/3,
        'constraint_length': 7,
    }
}

# Data Sizes (according to competition)
DATA_SIZES = {
    'phase1': [1e9, 10e9],      # 1GB, 10GB
    'phase2_3': [100e6, 1e9],   # 100MB, 1GB
}

# CNN Model Configuration
CNN_CONFIG = {
    'input_shape': (64, 1),      # Spectrum analysis input
    'hidden_layers': [128, 64, 32],
    'output_size': 5,            # 5 frequency bands
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
}

# GUI Configuration
GUI_CONFIG = {
    'update_interval': 100,      # ms
    'plot_history': 1000,        # points
    'refresh_rate': 10,          # Hz
}

# Simulation Parameters
SIMULATION_CONFIG = {
    'sampling_rate': 20e6,       # 20 MHz
    'symbol_time': 50e-6,        # 50 μs
    'noise_floor': -90,          # dBm
    'jammer_power': 20,          # dBm
}

# Performance Metrics
METRICS_CONFIG = {
    'target_ber': 1e-6,         # Target bit error rate
    'max_latency': 0.1,         # 100ms
    'min_throughput': 10e6,     # 10 Mbps
}
