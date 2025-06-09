# 🔐 5G Adaptive Communication System - TEKNOFEST 2025

A state-of-the-art wireless communication simulation system featuring AI-driven frequency hopping, advanced security algorithms, and comprehensive data integrity verification for the TEKNOFEST Wireless Communication Competition.

## 🚀 Overview

This system implements a complete 5G-compatible wireless communication solution with:
- **🧠 CNN-based intelligent frequency hopping** to avoid jamming attacks
- **🔄 Adaptive modulation schemes** from BPSK to 1024QAM based on channel conditions
- **📡 OFDM waveform generation** with realistic channel modeling and interference
- **🔒 Advanced security algorithms** with real-time data corruption detection and correction
- **📊 Sub-band SNR monitoring** across all 5 frequency bands with jammer impact visualization
- **🎮 Interactive GUI** with real-time performance monitoring and security event logging
- **🏆 Competition scenario simulation** for all three TEKNOFEST contest phases

## ✨ Enhanced Security & Monitoring Features

### 🔐 Security Algorithm Visualization
- **Real-time error detection and correction logging** using LDPC/Turbo coding
- **Security event notifications** with emoji-coded severity levels (🔒 INFO, ⚠️ WARNING)
- **Jammer detection and countermeasure tracking** with frequency hopping responses
- **Data integrity verification** using CRC-32 and SHA-256 checksums
- **Channel quality assessment** with adaptive coding scheme selection

### 📈 Sub-band SNR Analysis
- **Complete frequency spectrum monitoring** across all 5 bands (2.4-2.6 GHz)
- **Active transmission band highlighting** with green color coding
- **Jammer impact visualization** showing SNR degradation in affected bands
- **Real-time SNR estimation** for non-active bands with pattern jammer tracking
- **Color-coded status indicators**: 
  - 🟢 Green: Active transmission band
  - 🔵 Blue: Available bands with good SNR
  - 🔴 Red: Jammed or poor SNR bands

### 📡 Real Data Transmission
This system transmits **REAL DATA**, not simulated progress bars:
- **Actual file transmission** with packetization and CRC protection
- **Mixed content test files** containing both text and binary data
- **SHA-256 hash verification** for transmitted data integrity
- **Packet header structure** with sequence numbers and metadata
- **File reconstruction** from received packets with integrity checking
- **Transmission statistics** including packet loss, corruption rates, and success metrics

## 🔐 Security Algorithm Details

### Error Detection & Correction
The system implements multiple layers of data protection:

#### 1. Forward Error Correction (FEC)
- **LDPC Codes**: Low-Density Parity-Check with 95% correction efficiency
- **Turbo Codes**: Iterative decoding with 90% correction efficiency  
- **Basic Coding**: Reed-Solomon with 50% correction efficiency
- **Adaptive Selection**: Automatically chooses optimal coding based on channel quality

#### 2. Data Integrity Verification
- **CRC-32**: Packet-level integrity checking with IEEE 802.3 polynomial
- **SHA-256**: File-level hash verification for end-to-end authenticity
- **Sequence Numbers**: Packet ordering and loss detection
- **Timestamp Validation**: Transmission timing verification

#### 3. Security Event Logging
```
[12:34:56] 🔒 Security: LDPC corrected 45 errors (SNR: 18.2dB)
[12:34:57] ⚠️  Security: Pattern jammer detected, interference: 12.3
[12:34:58] 🔒 Security: Frequency hop 3→5 to avoid jammer
```

### Real Data Transmission Architecture

#### File Processing Pipeline
1. **File Reading**: Binary file content with metadata extraction
2. **Header Creation**: Filename, size, hash, timestamp encoding
3. **Packetization**: 1024-byte packets with 16-byte headers
4. **CRC Protection**: Individual packet and overall data integrity
5. **Transmission**: Real packet transmission with loss/corruption simulation
6. **Reconstruction**: Packet reassembly and file integrity verification

#### Packet Structure
```
Header (16 bytes):
├── Sync Word (4 bytes): 'SYNC'
├── Sequence Number (4 bytes): Packet ordering
├── Total Packets (4 bytes): Complete transmission size
├── Payload Length (2 bytes): Current packet data size
├── Packet Type (1 byte): Data/Control indicator
└── Reserved (1 byte): Future extensions

Payload (1008 bytes max): Actual file data
CRC (4 bytes): Packet integrity verification
```

#### Data Types Transmitted
- **Mixed Content Files**: Combination of text and binary data
- **Realistic Test Data**: Not fake progress bars - actual file transmission
- **Metadata Packets**: File information and transmission parameters
- **Control Packets**: System status and acknowledgments

### 📈 Sub-band SNR Monitoring

### Frequency Band Allocation
```
Band 1: 2.400 GHz (Stable, low frequency characteristics)
Band 2: 2.450 GHz (Balanced performance)  
Band 3: 2.500 GHz (Optimal for most conditions)
Band 4: 2.550 GHz (High performance, sensitive to interference)
Band 5: 2.600 GHz (Maximum throughput, high frequency)
```

### Visual Indicators
- **🟢 Green Bars**: Currently active transmission band
- **🔵 Blue Bars**: Available bands with good SNR (>10 dB)
- **🔴 Red Bars**: Jammed or poor SNR bands (<10 dB)
- **SNR Values**: Real-time measurements displayed on each bar
- **Active Band Highlight**: Clear indication of current transmission frequency

### Jammer Impact Modeling
- **Pattern Jammer**: Targets even-numbered bands (2, 4) with 15dB degradation
- **Random Jammer**: 30% probability per band with 10-20dB interference
- **Adaptive Response**: Automatic frequency hopping to avoid compromised bands
- **SNR Estimation**: Non-active bands estimated with realistic uncertainty

## 📁 Project Structure

```
sara-simulation/
├── core/                      # Core communication modules
│   ├── modulation.py          # OFDM and modulation schemes (BPSK-1024QAM)
│   ├── channel.py             # Channel modeling and interference simulation
│   ├── coding.py              # LDPC/Turbo coding implementation
│   ├── frequency_hopping.py   # CNN-based intelligent frequency selection
│   └── data_processing.py     # Data handling, CRC, and packet processing
├── ai/                        # AI/ML components
│   ├── cnn_model.py           # CNN architecture for frequency selection
│   ├── data_generator.py      # Synthetic training data generation
│   └── training.py            # Model training pipeline with PyTorch
├── gui/                       # Enhanced GUI with security monitoring
│   ├── main_window.py         # PyQt5 interface with 6 visualization tabs:
│   │                          #   - SNR/BER/Throughput real-time plots
│   │                          #   - Constellation diagram with live updates
│   │                          #   - Frequency hopping visualization
│   │                          #   - Sub-band SNR monitoring (NEW!)
│   └── __init__.py            # Security event logging integration
├── simulation/                # Competition simulation
│   ├── competition_simulator.py # Full competition environment simulation
│   └── __init__.py            # Package initialization
├── main.py                    # Main application entry point
├── config.py                  # System configuration settings
├── setup.py                   # Installation and dependency verification
├── test_system.py             # Comprehensive test suite
├── requirements.txt           # Python dependencies
└── YARISMA_SARTNAMESI.md      # Competition specifications (Turkish)
```

## ✨ Key Features

### 🧠 AI-Powered Frequency Hopping
- **CNN-based decision making** for optimal frequency selection
- **Real-time adaptation** to jamming patterns and channel conditions
- **Synthetic data generation** for training in various scenarios
- **Performance comparison** with rule-based hopping strategies

### 📡 Advanced Modulation
- **Adaptive modulation schemes**: BPSK, QPSK, 16QAM, 64QAM, 256QAM, 1024QAM
- **OFDM waveform generation** with cyclic prefix and windowing
- **Channel coding support**: LDPC and Turbo codes
- **Real-time constellation diagrams** and signal quality metrics

### 🎯 Competition Simulation
- **Three-phase competition scenarios** with increasing complexity
- **Multiple jammer types**: Pattern-based, random, and adaptive
- **Performance scoring** based on throughput, BER, and latency
- **Automated result analysis** and visualization

### 🖥️ Enhanced Interactive GUI
- **6 comprehensive visualization tabs**:
  1. **SNR Monitoring**: Real-time signal-to-noise ratio tracking
  2. **BER Analysis**: Bit error rate with logarithmic scaling
  3. **Throughput Metrics**: Data transmission rates (Mbps)
  4. **Constellation Diagrams**: Live modulation scheme visualization
  5. **Frequency Hopping**: Band selection dynamics over time
  6. **Sub-band SNR Analysis**: Complete spectrum monitoring with jammer impact (NEW!)
- **Security event logging** with real-time corruption detection alerts
- **Scenario control panels** for jammer type and file size selection
- **Performance dashboards** with current status indicators
- **Simulation speed control** with 1x-10x acceleration options

### 🔬 Data Integrity & Security
- **Multi-layer error correction**: LDPC (95% efficiency), Turbo (90% efficiency), Basic (50% efficiency)
- **Adaptive coding scheme selection** based on channel quality assessment
- **Real-time security event tracking** with detailed logging timestamps
- **CRC-32 packet integrity verification** with automatic retransmission
- **File-level SHA-256 verification** ensuring end-to-end data authenticity
- **Jammer countermeasure logging** tracking frequency hop responses to interference

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Quick Setup
```powershell
# Clone or download the project
cd sara-simulation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python setup.py
```

### Manual Dependency Installation
```powershell
pip install torch torchvision torchaudio
pip install PyQt5 matplotlib numpy scipy
pip install tqdm commpy
```

## 🚀 Usage Guide

> **📝 Command Format Update**: This system now uses `--mode` based commands for better organization:
> - GUI: `python main.py --mode gui`
> - Simulation: `python main.py --mode simulation`
> - Training: `python main.py --mode training`
> - Data Generation: `python main.py --mode data`

### 1. Quick Start - Enhanced GUI Mode
```powershell
# Launch the enhanced GUI with all security features
python main.py --mode gui
```
🎮 **GUI Features**:
- Real-time security event monitoring with 🔒 and ⚠️ indicators
- Sub-band SNR visualization showing jammer impact across all 5 frequency bands
- Live constellation diagrams and performance metrics
- Interactive scenario switching between no jammer, pattern jammer, and random jammer
- File size selection (100MB, 1GB, 10GB) for realistic transmission testing

### 2. Competition Simulation
```powershell
# Run full competition simulation
python main.py --mode simulation

# Run specific competition phases
python main.py --mode simulation --phases 1 2 3

# Run with custom file sizes (in MB)
python main.py --mode simulation --phases 1 2 --file-sizes 100 500
```
🔐 **Simulation Features**:
- Complete competition environment with all three phases
- Automatic performance scoring and evaluation
- Real-time adaptation to jamming patterns
- Comprehensive results analysis and logging

### 3. AI Model Training
```powershell
# Train the AI frequency hopping model
python main.py --mode training --samples 5000

# Generate training data
python main.py --mode data --samples 3000
```
📊 **AI Training Features**:
- CNN-based frequency selection model training
- Synthetic data generation for various jamming scenarios
- Performance comparison with rule-based strategies
- Model validation and optimization

### 4. Data Transmission Verification
```powershell
# Verify real data transmission capabilities
python test_system.py --verbose
```
📁 **Real Data Transmission**:
- Creates test files with mixed text and binary content
- Performs actual packetization with CRC protection
- Transmits real data packets with headers and sequence numbers
- Reconstructs files from received packets
- Verifies data integrity end-to-end

## 📊 Enhanced Command Line Options

### Main Application (`main.py`)
```powershell
# Display comprehensive help
python main.py --help

# Launch GUI mode (default)
python main.py --mode gui

# Run full competition simulation
python main.py --mode simulation --verbose

# Run specific phases with custom settings
python main.py --mode simulation --phases 2 3 --file-sizes 500 1000

# Train AI model with custom sample size
python main.py --mode training --samples 3000 --verbose

# Generate training data
python main.py --mode data --samples 2000

# Run without AI (rule-based decisions only)
python main.py --mode simulation --no-ai
```

### System Testing Commands
```powershell
# Run comprehensive system tests
python test_system.py --verbose

# Test GUI functionality
python test_gui_ready.py

# Test enhanced frequency hopping
python test_enhanced_frequency_hopping.py

# Verify AI model authenticity
python verify_ai_authenticity.py

# Final system validation
python final_validation.py
```

### Configuration Options
- `--mode {gui,simulation,training,data}`: Application mode (default: gui)
- `--phases {1,2,3}`: Competition phases to run (can specify multiple)
- `--file-sizes`: File sizes in MB for each phase (space-separated)
- `--no-ai`: Disable AI model and use rule-based frequency hopping
- `--samples SIZE`: Number of samples for training/data generation (default: 5000)
- `--verbose, -v`: Enable detailed logging output

## 🔧 Technical Specifications

### System Requirements
- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible (optional, for training acceleration)
- **Storage**: 2GB free space for models and data

### Performance Characteristics
- **Frequency Range**: 2.4-2.5 GHz (competition band)
- **Bandwidth**: 20 MHz per channel
- **Modulation Support**: BPSK to 1024QAM adaptive
- **Coding Rates**: 1/2, 2/3, 3/4, 5/6 for LDPC
- **OFDM Parameters**: 1024 subcarriers, 256 data carriers
- **Frame Duration**: 1ms (1000 symbols/second)

### AI Model Architecture
```
CNN Frequency Selector:
├── Input: [batch_size, sequence_length, features]
├── Conv1D: 64 filters, kernel=3, ReLU
├── Conv1D: 128 filters, kernel=3, ReLU  
├── Conv1D: 256 filters, kernel=3, ReLU
├── Global Average Pooling
├── Dense: 512 units, ReLU, Dropout(0.3)
├── Dense: 256 units, ReLU, Dropout(0.3)
└── Output: [num_frequencies] with Softmax
```

## 🎮 GUI User Guide

### Main Interface
1. **Control Panel** (Left): Scenario selection, start/stop controls, parameter adjustment
2. **Real-time Plots** (Center): SNR, BER, throughput monitoring over time
3. **Constellation Diagram** (Right): Current modulation scheme visualization
4. **Status Bar** (Bottom): System status, current metrics, progress indicators

### Key Features
- **Scenario Selection**: Choose from three competition phases
- **Real-time Monitoring**: Live updates of all performance metrics
- **Parameter Control**: Adjust modulation, coding, and hopping parameters
- **Data Export**: Save simulation results and plots
- **Log Viewer**: Real-time system messages and warnings

### Keyboard Shortcuts
- `Ctrl+S`: Save current results
- `Ctrl+R`: Reset simulation
- `Ctrl+P`: Pause/Resume
- `Ctrl+Q`: Quit application
- `F1`: Show help dialog

## 🔍 Competition Scenarios

### Phase 1: Basic Communication
- **Duration**: 5 minutes
- **Jammers**: None
- **Objective**: Establish reliable communication
- **Success Criteria**: BER < 10^-3, Throughput > 1 Mbps

### Phase 2: Jammer Avoidance
- **Duration**: 10 minutes  
- **Jammers**: Pattern-based (3 types)
- **Objective**: Maintain communication under jamming
- **Success Criteria**: BER < 10^-2, Throughput > 500 kbps

### Phase 3: Advanced Scenarios
- **Duration**: 15 minutes
- **Jammers**: Adaptive and random (5+ types)
- **Objective**: Optimize performance under severe interference
- **Success Criteria**: BER < 5×10^-2, Throughput > 200 kbps

## 🐛 Troubleshooting

### Common Issues

#### Installation Problems
```powershell
# Missing PyQt5
pip install PyQt5

# CUDA issues (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Missing system libraries on Windows
# Ensure Visual C++ Build Tools are installed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Runtime Errors
```powershell
# Module import errors
$env:PYTHONPATH += ";$(pwd)"

# Memory issues during training
python main.py --mode training --samples 1000  # Reduce sample size

# GUI display problems on Windows
# Usually resolved by installing latest PyQt5
pip install --upgrade PyQt5
```

#### Performance Issues
- **Slow training**: Enable CUDA if available, reduce sample count
- **GUI lag**: Close unnecessary applications, reduce plot update frequency
- **Memory usage**: Limit data generation samples with `--samples` parameter

### Debug Mode
```powershell
# Enable verbose logging
python main.py --mode gui --verbose

# Check system status
python setup.py

# Run diagnostics
python test_system.py --verbose

# Test specific components
python test_gui_ready.py
python verify_ai_authenticity.py
```

## 🧪 Testing and Validation

### Test Categories
1. **Unit Tests**: Individual module functionality
2. **Integration Tests**: Cross-module communication
3. **Performance Tests**: Speed and memory benchmarks
4. **GUI Tests**: Interface responsiveness and accuracy
5. **End-to-End Tests**: Complete workflow validation

### Running Tests
```powershell
# Full test suite
python test_system.py --verbose

# Test GUI components
python test_gui_ready.py

# Test frequency hopping algorithms
python test_enhanced_frequency_hopping.py

# Verify AI model authenticity
python verify_ai_authenticity.py

# Run final system validation
python final_validation.py

# Test with coverage (optional)
pip install coverage
coverage run test_system.py
coverage report -m
```

## 📈 Performance Optimization

### Training Optimization
```powershell
# Train AI model with GPU acceleration (if available)
python main.py --mode training --samples 5000 --verbose

# Generate larger training datasets
python main.py --mode data --samples 10000

# Run simulation without AI for comparison
python main.py --mode simulation --no-ai --verbose
```

### Runtime Optimization
- **CPU**: Use multiprocessing for signal processing
- **Memory**: Implement batch processing for large datasets
- **I/O**: Cache frequently accessed data
- **GUI**: Optimize plot update frequencies

## 🔬 Development and Extension

### Adding New Modulation Schemes
1. Edit `core/modulation.py`
2. Add modulation class inheriting from `BaseModulation`
3. Implement `modulate()` and `demodulate()` methods
4. Update configuration in `config.py`

### Extending AI Models
1. Create new model in `ai/` directory
2. Inherit from `torch.nn.Module`
3. Update training pipeline in `ai/training.py`
4. Add model selection in configuration

### Custom Jamming Patterns
1. Edit `core/channel.py`
2. Add jammer class inheriting from `BaseJammer`
3. Implement `generate_interference()` method
4. Register in scenario configurations

## ✅ Quick Validation

After installation, verify everything works correctly:

```powershell
# Check system dependencies
python setup.py

# Test basic functionality
python test_system.py --verbose

# Launch GUI to verify interface
python main.py --mode gui

# Test AI model functionality
python verify_ai_authenticity.py

# Run a quick simulation
python main.py --mode simulation --phases 1 --file-sizes 10 --verbose
```

If all commands execute without errors, your installation is complete and ready for the competition!

## 📚 References and Documentation

### Competition Resources
- [TEKNOFEST Official Website](https://www.teknofest.org/)
- 5G NR specifications: 3GPP TS 38.211-214

### Technical References
- **OFDM**: "OFDM for Wireless Communications" by van Nee & Prasad
- **Channel Coding**: "Channel Coding: Theory, Algorithms, and Applications" by Glover & Grant  
- **Machine Learning**: "Deep Learning" by Goodfellow, Bengio & Courville
- **Software Radio**: "Software Radio Architecture" by Tuttlebee

## 📄 License

This project is developed for the TEKNOFEST Wireless Communication Competition. All rights reserved.

## 🤝 Contributing

This is a competition project developed for TEKNOFEST 2025. For questions or suggestions, please contact the development team.

### 👨‍💻 Project Developers
- **[@RAhsencicek](https://github.com/RAhsencicek)** - Lead Developer & AI Systems Engineer
- **[@ihoflaz](https://github.com/ihoflaz)** - Core Systems Developer & GUI Architect

For technical questions, bug reports, or collaboration inquiries, feel free to reach out through GitHub.

---

**Last Updated**: June 2025  
**Version**: 1.0.0  
**Competition**: TEKNOFEST 2025 Wireless Communication Challenge
