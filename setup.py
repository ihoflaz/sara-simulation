# Setup and Installation Script

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        return False
    
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def check_installed_packages():
    """Check if all required packages are installed"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        with open(requirements_file, 'r') as f:
            requirements = f.read().splitlines()
        
        # Remove comments and empty lines
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
        
        missing_packages = []
        for requirement in requirements:
            try:
                pkg_resources.require(requirement)
                print(f"✓ {requirement}")
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
                print(f"✗ {requirement} - {str(e)}")
                missing_packages.append(requirement)
        
        if missing_packages:
            print(f"\nMissing packages: {missing_packages}")
            return False
        else:
            print("\n✓ All required packages are installed")
            return True
            
    except Exception as e:
        print(f"Error checking packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'logs', 'data']
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    
    test_imports = [
        'numpy',
        'matplotlib',
        'scipy',
        'torch', 
        'sklearn',
        'PyQt5.QtWidgets',
        'config',
        'core.modulation',
        'core.channel',
        'core.frequency_hopping',
        'core.coding',
        'core.data_processing',
        'ai.cnn_model',
        'ai.data_generator',
        'ai.training'
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {str(e)}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        return False
    else:
        print("\n✓ All modules imported successfully")
        return True

def run_quick_test():
    """Run a quick functionality test"""
    print("Running quick functionality test...")
    
    try:
        # Test OFDM system
        from core.modulation import OFDMSystem
        ofdm = OFDMSystem()
        print("✓ OFDM system initialized")
          # Test channel simulator        
        from core.channel import ChannelModel
        channel = ChannelModel()
        print("✓ Channel simulator initialized")
        
        # Test AI model
        from ai.cnn_model import FrequencyHoppingCNN
        model = FrequencyHoppingCNN()
        print("✓ AI model initialized")
        
        # Test data generator
        from ai.data_generator import SyntheticDataGenerator
        generator = SyntheticDataGenerator()
        print("✓ Data generator initialized")
        
        print("\n✓ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Quick test failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("  TEKNOFEST 5G Communication System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    print("\n" + "-" * 40)
    print("Creating directories...")
    create_directories()
    
    print("\n" + "-" * 40)
    print("Checking installed packages...")
    packages_ok = check_installed_packages()
    
    if not packages_ok:
        print("\nInstalling missing packages...")
        if not install_requirements():
            return False
        
        # Check again after installation
        if not check_installed_packages():
            return False
    
    print("\n" + "-" * 40)
    print("Testing imports...")
    if not test_imports():
        return False
    
    print("\n" + "-" * 40)
    print("Running functionality test...")
    if not run_quick_test():
        return False
    
    print("\n" + "=" * 60)
    print("✓ Setup completed successfully!")
    print("=" * 60)
    
    print("\nYou can now run the application using:")
    print("  python main.py --mode gui                    # GUI mode")
    print("  python main.py --mode simulation            # Simulation mode")
    print("  python main.py --mode training --samples 2000  # Train AI model")
    print("  python main.py --help                       # Show all options")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
