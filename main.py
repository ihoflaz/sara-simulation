# Main Application Entry Point for 5G Communication System

import sys
import os
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'logs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_gui_mode():
    """Run application in GUI mode"""
    print("Starting GUI mode...")
    
    try:
        from gui.main_window import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"GUI dependencies not available: {e}")
        print("Please install PyQt5: pip install PyQt5")
        return False
    except Exception as e:
        print(f"Error starting GUI: {e}")
        return False
    
    return True

def run_simulation_mode(enable_ai: bool = True, phases: list = None, file_sizes: list = None):
    """Run application in simulation mode"""
    print("Starting simulation mode...")
    
    from simulation.competition_simulator import CompetitionSimulator
    
    # Create simulator
    simulator = CompetitionSimulator(enable_ai=enable_ai, enable_gui=False)
    
    if phases is None:
        # Run full competition
        print("Running full competition simulation...")
        results = simulator.run_full_competition(file_sizes)
        
        # Generate plots
        try:
            import matplotlib.pyplot as plt
            fig = simulator.plot_results(results, save_plots=True)
            # Don't show plots in headless mode
            # plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
            
    else:
        # Run specific phases
        results = {'phases': {}, 'overall_score': 0, 'success_phases': 0}
        
        default_file_sizes = [100e6, 1e9, 10e9]  # 100MB, 1GB, 10GB
        
        for phase in phases:
            if 1 <= phase <= 3:
                file_size = file_sizes[phase-1] if file_sizes and len(file_sizes) >= phase else default_file_sizes[phase-1]
                
                print(f"\nRunning Phase {phase}...")
                phase_results = simulator.simulate_phase(
                    phase_number=phase,
                    file_size=file_size,
                    duration=120.0
                )
                
                results['phases'][phase] = phase_results
                phase_score = simulator.calculate_phase_score(phase_results)
                results['overall_score'] += phase_score
                
                if phase_results['success']:
                    results['success_phases'] += 1
                    
        # Save results
        simulator.save_results(results)
    
    return results

def run_training_mode(num_samples: int = 5000):
    """Run AI model training mode"""
    print("Starting AI model training...")
    
    from ai.training import ModelTrainer
    from ai.cnn_model import FrequencyHoppingCNN
      # Create model and trainer
    model = FrequencyHoppingCNN(input_size=30, sequence_length=100, num_frequencies=5)
    trainer = ModelTrainer(model)
    
    # Train model
    results = trainer.train(
        num_samples=num_samples,
        save_path='models/frequency_hopping_model.pth'
    )
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        fig = trainer.plot_training_curves('models/training_curves.png')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not generate training plots: {e}")
    
    print(f"Training completed! Final test accuracy: {results['test_accuracy']:.4f}")
    return results

def run_data_generation_mode(num_samples: int = 1000):
    """Run synthetic data generation mode"""
    print("Starting data generation...")
    
    from ai.data_generator import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator()
    
    # Generate dataset
    features, labels = generator.generate_dataset(
        num_samples=num_samples,
        save_path='data/training_data.npz'
    )
    
    # Visualize sample
    try:
        import matplotlib.pyplot as plt
        fig = generator.visualize_sample(features, labels, sample_idx=0)
        plt.savefig('data/sample_visualization.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
    
    print(f"Generated {num_samples} samples and saved to data/training_data.npz")
    return features, labels

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'), 
        ('torch', 'torch'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              TEKNOFEST 5G Communication System              ║
    ║                                                              ║
    ║     Adaptive Modulation & AI-based Frequency Hopping        ║
    ║              for Wireless Communication Competition         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main application entry point"""
    
    print_banner()
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='5G Adaptive Communication System for TEKNOFEST Competition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode gui                    # Start GUI
  python main.py --mode simulation            # Run full competition
  python main.py --mode simulation --phases 1 2   # Run specific phases
  python main.py --mode training --samples 3000    # Train AI model
  python main.py --mode data --samples 2000        # Generate training data
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['gui', 'simulation', 'training', 'data'], 
                       default='gui',
                       help='Application mode (default: gui)')
    
    parser.add_argument('--phases', 
                       nargs='+', 
                       type=int, 
                       choices=[1, 2, 3],
                       help='Competition phases to run (1, 2, 3)')
    
    parser.add_argument('--file-sizes', 
                       nargs='+', 
                       type=float,
                       help='File sizes in MB for each phase')
    
    parser.add_argument('--no-ai', 
                       action='store_true',
                       help='Disable AI model (use rule-based decisions)')
    
    parser.add_argument('--samples', 
                       type=int, 
                       default=5000,
                       help='Number of samples for training/data generation')
    
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create necessary directories
    create_directories()
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting application in {args.mode} mode")
    
    try:
        if args.mode == 'gui':
            success = run_gui_mode()
            return 0 if success else 1
            
        elif args.mode == 'simulation':
            file_sizes = None
            if args.file_sizes:
                file_sizes = [size * 1e6 for size in args.file_sizes]  # Convert MB to bytes
                
            results = run_simulation_mode(
                enable_ai=not args.no_ai,
                phases=args.phases,
                file_sizes=file_sizes
            )
            
            print(f"\nSimulation completed!")
            if 'final_score' in results:
                print(f"Final Score: {results['final_score']:.2f}/100")
            
            return 0
            
        elif args.mode == 'training':
            results = run_training_mode(args.samples)
            return 0
            
        elif args.mode == 'data':
            features, labels = run_data_generation_mode(args.samples)
            return 0
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
