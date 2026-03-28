#%%
#!/usr/bin/env python3
"""
Installation Verification Script for DRR2-HUmap

This script verifies that all dependencies are correctly installed and 
the package is ready for use.

Usage:
    python scripts/verify_installation.py
"""

import sys
import os
import subprocess
from pathlib import Path
import importlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {text}")
    print(f"{'='*60}")

def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")

def print_warning(text: str):
    """Print warning message."""
    print(f"⚠️  {text}")

def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")

def check_python_version():
    """Check Python version compatibility."""
    print_header("Python Version Check")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 9):
        print_success("Python version is compatible (≥3.9)")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} is not supported. Requires Python ≥3.9")
        return False

def check_core_dependencies():
    """Check if core dependencies are installed."""
    print_header("Core Dependencies Check")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'numpy': 'NumPy',
        'cv2': 'OpenCV (cv2)',
        'skimage': 'scikit-image',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'pickle': 'Pickle (built-in)',
        'pathlib': 'Pathlib (built-in)'
    }
    
    failed_imports = []
    
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
                print_success(f"{name} - version {cv2.__version__}")
            elif module == 'torch':
                import torch
                print_success(f"{name} - version {torch.__version__}")
            elif module == 'numpy':
                import numpy as np
                print_success(f"{name} - version {np.__version__}")
            elif module == 'matplotlib':
                import matplotlib
                print_success(f"{name} - version {matplotlib.__version__}")
            elif module == 'sklearn':
                import sklearn
                print_success(f"{name} - version {sklearn.__version__}")
            elif module == 'skimage':
                import skimage
                print_success(f"{name} - version {skimage.__version__}")
            else:
                importlib.import_module(module)
                print_success(f"{name} - available")
        except ImportError as e:
            print_error(f"{name} - NOT FOUND: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def check_pytorch_acceleration():
    """Check PyTorch acceleration capabilities."""
    print_header("PyTorch Acceleration Check")
    
    try:
        import torch
        
        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print_success("MPS (Apple Silicon) acceleration available")
            device = torch.device("mps")
            try:
                # Test MPS with a simple operation
                x = torch.randn(3, 3).to(device)
                y = torch.matmul(x, x)
                print_success("MPS acceleration test passed")
                return True
            except Exception as e:
                print_warning(f"MPS available but test failed: {e}")
                print_warning("Falling back to CPU - this will be slower")
                return False
        
        # Check CUDA (NVIDIA)
        elif torch.cuda.is_available():
            print_success(f"CUDA acceleration available")
            print_success(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print_success(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            
            # Test CUDA with a simple operation
            try:
                device = torch.device("cuda:0")
                x = torch.randn(3, 3).to(device)
                y = torch.matmul(x, x)
                print_success("CUDA acceleration test passed")
                return True
            except Exception as e:
                print_warning(f"CUDA available but test failed: {e}")
                print_warning("Falling back to CPU - this will be slower")
                return False
        
        else:
            print_warning("No GPU acceleration available - will use CPU only")
            print_warning("Training and inference will be significantly slower")
            return True  # CPU is still functional, just slower
        
    except ImportError:
        print_error("PyTorch not available")
        return False

def check_package_structure():
    """Check if the package structure is correct."""
    print_header("Package Structure Check")
    
    base_path = Path(__file__).parent.parent
    required_paths = [
        "src/__init__.py",
        "src/config.py",
        "src/00_models/attention_unet.py",
        "src/00_models/cnn_pca.py",
        "src/01_DRR_segmentation/drr_segmentation.py",
        "src/02_PCA_CNN_regression/pca_cnn_regression.py",
        "src/utils/model_utils.py",
        "src/utils/metrics.py",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    missing_files = []
    
    for path in required_paths:
        full_path = base_path / path
        if full_path.exists():
            print_success(f"{path}")
        else:
            print_error(f"{path} - MISSING")
            missing_files.append(path)
    
    return len(missing_files) == 0

def check_imports():
    """Check if package imports work correctly."""
    print_header("Package Import Check")
    
    # Add src directory to Python path
    base_path = Path(__file__).parent.parent
    src_path = base_path / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    imports_to_test = [
        ("00_models.attention_unet", "AttentionUNet"),
        ("00_models.cnn_pca", "PC1Regressor"),
        ("utils.model_utils", "load_segmentation_model"),
        ("utils.metrics", "compute_dice_score"),
        ("01_DRR_segmentation.drr_segmentation", "drr_segmentation"),
        ("02_PCA_CNN_regression.pca_cnn_regression", "pca_cnn_prediction")
    ]
    
    failed_imports = []
    
    for module_name, function_name in imports_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, function_name):
                print_success(f"{module_name}.{function_name}")
            else:
                print_error(f"{module_name}.{function_name} - Function not found")
                failed_imports.append(f"{module_name}.{function_name}")
        except ImportError as e:
            print_error(f"{module_name} - Import failed: {e}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def check_example_data():
    """Check if example data is available."""
    print_header("Example Data Check")
    
    base_path = Path(__file__).parent.parent
    example_data_path = base_path / "src" / "example_data"
    
    if example_data_path.exists():
        files = list(example_data_path.glob("*"))
        if files:
            print_success(f"Example data directory found with {len(files)} files")
            for file in files[:5]:  # Show first 5 files
                print_success(f"  {file.name}")
            if len(files) > 5:
                print_success(f"  ... and {len(files) - 5} more files")
            return True
        else:
            print_warning("Example data directory exists but is empty")
            return False
    else:
        print_warning("Example data directory not found")
        print_warning("You'll need to provide your own data for testing")
        return False

def run_verification():
    """Run all verification checks."""
    print("🚀 DRR2-HUmap Installation Verification")
    print("This script will verify that your installation is ready to use.")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_core_dependencies),
        ("PyTorch Acceleration", check_pytorch_acceleration),
        ("Package Structure", check_package_structure),
        ("Package Imports", check_imports),
        ("Example Data", check_example_data)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        if result:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
    
    print(f"\n📊 Overall Result: {passed}/{total} checks passed")
    
    if passed == total:
        print_success("🎉 Installation verification completed successfully!")
        print_success("Your DRR2-HUmap installation is ready to use.")
    elif passed >= total - 2:
        print_warning("⚠️  Installation mostly successful with minor issues.")
        print_warning("Some features may not work optimally.")
    else:
        print_error("❌ Installation verification failed.")
        print_error("Please fix the issues above before using DRR2-HUmap.")
    
    return passed == total

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1) 
# %%
