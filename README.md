# DRR2-HUmap: Deep Learning Pipeline for DRR Segmentation and HU Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Published article:** [https://doi.org/10.1016/j.compbiomed.2025.111434](https://doi.org/10.1016/j.compbiomed.2025.111434) (*Computer Methods and Programs in Biomedicine*).

A comprehensive deep learning pipeline for automated segmentation of anatomical structures in Digitally Reconstructed Radiographs (DRRs) and subsequent Hounsfield Unit (HU) intensity prediction using CNN-PCA models.

## 🎯 Overview

This pipeline combines state-of-the-art deep learning techniques to:

1. **Segment anatomical structures** (e.g., tibia) in DRR images using Attention U-Net
2. **Extract and preprocess** segmented regions with optimal cropping and resizing
3. **Predict HU intensities** using CNN-PCA models with uncertainty quantification
4. **Reconstruct 3D CT volumes** from predicted principal components

## 🏗️ Architecture

### Segmentation Module
- **Attention U-Net**: Enhanced U-Net with attention gates for precise anatomical segmentation
- **CLAHE Preprocessing**: Contrast Limited Adaptive Histogram Equalization for improved image quality
- **Multi-view Support**: Handles both Anteroposterior (AP) and Mediolateral (ML) projections

### CNN-PCA Module
- **Principal Component Analysis**: Dimensionality reduction for efficient HU representation
- **Monte Carlo Dropout**: Uncertainty quantification in predictions
- **Multi-channel CNN**: Uses raw and CLAHE-enhanced DRR channels for robust prediction

## 📁 Repository Structure

```
DRR2-HUmap/
├── README.md                     # This documentation
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
│
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   │
│   ├── 00_models/               # Model architectures
│   │   ├── attention_unet.py    # Attention U-Net implementation
│   │   ├── cnn_pca.py          # CNN-PCA models (PC1Regressor)
│   │   └── __init__.py
│   │
│   ├── 01_DRR_segmentation/     # Segmentation pipeline
│   │   ├── drr_segmentation.py  # Core segmentation functions
│   │   ├── run_segmentation.py  # Main segmentation script
│   │   ├── eval_segmentation.py # Evaluation utilities
│   │   └── __pycache__/
│   │
│   ├── 02_PCA_CNN_regression/   # HU prediction pipeline
│   │   ├── pca_cnn_regression.py     # Core PCA-CNN functions
│   │   ├── run_pca_cnn_prediction.py # Main prediction script
│   │   └── __pycache__/
│   │
│   ├── 03_evaluation/           # Evaluation tools
│   │
│   ├── utils/                   # Utility functions
│   │   ├── model_utils.py       # Model loading and management
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── __init__.py
│   │
│   └── example_data/            # Sample input data
│       ├── drr_AP.npy          # Example AP DRR
│       ├── drr_ML.npy          # Example ML DRR
│       ├── mask_AP.npy         # Example AP mask
│       ├── mask_ML.npy         # Example ML mask
│       ├── pca_AP_drr.npy      # Preprocessed AP DRR
│       └── pca_ML_drr.npy      # Preprocessed ML DRR
│
├── training/                     # Training scripts and configurations
│   ├── segmentation/            # Segmentation model training
│   │   ├── train_ap_segmentation.ipynb  # AP view training notebook
│   │   └── train_ml_segmentation.ipynb  # ML view training notebook
│   │
│   ├── CNN_PCA/                 # PCA-CNN model training
│   │   ├── run.py              # Main training script
│   │   └── mymodules/          # Training utilities
│   │
│   └── configs/                # Training configurations
│
├── configs/                     # Configuration files
│   ├── default.yaml            # Default pipeline configuration
│   └── experiment_configs/     # Experiment-specific configs
│
├── scripts/                     # Utility scripts
│   └── verify_installation.py  # Installation verification
│
└── figures/                     # Figure generation and analysis
    ├── modules/                # Figure generation modules
    ├── generate_all_figures.py # Generate all figures script
    └── [various figure scripts] # Individual figure generation
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+ (with MPS support for Apple Silicon or CUDA for NVIDIA GPUs)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/DRR2-HUmap.git
   cd DRR2-HUmap
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv drr2humap_env
   source drr2humap_env/bin/activate  # On Windows: drr2humap_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python scripts/verify_installation.py
   ```

## ⚠️ Important: Trained Models Not Included

**This repository does NOT include pre-trained models.** You must train your own models using the provided training code before running the inference scripts.

### Required Trained Models

You need to train the following models:

1. **Segmentation Models**:
   - AP view segmentation model (`AttentionUNet_V02_AP.pt`)
   - ML view segmentation model (`AttentionUNet_V02_ML.pt`)

2. **PCA-CNN Models**:
   - PC regressor models: `model_pc1_sequential.pt` through `model_pc10_sequential.pt`
   - PCA components: `pca.pkl`
   - Data scaler: `scaler.pkl`
   - Voxel indices: `idx.npy`

## 🎓 Training Your Own Models

### 1. Segmentation Model Training

#### AP View Segmentation
```bash
# Open and run the Jupyter notebook
cd training/segmentation
jupyter notebook train_ap_segmentation.ipynb
```

#### ML View Segmentation
```bash
# Open and run the Jupyter notebook
cd training/segmentation
jupyter notebook train_ml_segmentation.ipynb
```

**Training Requirements:**
- Training data: Pairs of DRR images (`.npy`) and corresponding segmentation masks
- Data format: DRR images should be 256×512 pixels, normalized
- Hardware: GPU recommended for faster training (MPS on Apple Silicon, CUDA on NVIDIA)

### 2. PCA-CNN Model Training

```bash
cd training/CNN_PCA
python run.py
```

**Training Requirements:**
- Training data: Pairs of segmented DRR images (AP and ML views) and corresponding CT volumes
- Data organization: The script expects specific directory structure with CT data
- Processing: Automatically computes PCA components and trains individual PC regressors
- Output: Creates all 10 PC models plus PCA components

**What the training script does:**
1. Computes PCA and scaler on the first 10 modes of CT scans
2. Trains separate CNN models for each principal component (PC1-PC10)
3. Saves all models and PCA components to the output directory
4. Performs PCA analysis and saves performance metrics

### 3. Training Data Requirements

**Segmentation Training Data:**
- DRR images: `.npy` format, shape (256, 512), floating point values
- Segmentation masks: `.npy` format, binary masks with same shape as DRRs
- Views: Both AP (Anteroposterior) and ML (Mediolateral) views required

**PCA-CNN Training Data:**
- Segmented DRR pairs: AP and ML views after segmentation preprocessing
- CT volumes: Corresponding 3D CT scans for HU value extraction
- Data splits: Train/validation/test splits (handled by training script)

## 📖 Usage (After Training)

### Quick Start

Once you have trained your models, update the file paths in the run scripts:

#### 1. DRR Segmentation

Edit `src/01_DRR_segmentation/run_segmentation.py` and update:
```python
# Update these paths to your trained models and data
drr_path = "path/to/your/drr.npy"
mask_path = "path/to/your/mask.npy"  # Optional ground truth
segmentation_model_path = "path/to/your/AttentionUNet_model.pt"
```

Then run:
```bash
cd src/01_DRR_segmentation
python run_segmentation.py
```

#### 2. HU Volume Prediction

Edit `src/02_PCA_CNN_regression/run_pca_cnn_prediction.py` and update:
```python
# Update these paths to your trained models and data
ap_drr_path = "path/to/your/pca_AP_drr.npy"
ml_drr_path = "path/to/your/pca_ML_drr.npy"

# Update PC model paths
pc_model_paths = {
    1: "path/to/model_pc1_sequential.pt",
    2: "path/to/model_pc2_sequential.pt",
    # ... up to pc10
}

# Update PCA component paths
pca_model_path = "path/to/pca.pkl"
scaler_path = "path/to/scaler.pkl"
indices_path = "path/to/idx.npy"
```

Then run:
```bash
cd src/02_PCA_CNN_regression
python run_pca_cnn_prediction.py
```

## 📋 Configuration

The repository includes comprehensive configuration files:

- `configs/default.yaml`: Default parameters for all pipeline components
- `configs/experiment_configs/`: Experiment-specific configurations
- Customize these files to match your data and training requirements

## 🔧 Data Format Specifications

### DRR Images
- **Format**: NumPy arrays saved as `.npy` files
- **Dimensions**: 256×512 pixels (height × width)
- **Data type**: `float32`
- **Value range**: Normalized attenuation coefficients

### Segmentation Masks
- **Format**: NumPy arrays saved as `.npy` files
- **Dimensions**: Same as corresponding DRR (256×512)
- **Data type**: `uint8` or `float32`
- **Values**: Binary masks (0 for background, 1 for foreground)

### CT Volumes (for training)
- **Format**: NumPy arrays
- **Dimensions**: Typically 64×64×256 (x×y×z)
- **Data type**: `float32`
- **Values**: Hounsfield Units (HU)

## ⚡ Hardware Requirements

### Minimum
- **CPU**: Multi-core processor
- **RAM**: 8GB system memory
- **Storage**: 5GB free space

### Recommended
- **GPU**: 
  - Apple Silicon Mac with 16GB+ unified memory (uses MPS acceleration)
  - NVIDIA GPU with 8GB+ VRAM (uses CUDA acceleration)
- **RAM**: 32GB+ for large datasets
- **Storage**: 20GB+ for training data and models

### Training Requirements
- **GPU strongly recommended** for reasonable training times
- **Large memory** for processing CT volumes during PCA computation
- **Sufficient storage** for training datasets (typically several GB)

## 🔍 Troubleshooting

### Common Issues

1. **Missing trained models**
   - **Error**: FileNotFoundError when running inference scripts
   - **Solution**: Train models using the provided training notebooks/scripts

2. **Memory errors during training**
   - **Error**: CUDA out of memory or system memory errors
   - **Solution**: Reduce batch size, use smaller datasets, or add more RAM/VRAM

3. **Import errors**
   - **Error**: ModuleNotFoundError
   - **Solution**: Ensure you're running from the correct directory and dependencies are installed

4. **MPS/CUDA errors**
   - **Error**: MPS/CUDA device errors
   - **Solution**: Set device to CPU in scripts or update PyTorch version

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-username/DRR2-HUmap/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/DRR2-HUmap/discussions)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite the [peer-reviewed paper](https://doi.org/10.1016/j.compbiomed.2025.111434) (DOI: [10.1016/j.compbiomed.2025.111434](https://doi.org/10.1016/j.compbiomed.2025.111434)) and/or the repository below.

```bibtex
@misc{huppe2024drr2humap,
  title={DRR2-HUmap: Deep Learning Pipeline for DRR Segmentation and HU Prediction},
  author={Huppe, Maxime},
  year={2024},
  institution={Imperial College London},
  url={https://github.com/your-username/DRR2-HUmap},
  doi={10.1016/j.compbiomed.2025.111434}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 Quick Start Checklist

Before using this pipeline:

- [ ] ✅ Install dependencies (`pip install -r requirements.txt`)
- [ ] ✅ Verify installation (`python scripts/verify_installation.py`)
- [ ] ⚠️ **Train segmentation models** (use `training/segmentation/` notebooks)
- [ ] ⚠️ **Train PCA-CNN models** (run `training/CNN_PCA/run.py`)
- [ ] ✅ Update file paths in run scripts
- [ ] ✅ Run inference scripts

**Remember**: This repository provides the framework and training code, but you must train your own models with your data before running inference! 