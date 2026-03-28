"""
DRR2-HUmap: Deep Learning Pipeline for DRR Segmentation and HU Prediction

A comprehensive pipeline for automated segmentation of anatomical structures 
in Digitally Reconstructed Radiographs (DRRs) and subsequent Hounsfield Unit 
(HU) intensity prediction using CNN-PCA models.

Author: Maxime Huppe
Institution: Imperial College London
"""

__version__ = "1.0.0"
__author__ = "Maxime Huppe"
__email__ = "your.email@domain.com"

# Core imports - using importlib to handle numeric module names
import importlib

# Import models from 00_models
models_module = importlib.import_module('.00_models', package=__name__)
AttentionUNet = models_module.attention_unet.AttentionUNet
MCDropout = models_module.cnn_pca.MCDropout  
PC1Regressor = models_module.cnn_pca.PC1Regressor

# Import segmentation functions from 01_DRR_segmentation
segmentation_module = importlib.import_module('.01_DRR_segmentation.drr_segmentation', package=__name__)
drr_preprocess = segmentation_module.drr_preprocess
drr_segmentation = segmentation_module.drr_segmentation
crop_and_resize = segmentation_module.crop_and_resize
process_views = segmentation_module.process_views

# Configuration
from .config import get_default_config, ConfigManager

# Try to import optional modules
try:
    from .utils.visualization import plot_segmentation_results, plot_prediction_results
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from .utils.metrics import calculate_segmentation_metrics, calculate_prediction_metrics  
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

__all__ = [
    # Core models
    "AttentionUNet",
    "PC1Regressor",
    "MCDropout",
    
    # Segmentation functions
    "drr_preprocess",
    "drr_segmentation",
    "crop_and_resize", 
    "process_views",
    
    # Configuration
    "get_default_config",
    "ConfigManager",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]

# Add optional imports to __all__ if available
if VISUALIZATION_AVAILABLE:
    __all__.extend(["plot_segmentation_results", "plot_prediction_results"])
    
if METRICS_AVAILABLE:
    __all__.extend(["calculate_segmentation_metrics", "calculate_prediction_metrics"]) 