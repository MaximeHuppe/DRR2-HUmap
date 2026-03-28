"""
Model architectures for DRR2-HUmap pipeline.

This module contains the neural network architectures used in the DRR2-HUmap pipeline:
- AttentionUNet: For anatomical structure segmentation
- CNN-PCA models: For HU intensity prediction with uncertainty quantification
"""

from .attention_unet import AttentionUNet, AttentionBlock, ConvBlock
from .cnn_pca import PC1Regressor, MCDropout

__all__ = [
    # Segmentation models (clean pipeline)
    "AttentionUNet",
    "AttentionBlock", 
    "ConvBlock",

    # CNN-PCA models
    "PC1Regressor",  # Legacy compatibility
    "MCDropout",
] 