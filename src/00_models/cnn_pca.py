"""
CNN-PCA Models for HU Intensity Prediction

This module implements CNN-based models for predicting Hounsfield Unit (HU) intensities
from segmented DRR images using Principal Component Analysis (PCA) for dimensionality
reduction and Monte Carlo Dropout for uncertainty quantification.

Author: Maxime Huppe
Institution: Imperial College London
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as tv_models

DROPOUT_START = 0.5

class MCDropout(nn.Dropout):
    """Monte Carlo Dropout layer."""
    def forward(self, x):
        return F.dropout(x, self.p, training=True, inplace=self.inplace)

class PC1Regressor(nn.Module):
    """Neural network for predicting PCA coefficients from DRRs."""
    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            MCDropout(DROPOUT_START),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        return self.head(f).squeeze(1) 