#%% ──────────────────────────────────────────────────────────────────────────────
# Author: Maxime Huppe
# Date: 2025-05-15
# Institute: Imperial College London

# PURPOSE:
#   Define the models for the PCA regression pipeline

# Description : 
"""
    MCDropout: 
        A dropout layer that is used to regularize the model.
        It is a dropout layer that is used to regularize the model(dropout_start).

    PC1Regressor: 
        A regressor that is used to regress the PC1 of the CT scans. 
        It is a resnet18 model with a dropout layer and a linear layer.
        The input is a 2 channel image (CT and mask), and the output is a scalar.
        The model is trained to regress a single PC of the CT scans at a time.
"""
#%% ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mymodules.config import *

class MCDropout(nn.Dropout):
    def forward(self,x): return F.dropout(x, self.p, training=True, inplace=self.inplace)


class PC1Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(2,64,7,2,3,bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head     = nn.Sequential(
            nn.Linear(512,256), nn.ReLU(),
            MCDropout(DROPOUT_START),
            nn.Linear(256,1)
        )
    def forward(self,x):
        f = self.backbone(x).view(x.size(0),-1)
        return self.head(f).squeeze(1)