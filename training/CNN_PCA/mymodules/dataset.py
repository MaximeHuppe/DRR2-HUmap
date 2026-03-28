#%% ──────────────────────────────────────────────────────────────────────────────
# Author: Maxime Huppe
# Date: 2025-05-15
# Institute: Imperial College London

# PURPOSE:
#   Define the dataset for the PCA regression pipeline

# Description : 
"""
    TibiaMultiPCDataset: 
        A dataset that is used to train the PCA regression pipeline.
        The dataset is a list of directories, each containing a CT scan and a DRR.
        The dataset is used to train the PCA regression pipeline.
        The input is a 2 channel image (CT and mask), and the output is a scalar.
        The model is trained to regress a single PC of the CT scans at a time.

    clahe:
        A function that applies CLAHE to the DRRs.
        The input is a numpy array, and the output is a numpy array.
        The CLAHE is applied to the DRRs to enhance the contrast.
"""
#%% ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
from skimage import exposure
from torch.utils.data import Dataset
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mymodules.config import *

def clahe(img: np.ndarray) -> np.ndarray:
    img01 = exposure.rescale_intensity(img, in_range='image', out_range=(0.0,1.0)).astype(np.float32)
    out   = exposure.equalize_adapthist(img01, clip_limit=CLAHE_CLIP, kernel_size=CLAHE_NTILES)
    return out.astype(np.float32)

class TibiaMultiPCDataset(Dataset):
    def __init__(self, dirs, pca, scaler, idx, pc_idx, augment=False):
        self.dirs   = dirs
        self.pca    = pca
        self.scaler = scaler
        self.idx    = idx
        self.pc_idx = pc_idx
        self.augment= augment

    def __len__(self): return len(self.dirs)

    def __getitem__(self, i):
        base = self.dirs[i]
        ap   = np.load(os.path.join(base,"DRR","AP.npy")).astype(np.float32)
        ml   = np.load(os.path.join(base,"DRR","ML.npy")).astype(np.float32)

        # raw normalize
        rap = (ap - DRR_MEAN)/DRR_STD
        rml = (ml - DRR_MEAN)/DRR_STD

        # CLAHE
        cap = clahe(ap); cml = clahe(ml)

        drr = np.stack([rap, rml, cap, cml], axis=0)
        if self.augment:
            drr += np.random.randn(*drr.shape)*0.02

        # PCA targets
        ct     = np.load(os.path.join(base,"CT","CT_reg.npy")).astype(np.float32)
        vox    = ct.flatten()[self.idx].reshape(1,-1)
        coeffs = self.pca.transform(vox)
        scaled = self.scaler.transform(coeffs)[0,self.pc_idx].astype(np.float32)
        if self.augment:
            scaled += np.random.randn()*0.05

        return torch.from_numpy(drr), torch.tensor(scaled, dtype=torch.float32), torch.from_numpy(ct)
