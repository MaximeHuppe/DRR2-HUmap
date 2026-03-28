#%% ──────────────────────────────────────────────────────────────────────────────
# Author: Maxime Huppe
# Date: 2025-05-15
# Institute: Imperial College London

# PURPOSE:
#   Configuration for the PCA regression pipeline

# INPUTS : 
"""    #   Data_root           : path to the dataset
    #   Split_path          : path to the split file
    #   Out_root            : path to the output directory
    #   Num_pca_modes       : number of PCA modes to use
    #   Threshold           : threshold for the PCA modes
    #   Stride              : stride for the voxel sampling
    #   Ct_shape            : shape of the CT scans
    #   Batch_size          : batch size for the training
    #   Epochs_pc1          : number of epochs for the PC1 regression
    #   Epochs_head         : number of epochs for the head regression
    #   Epochs_joint        : number of epochs for the joint regression
    #   Lr_pc1              : learning rate for the PC1 regression
    #   Lr_head             : learning rate for the head regression
    #   Lr_joint            : learning rate for the joint regression
    #   Device              : device to use for the training
    #   Drr_mean            : mean of the DRRs
    #   Drr_std             : standard deviation of the DRRs
    #   Clahe_clip          : clip for the CLAHE
    #   Clahe_ntiles        : number of tiles for the CLAHE
    #   Dropout_start       : start of the dropout schedule
    #   Dropout_end         : end of the dropout schedule
    #   Max_grad_norm       : maximum gradient norm
"""
#%% ──────────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np 
from tqdm import tqdm
import pickle

def compute_drr_mean_std(train_dirs):
    ap_list, ml_list = [], []
    for d in tqdm(train_dirs, desc="Computing DRR mean/std"):
        ap = np.load(f"{d}/DRR/AP.npy").astype(np.float32)
        ml = np.load(f"{d}/DRR/ML.npy").astype(np.float32)
        ap_list.append(ap)
        ml_list.append(ml)
    ap_stack = np.stack(ap_list)
    ml_stack = np.stack(ml_list)
    drr_mean = np.mean(np.concatenate([ap_stack, ml_stack]))
    drr_std = np.std(np.concatenate([ap_stack, ml_stack]))
    print(f"DRR mean : {drr_mean:.4f}")
    print(f"DRR std : {drr_std:.4f}")
    return drr_mean, drr_std

DATA_ROOT      = "/Volumes/Maxime Imperial Backup/Max Data/CNN_Dataset"
SPLIT_PATH     = '/Volumes/Maxime Imperial Backup/Code/4.PCA_regression copy/models/data_splitting_dir.pkl'
OUT_ROOT       = "/Volumes/Maxime Imperial Backup/Code/4.PCA_regression copy/models_V2"
OUT_ROOT_FIG   = "/Volumes/Maxime Imperial Backup/Code/4.PCA_regression copy/figures_V2"
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(OUT_ROOT_FIG, exist_ok=True)

# PCA + voxel sampling
NUM_PCA_MODES  = 10        # now only first 5 PCs
THRESHOLD      = -1000
STRIDE         = 1
CT_SHAPE       = (64,64,256)

# training
BATCH_SIZE     = 8
EPOCHS_PC1     = 150
EPOCHS_HEAD    = 150
EPOCHS_JOINT   = 150

LR_PC1         = 5e-5
LR_HEAD        = 5e-5
LR_JOINT       = 5e-5

DEVICE         = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# DRR normalization & augmentation
splits = pickle.load(open(SPLIT_PATH, "rb"))

train_dirs = splits["train"]

DRR_MEAN, DRR_STD = compute_drr_mean_std(train_dirs=train_dirs)
CLAHE_CLIP     = 0.01
CLAHE_NTILES   = (16,16)

# dropout schedule
DROPOUT_START  = 0.5
DROPOUT_END    = 0.1

# gradient clipping
MAX_GRAD_NORM  = 1.0

# Data dimensions
DRR_SHAPE = (64, 64)  # Height, Width

# PCA parameters
PCA_VARIANCE_THRESHOLD = 0.95  # Cumulative variance threshold

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# Model parameters
DROPOUT_RATE = 0.2
NUM_MC_DRAWS = 10  # Number of Monte Carlo draws for uncertainty estimation
# %%
