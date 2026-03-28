#%% ──────────────────────────────────────────────────────────────────────────────
# Author: Maxime Huppe
# Date: 2025-05-15
# Institute: Imperial College London

# PURPOSE:
#   Run the PCA regression pipeline

# Description : 
"""
    This script runs the PCA regression pipeline.
    It computes the PCA and scaler on the first 5 modes of the CT scans.
    It then trains a separate model for each PC.
    The models are trained sequentially, one PC at a time.
    The models are trained using the raw DRR channels (AP, ML).
"""

#%% ──────────────────────────────────────────────────────────────────────────────
# RUN ROUTINES
# ──────────────────────────────────────────────────────────────────────────────

import os
import pickle

from mymodules.config import *
from mymodules.training import *
from mymodules.util import *

print_parameters()
print("Starting sequential PC training with PC1Regressor")
print(f"Training the model for {NUM_PCA_MODES} PCs ! ")


# Create output directory
os.makedirs(OUT_ROOT, exist_ok=True); print(f"Output directory created: {OUT_ROOT}")

# Load splits and compute PCA
splits = pickle.load(open(SPLIT_PATH,"rb"))
#train_dirs = [d.split("/CT")[0] for d in splits["train"] if 'Aug_0' in d]
train_dirs = splits["train"]
print(f"Training on {len(train_dirs)} CTs for PCA computation")

# 1) Compute PCA & scaler on first 5 modes
print("Computing PCA and scaler...")
pca, scaler, idx = compute_and_save_pca(train_dirs)
print(f"PCA and scaler computed and saved to {OUT_ROOT}")

# Analyze PCA results
print("\nAnalyzing PCA results...")
out_PCA_perf = "/Volumes/Maxime Imperial Backup/Code/4.PCA_regression/output/PCA_perf"
os.makedirs(out_PCA_perf, exist_ok=True)
analysis_metrics = analyze_pca(pca, scaler, idx,train_dirs, out_PCA_perf)
print(f"PCA analysis complete! Results saved to {out_PCA_perf}")
print(f"Components needed for 95% variance: {analysis_metrics['components_to_95_variance']}")
print(f"Total variance explained: {analysis_metrics['total_variance_explained']:.4f}")

# 2) Train a separate model for each PC
models = []
for pc_idx in range(NUM_PCA_MODES):
    print(f"\nTraining model for PC{pc_idx+1}...")
    current_model = train_pc1_sequential(pca, scaler, idx, pc_idx)
    models.append(current_model)

print("\nTraining complete! Models saved as:")
for pc_idx in range(NUM_PCA_MODES):
    print(f"- model_pc{pc_idx+1}_sequential.pt")
# %%
