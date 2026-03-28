#%% ──────────────────────────────────────────────────────────────────────────────
# Author: Maxime Huppe
# Date: 2025-05-15
# Institute: Imperial College London

# PURPOSE:
#   Utility functions for the PCA regression pipeline

# Description : 
"""
    sample_voxels: 
        sample voxels from the union of the CTs. The voxels are sampled
        from the union of the CTs, and the voxels are sampled at a stride of STRIDE.
        The voxels are sampled from the union of the CTs, and the voxels are sampled at a
        stride of STRIDE.

    compute_and_save_pca: 
        compute the PCA and save the results. The PCA is computedon the union of the CTs, 
        and the PCA is saved in the OUT_ROOT directory.
"""
#%% ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from pathlib import Path

from mymodules.config import *

def sample_voxels(union_mask, shape, stride):
    fg = np.array(np.where(union_mask)).T
    voxels = []
    for z in range(shape[2]):
        coords = fg[fg[:,2]==z,:2]
        mask2d = np.zeros(shape[:2],bool)
        mask2d[coords[:,0],coords[:,1]] = True
        for x in range(0,shape[0],stride):
            for y in range(0,shape[1],stride):
                cx,cy = x+stride//2, y+stride//2
                if cx<shape[0] and cy<shape[1] and mask2d[cx,cy]:
                    voxels.append((cx,cy,z))
    return np.array(voxels)

def compute_and_save_pca(train_dirs):
    union = None
    vols  = []
    for p in tqdm(train_dirs, desc="Loading CTs"):
        ct   = np.load(os.path.join(p,"CT","CT_reg.npy"))
        mask = ct > THRESHOLD
        union = mask if union is None else union | mask
        vols.append(ct)
    voxels = sample_voxels(union, CT_SHAPE, STRIDE)
    idx    = np.ravel_multi_index(voxels.T, CT_SHAPE)
    X      = np.stack([v.flatten()[idx] for v in vols], axis=0)

    pca    = PCA(n_components=NUM_PCA_MODES)
    X_pca  = pca.fit_transform(X)
    scaler = StandardScaler().fit(X_pca)

    # save
    with open(os.path.join(OUT_ROOT,"pca.pkl"),   "wb") as f: pickle.dump(pca, f)
    with open(os.path.join(OUT_ROOT,"scaler.pkl"),"wb") as f: pickle.dump(scaler, f)
    np.save(os.path.join(OUT_ROOT,"idx.npy"), idx)

    return pca, scaler, idx

def print_parameters():
    print("--------------------------------")
    print("---- PARAMETERS ----------------")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"SPLIT_PATH: {SPLIT_PATH}")
    print(f"NUM_PCA_MODES: {NUM_PCA_MODES}")
    print("--------------------------------")
    print("--- MODEL PARAMETERS ------------")
    print(f"DROPOUT_START: {DROPOUT_START}")
    print(f"DROPOUT_END: {DROPOUT_END}")
    print(f"LR_PC1: {LR_PC1}")
    print(f"LR_HEAD: {LR_HEAD}")
    print(f"LR_JOINT: {LR_JOINT}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCHS_PC1: {EPOCHS_PC1}")
    print(f"EPOCHS_HEAD: {EPOCHS_HEAD}")
    print(f"EPOCHS_JOINT: {EPOCHS_JOINT}")
    print(f"DEVICE: {DEVICE}")
    print("--------------------------------")

def analyze_pca(pca, scaler, idx, train_dirs, out_dir=None):
    """
    Analyze PCA results with multiple visualizations to understand the components.
    
    Args:
        pca: Fitted PCA model
        scaler: Fitted StandardScaler
        train_dirs: List of training directories containing CT scans
        out_dir: Directory to save plots (if None, uses OUT_ROOT_FIG from config)
    
    Returns:
        dict: Dictionary containing analysis metrics
    """
    if out_dir is None:
        out_dir = OUT_ROOT_FIG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Variance Explained Plot
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir / 'pca_variance_explained.png')
    plt.close()
    
    # 2. Individual Component Variance
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Component Variance')
    plt.grid(True)
    plt.savefig(out_dir / 'pca_individual_variance.png')
    plt.close()
    
    # 3. Component Coefficients Heatmap (first 10 components)
    n_components_plot = min(10, pca.n_components_)
    plt.figure(figsize=(15, 8))
    sns.heatmap(pca.components_[:n_components_plot], 
                cmap='RdBu_r',
                center=0,
                xticklabels=False,
                yticklabels=[f'PC{i+1}' for i in range(n_components_plot)])
    plt.title('PCA Component Coefficients (First 10 Components)')
    plt.savefig(out_dir / 'pca_coefficients_heatmap.png')
    plt.close()
    
    # 4. Reconstruction Error Analysis
    reconstruction_errors = []
    for p in train_dirs:
        ct = np.load(os.path.join(p, "CT", "CT_reg.npy"))
        vox = ct.flatten()[idx].reshape(1, -1)
        coeffs = pca.transform(vox)
        reconstructed = pca.inverse_transform(coeffs)
        mae = mean_absolute_error(vox, reconstructed)
        reconstruction_errors.append(mae)
    
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors, bins=30)
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.grid(True)
    plt.savefig(out_dir / 'pca_reconstruction_errors.png')
    plt.close()
    
    # 5. Component Correlation Matrix
    plt.figure(figsize=(12, 10))
    component_corr = np.corrcoef(pca.components_)
    sns.heatmap(component_corr, 
                cmap='RdBu_r',
                center=0,
                xticklabels=[f'PC{i+1}' for i in range(pca.n_components_)],
                yticklabels=[f'PC{i+1}' for i in range(pca.n_components_)])
    plt.title('PCA Component Correlation Matrix')
    plt.savefig(out_dir / 'pca_component_correlation.png')
    plt.close()
    
    # Return analysis metrics
    analysis_metrics = {
        'n_components': pca.n_components_,
        'total_variance_explained': cumulative_variance[-1],
        'components_to_95_variance': np.argmax(cumulative_variance >= 0.95) + 1,
        'mean_reconstruction_error': np.mean(reconstruction_errors),
        'std_reconstruction_error': np.std(reconstruction_errors),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_
    }
    
    # Save metrics to file
    with open(out_dir / 'pca_analysis_metrics.txt', 'w') as f:
        f.write("PCA Analysis Metrics\n")
        f.write("===================\n\n")
        f.write(f"Number of Components: {analysis_metrics['n_components']}\n")
        f.write(f"Total Variance Explained: {analysis_metrics['total_variance_explained']:.4f}\n")
        f.write(f"Components needed for 95% variance: {analysis_metrics['components_to_95_variance']}\n")
        f.write(f"Mean Reconstruction Error: {analysis_metrics['mean_reconstruction_error']:.4f}\n")
        f.write(f"Std Reconstruction Error: {analysis_metrics['std_reconstruction_error']:.4f}\n\n")
        f.write("Explained Variance Ratio per Component:\n")
        for i, var in enumerate(analysis_metrics['explained_variance_ratio']):
            f.write(f"PC{i+1}: {var:.4f}\n")
    
    return analysis_metrics
