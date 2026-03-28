import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import pickle
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from scipy.stats import wilcoxon, shapiro

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CT_SHAPE = (64, 64, 256)
THRESHOLD = -1000
NUM_PCA_MODES = 10
OUT_ROOT = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2"
DROPOUT_START = 0.5

# CT value range based on percentiles
CT_MIN = -145  # 1% percentile
CT_MAX = 2172  # 99% percentile

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def apply_gaussian_volume(volume, sigma=1):
    """Apply Gaussian smoothing to a 3D volume."""
    return gaussian_filter(volume, sigma=sigma, radius=3)

def compute_drr_mean_std(train_dirs):
    """Compute mean and standard deviation of DRR images from training data."""
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

    print("DRR Mean : " ,drr_mean)
    print("DRR std : " ,drr_std)

    return drr_mean, drr_std

# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================
def compute_voxel_metrics(ct_masked, pred_masked):
    """Compute various metrics for comparing aligned voxel sets."""
    # Normalize data to [0,1] range for better metric computation using percentile-based range
    ct_norm = (ct_masked - CT_MIN) / (CT_MAX - CT_MIN)
    pred_norm = (pred_masked - CT_MIN) / (CT_MAX - CT_MIN)
    
    # Normalized Cross-Correlation (unchanged as it's already good)
    ct_norm_std = (ct_masked - np.mean(ct_masked)) / np.std(ct_masked)
    pred_norm_std = (pred_masked - np.mean(pred_masked)) / np.std(pred_masked)
    ncc = np.mean(ct_norm_std * pred_norm_std)
    
    # Mutual Information with improved binning and normalization
    num_bins = min(50, int(np.sqrt(len(ct_masked))))  # Reduced number of bins
    hist_2d, x_edges, y_edges = np.histogram2d(ct_norm, pred_norm, 
                                             bins=num_bins, 
                                             range=[[0, 1], [0, 1]], 
                                             density=True)
    
    # Normalize the 2D histogram
    hist_2d = hist_2d / np.sum(hist_2d)
    
    # Compute marginal histograms
    hist_ct, _ = np.histogram(ct_norm, bins=x_edges, density=True)
    hist_pred, _ = np.histogram(pred_norm, bins=y_edges, density=True)
    
    # Normalize marginal histograms
    hist_ct = hist_ct / np.sum(hist_ct)
    hist_pred = hist_pred / np.sum(hist_pred)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_2d = hist_2d + eps
    hist_ct = hist_ct + eps
    hist_pred = hist_pred + eps
    
    # Compute MI using the joint and marginal distributions
    # Scale MI to be between 0 and 1
    mi = np.sum(hist_2d * np.log(hist_2d / (hist_ct[:, None] * hist_pred[None, :])))
    mi = mi / np.log(num_bins)  # Normalize by maximum possible MI
    
    # Earth Mover's Distance on normalized data, scaled back to HU range
    emd = wasserstein_distance(ct_norm, pred_norm) * (CT_MAX - CT_MIN)
    
    # Histogram Intersection with normalized data
    hist_intersection = np.sum(np.minimum(hist_ct, hist_pred))
    
    return {
        'ncc': ncc,
        'mi': mi,
        'emd': emd,
        'hist_intersection': hist_intersection
    }

def compute_basic_metrics(ct_masked, pred_masked):
    """Compute basic reconstruction metrics (MAE, RMSE)."""
    mae = np.mean(np.abs(ct_masked - pred_masked))
    rmse = np.sqrt(np.mean((ct_masked - pred_masked) ** 2))
    return mae, rmse

def compute_ssim_metric(ct_masked, pred_masked):
    """Compute SSIM metric on masked CT data."""
    # Normalize masked data for SSIM
    ct_masked_norm = (ct_masked - CT_MIN) / (CT_MAX - CT_MIN)
    pred_masked_norm = (pred_masked - CT_MIN) / (CT_MAX - CT_MIN)
    
    # Reshape to 2D for SSIM computation (using square root of length)
    side_length = int(np.sqrt(len(ct_masked_norm)))
    # Pad to nearest square
    pad_length = side_length * side_length - len(ct_masked_norm)
    if pad_length > 0:
        ct_masked_norm = np.pad(ct_masked_norm, (0, pad_length), mode='edge')
        pred_masked_norm = np.pad(pred_masked_norm, (0, pad_length), mode='edge')
    
    # Reshape to square
    ct_2d = ct_masked_norm[:side_length*side_length].reshape(side_length, side_length)
    pred_2d = pred_masked_norm[:side_length*side_length].reshape(side_length, side_length)
    
    # Compute SSIM on 2D data
    ssim = structural_similarity(ct_2d, pred_2d,
                               data_range=1.0,
                               gaussian_weights=True,
                               use_sample_covariance=True,
                               win_size=7)
    
    return ssim

def compute_psnr_metric(ct_masked, pred_masked):
    """Compute PSNR metric using percentile-based CT range."""
    mse = np.mean((ct_masked - pred_masked) ** 2)
    if mse > 0:
        # Use the percentile-based HU range
        psnr = 10 * np.log10((CT_MAX - CT_MIN)**2 / mse)
    else:
        psnr = float('inf')
    return psnr

def compute_all_metrics_for_sample(ct, pred_ct, volume_mask):
    """Compute all metrics for a single sample."""
    metrics = {}
    
    # Get masked regions
    ct_masked = ct[volume_mask]
    pred_masked = pred_ct[volume_mask]
    
    # Basic metrics
    mae, rmse = compute_basic_metrics(ct_masked, pred_masked)
    metrics['mae'] = mae
    metrics['rmse'] = rmse
    
    # Voxel-based metrics
    voxel_metrics = compute_voxel_metrics(ct_masked, pred_masked)
    metrics.update(voxel_metrics)
    
    # SSIM
    metrics['ssim'] = compute_ssim_metric(ct_masked, pred_masked)
    
    # PSNR
    metrics['psnr'] = compute_psnr_metric(ct_masked, pred_masked)
    
    return metrics

# ============================================================================
# MODEL RELATED FUNCTIONS
# ============================================================================
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

def load_models():
    """Load all trained PC models."""
    models = []
    for pc_idx in range(NUM_PCA_MODES):
        model = PC1Regressor().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(OUT_ROOT, "PC_models", f"model_pc{pc_idx+1}_sequential.pt"),
                                       map_location=DEVICE))
        model.eval()
        models.append(model)
    return models

def predict_pcs(drr, models):
    """
    Predict all PC coefficients for a single DRR pair.
    Args:
        drr: DRR images (AP, ML) shape (2, H, W)
        models: list of trained PC models
    Returns:
        pred_pcs: predicted PC coefficients
    """
    with torch.no_grad():
        pred_pcs = []
        for model in models:
            # Get mean prediction from MC dropout
            draws = torch.stack([model(drr[None].to(DEVICE)) for _ in range(20)])
            pred = draws.mean(0).cpu().numpy()
            pred_pcs.append(pred[0])  # Extract scalar value from prediction
    return np.array(pred_pcs)

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
def load_and_preprocess_ct(data_dir, gaussian_smoothing=False):
    """Load and optionally preprocess CT data."""
    ct = np.load(os.path.join(data_dir, "CT", "CT_reg.npy")).astype(np.float32)
    if gaussian_smoothing:
        ct = apply_gaussian_volume(ct, sigma=0.4)
    return ct

def load_and_normalize_drr(data_dir, drr_mean, drr_std):
    """Load and normalize DRR data."""
    ap = np.load(os.path.join(data_dir, "DRR", "AP.npy")).astype(np.float32)
    ml = np.load(os.path.join(data_dir, "DRR", "ML.npy")).astype(np.float32)
    rap = (ap - drr_mean) / drr_std
    rml = (ml - drr_mean) / drr_std
    return np.stack([rap, rml])

def reconstruct_ct_from_pcs(pca, scaler, pred_pcs, idx, n_components):
    """Reconstruct CT volume from predicted PC coefficients."""
    full_pred_pcs = np.zeros(NUM_PCA_MODES)
    full_pred_pcs[:n_components] = pred_pcs
    pred_vox = pca.inverse_transform(scaler.inverse_transform(full_pred_pcs.reshape(1,-1))).reshape(-1)
    pred_ct = np.zeros(CT_SHAPE, dtype=np.float32)
    pred_ct.flat[idx] = pred_vox
    return pred_ct

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================
def evaluate_reconstruction_performance(pca, scaler, idx, models, val_dirs, drr_mean, drr_std, 
                                       gaussian_smoothing=False):
    """
    Evaluate reconstruction performance using all 10 PCs across all validation samples.
    
    Returns:
        Dictionary containing lists of metrics for each sample
    """
    all_metrics = {
        'mae': [],
        'rmse': [],
        'ncc': [],
        'mi': [],
        'emd': [],
        'hist_intersection': [],
        'ssim': [],
        'psnr': []
    }

    print("Evaluating reconstruction performance with all 10 PCs...")
    
    for data_dir in tqdm(val_dirs, desc="Processing samples"):
        # Load and preprocess data
        ct = load_and_preprocess_ct(data_dir, gaussian_smoothing)
        drr = load_and_normalize_drr(data_dir, drr_mean, drr_std)
        
        # Predict and reconstruct using all 10 PCs
        pred_pcs = predict_pcs(torch.from_numpy(drr), models)
        pred_ct = reconstruct_ct_from_pcs(pca, scaler, pred_pcs, idx, NUM_PCA_MODES)
        
        # Create volume mask
        volume_mask = ct > THRESHOLD
        
        # Compute metrics only if there are enough valid pixels
        if np.sum(volume_mask) > 1000:
            metrics = compute_all_metrics_for_sample(ct, pred_ct, volume_mask)
            
            # Store all metrics
            all_metrics['mae'].append(metrics['mae'])
            all_metrics['rmse'].append(metrics['rmse'])
            all_metrics['ncc'].append(metrics['ncc'])
            all_metrics['mi'].append(metrics['mi'])
            all_metrics['emd'].append(metrics['emd'])
            all_metrics['hist_intersection'].append(metrics['hist_intersection'])
            all_metrics['ssim'].append(metrics['ssim'])
            all_metrics['psnr'].append(metrics['psnr'])
    
    # Print summary statistics
    print(f"\nProcessed {len(all_metrics['mae'])} samples")
    print(f"Mean MAE: {np.mean(all_metrics['mae']):.2f} ± {np.std(all_metrics['mae']):.2f} HU")
    print(f"Mean RMSE: {np.mean(all_metrics['rmse']):.2f} ± {np.std(all_metrics['rmse']):.2f} HU")
    print(f"Mean SSIM: {np.mean(all_metrics['ssim']):.4f} ± {np.std(all_metrics['ssim']):.4f}")
    print(f"Mean PSNR: {np.mean(all_metrics['psnr']):.2f} ± {np.std(all_metrics['psnr']):.2f} dB")
    print(f"Mean NCC: {np.mean(all_metrics['ncc']):.4f} ± {np.std(all_metrics['ncc']):.4f}")
    print(f"Mean MI: {np.mean(all_metrics['mi']):.4f} ± {np.std(all_metrics['mi']):.4f}")
    
    return all_metrics

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_performance_table(metrics_dict):
    """Display a formatted table of reconstruction performance metrics."""
    
    def compute_stats(values):
        """Compute mean, std, median, min, max for a list of values."""
        values = np.array(values)
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Compute statistics for all metrics (using 10 PCs)
    final_metrics = {
        'MAE (HU)': compute_stats(metrics_dict['mae']),
        'RMSE (HU)': compute_stats(metrics_dict['rmse']),
        'NCC': compute_stats(metrics_dict['ncc']),
        'MI': compute_stats(metrics_dict['mi']),
        'EMD': compute_stats(metrics_dict['emd']),
        'HI': compute_stats(metrics_dict['hist_intersection']),
        'SSIM': compute_stats(metrics_dict['ssim']),
        'PSNR (dB)': compute_stats(metrics_dict['psnr'])
    }
    
    # Print the formatted table
    print("\n" + "="*80)
    print("RECONSTRUCTION PERFORMANCE TABLE")
    print("="*80)
    print(f"{'Metric':<15} {'Mean ± Std':<20} {'Median':<10} {'Range':<20}")
    print("-"*80)
    
    for metric_name, stats in final_metrics.items():
        # Format mean ± std
        if metric_name in ['MAE (HU)', 'RMSE (HU)', 'EMD']:
            mean_std_str = f"{stats['mean']:.2f} ± {stats['std']:.2f}"
        else:
            mean_std_str = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
        
        # Format median and range (only for certain metrics)
        if metric_name in ['MAE (HU)', 'RMSE (HU)', 'SSIM', 'PSNR (dB)']:
            if metric_name in ['MAE (HU)', 'RMSE (HU)']:
                median_str = f"{stats['median']:.2f}"
                range_str = f"[{stats['min']:.2f}, {stats['max']:.2f}]"
            else:
                median_str = f"{stats['median']:.3f}"
                range_str = f"[{stats['min']:.3f}, {stats['max']:.3f}]"
        else:
            median_str = "-"
            range_str = "-"
        
        print(f"{metric_name:<15} {mean_std_str:<20} {median_str:<10} {range_str:<20}")
    
    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load data and models
    pca_path = os.path.join(OUT_ROOT, "pca.pkl")
    scaler_path = os.path.join(OUT_ROOT, "scaler.pkl")
    idx_path = os.path.join(OUT_ROOT, "idx.npy")
    splits_path = os.path.join(OUT_ROOT, "data_splitting_dir.pkl")
    
    pca = pickle.load(open(pca_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    idx = np.load(idx_path)
    models = load_models()
    splits = pickle.load(open(splits_path, "rb"))
    test_dirs = splits["test"]
    train_dirs = splits["train"]
    
    # Compute DRR normalization parameters
    drr_mean, drr_std = compute_drr_mean_std(train_dirs)
    
    # Evaluate reconstruction performance
    metrics = evaluate_reconstruction_performance(
        pca, scaler, idx, models, test_dirs, drr_mean, drr_std,
        gaussian_smoothing=False
    )
    
    # Display results
    display_performance_table(metrics)

if __name__ == "__main__":
    main()
