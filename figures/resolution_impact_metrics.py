import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import os
from scipy.ndimage import gaussian_filter

# ============================================================================
# PATHS AND PARAMETERS
# ============================================================================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CT_SHAPE = (64, 64, 256)
THRESHOLD = -1000
NUM_PCA_MODES = 10
OUT_ROOT = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2"
DROPOUT_START = 0.5

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
# COMPUTE FUNCTIONS
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

def compute_pred_and_baseline_mae(
    test_dirs, models, pca, scaler, idx, ct_shape, threshold, device, drr_mean, drr_std):
   
    for resolution in [1,2,4,8]:
        for d in tqdm(test_dirs, desc="Test set"):
            # Load GT CT
            ct = np.load(f"{d}/CT/CT_reg.npy").astype(np.float32)
            #ct_blur = apply_gaussian_volume(ct, sigma=0.4)
            # Load DRRs
            ap = np.load(f"{d}/DRR/AP.npy").astype(np.float32)
            ml = np.load(f"{d}/DRR/ML.npy").astype(np.float32)
            rap = (ap - drr_mean) / drr_std
            rml = (ml - drr_mean) / drr_std
            drr = np.stack([rap, rml])
            drr_tensor = torch.from_numpy(drr).unsqueeze(0).to(device)
            
            # Predict PCs and reconstruct CT
            with torch.no_grad():
                pred_pcs = []
                for model in models:
                    draws = torch.stack([model(drr_tensor) for _ in range(20)])
                    pred = draws.mean(0).cpu().numpy()
                    pred_pcs.append(pred[0])
            pred_pcs = np.array(pred_pcs)
            pred_vox = pca.inverse_transform(scaler.inverse_transform(pred_pcs.reshape(1, -1))).reshape(-1)
            pred_ct = np.zeros(ct_shape, dtype=np.float32)
            pred_ct.flat[idx] = pred_vox
            
            # Compute overall mask
            mask = ct > threshold
            
            # Resolution impact (reduce the resolution of the CT by a factor of resolution)
            ct_reduced = ct[::resolution,::resolution,::resolution]
            pred_ct_reduced = pred_ct[::resolution,::resolution,::resolution]
            mask_reduced = ct_reduced > threshold
            mae_reduced = np.mean(np.abs(ct_reduced[mask_reduced] - pred_ct_reduced[mask_reduced]))
            print(f"Resolution {resolution} MAE: {mae_reduced:.2f} HU")
            
            
            # Compute slice-wise MAE
            for i in range(n_slices):
                slice_mask = ct[:,:,i] > threshold
                if np.any(slice_mask):  # Only compute if there are bone voxels in the slice
                    mae_slice_pred = np.mean(np.abs(ct_blur[:,:,i][slice_mask] - pred_ct[:,:,i][slice_mask]))
                    mae_slice_baseline = np.mean(np.abs(ct[:,:,i][slice_mask] - mean_ct[:,:,i][slice_mask]))
                    slice_mae_samples[i].append(mae_slice_pred)
                    slice_mae_baseline_samples[i].append(mae_slice_baseline)

    # Convert overall MAE to numpy arrays
    pred_mae = np.array(pred_mae)
    baseline_mae = np.array(baseline_mae)
    
    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"MAE baseline: {np.mean(baseline_mae):.2f} ± {np.std(baseline_mae):.2f} HU")
    print(f"MAE predicted: {np.mean(pred_mae):.2f} ± {np.std(pred_mae):.2f} HU")
    improvement = ((np.mean(baseline_mae) - np.mean(pred_mae)) / np.mean(baseline_mae)) * 100
    print(f"Improvement: {improvement:.2f}%")

    # Compute slice-wise statistics
    slice_mae_mean = np.zeros(n_slices)
    slice_mae_max = np.zeros(n_slices)
    slice_mae_min = np.zeros(n_slices)
    slice_mae_baseline_mean = np.zeros(n_slices)
    slice_mae_baseline_max = np.zeros(n_slices)
    slice_mae_baseline_min = np.zeros(n_slices)

    for i in range(n_slices):
        if len(slice_mae_samples[i]) > 0:
            slice_mae_mean[i] = np.mean(slice_mae_samples[i])
            slice_mae_max[i] = np.max(slice_mae_samples[i]) 
            slice_mae_min[i] = np.min(slice_mae_samples[i])
        if len(slice_mae_baseline_samples[i]) > 0:
            slice_mae_baseline_mean[i] = np.mean(slice_mae_baseline_samples[i])
            slice_mae_baseline_max[i] = np.max(slice_mae_baseline_samples[i])
            slice_mae_baseline_min[i] = np.min(slice_mae_baseline_samples[i])

    metrics = {
               
               "pred_mae": pred_mae,
               "baseline_mae":baseline_mae,
               "slice_mae_mean":slice_mae_mean,
               "slice_mae_max":slice_mae_max,
               "slice_mae_min": slice_mae_min, 
               "slice_mae_baseline_mean":slice_mae_baseline_mean,
               "slice_mae_baseline_max":slice_mae_baseline_max, 
               "slice_mae_baseline_min": slice_mae_baseline_min

               }
  
    return metrics

