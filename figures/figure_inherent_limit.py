import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# ============================================================================
# PATHS AND PARAMETERS
# ============================================================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OUT_ROOT = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2"
SPLIT_PATH = os.path.join(OUT_ROOT, "data_splitting_dir.pkl")
splits = pickle.load(open(SPLIT_PATH, "rb"))
train_dirs = splits["train"]
test_dirs = splits["test"]

THRESHOLD = -1000
DROPOUT_START = 0.5
NUM_PCA_MODES = 10
CT_SHAPE = (64, 64, 256)

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

    print("DRR Mean : " ,drr_mean)
    print("DRR std : " ,drr_std)

    return drr_mean, drr_std

def evaluate_progressive_reconstruction(pca, scaler, idx, models, val_dirs, drr_mean, drr_std):
    """
    Evaluate reconstruction quality using progressive number of components.
    Also computes inherent PCA performance for comparison.
    Args:
        pca: PCA model
        scaler: StandardScaler
        idx: voxel indices
        models: list of trained models
        val_dirs: validation directories
    Returns:
        pred_mae_values: list of MAE values for predicted reconstructions
        inherent_mae_values: list of MAE values for inherent PCA performance
    """
    pred_mae_values = []
    inherent_mae_values = []

    print("\nEvaluating progressive reconstructions...")
    for n_components in range(1, NUM_PCA_MODES + 1):
        print(f"\nUsing {n_components} component(s)...")
        all_true_cts = []
        all_pred_cts = []
        all_inherent_cts = []
        
        for d in tqdm(val_dirs, desc=f"Processing cases (n={n_components})"):
            ct = np.load(os.path.join(d, "CT", "CT_reg.npy")).astype(np.float32)
            vox = ct.flatten()[idx].reshape(1,-1)
            true_pcs = scaler.transform(pca.transform(vox))[0]
            

            ap = np.load(os.path.join(d, "DRR", "AP.npy")).astype(np.float32)
            ml = np.load(os.path.join(d, "DRR", "ML.npy")).astype(np.float32)
            rap = (ap - drr_mean)/drr_std
            rml = (ml - drr_mean)/drr_std
            #drr = np.stack([rap, rml, cap, cml])
            drr = np.stack([rap, rml])
            pred_pcs = predict_pcs(torch.from_numpy(drr), models[:n_components])
            full_pred_pcs = np.zeros(NUM_PCA_MODES)
            full_pred_pcs[:n_components] = pred_pcs
            pred_vox = pca.inverse_transform(scaler.inverse_transform(full_pred_pcs.reshape(1,-1))).reshape(-1)
            pred_ct = np.zeros(CT_SHAPE, dtype=np.float32)
            pred_ct.flat[idx] = pred_vox
            inherent_pcs = np.zeros(NUM_PCA_MODES)
            inherent_pcs[:n_components] = true_pcs[:n_components]
            inherent_vox = pca.inverse_transform(scaler.inverse_transform(inherent_pcs.reshape(1,-1))).reshape(-1)
            inherent_ct = np.zeros(CT_SHAPE, dtype=np.float32)
            inherent_ct.flat[idx] = inherent_vox
            all_true_cts.append(ct)
            all_pred_cts.append(pred_ct)
            all_inherent_cts.append(inherent_ct)
        
        # Convert to arrays
        all_true_cts = np.stack(all_true_cts)
        all_pred_cts = np.stack(all_pred_cts)
        all_inherent_cts = np.stack(all_inherent_cts)
        
        # Compute MAE only on voxels above threshold
        mask = all_true_cts > THRESHOLD
        pred_mae = np.mean(np.abs(all_true_cts[mask] - all_pred_cts[mask]))
        inherent_mae = np.mean(np.abs(all_true_cts[mask] - all_inherent_cts[mask]))
        
        pred_mae_values.append(pred_mae)
        inherent_mae_values.append(inherent_mae)
        
        print(f"MAE with {n_components} component(s):")
        print(f"  - Predicted: {pred_mae:.4f}")
        print(f"  - Inherent PCA: {inherent_mae:.4f}")
    
    return pred_mae_values, inherent_mae_values

# ============================================================================
# PLOT
# ============================================================================

def plot_volumetric_mae(pred_mae_values, inherent_mae_values, save_path=None):
    """
    Plot volumetric MAE vs number of components for both predicted and inherent performance.
    Args:
        pred_mae_values: list of MAE values for predicted reconstructions
        inherent_mae_values: list of MAE values for inherent PCA performance
        save_path: optional path to save the plot
    """
    plt.figure(figsize=(12, 7))
    components = range(1, len(pred_mae_values) + 1)
    
    # Plot both lines
    plt.plot(components, pred_mae_values, 'bo-', linewidth=2, markersize=8, label='Predicted')
    plt.plot(components, inherent_mae_values, 'ro-', linewidth=2, markersize=8, label='Inherent PCA')
    
    plt.xlabel('Number of PCA Components', fontsize=12)
    plt.ylabel('Volumetric MAE', fontsize=12)
    plt.title('Reconstruction Error vs Number of Components', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add value labels on points
    for i, (pred_mae, inherent_mae) in enumerate(zip(pred_mae_values, inherent_mae_values)):
        plt.text(components[i], pred_mae, f'{pred_mae:.1f}', 
                ha='center', va='bottom', fontsize=10, color='blue')
        plt.text(components[i], inherent_mae, f'{inherent_mae:.1f}', 
                ha='center', va='top', fontsize=10, color='red')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    pca = pickle.load(open(os.path.join(OUT_ROOT, "pca.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(OUT_ROOT, "scaler.pkl"), "rb"))
    idx_path = os.path.join(OUT_ROOT, "idx.npy")
    idx = np.load(idx_path)
    models = load_models()
    drr_mean, drr_std = compute_drr_mean_std(splits["train"])
    pred_mae_values, inherent_mae_values = evaluate_progressive_reconstruction(pca=pca, 
                                                                           scaler=scaler, 
                                                                           idx=idx, 
                                                                           models=models, 
                                                                           val_dirs=test_dirs,
                                                                           drr_mean=drr_mean,
                                                                           drr_std=drr_std)

    plot_volumetric_mae(pred_mae_values=pred_mae_values, 
                    inherent_mae_values=inherent_mae_values, 
                    save_path=None)

if __name__ == "__main__":
    main()