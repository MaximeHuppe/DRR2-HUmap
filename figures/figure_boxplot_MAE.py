import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
import pickle
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
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
NUM_PCA_MODES = 10
CT_SHAPE = (64, 64, 256)
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

def calculate_and_plot_sample_mae(pca, 
                                  scaler, 
                                  idx, 
                                  models, 
                                  val_dirs,
                                  drr_mean,
                                  drr_std,
                                  n_components=10, 
                                  gaussian_smoothing=False):
    """
    Calculate and plot MAE for all samples across different numbers of components.
    
    Args:
        pca: Fitted PCA model
        scaler: Fitted StandardScaler
        idx: Voxel indices
        models: List of trained models
        val_dirs: List of validation directories
        n_components: Maximum number of components to evaluate
        save_path: Optional path to save the plot
    """
    all_sample_mae = []  # List of lists: each sublist is MAE for all samples at a given n_components
    

    for n_components in range(1, NUM_PCA_MODES + 1):
        sample_mae = []
        
        for d in tqdm(val_dirs, desc="Computing : "):
            ct = np.load(os.path.join(d, "CT", "CT_reg.npy")).astype(np.float32)
            
            if gaussian_smoothing:
                ct = apply_gaussian_volume(ct, sigma=0.4)

            vox = ct.flatten()[idx].reshape(1,-1)
            true_pcs = scaler.transform(pca.transform(vox))[0]

            ap = np.load(os.path.join(d, "DRR", "AP.npy")).astype(np.float32)
            ml = np.load(os.path.join(d, "DRR", "ML.npy")).astype(np.float32)
            rap = (ap - drr_mean)/drr_std
            rml = (ml - drr_mean)/drr_std
            drr = np.stack([rap, rml])
            pred_pcs = predict_pcs(torch.from_numpy(drr), models[:n_components])
            full_pred_pcs = np.zeros(NUM_PCA_MODES)
            full_pred_pcs[:n_components] = pred_pcs
            pred_vox = pca.inverse_transform(scaler.inverse_transform(full_pred_pcs.reshape(1,-1))).reshape(-1)
            pred_ct = np.zeros(CT_SHAPE, dtype=np.float32)
            pred_ct.flat[idx] = pred_vox

            mask = ct > THRESHOLD
            mae = np.mean(np.abs(ct[mask] - pred_ct[mask]))
            sample_mae.append(mae)

        print(f"Mean MAE with {n_components} component(s): {np.mean(sample_mae):.4f} ± {np.std(sample_mae):.4f}")
        
        all_sample_mae.append(sample_mae)

    return all_sample_mae


# ============================================================================
# PLOT
# ============================================================================

def plot_boxplot_mae(train_dirs, test_dirs, sample_mae_values):
    mean_volume = []

    for d in tqdm(train_dirs, desc="Computing mean volume"):
        ct = np.load(os.path.join(d, "CT", "CT_reg.npy")).astype(np.float32)
        mean_volume.append(ct)

    mean_volume = np.stack(mean_volume)
    mean_volume = np.mean(mean_volume, axis=0)

    mae_values_mean = []
    for d in tqdm(test_dirs, desc="Computing MAE"):
        ct = np.load(os.path.join(d, "CT", "CT_reg.npy")).astype(np.float32)
        mask = ct > THRESHOLD
        ct = ct[mask]
        mean_volume_masked = mean_volume[mask]
        mae = np.mean(np.abs(ct - mean_volume_masked))
        mae_values_mean.append(mae)

    box_plot_mae = [mae_values_mean, 
                    sample_mae_values[0], 
                    sample_mae_values[1], 
                    sample_mae_values[2], 
                    sample_mae_values[3], 
                    sample_mae_values[4], 
                    sample_mae_values[5], 
                    sample_mae_values[6], 
                    sample_mae_values[7], 
                    sample_mae_values[8], 
                    sample_mae_values[9]]

    # Fix box_plot_mae[10] by deleting every n+1 value so its length matches box_plot_mae[9]
    if len(box_plot_mae[10]) > len(box_plot_mae[9]):
        n = len(box_plot_mae[9])
        # Remove every (n+1)th value from box_plot_mae[10] to match length
        box_plot_mae[10] = [v for i, v in enumerate(box_plot_mae[10]) if (i % (len(box_plot_mae[10]) // n)) != 0][:n]

    # Lighter blue for PCA boxes, white for mean volume
    light_blue = mpl.cm.Blues(0.4)
    box_colors = ["white"] + [light_blue] * (len(box_plot_mae) - 1)

    plt.figure(figsize=(20, 15), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # Create box plot
    box = plt.boxplot(
        box_plot_mae,
        positions=range(len(box_plot_mae)),
        showmeans=False,
        meanline=False,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2.5),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=7, alpha=0.5),
        boxprops=dict(linewidth=2.0, color='black'),
        whiskerprops=dict(linewidth=2.0, color='black'),
        capprops=dict(linewidth=2.0, color='black')
    )
    fontsize = 30
    # Color the boxes
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)

    # Highlight the box with the lowest mean MAE (subtle marker)
    means = [np.mean(data) for data in box_plot_mae]
    min_idx = int(np.argmin(means))
    ax.scatter(min_idx, means[min_idx], color='black', marker='*', s=120, zorder=5, label='Lowest Mean MAE')

    # Customize x-axis labels
    x_labels = ['µ'] + [f'PC {i+1}' for i in range(len(box_plot_mae) - 1)]
    plt.xticks(range(len(box_plot_mae)), x_labels, rotation=30, ha='right', fontsize=fontsize, color='black')
    plt.yticks(fontsize=fontsize*0.8)
    # Add labels and title (all in black)
    #plt.xlabel('Reconstruction Method', fontsize=15, labelpad=12, color='black')
    plt.ylabel('Mean Absolute Error (HU)', fontsize=fontsize, labelpad=22, color='black')
    #plt.title('MAE vs Number of PCA Modes', fontsize=18, pad=22, color='black')

    # Improved grid
    ymin, ymax = ax.get_ylim()
    plt.grid(True, linestyle='--', alpha=0.18, axis='y')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend for the star marker (upper right, text in black)
    #legend1 = ax.legend(loc='upper right', framealpha=0.97, fontsize=12, labelcolor='black', facecolor='white', edgecolor='black')
    #ax.add_artist(legend1)

    # --- Secondary y-axis: cumulative MAE improvement ---
    mae_means = np.array(means)
    cumulative_improvement = mae_means[0] - mae_means  # Starts at 0, increases

    ax2 = ax.twinx()
    ax2.plot(range(len(box_plot_mae)), cumulative_improvement, color='orange', marker='o',linewidth=3, label='Cumulative MAE Improvement')
    ax2.set_ylabel('Cumulative MAE Improvement (HU)', fontsize=fontsize, labelpad=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(bottom=0)
    plt.yticks(fontsize=fontsize*0.8)

    # Add a legend for the secondary axis (upper left, text in black)
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc='upper left', framealpha=0.97, fontsize=fontsize, labelcolor='black', facecolor='white', edgecolor='black')

    plt.tight_layout()
    #plt.savefig(os.path.join(OUT_ROOT, 'mae_comparison_boxplot_scientific.png'), dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    pca_path = os.path.join(OUT_ROOT, "pca.pkl")
    scaler_path = os.path.join(OUT_ROOT, "scaler.pkl")
    pca = pickle.load(open(pca_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    idx_path = os.path.join(OUT_ROOT, "idx.npy")
    idx = np.load(idx_path)
    models = load_models()
    drr_mean, drr_std = compute_drr_mean_std(splits["train"])
    all_sample_mae = calculate_and_plot_sample_mae(pca, scaler, idx, models, splits["test"], drr_mean, drr_std)
    plot_boxplot_mae(splits["train"], splits["test"], all_sample_mae)


if __name__ == "__main__":
    main()