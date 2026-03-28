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

# Reconstruction parameters 
CT_SHAPE = (64, 64, 256)
THRESHOLD = -1000

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

def apply_gaussian_volume(volume, sigma=1):
    """Apply Gaussian smoothing to a 3D volume."""
    return gaussian_filter(volume, sigma=sigma, radius=3)

def reconstruct_ct_from_pcs(test_dir, 
                       test_idx,  
                       blur = False, 
                       drr_mean = None, 
                       drr_std = None, 
                       device = DEVICE, 
                       models = None, 
                       pca = None, 
                       scaler = None, 
                       idx = None, 
                       ct_shape = CT_SHAPE):
    """
    Plots a 2D heatmap of absolute error for a given slice and plane.
    plane: 'axial', 'coronal', or 'sagittal'
    """
    ct_true = np.load(f"{test_dir[test_idx]}/CT/CT_reg.npy").astype(np.float32)
    
    if blur == True: 
        ct_true = apply_gaussian_volume(ct_true, sigma=0.4)

    # Load DRRs
    ap = np.load(f"{test_dir[test_idx]}/DRR/AP.npy").astype(np.float32)
    ml = np.load(f"{test_dir[test_idx]}/DRR/ML.npy").astype(np.float32)

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
    pred_ct = np.ones(ct_shape, dtype=np.float32)*-1000
    pred_ct.flat[idx] = pred_vox
    pred_ct = pred_ct.reshape(ct_shape)


    return ct_true, pred_ct
# ============================================================================
# PLOT FUNCTIONS
# ============================================================================
def plot_case_comparison(ct_true, ct_pred, slice_idx, plane='axial', save_path=None, dpi=300, 
                        fixed_hist_limits=None):
    """
    Creates a publication-ready comparison plot of ground truth, prediction, and error analysis.
    
    Args:
        ct_true: Ground truth CT volume
        ct_pred: Predicted CT volume
        slice_idx: Slice index to visualize
        plane: Viewing plane ('axial', 'coronal', or 'sagittal')
        save_path: Optional path to save the figure
        dpi: Resolution for saved figure
        fixed_hist_limits: Tuple (x_max, y_max) for fixed histogram scaling
    """
    # Set publication-ready style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Extract slice based on plane
    if plane == 'axial':
        gt = ct_true[:, :, slice_idx]
        pred = ct_pred[:, :, slice_idx]
        plane_label = f'Axial (z={slice_idx})'
    elif plane == 'coronal':
        gt = ct_true[:, slice_idx, :]
        pred = ct_pred[:, slice_idx, :]
        plane_label = f'Coronal (y={slice_idx})'
    elif plane == 'sagittal':
        gt = ct_true[slice_idx, :, :]
        pred = ct_pred[slice_idx, :, :]
        plane_label = f'Sagittal (x={slice_idx})'
    else:
        raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'")
    
    # Compute error metrics
    error = np.abs(gt - pred)
    error_signed = gt - pred
    error_signed = np.where(error_signed == 0, np.nan, error_signed)
    mask_slice = gt > -1000
    
    # Calculate statistics
    mean_error = np.mean(error[mask_slice])
    error_values = error[mask_slice].flatten()
    percentile_80 = np.percentile(error_values, 80)
    below_150_pct = (np.sum(error_values < 150) / len(error_values)) * 100
    
    print(f'Mean Error: {mean_error:.2f} HU')
    print(f"80th percentile error: {percentile_80:.2f} HU")
    print(f"{below_150_pct:.1f}% of errors below 150 HU")

    # Create figure with clean 2x2 layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.25, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # CT display parameters
    vmin, vmax = -1000, 1000  # Standard CT window for bone
    
    # Ground Truth (top left)
    im1 = axs[0, 0].imshow(gt, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
    axs[0, 0].set_title('Ground Truth', fontweight='bold', pad=10)
    axs[0, 0].axis('off')
    
    # Reconstruction (top right)
    im2 = axs[0, 1].imshow(pred, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
    axs[0, 1].set_title('Reconstruction', fontweight='bold', pad=10)
    axs[0, 1].axis('off')
    
    # Error Map (bottom left)
    im3 = axs[1, 0].imshow(error_signed, cmap='RdBu_r', vmin=-400, vmax=400, aspect='equal')
    axs[1, 0].set_title('Error Map', fontweight='bold', pad=10)
    axs[1, 0].axis('off')
    
    # Add colorbar for error map
    pos3 = axs[1, 0].get_position()
    cbar_ax = fig.add_axes([pos3.x0 - 0.05, pos3.y0, 0.015, pos3.height])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Error (HU)', rotation=90, labelpad=12, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    # Error Distribution Histogram (bottom right)
    ax_hist = axs[1, 1]
    
    # Create histogram with better styling
    n, bins, patches = ax_hist.hist(error_values, bins=50, alpha=0.7, color='#2E86AB', 
                                   edgecolor='black', linewidth=0.5, density=True)
    
    # Add statistical annotations
    ax_hist.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.1f} HU')
    ax_hist.axvline(percentile_80, color='orange', linestyle='--', linewidth=2, 
                   label=f'80th percentile: {percentile_80:.1f} HU')
    ax_hist.axvline(150, color='green', linestyle='--', linewidth=2, 
                   label=f'{below_150_pct:.1f}% < 150 HU')
    
    ax_hist.set_xlabel('Absolute Error (HU)', fontweight='bold')
    ax_hist.set_ylabel('Probability Density', fontweight='bold')
    ax_hist.set_title('Error Distribution', fontweight='bold', pad=10)
    ax_hist.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_hist.legend(frameon=True, fancybox=True, shadow=True, fontsize=9, loc='upper right')
    
    # Set histogram limits - use fixed limits if provided, otherwise calculate
    if fixed_hist_limits:
        x_max, y_max = fixed_hist_limits
        ax_hist.set_xlim(0, x_max)
        ax_hist.set_ylim(0, y_max)
    else:
        ax_hist.set_xlim(0, min(500, np.percentile(error_values, 99)))
    
    # Add overall figure title
    fig.suptitle(f'CT Reconstruction Analysis - {plane_label}', 
                fontsize=16, fontweight='bold', y=0.97)
    
    # Add text box with key metrics
    textstr = f'MAE: {mean_error:.1f} HU\nP80: {percentile_80:.1f} HU\n<150 HU: {below_150_pct:.1f}%'
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax_hist.text(0.98, 0.98, textstr, transform=ax_hist.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Reset matplotlib parameters to default
    plt.rcdefaults()


def create_tiff_sequence(ct_true, ct_pred, output_dir, plane='axial', dpi=150):
    """
    Creates an optimized sequence of TIFF images for all slices with consistent scaling.
    
    Args:
        ct_true: Ground truth CT volume
        ct_pred: Predicted CT volume  
        output_dir: Directory to save TIFF sequence
        plane: Viewing plane ('axial', 'coronal', or 'sagittal')
        dpi: Resolution for saved figures (reduced default for speed)
    """
    import os
    from tqdm import tqdm
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine slice parameters based on plane
    if plane == 'axial':
        n_slices = ct_true.shape[2]
        def get_slice(idx): return ct_true[:, :, idx], ct_pred[:, :, idx]
    elif plane == 'coronal':
        n_slices = ct_true.shape[1]
        def get_slice(idx): return ct_true[:, idx, :], ct_pred[:, idx, :]
    elif plane == 'sagittal':
        n_slices = ct_true.shape[0]
        def get_slice(idx): return ct_true[idx, :, :], ct_pred[idx, :, :]
    else:
        raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'")
    
    print(f"Processing {n_slices} {plane} slices...")
    
    # Fast computation of global statistics using vectorized operations
    print("Computing global error statistics...")
    
    # Sample every 5th slice for statistics to speed up
    sample_indices = range(0, n_slices, max(1, n_slices // 50))  # Sample ~50 slices
    all_errors = []
    max_density = 0
    
    for i in tqdm(sample_indices, desc="Sampling slices for statistics"):
        gt_slice, pred_slice = get_slice(i)
        mask = gt_slice > -1000
        
        if np.sum(mask) > 100:  # Only process slices with sufficient tissue
            error = np.abs(gt_slice[mask] - pred_slice[mask])
            all_errors.extend(error.flatten())
            
            # Quick histogram for max density
            if len(error) > 10:
                hist_vals, _ = np.histogram(error, bins=30, density=True)
                max_density = max(max_density, np.max(hist_vals))
    
    # Set global limits
    global_error_max = min(400, np.percentile(all_errors, 98)) if all_errors else 300
    global_density_max = max_density * 1.2 if max_density > 0 else 0.01
    fixed_limits = (global_error_max, global_density_max)
    
    print(f"Global limits: error_max={global_error_max:.1f}, density_max={global_density_max:.4f}")
    
    # Set optimized plotting parameters
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight'
    })
    
    # Generate TIFF sequence with optimized plotting
    print("Generating TIFF sequence...")
    plt.ioff()  # Turn off interactive plotting
    
    valid_slices = 0
    for i in tqdm(range(n_slices), desc=f"Creating {plane} TIFFs"):
        gt_slice, pred_slice = get_slice(i)
        mask = gt_slice > -1000
        
        # Skip slices with insufficient tissue
        if np.sum(mask) < 100:
            continue
        
        # Generate optimized plot
        filename = f"{plane}_slice_{i:03d}.tiff"
        save_path = os.path.join(output_dir, filename)
        
        success = _create_optimized_slice_plot(gt_slice, pred_slice, i, plane, 
                                             save_path, fixed_limits)
        if success:
            valid_slices += 1
        
        # Clear memory every 10 slices
        if i % 10 == 0:
            plt.close('all')
    
    plt.ion()  # Turn interactive plotting back on
    plt.rcdefaults()  # Reset parameters
    
    print(f"\nTIFF sequence completed!")
    print(f"Generated {valid_slices} TIFF files in: {output_dir}")
    
    return output_dir


def _create_optimized_slice_plot(gt_slice, pred_slice, slice_idx, plane, save_path, fixed_limits):
    """Optimized plotting function for batch TIFF generation."""
    try:
        # Compute error metrics
        error = np.abs(gt_slice - pred_slice)
        error_signed = gt_slice - pred_slice
        mask = gt_slice > -1000
        error_values = error[mask].flatten()
        
        if len(error_values) < 10:
            return False
        
        # Calculate quick statistics
        mean_error = np.mean(error_values)
        percentile_80 = np.percentile(error_values, 80)
        below_150_pct = (np.sum(error_values < 150) / len(error_values)) * 100
        
        # Create optimized figure
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Smaller figure
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        
        # CT display parameters
        vmin, vmax = -1000, 1000
        
        # Ground Truth
        axs[0, 0].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        axs[0, 0].set_title('Ground Truth', fontweight='bold')
        axs[0, 0].axis('off')
        
        # Reconstruction
        axs[0, 1].imshow(pred_slice, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        axs[0, 1].set_title('Reconstruction', fontweight='bold')
        axs[0, 1].axis('off')
        
        # Error Map
        im = axs[1, 0].imshow(error_signed, cmap='RdBu_r', vmin=-300, vmax=300, aspect='equal')
        axs[1, 0].set_title('Error Map', fontweight='bold')
        axs[1, 0].axis('off')
        
        # Simple colorbar
        plt.colorbar(im, ax=axs[1, 0], shrink=0.8, label='Error (HU)')
        
        # Optimized histogram
        ax_hist = axs[1, 1]
        ax_hist.hist(error_values, bins=30, alpha=0.7, color='#2E86AB', density=True)
        
        # Add key statistics only
        ax_hist.axvline(mean_error, color='red', linestyle='--', 
                       label=f'Mean: {mean_error:.1f}')
        ax_hist.axvline(150, color='green', linestyle='--', 
                       label=f'{below_150_pct:.1f}% < 150')
        
        ax_hist.set_xlabel('Absolute Error (HU)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Error Distribution')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)
        
        # Apply fixed limits
        ax_hist.set_xlim(0, fixed_limits[0])
        ax_hist.set_ylim(0, fixed_limits[1])
        
        # Add figure title
        plane_label = f'{plane.title()} (slice {slice_idx})'
        fig.suptitle(f'CT Analysis - {plane_label}', fontsize=14, fontweight='bold')
        
        # Save figure
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"Error processing slice {slice_idx}: {e}")
        plt.close('all')
        return False


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main execution function."""
    # Load data and models
    pca_path = os.path.join(OUT_ROOT, "pca.pkl")
    scaler_path = os.path.join(OUT_ROOT, "scaler.pkl")
    idx_path = os.path.join(OUT_ROOT, "idx.npy")
    splits_path = os.path.join(OUT_ROOT, "data_splitting_dir.pkl")

    splits = pickle.load(open(splits_path, "rb"))
    test_dirs = splits["test"]
    drr_mean, drr_std = compute_drr_mean_std(splits["train"])
    models = load_models()
    pca = pickle.load(open(pca_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    idx = np.load(idx_path)

    # Reconstruct CT
    ground_truth_CT, predicted_CT= reconstruct_ct_from_pcs(test_dirs, 
                                      test_idx = 0, 
                                      blur = False, 
                                      drr_mean = drr_mean, 
                                      drr_std = drr_std, 
                                      device = DEVICE, 
                                      models = models, 
                                      pca = pca, 
                                      scaler = scaler, 
                                      idx = idx, 
                                      ct_shape = CT_SHAPE)

    # Plot single case comparison with option to save
    save_path = os.path.join(OUT_ROOT, "reconstruction_analysis_figure.png")
    plot_case_comparison(ground_truth_CT, predicted_CT, slice_idx=242, plane='axial', 
                        save_path=save_path, dpi=600)
    
    # Optional: Create TIFF sequence for all slices (uncomment to generate)
    tiff_output_dir = os.path.join(OUT_ROOT, "tiff_sequence_axial")
    create_tiff_sequence(ground_truth_CT, predicted_CT, tiff_output_dir, 
                         plane='axial', dpi=300)

if __name__ == "__main__":
    main()