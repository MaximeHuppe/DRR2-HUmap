import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# ============================================================================
# CONFIGURATION AND CONSTANTS
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

def predict_single_pc(drr, model, n_samples=20):
    """
    Predict a single PC coefficient with MC dropout.
    Args:
        drr: DRR images (AP, ML) shape (2, H, W)
        model: trained PC model
        n_samples: number of MC dropout samples
    Returns:
        predicted PC coefficient
    """
    with torch.no_grad():
        draws = torch.stack([model(drr[None].to(DEVICE)) for _ in range(n_samples)])
        pred = draws.mean(0).cpu().numpy()
        return pred[0]  # Extract scalar value

# ============================================================================
# UTILITY FUNCTIONS
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

# ============================================================================
# PC PERFORMANCE ANALYSIS
# ============================================================================
def evaluate_pc_regression_performance(pca, scaler, idx, models, test_dirs, drr_mean, drr_std):
    """
    Evaluate the regression performance for each individual PC.
    
    Returns:
        Dictionary with performance metrics for each PC
    """
    print("Evaluating PC regression performance...")
    
    # Initialize storage for true and predicted PCs
    true_pcs_all = []
    pred_pcs_all = [[] for _ in range(NUM_PCA_MODES)]
    
    # Process each test sample
    for data_dir in tqdm(test_dirs, desc="Processing test samples"):
        # Load ground truth CT
        ct = np.load(os.path.join(data_dir, "CT", "CT_reg.npy")).astype(np.float32)
        
        # Compute true PCs
        vox = ct.flatten()[idx].reshape(1, -1)
        true_pcs = scaler.transform(pca.transform(vox))[0]
        true_pcs_all.append(true_pcs)
        
        # Load and normalize DRRs
        ap = np.load(os.path.join(data_dir, "DRR", "AP.npy")).astype(np.float32)
        ml = np.load(os.path.join(data_dir, "DRR", "ML.npy")).astype(np.float32)
        rap = (ap - drr_mean) / drr_std
        rml = (ml - drr_mean) / drr_std
        drr = np.stack([rap, rml])
        drr_tensor = torch.from_numpy(drr)
        
        # Predict each PC individually
        for pc_idx, model in enumerate(models):
            pred_pc = predict_single_pc(drr_tensor, model)
            pred_pcs_all[pc_idx].append(pred_pc)
    
    # Convert to numpy arrays
    true_pcs_all = np.array(true_pcs_all)  # Shape: (n_samples, n_components)
    pred_pcs_all = [np.array(pred_list) for pred_list in pred_pcs_all]  # List of arrays
    
    # Compute metrics for each PC
    results = {}
    explained_variances = pca.explained_variance_ratio_
    
    for pc_idx in range(NUM_PCA_MODES):
        true_pc = true_pcs_all[:, pc_idx]
        pred_pc = pred_pcs_all[pc_idx]
        
        # Compute metrics
        mae_pc = mean_absolute_error(true_pc, pred_pc)
        r2_pc = r2_score(true_pc, pred_pc)
        explained_var = explained_variances[pc_idx]
        
        results[f'PC{pc_idx+1}'] = {
            'explained_variance': explained_var,
            'mae_pc_space': mae_pc,
            'r2_score': r2_pc
        }
        
        print(f"PC{pc_idx+1}: Explained Var = {explained_var:.3f}, "
              f"MAE = {mae_pc:.2f}, R² = {r2_pc:.3f}")
    
    return results

def create_pc_performance_table(results):
    """
    Create and display a formatted table of PC regression performance.
    
    Args:
        results: Dictionary with performance metrics for each PC
    """
    print("\n" + "="*60)
    print("PC REGRESSION PERFORMANCE TABLE")
    print("="*60)
    
    # Create formatted table
    print(f"{'Component':<12} {'Explained Var':<15} {'MAE (PC space)':<18} {'R² Score':<10}")
    print("-"*60)
    
    for pc_name, metrics in results.items():
        explained_var = metrics['explained_variance']
        mae_pc = metrics['mae_pc_space']
        r2_score_val = metrics['r2_score']
        
        print(f"{pc_name:<12} {explained_var:<15.3f} {mae_pc:<18.2f} {r2_score_val:<10.3f}")
    
    print("="*60)
    
    # Create pandas DataFrame for better formatting
    df_data = []
    for pc_name, metrics in results.items():
        df_data.append({
            'Component': pc_name,
            'Explained Var': f"{metrics['explained_variance']:.3f}",
            'MAE (PC space)': f"{metrics['mae_pc_space']:.2f}",
            'R² Score': f"{metrics['r2_score']:.3f}"
        })
    
    df = pd.DataFrame(df_data)
    print("\nPandas DataFrame format:")
    print(df.to_string(index=False))
    
    return df

def save_pc_performance_table(results, save_path):
    """
    Save the PC performance table as CSV and formatted text.
    
    Args:
        results: Dictionary with performance metrics for each PC
        save_path: Base path for saving (without extension)
    """
    # Create DataFrame
    df_data = []
    for pc_name, metrics in results.items():
        df_data.append({
            'Component': pc_name,
            'Explained_Variance': metrics['explained_variance'],
            'MAE_PC_Space': metrics['mae_pc_space'],
            'R2_Score': metrics['r2_score']
        })
    
    df = pd.DataFrame(df_data)
    
    # Save as CSV
    csv_path = f"{save_path}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Table saved as CSV: {csv_path}")
    
    # Save as formatted text
    txt_path = f"{save_path}.txt"
    with open(txt_path, 'w') as f:
        f.write("PC REGRESSION PERFORMANCE TABLE\n")
        f.write("="*60 + "\n")
        f.write(f"{'Component':<12} {'Explained Var':<15} {'MAE (PC space)':<18} {'R² Score':<10}\n")
        f.write("-"*60 + "\n")
        
        for pc_name, metrics in results.items():
            f.write(f"{pc_name:<12} {metrics['explained_variance']:<15.3f} "
                   f"{metrics['mae_pc_space']:<18.2f} {metrics['r2_score']:<10.3f}\n")
        
        f.write("="*60 + "\n")
    
    print(f"Table saved as text: {txt_path}")
    
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_pc_performance_metrics(results, save_path=None):
    """
    Create visualization plots for PC performance metrics.
    
    Args:
        results: Dictionary with performance metrics for each PC
        save_path: Optional path to save the figure
    """
    # Extract data for plotting
    components = list(results.keys())
    explained_vars = [results[pc]['explained_variance'] for pc in components]
    mae_values = [results[pc]['mae_pc_space'] for pc in components]
    r2_values = [results[pc]['r2_score'] for pc in components]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Explained Variance
    bars1 = ax1.bar(components, explained_vars, color='steelblue', alpha=0.7)
    ax1.set_title('Explained Variance by PC', fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_ylim(0, max(explained_vars) * 1.1)
    for i, v in enumerate(explained_vars):
        ax1.text(i, v + max(explained_vars)*0.01, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: MAE in PC Space
    bars2 = ax2.bar(components, mae_values, color='coral', alpha=0.7)
    ax2.set_title('MAE in PC Space', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_ylim(0, max(mae_values) * 1.1)
    for i, v in enumerate(mae_values):
        ax2.text(i, v + max(mae_values)*0.01, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 3: R² Score
    bars3 = ax3.bar(components, r2_values, color='mediumseagreen', alpha=0.7)
    ax3.set_title('R² Score by PC', fontweight='bold')
    ax3.set_ylabel('R² Score')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for i, v in enumerate(r2_values):
        ax3.text(i, v + (max(r2_values) - min(r2_values))*0.02, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved: {save_path}")
    
    plt.show()

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
    
    # Evaluate PC regression performance
    results = evaluate_pc_regression_performance(
        pca, scaler, idx, models, test_dirs, drr_mean, drr_std
    )
    
    # Create and display the performance table
    df = create_pc_performance_table(results)
    
    # Save the table
    save_base_path = os.path.join(OUT_ROOT, "pc_regression_performance_table")
    save_pc_performance_table(results, save_base_path)
    
    # Create performance visualization
    plot_save_path = os.path.join(OUT_ROOT, "pc_performance_metrics.png")
    plot_pc_performance_metrics(results, plot_save_path)

if __name__ == "__main__":
    main()
