#%%
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                       IMPORT THE FUNCTIONS 
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from skimage.measure import block_reduce
import scipy.ndimage as ndi
from scipy import ndimage
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim

# Add the src directory to Python path for imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_path)

# Import functions using importlib for numeric module names
import importlib

# Import PCA-CNN regression functions from current module
try:
    pca_module = importlib.import_module('02_PCA_CNN_regression.pca_cnn_regression')
    pca_cnn_prediction = pca_module.pca_cnn_prediction
    preprocess_drr_for_cnn = pca_module.preprocess_drr_for_cnn
    load_pc_models = pca_module.load_pc_models
    predict_pc_coefficients = pca_module.predict_pc_coefficients
    reconstruct_volume_from_pcs = pca_module.reconstruct_volume_from_pcs
except ImportError as e:
    print(f"Failed to import PCA-CNN functions: {e}")
    raise

# Import PC1Regressor from models
try:
    models_module = importlib.import_module('00_models.cnn_pca')
    PC1Regressor = models_module.PC1Regressor
    MCDropout = models_module.MCDropout
except ImportError as e:
    print(f"Failed to import CNN-PCA models: {e}")
    raise


#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                          REQUIRED INPUTS 
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1. Segmented DRR images (AP and ML views) obtained from the segmentation step
# 2. Trained PC regressor models (one for each principal component)
# 3. PCA model, scaler, and voxel indices from training
# 4. Expected volume dimensions and preprocessing parameters


#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                                RUN 
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Set-up the parameters : 
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Input DRR and mask paths (typically from segmentation step output)
ap_drr_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/DRR2-HUmap-Clean/src/drr2humap/example_data/pca_AP_drr.npy"
ml_drr_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/DRR2-HUmap-Clean/src/drr2humap/example_data/pca_ML_drr.npy"  # You may need to add this
ap_mask_path = None
ml_mask_path = None  # You may need to add this

# PC regressor model paths (you'll need to adjust these to your actual model paths)
pc_model_paths = {
    1: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc1_sequential.pt",
    2: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc2_sequential.pt",
    3: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc3_sequential.pt",
    4: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc4_sequential.pt",
    5: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc5_sequential.pt",
    6: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc6_sequential.pt",
    7: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc7_sequential.pt",
    8: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc8_sequential.pt",
    9: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc9_sequential.pt",
    10: "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/PC_models/model_pc10_sequential.pt"
}

# PCA components and preprocessing files
pca_model_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/pca.pkl"
scaler_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/scaler.pkl"
indices_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/idx.npy"

# Processing parameters
target_size = (256, 64)  # Size for CNN input
volume_shape = (64, 64, 256)  # Expected output volume shape
drr_mean = 0.0425  # DRR normalization statistics from training
drr_std = 0.0924
n_components = 10  # Number of PC components to predict

print("="*60)
print("🚀 STARTING PCA-CNN HU VOLUME PREDICTION")
print("="*60)

# Load input data
print("📁 Loading input DRR images and masks...")
try:
    ap_drr = np.load(ap_drr_path)
    print(f"   ✅ Loaded AP DRR: {ap_drr.shape}")
    
    # Try to load ML DRR, create dummy if not available
    if os.path.exists(ml_drr_path):
        ml_drr = np.load(ml_drr_path)
        print(f"   ✅ Loaded ML DRR: {ml_drr.shape}")
    else:
        print(f"   ⚠️  ML DRR not found, creating dummy ML view")
        ml_drr = np.zeros_like(ap_drr)  # Dummy ML for demonstration
    
    # Handle masks - load if paths provided, otherwise use None
    if ap_mask_path is not None and os.path.exists(ap_mask_path):
        ap_mask = np.load(ap_mask_path)
        print(f"   ✅ Loaded AP mask: {ap_mask.shape}")
    else:
        ap_mask = None
        print(f"   ℹ️  No AP mask provided - will use full AP DRR")
    
    if ml_mask_path is not None and os.path.exists(ml_mask_path):
        ml_mask = np.load(ml_mask_path)
        print(f"   ✅ Loaded ML mask: {ml_mask.shape}")
    else:
        ml_mask = None
        print(f"   ℹ️  No ML mask provided - will use full ML DRR")

except FileNotFoundError as e:
    print(f"❌ Error loading input files: {e}")
    print("💡 Please ensure you have:")
    print("   - DRR images (AP and optionally ML)")
    print("   - Segmentation masks are optional")
    sys.exit(1)

# Check which PC models are available
print("\n📦 Checking PC model availability...")
available_models = {}
for pc_idx, model_path in pc_model_paths.items():
    if os.path.exists(model_path):
        available_models[pc_idx] = model_path
        print(f"   ✅ PC{pc_idx} model found")
    else:
        print(f"   ❌ PC{pc_idx} model not found: {os.path.basename(model_path)}")

if not available_models:
    print("❌ No PC models found! Please ensure PC regressor models are available.")
    print("💡 Expected model paths (adjust as needed):")
    for pc_idx, path in pc_model_paths.items():
        print(f"   PC{pc_idx}: {path}")
    sys.exit(1)

print(f"   📊 Will use {len(available_models)} available PC models")

# Check PCA components availability
print("\n🔧 Checking PCA components availability...")
missing_files = []
for name, path in [("PCA model", pca_model_path), ("Scaler", scaler_path), ("Indices", indices_path)]:
    if os.path.exists(path):
        print(f"   ✅ {name} found")
    else:
        print(f"   ❌ {name} not found: {os.path.basename(path)}")
        missing_files.append(name)

if missing_files:
    print(f"❌ Missing required files: {', '.join(missing_files)}")
    print("💡 Please ensure you have the PCA training outputs:")
    print(f"   - PCA model: {pca_model_path}")
    print(f"   - Scaler: {scaler_path}")
    print(f"   - Indices: {indices_path}")
    sys.exit(1)

# Run the complete PCA-CNN prediction pipeline
print("\n" + "="*60)
print("🎯 RUNNING PCA-CNN PREDICTION PIPELINE")
print("="*60)

try:
    results = pca_cnn_prediction(
        ap_drr=ap_drr,
        ml_drr=ml_drr,
        ap_mask=ap_mask,
        ml_mask=ml_mask,
        pc_model_paths=available_models,
        pca_model_path=pca_model_path,
        scaler_path=scaler_path,
        indices_path=indices_path,
        target_size=target_size,
        volume_shape=volume_shape,
        drr_mean=drr_mean,
        drr_std=drr_std,
        n_components=len(available_models),
        use_uncertainty=True,  # Enable uncertainty estimation
        show=True
    )
    
    # Extract results
    predicted_volume = results['volume']
    volume_mask = results['mask']
    volume_stats = results['volume_stats']
    pc_results = results['pc_results']
    
    print("\n" + "="*60)
    print("📊 PREDICTION RESULTS SUMMARY")
    print("="*60)
    print(f"🏗️  Reconstructed Volume:")
    print(f"   • Shape: {predicted_volume.shape}")
    print(f"   • HU range: [{volume_stats['hu_min']:.1f}, {volume_stats['hu_max']:.1f}]")
    print(f"   • HU mean ± std: {volume_stats['hu_mean']:.1f} ± {volume_stats['hu_std']:.1f}")
    print(f"   • Reconstructed voxels: {volume_stats['n_voxels']:,}")
    print(f"   • Volume coverage: {volume_stats['volume_percentage']:.1f}%")
    
    print(f"\n🧠 PC Coefficients:")
    for i, coeff in enumerate(pc_results['coefficients'][:10]):  # Show first 10
        uncertainty_str = ""
        if 'uncertainty' in pc_results:
            uncertainty_str = f" ± {pc_results['uncertainty'][i]:.4f}"
        print(f"   • PC{i+1}: {coeff:.4f}{uncertainty_str}")
    
    if 'confidence' in pc_results:
        confidence_level = pc_results['confidence']
        print(f"\n📈 Prediction Confidence: {confidence_level:.4f}")
        if confidence_level > 0.8:
            print("   🎉 High confidence prediction!")
        elif confidence_level > 0.6:
            print("   👍 Good confidence prediction")
        else:
            print("   ⚠️  Lower confidence - consider investigating")
    
    # Optional: Save results
    save_results = False  # Set to True if you want to save outputs
    if save_results:
        output_dir = "pca_cnn_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predicted volume
        np.save(os.path.join(output_dir, "predicted_hu_volume.npy"), predicted_volume)
        np.save(os.path.join(output_dir, "volume_mask.npy"), volume_mask)
        np.save(os.path.join(output_dir, "pc_coefficients.npy"), pc_results['coefficients'])
        
        print(f"\n💾 Results saved to: {output_dir}/")
        print(f"   • predicted_hu_volume.npy")
        print(f"   • volume_mask.npy")
        print(f"   • pc_coefficients.npy")
    
    print("\n✅ PCA-CNN prediction completed successfully!")
    
except Exception as e:
    print(f"❌ Error in PCA-CNN prediction: {str(e)}")
    print("💡 Common issues to check:")
    print("   - Ensure all model files exist and are loadable")
    print("   - Check DRR and mask shapes are compatible")
    print("   - Verify PCA components match training setup")
    print("   - Check available GPU/CPU memory")
    raise

# %% 

# Full resolution 
CT_pred = results['volume']
CT_gt = np.load('/Volumes/Maxime Imperial Backup/Max Data/CNN_Dataset/f_001_left/Aug_0/CT/CT_reg.npy')
diff_full = CT_pred - CT_gt
# Compute MAE excluding -1000 HU background voxels
mask_full = CT_gt != -1000
mae_full = np.mean(np.abs(diff_full[mask_full]))
diff_full = np.where(diff_full == 0, np.nan, diff_full)


# Resolution impact - IMPROVED downsampling using Gaussian + subsample
from skimage.measure import block_reduce
import scipy.ndimage as ndi

def gaussian_downsample(volume, factor):
    """Improved downsampling using Gaussian blur + subsampling (anti-aliasing)"""
    sigma = factor / 2.0  # Gaussian sigma based on downsampling factor
    blurred = ndi.gaussian_filter(volume, sigma=(sigma, sigma, 0))
    return blurred[::factor, ::factor, :]

# Voxel size 2mm - IMPROVED METHOD
CT_pred_2mm = gaussian_downsample(CT_pred, factor=2)
CT_gt_2mm = gaussian_downsample(CT_gt, factor=2)
diff_2mm = CT_pred_2mm - CT_gt_2mm
# Compute MAE excluding -1000 HU background voxels
mask_2mm = CT_gt_2mm != -1000
mae_2mm = np.mean(np.abs(diff_2mm[mask_2mm]))
diff_2mm = np.where(diff_2mm == 0, np.nan, diff_2mm)

# Voxel size 4mm - IMPROVED METHOD  
CT_pred_4mm = gaussian_downsample(CT_pred, factor=4)
CT_gt_4mm = gaussian_downsample(CT_gt, factor=4)
diff_4mm = CT_pred_4mm - CT_gt_4mm
# Compute MAE excluding -1000 HU background voxels
mask_4mm = CT_gt_4mm != -1000
mae_4mm = np.mean(np.abs(diff_4mm[mask_4mm]))
diff_4mm = np.where(diff_4mm == 0, np.nan, diff_4mm)

# Voxel size 8mm - IMPROVED METHOD
CT_pred_8mm = gaussian_downsample(CT_pred, factor=8)  
CT_gt_8mm = gaussian_downsample(CT_gt, factor=8)
diff_8mm = CT_pred_8mm - CT_gt_8mm
# Compute MAE excluding -1000 HU background voxels
mask_8mm = CT_gt_8mm != -1000
mae_8mm = np.mean(np.abs(diff_8mm[mask_8mm]))
diff_8mm = np.where(diff_8mm == 0, np.nan, diff_8mm)

print(f"📈 IMPROVED MAE Results:")
print(f"   2mm resolution: {mae_2mm:.2f}")
print(f"   4mm resolution: {mae_4mm:.2f}")  
print(f"   8mm resolution: {mae_8mm:.2f}")

# Diagnostic: Show tissue vs background percentages
total_voxels_full = CT_gt.size
tissue_voxels_full = np.sum(CT_gt != -1000)
background_percentage_full = (1 - tissue_voxels_full/total_voxels_full) * 100

total_voxels_2mm = CT_gt_2mm.size
tissue_voxels_2mm = np.sum(CT_gt_2mm != -1000)
background_percentage_2mm = (1 - tissue_voxels_2mm/total_voxels_2mm) * 100

print(f"\n🔍 Tissue vs Background Analysis:")
print(f"   Full resolution: {tissue_voxels_full:,} tissue voxels ({100-background_percentage_full:.1f}% tissue)")
print(f"   2mm resolution: {tissue_voxels_2mm:,} tissue voxels ({100-background_percentage_2mm:.1f}% tissue)")
print(f"   Background (-1000 HU) excluded from MAE calculation")

# %% MAE Evolution Analysis Across Full Volume

def compute_mae_per_slice(pred_volume, gt_volume, axis=2, background_hu=-1000):
    """Compute MAE for each slice along specified axis, excluding background voxels"""
    mae_per_slice = []
    for i in range(pred_volume.shape[axis]):
        if axis == 2:  # z-axis (most common)
            pred_slice = pred_volume[:, :, i]
            gt_slice = gt_volume[:, :, i]
        elif axis == 1:  # y-axis
            pred_slice = pred_volume[:, i, :]
            gt_slice = gt_volume[:, i, :]
        elif axis == 0:  # x-axis
            pred_slice = pred_volume[i, :, :]
            gt_slice = gt_volume[i, :, :]
        
        # Compute MAE for this slice (ignore -1000 HU background areas)
        diff = np.abs(pred_slice - gt_slice)
        # Only compute MAE where there's actual tissue (not -1000 HU background)
        mask = gt_slice != background_hu
        if np.sum(mask) > 0:
            mae_slice = np.mean(diff[mask])
        else:
            mae_slice = 0
        mae_per_slice.append(mae_slice)
    
    return np.array(mae_per_slice)

def compute_psnr_per_slice(pred_volume, gt_volume, axis=2, background_hu=-1000):
    """Compute PSNR for each slice along specified axis, excluding background voxels"""
    psnr_per_slice = []
    
    # Determine dynamic range for PSNR calculation
    tissue_mask = gt_volume != background_hu
    if np.sum(tissue_mask) > 0:
        hu_min = np.min(gt_volume[tissue_mask])
        hu_max = np.max(gt_volume[tissue_mask])
        dynamic_range = hu_max - hu_min
    else:
        dynamic_range = 4095  # Default for 12-bit CT
    
    for i in range(pred_volume.shape[axis]):
        if axis == 2:  # z-axis (most common)
            pred_slice = pred_volume[:, :, i]
            gt_slice = gt_volume[:, :, i]
        elif axis == 1:  # y-axis
            pred_slice = pred_volume[:, i, :]
            gt_slice = gt_volume[:, i, :]
        elif axis == 0:  # x-axis
            pred_slice = pred_volume[i, :, :]
            gt_slice = gt_volume[i, :, :]
        
        # Compute PSNR for this slice (ignore -1000 HU background areas)
        mask = gt_slice != background_hu
        if np.sum(mask) > 0:
            mse = np.mean((pred_slice[mask] - gt_slice[mask])**2)
            if mse > 0:
                psnr_slice = 20 * np.log10(dynamic_range / np.sqrt(mse))
            else:
                psnr_slice = 100  # Perfect match
        else:
            psnr_slice = 0
        psnr_per_slice.append(psnr_slice)
    
    return np.array(psnr_per_slice)

def compute_ssim_per_slice(pred_volume, gt_volume, axis=2, background_hu=-1000):
    """Compute SSIM for each slice along specified axis, excluding background voxels"""
    ssim_per_slice = []
    
    # Pre-calculate tissue mask for the entire volume
    tissue_mask = gt_volume != background_hu
    
    for i in range(pred_volume.shape[axis]):
        if axis == 2:  # z-axis (most common)
            pred_slice = pred_volume[:, :, i]
            gt_slice = gt_volume[:, :, i]
            mask = tissue_mask[:, :, i]
        elif axis == 1:  # y-axis
            pred_slice = pred_volume[:, i, :]
            gt_slice = gt_volume[:, i, :]
            mask = tissue_mask[:, i, :]
        elif axis == 0:  # x-axis
            pred_slice = pred_volume[i, :, :]
            gt_slice = gt_volume[i, :, :]
            mask = tissue_mask[i, :, :]
        
        # Compute SSIM for this slice
        if np.sum(mask) > 100:  # Need sufficient pixels for SSIM
            # Use only tissue regions for SSIM calculation
            try:
                # Compute data range from tissue only
                tissue_min = np.min(gt_slice[mask])
                tissue_max = np.max(gt_slice[mask])
                data_range = tissue_max - tissue_min
                
                if data_range > 0:
                    # Use smaller window size for faster computation
                    ssim_slice = ssim(pred_slice, gt_slice, 
                                    data_range=data_range,
                                    win_size=min(7, min(pred_slice.shape)//4*2+1))  # Adaptive window size
                else:
                    ssim_slice = 1.0  # Perfect match if no variation
            except:
                ssim_slice = 0.0  # Fallback if SSIM computation fails
        else:
            ssim_slice = 0.0
        ssim_per_slice.append(ssim_slice)
    
    return np.array(ssim_per_slice)

print("\n🔬 Computing MAE, PSNR, and SSIM evolution across all slices...")

# Compute metrics per slice for different resolutions
mae_full_slices = compute_mae_per_slice(CT_pred, CT_gt)
mae_2mm_slices = compute_mae_per_slice(CT_pred_2mm, CT_gt_2mm)
mae_4mm_slices = compute_mae_per_slice(CT_pred_4mm, CT_gt_4mm)
mae_8mm_slices = compute_mae_per_slice(CT_pred_8mm, CT_gt_8mm)

psnr_full_slices = compute_psnr_per_slice(CT_pred, CT_gt)
psnr_2mm_slices = compute_psnr_per_slice(CT_pred_2mm, CT_gt_2mm)
psnr_4mm_slices = compute_psnr_per_slice(CT_pred_4mm, CT_gt_4mm)
psnr_8mm_slices = compute_psnr_per_slice(CT_pred_8mm, CT_gt_8mm)

ssim_full_slices = compute_ssim_per_slice(CT_pred, CT_gt)
ssim_2mm_slices = compute_ssim_per_slice(CT_pred_2mm, CT_gt_2mm)
ssim_4mm_slices = compute_ssim_per_slice(CT_pred_4mm, CT_gt_4mm)
ssim_8mm_slices = compute_ssim_per_slice(CT_pred_8mm, CT_gt_8mm)

# Visualization 
plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(CT_gt[:,:,240], cmap='gray')
plt.title('GT')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(CT_pred[:,:,240], cmap='gray')
plt.title('Pred')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(CT_pred[:,:,240] - CT_gt[:,:,240], cmap='RdBu_r', vmin=-300, vmax=300); plt.colorbar()
plt.title('Diff')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(CT_gt_2mm[:,:,240], cmap='gray')
plt.title('GT 2mm')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(CT_pred_2mm[:,:,240], cmap='gray')
plt.title('Pred 2mm')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(CT_pred_2mm[:,:,240] - CT_gt_2mm[:,:,240], cmap='RdBu_r', vmin=-300, vmax=300); plt.colorbar()
plt.title('Diff 2mm')
plt.axis('off')


# Statistics
print(f"📊 Metrics Statistics Across All Slices:")
print(f"   Full Resolution (1mm):")
print(f"      MAE:  {np.mean(mae_full_slices):.2f} ± {np.std(mae_full_slices):.2f} HU")
print(f"      PSNR: {np.mean(psnr_full_slices):.2f} ± {np.std(psnr_full_slices):.2f} dB")
print(f"      SSIM: {np.mean(ssim_full_slices):.3f} ± {np.std(ssim_full_slices):.3f}")
print(f"   2mm Resolution:")
print(f"      MAE:  {np.mean(mae_2mm_slices):.2f} ± {np.std(mae_2mm_slices):.2f} HU")
print(f"      PSNR: {np.mean(psnr_2mm_slices):.2f} ± {np.std(psnr_2mm_slices):.2f} dB")
print(f"      SSIM: {np.mean(ssim_2mm_slices):.3f} ± {np.std(ssim_2mm_slices):.3f}")
print(f"   4mm Resolution:")
print(f"      MAE:  {np.mean(mae_4mm_slices):.2f} ± {np.std(mae_4mm_slices):.2f} HU")
print(f"      PSNR: {np.mean(psnr_4mm_slices):.2f} ± {np.std(psnr_4mm_slices):.2f} dB")
print(f"      SSIM: {np.mean(ssim_4mm_slices):.3f} ± {np.std(ssim_4mm_slices):.3f}")
print(f"   8mm Resolution:")
print(f"      MAE:  {np.mean(mae_8mm_slices):.2f} ± {np.std(mae_8mm_slices):.2f} HU")
print(f"      PSNR: {np.mean(psnr_8mm_slices):.2f} ± {np.std(psnr_8mm_slices):.2f} dB")
print(f"      SSIM: {np.mean(ssim_8mm_slices):.3f} ± {np.std(ssim_8mm_slices):.3f}")

# Plot metrics evolution
plt.figure(figsize=(20, 15))

# MAE evolution plot
plt.subplot(3, 3, 1)
slice_indices = np.arange(len(mae_full_slices))
plt.plot(slice_indices, mae_full_slices, 'b-', linewidth=2, label='Full Res (1mm)', alpha=0.8)
plt.plot(slice_indices, mae_2mm_slices, 'g-', linewidth=2, label='2mm', alpha=0.8)
plt.plot(slice_indices, mae_4mm_slices, 'orange', linewidth=2, label='4mm', alpha=0.8)
plt.plot(slice_indices, mae_8mm_slices, 'r-', linewidth=2, label='8mm', alpha=0.8)
plt.xlabel('Slice Index (z-direction)')
plt.ylabel('MAE (HU)')
plt.title('MAE Evolution Across Volume')
plt.legend()
plt.grid(True, alpha=0.3)

# PSNR evolution plot
plt.subplot(3, 3, 2)
plt.plot(slice_indices, psnr_full_slices, 'b-', linewidth=2, label='Full Res (1mm)', alpha=0.8)
plt.plot(slice_indices, psnr_2mm_slices, 'g-', linewidth=2, label='2mm', alpha=0.8)
plt.plot(slice_indices, psnr_4mm_slices, 'orange', linewidth=2, label='4mm', alpha=0.8)
plt.plot(slice_indices, psnr_8mm_slices, 'r-', linewidth=2, label='8mm', alpha=0.8)
plt.xlabel('Slice Index (z-direction)')
plt.ylabel('PSNR (dB)')
plt.title('PSNR Evolution Across Volume')
plt.legend()
plt.grid(True, alpha=0.3)

# SSIM evolution plot
plt.subplot(3, 3, 3)
plt.plot(slice_indices, ssim_full_slices, 'b-', linewidth=2, label='Full Res (1mm)', alpha=0.8)
plt.plot(slice_indices, ssim_2mm_slices, 'g-', linewidth=2, label='2mm', alpha=0.8)
plt.plot(slice_indices, ssim_4mm_slices, 'orange', linewidth=2, label='4mm', alpha=0.8)
plt.plot(slice_indices, ssim_8mm_slices, 'r-', linewidth=2, label='8mm', alpha=0.8)
plt.xlabel('Slice Index (z-direction)')
plt.ylabel('SSIM')
plt.title('SSIM Evolution Across Volume')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plots for MAE
plt.subplot(3, 3, 4)
mae_data = [mae_full_slices, mae_2mm_slices, mae_4mm_slices, mae_8mm_slices]
labels = ['Full\n(1mm)', '2mm', '4mm', '8mm']
bp = plt.boxplot(mae_data, tick_labels=labels, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.ylabel('MAE (HU)')
plt.title('MAE Distribution')
plt.grid(True, alpha=0.3)

# Box plots for PSNR
plt.subplot(3, 3, 5)
psnr_data = [psnr_full_slices, psnr_2mm_slices, psnr_4mm_slices, psnr_8mm_slices]
bp = plt.boxplot(psnr_data, tick_labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.ylabel('PSNR (dB)')
plt.title('PSNR Distribution')
plt.grid(True, alpha=0.3)

# Box plots for SSIM
plt.subplot(3, 3, 6)
ssim_data = [ssim_full_slices, ssim_2mm_slices, ssim_4mm_slices, ssim_8mm_slices]
bp = plt.boxplot(ssim_data, tick_labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.ylabel('SSIM')
plt.title('SSIM Distribution')
plt.grid(True, alpha=0.3)

# Histogram of MAE values
plt.subplot(3, 3, 7)
plt.hist(mae_full_slices, bins=30, alpha=0.7, label='Full Res', color='blue')
plt.hist(mae_2mm_slices, bins=30, alpha=0.7, label='2mm', color='green')
plt.hist(mae_4mm_slices, bins=30, alpha=0.7, label='4mm', color='orange')
plt.hist(mae_8mm_slices, bins=30, alpha=0.7, label='8mm', color='red')
plt.xlabel('MAE (HU)')
plt.ylabel('Number of Slices')
plt.title('MAE Distribution Histogram')
plt.legend()
plt.grid(True, alpha=0.3)

# Histogram of PSNR values
plt.subplot(3, 3, 8)
plt.hist(psnr_full_slices, bins=30, alpha=0.7, label='Full Res', color='blue')
plt.hist(psnr_2mm_slices, bins=30, alpha=0.7, label='2mm', color='green')
plt.hist(psnr_4mm_slices, bins=30, alpha=0.7, label='4mm', color='orange')
plt.hist(psnr_8mm_slices, bins=30, alpha=0.7, label='8mm', color='red')
plt.xlabel('PSNR (dB)')
plt.ylabel('Number of Slices')
plt.title('PSNR Distribution Histogram')
plt.legend()
plt.grid(True, alpha=0.3)

# Histogram of SSIM values
plt.subplot(3, 3, 9)
plt.hist(ssim_full_slices, bins=30, alpha=0.7, label='Full Res', color='blue')
plt.hist(ssim_2mm_slices, bins=30, alpha=0.7, label='2mm', color='green')
plt.hist(ssim_4mm_slices, bins=30, alpha=0.7, label='4mm', color='orange')
plt.hist(ssim_8mm_slices, bins=30, alpha=0.7, label='8mm', color='red')
plt.xlabel('SSIM')
plt.ylabel('Number of Slices')
plt.title('SSIM Distribution Histogram')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find best and worst performing slices
best_slice_full = np.argmin(mae_full_slices)
worst_slice_full = np.argmax(mae_full_slices)

print(f"\n🎯 Slice Analysis:")
print(f"   Best performing slice: {best_slice_full} (MAE: {mae_full_slices[best_slice_full]:.2f})")
print(f"   Worst performing slice: {worst_slice_full} (MAE: {mae_full_slices[worst_slice_full]:.2f})")
print(f"   Performance ratio: {mae_full_slices[worst_slice_full]/mae_full_slices[best_slice_full]:.1f}x worse")

# %% Show best and worst slices visually
plt.figure(figsize=(15, 8))

# Best slice
plt.subplot(2, 4, 1)
plt.imshow(CT_gt[:,:,best_slice_full], cmap='gray')
plt.title(f'GT - Best Slice {best_slice_full}')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(CT_pred[:,:,best_slice_full], cmap='gray')
plt.title(f'Pred - Best Slice {best_slice_full}')
plt.axis('off')

plt.subplot(2, 4, 3)
diff_best = CT_pred[:,:,best_slice_full] - CT_gt[:,:,best_slice_full]
plt.imshow(diff_best, cmap='RdBu_r', vmin=-200, vmax=200)
plt.title(f'MAE: {mae_full_slices[best_slice_full]:.1f}')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 4, 4)
plt.hist(diff_best.flatten(), bins=50, alpha=0.7, color='green')
plt.yscale('log')  # Log scale for histogram
plt.title('Difference Histogram (Best)')
plt.xlabel('HU Difference')
plt.ylabel('Count (log scale)')

# Worst slice  
plt.subplot(2, 4, 5)
plt.imshow(CT_gt[:,:,worst_slice_full], cmap='gray')
plt.title(f'GT - Worst Slice {worst_slice_full}')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(CT_pred[:,:,worst_slice_full], cmap='gray')
plt.title(f'Pred - Worst Slice {worst_slice_full}')
plt.axis('off')

plt.subplot(2, 4, 7)
diff_worst = CT_pred[:,:,worst_slice_full] - CT_gt[:,:,worst_slice_full]
plt.imshow(diff_worst, cmap='RdBu_r', vmin=-200, vmax=200)
plt.title(f'MAE : {mae_full_slices[worst_slice_full]:.1f}')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 4, 8)
plt.hist(diff_worst.flatten(), bins=50, alpha=0.7, color='red')
plt.yscale('log')  # Log scale for histogram
plt.title('Difference Histogram (Worst)')
plt.xlabel('HU Difference')
plt.ylabel('Count (log scale)')

plt.tight_layout()
plt.show()

# %%
