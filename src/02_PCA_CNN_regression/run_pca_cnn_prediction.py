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
pca_model_path = "DRR2-HUmap/src/02_PCA_CNN_regression/models/pca.pkl"
scaler_path = "DRR2-HUmap/src/02_PCA_CNN_regression/models/scaler.pkl"
indices_path = "DRR2-HUmap/src/02_PCA_CNN_regression/models/idx.npy"

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
