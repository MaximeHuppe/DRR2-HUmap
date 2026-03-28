`#%%
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                       IMPORT THE FUNCTIONS 
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from skimage.measure import find_contours
from matplotlib.lines import Line2D
import sys
import os

# Add the src directory to the Python path so we can import modules
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_path)

# Import functions using importlib for numeric module names
import importlib

# Import segmentation functions from current module
segmentation_module = importlib.import_module('01_DRR_segmentation.drr_segmentation')
drr_segmentation = segmentation_module.drr_segmentation
crop_and_resize = segmentation_module.crop_and_resize
process_views = segmentation_module.process_views
drr_preprocess = segmentation_module.drr_preprocess

# Import AttentionUNet from models
models_module = importlib.import_module('00_models.attention_unet')
AttentionUNet = models_module.AttentionUNet

# Import utilities
utils_module = importlib.import_module('utils.model_utils')
load_segmentation_model = utils_module.load_segmentation_model

metrics_module = importlib.import_module('utils.metrics')
evaluate_segmentation_from_paths = metrics_module.evaluate_segmentation_from_paths
print_segmentation_metrics = metrics_module.print_segmentation_metrics


#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                          REQUIRED INPUTS 
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 1. Raw DRR of the leg obtained using beer lambert attenuation law (.npy format). The input size is 256x512. DRR units : none (attenuation coeff)
# 2. Trained regression model (for either the Antero-Posterior (AP) or Medio-Lateral (ML) side)


#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%                                RUN 
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Set-up the parameters : 
device =  torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
drr_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/DRR2-HUmap-Clean/src/drr2humap/example_data/drr_ML.npy"
mask_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/DRR2-HUmap-Clean/src/drr2humap/example_data/mask_ML.npy"
segmentation_model_path = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_2.Segmentation/AttentionUNet_V02_ML.pt"

# Load the trained segmentation model
print("Loading segmentation model...")
model = load_segmentation_model(segmentation_model_path, device)

# Segment the DRR
segmented_drr = drr_segmentation(model=model, 
                                 drr_path=drr_path, 
                                 mask_path=mask_path,
                                 threshold=0.5, 
                                 show=True)

# Evaluate segmentation performance
print("\n" + "="*60)
print("🔍 EVALUATING SEGMENTATION PERFORMANCE")
print("="*60)

if mask_path:
    try:
        # Compute comprehensive metrics
        metrics = evaluate_segmentation_from_paths(segmented_drr, mask_path)
        
        # Display results with nice formatting
        print_segmentation_metrics(metrics, "AP View Segmentation Results")
        
        # Quick summary
        print(f"📊 Quick Summary:")
        print(f"   • Dice Score: {metrics['dice']:.4f} {'🎉' if metrics['dice'] > 0.9 else '⚠️' if metrics['dice'] > 0.7 else '❌'}")
        print(f"   • IoU Score:  {metrics['iou']:.4f} {'🎉' if metrics['iou'] > 0.8 else '⚠️' if metrics['iou'] > 0.6 else '❌'}")
        
        # Performance interpretation
        if metrics['dice'] > 0.9:
            print(f"   🏆 Excellent segmentation quality!")
        elif metrics['dice'] > 0.7:
            print(f"   👍 Good segmentation quality")
        elif metrics['dice'] > 0.5:
            print(f"   🤔 Moderate segmentation quality - consider adjusting threshold")
        else:
            print(f"   🚨 Poor segmentation quality - check model and data")
            
    except Exception as e:
        print(f"❌ Error computing metrics: {e}")
        print("ℹ️  Make sure the ground truth mask file exists and is valid")
else:
    print("ℹ️  No ground truth mask provided - skipping evaluation")

# Post process the segmented DRR for PCA/CNN regression 
print("\n" + "="*60)
print("🔧 POST-PROCESSING FOR CNN-PCA")
print("="*60)

_, _, _, pca_ready_DRR = crop_and_resize(mask=segmented_drr,
                                         drr=np.load(drr_path),
                                         target_size=(256,64), 
                                         show=True)

print("\n✅ Segmentation pipeline completed successfully!")
print(f"📈 Final output shape for CNN-PCA: {pca_ready_DRR.shape}")

# %%
