"""
Model Utilities for DRR Segmentation

This module contains utility functions for loading models and orchestrating
the complete segmentation pipeline workflow.
"""

import torch
import numpy as np
import os
from pathlib import Path
import importlib

# Import AttentionUNet using importlib to handle numeric module names
import sys
import os

# Add src directory to path for imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_path)

try:
    models_module = importlib.import_module('00_models.attention_unet')
    AttentionUNet = models_module.AttentionUNet
except ImportError as e:
    print(f"Failed to import AttentionUNet: {e}")
    raise

# Import segmentation functions using importlib for numeric module names
try:
    segmentation_module = importlib.import_module('01_DRR_segmentation.drr_segmentation')
    drr_segmentation = segmentation_module.drr_segmentation
    process_views = segmentation_module.process_views
    crop_and_resize = segmentation_module.crop_and_resize
except ImportError as e:
    print(f"Failed to import segmentation functions: {e}")
    raise

# # Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else 
                   "cuda" if torch.cuda.is_available() else "cpu")


def compute_drr_mean_std_from_training_data(train_dirs):
    """
    Compute DRR mean and std from training data to match preprocessing exactly.
    
    This function should be called once to compute the normalization constants
    that were used during CNN-PCA training.
    
    Args:
        train_dirs (list): List of training directories containing DRR files
        
    Returns:
        tuple: (drr_mean, drr_std) normalization constants
    """
    ap_list, ml_list = [], []
    
    for d in train_dirs:
        ap_path = os.path.join(d, "DRR", "AP.npy")
        ml_path = os.path.join(d, "DRR", "ML.npy")
        
        if os.path.exists(ap_path) and os.path.exists(ml_path):
            ap = np.load(ap_path).astype(np.float32)
            ml = np.load(ml_path).astype(np.float32)
            ap_list.append(ap)
            ml_list.append(ml)
    
    if not ap_list:
        raise ValueError("No training DRR files found")
    
    ap_stack = np.stack(ap_list)
    ml_stack = np.stack(ml_list)
    drr_mean = np.mean(np.concatenate([ap_stack, ml_stack]))
    drr_std = np.std(np.concatenate([ap_stack, ml_stack]))
    
    print(f"📊 Computed DRR normalization from {len(ap_list)} training samples:")
    print(f"   Mean: {drr_mean:.6f}")
    print(f"   Std:  {drr_std:.6f}")
    
    return drr_mean, drr_std


def apply_preprocessing_rotation(drr_array, apply_rotation=True):
    """
    Apply the same rotation as in preprocessing pipeline.
    
    Based on preprocessing code: np.save(..., np.rot90(drr_final))
    
    Args:
        drr_array (np.ndarray): DRR array to rotate
        apply_rotation (bool): Whether to apply rotation
        
    Returns:
        np.ndarray: Rotated DRR array (if apply_rotation=True)
    """
    if apply_rotation:
        return np.rot90(drr_array)
    return drr_array


def prepare_drr_for_cnn_pca(drr_ap, drr_ml, drr_mean, drr_std, apply_rotation=True):
    """
    Prepare DRR pair for CNN-PCA prediction exactly like preprocessing.
    
    Steps:
    1. Apply rotation (if specified)
    2. Normalize using training statistics
    3. Stack AP and ML views
    4. Convert to tensor
    
    Args:
        drr_ap (np.ndarray): AP view DRR (segmented and resized)
        drr_ml (np.ndarray): ML view DRR (segmented and resized)  
        drr_mean (float): Training data mean
        drr_std (float): Training data standard deviation
        apply_rotation (bool): Whether to apply preprocessing rotation
        
    Returns:
        torch.Tensor: Prepared tensor for CNN-PCA models, shape (2, H, W)
    """
    # Apply rotation to match preprocessing (if specified)
    if apply_rotation:
        drr_ap_rotated = apply_preprocessing_rotation(drr_ap, True)
        drr_ml_rotated = apply_preprocessing_rotation(drr_ml, True)
    else:
        drr_ap_rotated = drr_ap
        drr_ml_rotated = drr_ml
    
    # Normalize exactly like preprocessing
    rap = (drr_ap_rotated - drr_mean) / drr_std
    rml = (drr_ml_rotated - drr_mean) / drr_std
    
    # Stack views exactly like preprocessing
    drr_stacked = np.stack([rap, rml])
    
    # Convert to tensor
    drr_tensor = torch.from_numpy(drr_stacked).float()
    
    return drr_tensor


def load_segmentation_model(model_path, device=device):
    """
    Load a trained AttentionUNet model for segmentation.
    
    Args:
        model_path (str): Path to the saved model weights (.pt file)
        device (torch.device): Device to load the model on
        
    Returns:
        torch.nn.Module: Loaded and ready-to-use AttentionUNet model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    print(f"Loading segmentation model from: {model_path}")
    
    # Verify model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Initialize model architecture
        model = AttentionUNet(in_ch=1, out_ch=1)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Move to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully!")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def run_complete_pipeline(model, drr_path, mask_path=None, threshold=0.5, 
                         target_size=(64, 256), show_results=True):
    """
    Execute the complete DRR segmentation pipeline.
    
    This function orchestrates the entire workflow:
    1. Perform segmentation on DRR image
    2. Visualize results and compare with ground truth (if available)
    3. Crop and resize segmented regions
    
    Args:
        model (torch.nn.Module): Trained segmentation model
        drr_path (str): Path to input DRR image (.npy file)
        mask_path (str, optional): Path to ground truth mask for comparison
        threshold (float): Segmentation threshold (0.0 to 1.0)
        target_size (tuple): Target size for final output (height, width)
        show_results (bool): Whether to display intermediate visualizations
        
    Returns:
        dict: Complete pipeline results containing:
            - 'segmented_mask': Binary segmentation mask
            - 'masked_drr': Original DRR with mask applied
            - 'cropped_mask': Cropped mask region
            - 'cropped_drr': Cropped DRR region
            - 'resized_mask': Final resized mask
            - 'resized_drr': Final resized DRR with segmentation applied (ready for CNN-PCA)
            
    Raises:
        FileNotFoundError: If DRR file doesn't exist
        ValueError: If segmentation produces empty mask
    """
    print(f"\n🔄 Processing DRR: {Path(drr_path).name}")
    
    # Verify input file exists
    if not os.path.exists(drr_path):
        raise FileNotFoundError(f"DRR file not found: {drr_path}")
    
    # Step 1: Perform segmentation
    print("  Step 1/3: Performing segmentation...")
    segmented_mask = drr_segmentation(
        model=model, 
        drr_path=drr_path, 
        mask_path=mask_path,
        threshold=threshold, 
        show=show_results
    )
    
    # Verify we got a valid segmentation
    if np.sum(segmented_mask) == 0:
        raise ValueError("Segmentation produced empty mask. Try lowering the threshold.")
    
    # Step 2: Apply mask to original DRR
    print("  Step 2/3: Applying mask to original DRR...")
    drr_original = np.load(drr_path)
    masked_drr = drr_original * segmented_mask
    
    # Step 3: Compare with ground truth if available
    if mask_path and os.path.exists(mask_path) and show_results:
        print("  Bonus: Comparing with ground truth...")
        process_views(segmented_mask, mask_path, drr_path)
    
    # Step 4: Crop and resize for final output (matches preprocessing exactly)
    print("  Step 3/3: Cropping and resizing (preprocessing style)...")
    try:
        cropped_mask, cropped_drr, resized_mask, resized_drr_segmented = crop_and_resize(
            segmented_mask, masked_drr, target_size=target_size, show=show_results
        )
    except ValueError as e:
        print(f"  Warning: {e}")
        # Return original data if cropping fails
        cropped_mask = segmented_mask
        cropped_drr = masked_drr
        resized_mask = segmented_mask
        resized_drr_segmented = masked_drr
    
    # Package all results
    results = {
        'segmented_mask': segmented_mask,
        'masked_drr': masked_drr,
        'cropped_mask': cropped_mask,
        'cropped_drr': cropped_drr,
        'resized_mask': resized_mask,
        'resized_drr': resized_drr_segmented  # This now has mask applied, ready for CNN-PCA
    }
    
    print("✅ Pipeline completed successfully!")
    return results


def validate_configuration(config):
    """
    Validate pipeline configuration parameters.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Handle both single model path and separate AP/ML model paths
    if 'model_path' in config:
        required_keys = ['model_path', 'drr_path', 'threshold', 'target_size']
    else:
        required_keys = ['AP_model_path', 'ML_model_path', 'drr_AP_path', 'drr_ML_path', 'threshold', 'target_size']
    
    # Check required keys exist
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate file paths
    if 'model_path' in config:
        if not os.path.exists(config['model_path']):
            raise ValueError(f"Model file not found: {config['model_path']}")
        if not os.path.exists(config['drr_path']):
            raise ValueError(f"DRR file not found: {config['drr_path']}")
    else:
        if not os.path.exists(config['AP_model_path']):
            raise ValueError(f"AP model file not found: {config['AP_model_path']}")
        if not os.path.exists(config['ML_model_path']):
            raise ValueError(f"ML model file not found: {config['ML_model_path']}")
        if not os.path.exists(config['drr_AP_path']):
            raise ValueError(f"AP DRR file not found: {config['drr_AP_path']}")
        if not os.path.exists(config['drr_ML_path']):
            raise ValueError(f"ML DRR file not found: {config['drr_ML_path']}")
    
    # Validate threshold
    if not 0.0 <= config['threshold'] <= 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got: {config['threshold']}")
    
    # Validate target_size
    if not isinstance(config['target_size'], (tuple, list)) or len(config['target_size']) != 2:
        raise ValueError("target_size must be a tuple/list of (height, width)")
    
    if any(x <= 0 for x in config['target_size']):
        raise ValueError("target_size dimensions must be positive")
    
    return True 