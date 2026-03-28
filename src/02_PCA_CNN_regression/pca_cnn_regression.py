"""
PCA-CNN Regression for HU Volume Prediction

This module provides functions for predicting HU volumes from segmented DRR images
using PCA components and CNN regression models.

Functions:
    preprocess_drr_for_cnn: Preprocesses segmented DRRs for CNN input
    predict_pc_coefficients: Predicts PCA coefficients using trained models
    reconstruct_volume_from_pcs: Reconstructs HU volume from PC coefficients
    pca_cnn_prediction: Complete pipeline from DRRs to HU volume
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import pickle
import os
import sys
import importlib
from typing import Dict, List, Tuple, Optional, Union
from skimage import exposure
import matplotlib.pyplot as plt

# Add src directory to path for imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_path)

# Import PC1Regressor using importlib for numeric module names
try:
    models_module = importlib.import_module('00_models.cnn_pca')
    PC1Regressor = models_module.PC1Regressor
    MCDropout = models_module.MCDropout
except ImportError as e:
    print(f"Failed to import CNN-PCA models: {e}")
    raise

# Device configuration for PyTorch operations
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")


def apply_clahe_enhancement(drr_image: np.ndarray, 
                          clip_limit: float = 0.01, 
                          tile_grid_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to DRR image.
    
    Args:
        drr_image (np.ndarray): Input DRR image
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of the grid for histogram equalization
        
    Returns:
        np.ndarray: CLAHE enhanced image
        
    Example:
        >>> enhanced_drr = apply_clahe_enhancement(drr_image)
    """
    # Normalize to 0-1 range
    img_normalized = exposure.rescale_intensity(drr_image, in_range='image', out_range=(0.0, 1.0))
    
    # Apply CLAHE
    enhanced = exposure.equalize_adapthist(
        img_normalized.astype(np.float32), 
        clip_limit=clip_limit, 
        kernel_size=tile_grid_size
    )
    
    return enhanced.astype(np.float32)


def load_pc_models(model_paths: Dict[int, str], 
                   device: Union[str, torch.device] = device) -> Dict[int, nn.Module]:
    """
    Load multiple PC regressor models from file paths.
    
    Args:
        model_paths (dict): Dictionary mapping PC index to model file path
        device (str or torch.device): Device to load models on
        
    Returns:
        dict: Dictionary mapping PC index to loaded model objects
        
    Raises:
        FileNotFoundError: If any model file doesn't exist
        RuntimeError: If model loading fails
        
    Example:
        >>> model_paths = {1: "pc1.pt", 2: "pc2.pt", 3: "pc3.pt"}
        >>> models = load_pc_models(model_paths)
        >>> print(f"Loaded {len(models)} PC models")
    """
    print(f"🔄 Loading {len(model_paths)} PC regressor models...")
    
    if isinstance(device, str):
        device = torch.device(device)
    
    loaded_models = {}
    
    for pc_idx, model_path in model_paths.items():
        print(f"   Loading PC{pc_idx} model from: {os.path.basename(model_path)}")
        
        # Verify file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Initialize model architecture
            model = PC1Regressor()
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Move to device and set to evaluation mode
            model.to(device)
            model.eval()
            
            loaded_models[pc_idx] = model
            print(f"   ✅ PC{pc_idx} model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PC{pc_idx} model: {str(e)}")
    
    print(f"✅ All {len(loaded_models)} PC models loaded successfully!")
    return loaded_models


def preprocess_drr_for_cnn(ap_drr: np.ndarray, 
                          ml_drr: np.ndarray,
                          ap_mask: np.ndarray = None,
                          ml_mask: np.ndarray = None,
                          target_size: Tuple[int, int] = (64, 64),
                          drr_mean: float = 0.0425,
                          drr_std: float = 0.0924,
                          show: bool = True) -> torch.Tensor:
    """
    Preprocess DRR pair (AP and ML views) for CNN-PCA prediction.
    
    This function applies segmentation masks, resizes images, normalizes them,
    applies CLAHE enhancement, and stacks them into the format expected by CNN models.
    
    Args:
        ap_drr (np.ndarray): Antero-Posterior DRR image
        ml_drr (np.ndarray): Medio-Lateral DRR image  
        ap_mask (np.ndarray, optional): AP segmentation mask. Defaults to None.
        ml_mask (np.ndarray, optional): ML segmentation mask. Defaults to None.
        target_size (tuple): Target size for resizing (height, width). Defaults to (64, 64).
        drr_mean (float): DRR normalization mean. Defaults to 0.0425.
        drr_std (float): DRR normalization std. Defaults to 0.0924.
        show (bool): Whether to display preprocessing steps. Defaults to True.
        
    Returns:
        torch.Tensor: Preprocessed tensor ready for CNN input, shape (1, 4, H, W)
                     Channels: [raw_ap, raw_ml, clahe_ap, clahe_ml]
                     
    Example:
        >>> drr_tensor = preprocess_drr_for_cnn(ap_drr, ml_drr, ap_mask, ml_mask)
        >>> print(drr_tensor.shape)  # (1, 4, 64, 64)
    """
    print(f"🔧 Preprocessing DRR pair for CNN-PCA prediction...")
    print(f"   Input shapes: AP {ap_drr.shape}, ML {ml_drr.shape}")
    
    # Apply segmentation masks if provided
    if ap_mask is not None:
        ap_masked = ap_drr * ap_mask
        print(f"   Applied AP mask: {np.sum(ap_mask > 0.5)} segmented pixels")
    else:
        ap_masked = ap_drr
        print("   No AP mask provided - using full DRR")
        
    if ml_mask is not None:
        ml_masked = ml_drr * ml_mask
        print(f"   Applied ML mask: {np.sum(ml_mask > 0.5)} segmented pixels")
    else:
        ml_masked = ml_drr
        print("   No ML mask provided - using full DRR")
    
    # Resize to target size
    ap_resized = cv2.resize(ap_masked.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
    ml_resized = cv2.resize(ml_masked.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
    print(f"   Resized to: {target_size}")
    
    # Raw normalization using training statistics
    ap_raw = (ap_resized - drr_mean) / drr_std
    ml_raw = (ml_resized - drr_mean) / drr_std
    print(f"   Applied normalization: mean={drr_mean:.4f}, std={drr_std:.4f}")
    
    # CLAHE enhancement for improved contrast
    ap_clahe = apply_clahe_enhancement(ap_resized)
    ml_clahe = apply_clahe_enhancement(ml_resized)
    print(f"   Applied CLAHE enhancement")
    
    # Stack all channels: [raw_ap, raw_ml, clahe_ap, clahe_ml]
    drr_stack = np.stack([ap_raw, ml_raw, ap_clahe, ml_clahe], axis=0)
    
    # Convert to tensor and add batch dimension
    drr_tensor = torch.from_numpy(drr_stack).float().unsqueeze(0).to(device)
    print(f"   Final tensor shape: {drr_tensor.shape}")
    
    # Visualization
    if show:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original DRRs
        axes[0, 0].imshow(ap_drr, cmap='gray')
        axes[0, 0].set_title('Original AP DRR')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(ml_drr, cmap='gray')
        axes[1, 0].set_title('Original ML DRR')
        axes[1, 0].axis('off')
        
        # Masked DRRs
        axes[0, 1].imshow(ap_masked, cmap='gray')
        axes[0, 1].set_title('Masked AP DRR')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(ml_masked, cmap='gray')
        axes[1, 1].set_title('Masked ML DRR')
        axes[1, 1].axis('off')
        
        # Raw normalized
        axes[0, 2].imshow(ap_raw, cmap='gray')
        axes[0, 2].set_title('Normalized AP')
        axes[0, 2].axis('off')
        
        axes[1, 2].imshow(ml_raw, cmap='gray')
        axes[1, 2].set_title('Normalized ML')
        axes[1, 2].axis('off')
        
        # CLAHE enhanced
        axes[0, 3].imshow(ap_clahe, cmap='gray')
        axes[0, 3].set_title('CLAHE AP')
        axes[0, 3].axis('off')
        
        axes[1, 3].imshow(ml_clahe, cmap='gray')
        axes[1, 3].set_title('CLAHE ML')
        axes[1, 3].axis('off')
        
        plt.suptitle('DRR Preprocessing for CNN-PCA', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    return drr_tensor


def predict_pc_coefficients(drr_tensor: torch.Tensor,
                          pc_models: Dict[int, nn.Module],
                          n_components: int = 10,
                          use_uncertainty: bool = False,
                          n_mc_samples: int = 10) -> Dict[str, Union[np.ndarray, float]]:
    """
    Predict PCA coefficients from preprocessed DRR tensor using trained CNN models.
    
    Args:
        drr_tensor (torch.Tensor): Preprocessed DRR tensor from preprocess_drr_for_cnn
        pc_models (dict): Dictionary mapping PC index to loaded CNN models
        n_components (int): Number of PC components to predict. Defaults to 10.
        use_uncertainty (bool): Whether to compute uncertainty estimates. Defaults to False.
        n_mc_samples (int): Number of Monte Carlo samples for uncertainty. Defaults to 10.
        
    Returns:
        dict: Prediction results containing:
            - 'coefficients': Array of predicted PC coefficients
            - 'uncertainty': Standard deviation of predictions (if use_uncertainty=True)
            - 'confidence': Confidence score based on uncertainty
            
    Example:
        >>> results = predict_pc_coefficients(drr_tensor, pc_models)
        >>> print(f"Predicted coefficients: {results['coefficients']}")
    """
    print(f"🧠 Predicting PC coefficients using {len(pc_models)} trained models...")
    
    coefficients = np.zeros(n_components)
    uncertainties = np.zeros(n_components) if use_uncertainty else None
    
    with torch.no_grad():
        for pc_idx in range(1, n_components + 1):
            if pc_idx not in pc_models:
                print(f"   ⚠️  Warning: Model for PC{pc_idx} not found, using 0.0")
                continue
                
            model = pc_models[pc_idx]
            model.eval()
            
            if use_uncertainty:
                # Monte Carlo Dropout for uncertainty estimation
                model.train()  # Enable dropout
                predictions = []
                
                for _ in range(n_mc_samples):
                    # Use first 2 channels (raw AP, raw ML) for prediction
                    input_tensor = drr_tensor[:, :2, :, :]
                    pred = model(input_tensor)
                    predictions.append(pred.detach().cpu().numpy()[0])
                
                predictions = np.array(predictions)
                coefficients[pc_idx - 1] = np.mean(predictions)
                uncertainties[pc_idx - 1] = np.std(predictions)
                
                print(f"   PC{pc_idx}: {coefficients[pc_idx - 1]:.4f} ± {uncertainties[pc_idx - 1]:.4f}")
                
            else:
                # Single forward pass
                input_tensor = drr_tensor[:, :2, :, :]  # Use raw channels
                pred = model(input_tensor)
                coefficients[pc_idx - 1] = pred.detach().cpu().numpy()[0]
                
                print(f"   PC{pc_idx}: {coefficients[pc_idx - 1]:.4f}")
    
    # Compute overall confidence
    confidence = None
    if use_uncertainty and uncertainties is not None:
        # Higher confidence = lower uncertainty
        mean_uncertainty = np.mean(uncertainties)
        confidence = 1.0 / (1.0 + mean_uncertainty)
        print(f"   📊 Overall confidence: {confidence:.4f}")
    
    results = {
        'coefficients': coefficients,
        'n_components': n_components
    }
    
    if use_uncertainty:
        results['uncertainty'] = uncertainties
        results['confidence'] = confidence
    
    return results


def reconstruct_volume_from_pcs(pc_coefficients: np.ndarray,
                              pca_model_path: str,
                              scaler_path: str,
                              indices_path: str,
                              volume_shape: Tuple[int, int, int] = (64, 64, 256),
                              hu_threshold: float = -1000.0,
                              show: bool = True) -> Dict[str, np.ndarray]:
    """
    Reconstruct HU volume from PCA coefficients.
    
    Args:
        pc_coefficients (np.ndarray): Predicted PC coefficients
        pca_model_path (str): Path to fitted PCA model (.pkl file)
        scaler_path (str): Path to fitted StandardScaler (.pkl file)
        indices_path (str): Path to voxel indices array (.npy file)
        volume_shape (tuple): Expected CT volume shape. Defaults to (64, 64, 256).
        hu_threshold (float): HU threshold for background. Defaults to -1000.0.
        show (bool): Whether to display reconstruction results. Defaults to True.
        
    Returns:
        dict: Reconstruction results containing:
            - 'volume': Reconstructed HU volume
            - 'mask': Binary mask of reconstructed region
            - 'volume_stats': Statistics about the reconstructed volume
            
    Raises:
        FileNotFoundError: If any of the required files don't exist
        
    Example:
        >>> results = reconstruct_volume_from_pcs(coefficients, pca_path, scaler_path, idx_path)
        >>> hu_volume = results['volume']
        >>> print(f"Reconstructed volume shape: {hu_volume.shape}")
    """
    print(f"🏗️  Reconstructing HU volume from {len(pc_coefficients)} PC coefficients...")
    
    # Load PCA components and preprocessing objects
    try:
        with open(pca_model_path, 'rb') as f:
            pca = pickle.load(f)
        print(f"   ✅ Loaded PCA model: {pca.n_components_} components")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"   ✅ Loaded StandardScaler")
        
        indices = np.load(indices_path)
        print(f"   ✅ Loaded voxel indices: {len(indices)} voxels")
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading PCA components: {e}")
    
    # Ensure we have the right number of coefficients
    n_components_used = min(len(pc_coefficients), pca.n_components_)
    if len(pc_coefficients) > pca.n_components_:
        print(f"   ⚠️  Using first {n_components_used} coefficients (PCA has {pca.n_components_} components)")
        pc_coefficients = pc_coefficients[:n_components_used]
    elif len(pc_coefficients) < pca.n_components_:
        print(f"   ⚠️  Padding with zeros: {len(pc_coefficients)} -> {pca.n_components_} coefficients")
        padded_coeffs = np.zeros(pca.n_components_)
        padded_coeffs[:len(pc_coefficients)] = pc_coefficients
        pc_coefficients = padded_coeffs
    
    # Apply inverse scaling to PC coefficients first (if they were scaled during training)
    print(f"   🔄 Applying inverse scaling to PC coefficients...")
    try:
        # Try to inverse scale the PC coefficients
        pc_coefficients_scaled = scaler.inverse_transform(pc_coefficients.reshape(1, -1)).flatten()
        print(f"   ✅ Applied inverse scaling to PC coefficients")
    except Exception as e:
        # If scaling fails, use coefficients as-is (they might not have been scaled)
        print(f"   ⚠️  Scaling failed, using coefficients as-is: {e}")
        pc_coefficients_scaled = pc_coefficients
    
    # Reconstruct using PCA inverse transform
    print(f"   🔄 Applying PCA inverse transform...")
    reconstructed_data = pca.inverse_transform(pc_coefficients_scaled.reshape(1, -1))
    
    # The reconstructed data should now be the original feature space
    print(f"   ✅ Reconstructed data shape: {reconstructed_data.shape}")
    hu_values = reconstructed_data.flatten()
    
    # Create full volume
    print(f"   🔄 Creating full volume of shape {volume_shape}...")
    full_volume = np.full(np.prod(volume_shape), hu_threshold, dtype=np.float32)
    
    # Place reconstructed values at specified indices
    if len(indices) != len(hu_values):
        print(f"   ⚠️  Index/value mismatch: {len(indices)} indices, {len(hu_values)} values")
        min_len = min(len(indices), len(hu_values))
        indices = indices[:min_len]
        hu_values = hu_values[:min_len]
    
    full_volume[indices] = hu_values
    
    # Reshape to 3D volume
    volume = full_volume.reshape(volume_shape)
    
    # Create binary mask
    mask = (volume > hu_threshold).astype(np.uint8)
    
    # Compute statistics
    volume_stats = {
        'hu_min': float(np.min(volume[mask > 0])) if np.any(mask) else hu_threshold,
        'hu_max': float(np.max(volume[mask > 0])) if np.any(mask) else hu_threshold,
        'hu_mean': float(np.mean(volume[mask > 0])) if np.any(mask) else hu_threshold,
        'hu_std': float(np.std(volume[mask > 0])) if np.any(mask) else 0.0,
        'n_voxels': int(np.sum(mask)),
        'volume_percentage': float(np.sum(mask) / np.prod(volume_shape) * 100)
    }
    
    print(f"   📊 Reconstruction statistics:")
    print(f"      HU range: [{volume_stats['hu_min']:.1f}, {volume_stats['hu_max']:.1f}]")
    print(f"      HU mean ± std: {volume_stats['hu_mean']:.1f} ± {volume_stats['hu_std']:.1f}")
    print(f"      Reconstructed voxels: {volume_stats['n_voxels']} ({volume_stats['volume_percentage']:.1f}%)")
    
    # Visualization
    if show:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Volume slices
        mid_z = volume_shape[2] // 2
        mid_y = volume_shape[1] // 2
        mid_x = volume_shape[0] // 2
        
        # Axial slice
        axes[0, 0].imshow(volume[:, :, mid_z], cmap='gray', vmin=-1000, vmax=1000)
        axes[0, 0].set_title(f'Axial Slice (z={mid_z})')
        axes[0, 0].axis('off')
        
        # Coronal slice
        axes[0, 1].imshow(volume[:, mid_y, :], cmap='gray', vmin=-1000, vmax=1000)
        axes[0, 1].set_title(f'Coronal Slice (y={mid_y})')
        axes[0, 1].axis('off')
        
        # Sagittal slice
        axes[0, 2].imshow(volume[mid_x, :, :], cmap='gray', vmin=-1000, vmax=1000)
        axes[0, 2].set_title(f'Sagittal Slice (x={mid_x})')
        axes[0, 2].axis('off')
        
        # Corresponding masks
        axes[1, 0].imshow(mask[:, :, mid_z], cmap='gray')
        axes[1, 0].set_title('Axial Mask')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask[:, mid_y, :], cmap='gray')
        axes[1, 1].set_title('Coronal Mask')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(mask[mid_x, :, :], cmap='gray')
        axes[1, 2].set_title('Sagittal Mask')
        axes[1, 2].axis('off')
        
        plt.suptitle('Reconstructed HU Volume', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # HU histogram
        if np.any(mask):
            plt.figure(figsize=(10, 6))
            plt.hist(volume[mask > 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('HU Value')
            plt.ylabel('Frequency')
            plt.title('HU Distribution in Reconstructed Volume')
            plt.axvline(volume_stats['hu_mean'], color='red', linestyle='--', 
                       label=f"Mean: {volume_stats['hu_mean']:.1f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    return {
        'volume': volume,
        'mask': mask,
        'volume_stats': volume_stats
    }


def pca_cnn_prediction(ap_drr: np.ndarray,
                      ml_drr: np.ndarray, 
                      ap_mask: np.ndarray,
                      ml_mask: np.ndarray,
                      pc_model_paths: Dict[int, str],
                      pca_model_path: str,
                      scaler_path: str,
                      indices_path: str,
                      target_size: Tuple[int, int] = (64, 64),
                      volume_shape: Tuple[int, int, int] = (64, 64, 256),
                      drr_mean: float = 0.0425,
                      drr_std: float = 0.0924,
                      n_components: int = 10,
                      use_uncertainty: bool = False,
                      show: bool = True) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Complete pipeline from segmented DRRs to predicted HU volume.
    
    This function orchestrates the entire CNN-PCA prediction workflow:
    1. Load PC regressor models
    2. Preprocess DRR pair for CNN input
    3. Predict PC coefficients using CNN models
    4. Reconstruct HU volume from PC coefficients
    
    Args:
        ap_drr (np.ndarray): Antero-Posterior DRR image
        ml_drr (np.ndarray): Medio-Lateral DRR image
        ap_mask (np.ndarray): AP segmentation mask
        ml_mask (np.ndarray): ML segmentation mask
        pc_model_paths (dict): Dictionary mapping PC index to model file paths
        pca_model_path (str): Path to fitted PCA model
        scaler_path (str): Path to fitted StandardScaler
        indices_path (str): Path to voxel indices array
        target_size (tuple): Target size for DRR preprocessing. Defaults to (64, 64).
        volume_shape (tuple): Expected CT volume shape. Defaults to (64, 64, 256).
        drr_mean (float): DRR normalization mean. Defaults to 0.0425.
        drr_std (float): DRR normalization std. Defaults to 0.0924.
        n_components (int): Number of PC components to predict. Defaults to 10.
        use_uncertainty (bool): Whether to compute uncertainty estimates. Defaults to False.
        show (bool): Whether to display intermediate results. Defaults to True.
        
    Returns:
        dict: Complete prediction results containing:
            - 'volume': Reconstructed HU volume
            - 'mask': Binary mask of reconstructed region
            - 'volume_stats': Statistics about the reconstructed volume
            - 'pc_results': PC coefficient prediction results
            - 'preprocessing_info': Information about preprocessing steps
            
    Example:
        >>> results = pca_cnn_prediction(ap_drr, ml_drr, ap_mask, ml_mask, 
        ...                             pc_paths, pca_path, scaler_path, idx_path)
        >>> hu_volume = results['volume']
        >>> print(f"Predicted HU volume: {hu_volume.shape}")
    """
    print("🚀 Starting complete PCA-CNN prediction pipeline...")
    print("="*60)
    
    try:
        # Step 1: Load PC models
        print("📦 Step 1/4: Loading PC regressor models...")
        pc_models = load_pc_models(pc_model_paths, device)
        
        # Step 2: Preprocess DRRs
        print(f"\n🔧 Step 2/4: Preprocessing DRR pair...")
        drr_tensor = preprocess_drr_for_cnn(
            ap_drr=ap_drr,
            ml_drr=ml_drr, 
            ap_mask=ap_mask,
            ml_mask=ml_mask,
            target_size=target_size,
            drr_mean=drr_mean,
            drr_std=drr_std,
            show=show
        )
        
        # Step 3: Predict PC coefficients
        print(f"\n🧠 Step 3/4: Predicting PC coefficients...")
        pc_results = predict_pc_coefficients(
            drr_tensor=drr_tensor,
            pc_models=pc_models,
            n_components=n_components,
            use_uncertainty=use_uncertainty
        )
        
        # Step 4: Reconstruct volume
        print(f"\n🏗️  Step 4/4: Reconstructing HU volume...")
        volume_results = reconstruct_volume_from_pcs(
            pc_coefficients=pc_results['coefficients'],
            pca_model_path=pca_model_path,
            scaler_path=scaler_path,
            indices_path=indices_path,
            volume_shape=volume_shape,
            show=show
        )
        
        # Compile complete results
        results = {
            'volume': volume_results['volume'],
            'mask': volume_results['mask'],
            'volume_stats': volume_results['volume_stats'],
            'pc_results': pc_results,
            'preprocessing_info': {
                'target_size': target_size,
                'drr_normalization': {'mean': drr_mean, 'std': drr_std},
                'tensor_shape': drr_tensor.shape,
                'device': str(device)
            }
        }
        
        print("\n" + "="*60)
        print("✅ PCA-CNN prediction pipeline completed successfully!")
        print(f"📊 Final Results Summary:")
        print(f"   • Volume shape: {volume_results['volume'].shape}")
        print(f"   • HU range: [{volume_results['volume_stats']['hu_min']:.1f}, {volume_results['volume_stats']['hu_max']:.1f}]")
        print(f"   • HU mean: {volume_results['volume_stats']['hu_mean']:.1f}")
        print(f"   • Reconstructed voxels: {volume_results['volume_stats']['n_voxels']}")
        print(f"   • PC coefficients used: {len(pc_results['coefficients'])}")
        if use_uncertainty and 'confidence' in pc_results:
            print(f"   • Prediction confidence: {pc_results['confidence']:.4f}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in PCA-CNN prediction pipeline: {str(e)}")
        raise 