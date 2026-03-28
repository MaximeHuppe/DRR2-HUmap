"""
DRR Segmentation and Post-processing Utilities

This module provides functions for preprocessing DRR images, performing segmentation
using trained models, visualizing results, and post-processing segmented images.

Functions:
    drr_preprocess: Preprocesses DRR images with CLAHE enhancement
    drr_segmentation: Performs segmentation on DRR images using trained models
    process_views: Visualizes and compares segmentation results with ground truth
    crop_and_resize: Crops and resizes segmented regions for further processing
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from skimage.measure import find_contours
from matplotlib.lines import Line2D

# Device configuration for PyTorch operations
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")



def drr_preprocess(drr_path):
    """
    Preprocess DRR images with CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    This function applies CLAHE to enhance local contrast in DRR images, which is
    particularly useful for improving the visibility of anatomical structures.
    
    Args:
        drr_path (str): Path to the .npy file containing the DRR image
        
    Returns:
        np.ndarray: Preprocessed and normalized DRR image with values in [0, 1]
        
    Example:
        >>> drr_preprocessed = drr_preprocess("path/to/drr.npy")
        >>> print(drr_preprocessed.shape, drr_preprocessed.dtype)
    """
    # Load DRR image and convert to float32
    drr = np.load(drr_path).astype(np.float32)
    
    # Normalize to 0-255 range for CLAHE processing
    drr_normalized = drr / np.max(drr)
    drr_uint8 = np.uint8(255 * drr_normalized)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    drr_clahe = clahe.apply(drr_uint8)
    
    # Convert back to normalized float32
    drr_norm = drr_clahe.astype(np.float32) / 255.0
    
    return drr_norm


def drr_segmentation(model, drr_path, mask_path=None, threshold=0.5, show=True):
    """
    Perform DRR segmentation using a trained neural network model.
    
    This function preprocesses a DRR image and performs segmentation using a trained
    model (typically AttentionUNet). Optionally displays results and compares with
    ground truth if available.
    
    Args:
        model (torch.nn.Module): Trained segmentation model (should be in eval mode)
        drr_path (str): Path to the input DRR image (.npy file)
        mask_path (str, optional): Path to ground truth mask for comparison. Defaults to None.
        threshold (float, optional): Threshold for binary segmentation. Defaults to 0.5.
        show (bool, optional): Whether to display segmentation results. Defaults to True.
        
    Returns:
        np.ndarray: Binary segmentation mask with values in {0, 1}
        
    Example:
        >>> model = AttentionUNet()
        >>> model.load_state_dict(torch.load("model.pt"))
        >>> model.eval()
        >>> pred_mask = drr_segmentation(model, "drr.npy", "mask.npy", threshold=0.7)
    """
    # Preprocess the DRR image
    drr = drr_preprocess(drr_path)
    
    # Convert to tensor and add batch and channel dimensions
    # Shape: (1, 1, H, W) for batch_size=1, channels=1
    drr_tensor = torch.tensor(drr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Perform inference
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        pred = model(drr_tensor)
        pred_mask = (pred > threshold).float().squeeze().cpu().numpy()

    # Visualization
    if show:
        num_plots = 3 if mask_path else 2
        fig, axs = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
        
        # Handle single subplot case
        if num_plots == 2:
            axs = [axs[0], axs[1]]
        
        # Display input DRR
        axs[0].imshow(drr, cmap='gray')
        axs[0].set_title("Input DRR (Preprocessed)", fontsize=12)
        axs[0].axis('off')
        
        # Display predicted mask
        axs[1].imshow(pred_mask, cmap='gray')
        axs[1].set_title(f"Predicted Mask (t={threshold})", fontsize=12)
        axs[1].axis('off')
        
        # Display ground truth if available
        if mask_path:
            gt_mask = np.load(mask_path).astype(np.uint8)
            axs[2].imshow(gt_mask, cmap='gray')
            axs[2].set_title("Ground Truth Mask", fontsize=12)
            axs[2].axis('off')
        
        plt.tight_layout()
        plt.show()

    return pred_mask


def process_views(segmented_drr, mask_path, drr_path):
    """
    Visualize and compare segmentation contours with ground truth.
    
    This function extracts contours from both the segmented DRR and ground truth mask,
    then overlays them on the original DRR image for visual comparison.
    
    Args:
        segmented_drr (np.ndarray): Binary segmentation mask from model prediction
        mask_path (str): Path to ground truth mask file
        drr_path (str): Path to original DRR image file
        
    Example:
        >>> process_views(pred_mask, "ground_truth.npy", "original_drr.npy")
    """
    # Find contours for predicted segmentation
    contours_pred = find_contours(segmented_drr, level=0.5)
    
    # Load and find contours for ground truth mask
    mask_gt = np.load(mask_path)
    contours_gt = find_contours(mask_gt, level=0.5)

    # Load original DRR image
    drr_original = np.load(drr_path)

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Display original DRR as background
    ax.imshow(drr_original, cmap='bone', alpha=0.7)
    
    # Plot predicted contours in red
    for contour in contours_pred:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red', alpha=0.8)
    
    # Plot ground truth contours in blue
    for contour in contours_gt:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='blue', alpha=0.8)
    
    ax.set_title('Segmentation Comparison: Predicted vs Ground Truth', fontsize=14)
    ax.axis('off')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Predicted Segmentation'),
        Line2D([0], [0], color='blue', lw=2, label='Ground Truth Mask')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

def crop_and_resize(mask, drr, target_size=(256, 64), show=True):
    """
    Crop segmented region and resize to target dimensions.
    
    This function finds the bounding box of the segmented region, crops both the mask
    and DRR image to this region, and resizes them to the specified target size.
    
    Args:
        mask (np.ndarray): Binary segmentation mask
        drr (np.ndarray): Original or processed DRR image
        target_size (tuple, optional): Target size as (height, width). Defaults to (256, 64).
        show (bool, optional): Whether to display cropping results. Defaults to True.
        
    Returns:
        tuple: (cropped_mask, cropped_drr, resized_mask, resized_drr)
            - cropped_mask (np.ndarray): Cropped mask at original resolution
            - cropped_drr (np.ndarray): Cropped DRR at original resolution  
            - resized_mask (np.ndarray): Resized mask to target_size
            - resized_drr (np.ndarray): Resized DRR to target_size
            
    Example:
        >>> crop_mask, crop_drr, resize_mask, resize_drr = crop_and_resize(
        ...     pred_mask, drr_image, target_size=(256, 64)
        ... )
    """
    # print(f"DEBUG: Input mask shape: {mask.shape}")
    # print(f"DEBUG: Input mask dtype: {mask.dtype}")
    # print(f"DEBUG: Input mask unique values: {np.unique(mask)}")
    # print(f"DEBUG: Input mask min/max: {mask.min()}/{mask.max()}")
    # print(f"DEBUG: Input DRR shape: {drr.shape}")
    
    # Convert mask to binary if needed (handle float values from model output)
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        binary_mask = (mask > 0.5).astype(np.uint8)
        # print(f"DEBUG: Converted float mask to binary, threshold=0.5")
    else:
        binary_mask = (mask > 0).astype(np.uint8)
        # print(f"DEBUG: Using mask as-is with >0 threshold")
    
    # print(f"DEBUG: Binary mask unique values: {np.unique(binary_mask)}")
    # print(f"DEBUG: Number of segmented pixels: {np.sum(binary_mask)}")
    
    # Find bounding box of segmented region
    y_coords, x_coords = np.where(binary_mask > 0)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        # print("ERROR: No segmented region found in mask!")
        # print(f"DEBUG: y_coords length: {len(y_coords)}, x_coords length: {len(x_coords)}")
        raise ValueError("No segmented region found in mask")
    
    # Calculate bounding box coordinates
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # print(f"DEBUG: Original image size: {mask.shape}")
    # print(f"DEBUG: Bounding box: x=[{x_min}:{x_max+1}], y=[{y_min}:{y_max+1}]")
    # print(f"DEBUG: Bounding box size: {x_max-x_min+1} x {y_max-y_min+1}")
    # print(f"DEBUG: Cropping will reduce size from {mask.shape} to ({y_max-y_min+1}, {x_max-x_min+1})")

    # Crop both mask and DRR to bounding box
    cropped_mask = binary_mask[y_min:y_max+1, x_min:x_max+1]
    cropped_drr = drr[y_min:y_max+1, x_min:x_max+1]
    
    # Apply mask to DRR to get only segmented tibia pixels
    cropped_drr_segmented = cropped_drr * cropped_mask
    
    # print(f"DEBUG: After cropping - mask shape: {cropped_mask.shape}, drr shape: {cropped_drr.shape}")
    # print(f"DEBUG: Cropped mask has {np.sum(cropped_mask)} segmented pixels")
    # print(f"DEBUG: Segmented DRR range: [{cropped_drr_segmented.min():.3f}, {cropped_drr_segmented.max():.3f}]")

    # Resize to target dimensions - cv2.resize expects (width, height)
    # target_size is (height, width), so we need to reverse it for cv2.resize
    resized_mask = cv2.resize(cropped_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    resized_drr_segmented = cv2.resize(cropped_drr_segmented, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # print(f"DEBUG: After resizing - mask shape: {resized_mask.shape}, segmented drr shape: {resized_drr_segmented.shape}")
    # print(f"DEBUG: Target size was: {target_size} (height, width)")
    # print(f"DEBUG: cv2.resize called with: ({target_size[1]}, {target_size[0]}) (width, height)")
    # print(f"DEBUG: Final segmented DRR range: [{resized_drr_segmented.min():.3f}, {resized_drr_segmented.max():.3f}]")

    # Visualization
    if show:
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original images
        axs[0, 0].imshow(drr, cmap='bone')
        axs[0, 0].set_title(f'Original DRR\nSize: {drr.shape}', fontsize=12)
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(binary_mask, cmap='gray')
        axs[0, 1].set_title(f'Binary Mask\nSize: {binary_mask.shape}', fontsize=12)
        axs[0, 1].axis('off')
        
        # Show bounding box on original
        axs[0, 2].imshow(drr, cmap='bone')
        axs[0, 2].add_patch(plt.Rectangle((x_min, y_min), x_max-x_min+1, y_max-y_min+1, 
                                        fill=False, edgecolor='red', linewidth=2))
        axs[0, 2].set_title(f'Bounding Box\nBox: {x_max-x_min+1}x{y_max-y_min+1}', fontsize=12)
        axs[0, 2].axis('off')
        
        # Cropped versions
        axs[1, 0].imshow(cropped_drr_segmented, cmap='gray')
        axs[1, 0].set_title(f'Cropped Segmented Tibia\nSize: {cropped_drr_segmented.shape}', fontsize=12)
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(cropped_mask, cmap='gray')
        axs[1, 1].set_title(f'Cropped Mask\nSize: {cropped_mask.shape}', fontsize=12)
        axs[1, 1].axis('off')
        
        # Resized versions
        axs[1, 2].imshow(resized_drr_segmented, cmap='gray')
        axs[1, 2].set_title(f'Final Segmented Tibia\nSize: {resized_drr_segmented.shape}', fontsize=12)
        axs[1, 2].axis('off')
        
        plt.suptitle('Crop and Resize Process', fontsize=14)
        plt.tight_layout()
        plt.show()

    return cropped_mask, cropped_drr_segmented, resized_mask, resized_drr_segmented 