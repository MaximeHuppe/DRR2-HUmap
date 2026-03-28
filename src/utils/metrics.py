"""
Evaluation Metrics for DRR Segmentation

This module provides functions for evaluating segmentation performance
using common metrics like Dice score and Intersection over Union (IoU).

Author: Maxime Huppe
Institution: Imperial College London
"""

import numpy as np
import torch
from typing import Union, Tuple


def compute_dice_score(predicted: Union[np.ndarray, torch.Tensor], 
                      ground_truth: Union[np.ndarray, torch.Tensor],
                      smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient (F1 score) between predicted and ground truth masks.
    
    The Dice coefficient is defined as:
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        predicted (np.ndarray or torch.Tensor): Predicted binary mask
        ground_truth (np.ndarray or torch.Tensor): Ground truth binary mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Dice score between 0 and 1 (1 = perfect overlap)
        
    Example:
        >>> pred_mask = np.array([[1, 1, 0], [1, 0, 0]])
        >>> gt_mask = np.array([[1, 0, 0], [1, 1, 0]])
        >>> dice = compute_dice_score(pred_mask, gt_mask)
        >>> print(f"Dice score: {dice:.3f}")
    """
    # Convert to numpy arrays if tensors
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    
    # Ensure binary masks (threshold at 0.5 if needed)
    predicted = (predicted > 0.5).astype(np.float32)
    ground_truth = (ground_truth > 0.5).astype(np.float32)
    
    # Flatten arrays for easier computation
    pred_flat = predicted.flatten()
    gt_flat = ground_truth.flatten()
    
    # Compute intersection and union
    intersection = np.sum(pred_flat * gt_flat)
    pred_sum = np.sum(pred_flat)
    gt_sum = np.sum(gt_flat)
    
    # Compute Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    
    return float(dice)


def compute_iou_score(predicted: Union[np.ndarray, torch.Tensor], 
                     ground_truth: Union[np.ndarray, torch.Tensor],
                     smooth: float = 1e-6) -> float:
    """
    Compute Intersection over Union (IoU) between predicted and ground truth masks.
    
    The IoU coefficient is defined as:
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        predicted (np.ndarray or torch.Tensor): Predicted binary mask
        ground_truth (np.ndarray or torch.Tensor): Ground truth binary mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: IoU score between 0 and 1 (1 = perfect overlap)
        
    Example:
        >>> pred_mask = np.array([[1, 1, 0], [1, 0, 0]])
        >>> gt_mask = np.array([[1, 0, 0], [1, 1, 0]])
        >>> iou = compute_iou_score(pred_mask, gt_mask)
        >>> print(f"IoU score: {iou:.3f}")
    """
    # Convert to numpy arrays if tensors
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    
    # Ensure binary masks (threshold at 0.5 if needed)
    predicted = (predicted > 0.5).astype(np.float32)
    ground_truth = (ground_truth > 0.5).astype(np.float32)
    
    # Flatten arrays for easier computation
    pred_flat = predicted.flatten()
    gt_flat = ground_truth.flatten()
    
    # Compute intersection and union
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    
    # Compute IoU coefficient
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def compute_segmentation_metrics(predicted: Union[np.ndarray, torch.Tensor], 
                               ground_truth: Union[np.ndarray, torch.Tensor],
                               smooth: float = 1e-6) -> dict:
    """
    Compute comprehensive segmentation metrics.
    
    Args:
        predicted (np.ndarray or torch.Tensor): Predicted binary mask
        ground_truth (np.ndarray or torch.Tensor): Ground truth binary mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        dict: Dictionary containing all computed metrics:
            - 'dice': Dice coefficient
            - 'iou': Intersection over Union
            - 'sensitivity': True Positive Rate (Recall)
            - 'specificity': True Negative Rate
            - 'precision': Positive Predictive Value
            - 'accuracy': Overall accuracy
            
    Example:
        >>> metrics = compute_segmentation_metrics(pred_mask, gt_mask)
        >>> print(f"Dice: {metrics['dice']:.3f}, IoU: {metrics['iou']:.3f}")
    """
    # Convert to numpy arrays if tensors
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().numpy()
    
    # Ensure binary masks (threshold at 0.5 if needed)
    predicted = (predicted > 0.5).astype(np.float32)
    ground_truth = (ground_truth > 0.5).astype(np.float32)
    
    # Flatten arrays
    pred_flat = predicted.flatten()
    gt_flat = ground_truth.flatten()
    
    # Compute confusion matrix components
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))  # True Positives
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))  # True Negatives
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))  # False Positives
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))  # False Negatives
    
    # Compute metrics
    dice = compute_dice_score(predicted, ground_truth, smooth)
    iou = compute_iou_score(predicted, ground_truth, smooth)
    
    # Additional metrics
    sensitivity = (tp + smooth) / (tp + fn + smooth)  # Recall, True Positive Rate
    specificity = (tn + smooth) / (tn + fp + smooth)  # True Negative Rate
    precision = (tp + smooth) / (tp + fp + smooth)    # Positive Predictive Value
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)  # Overall accuracy
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def evaluate_segmentation_from_paths(predicted_mask: Union[np.ndarray, str], 
                                   ground_truth_path: str,
                                   smooth: float = 1e-6) -> dict:
    """
    Evaluate segmentation metrics when ground truth is provided as a file path.
    
    Args:
        predicted_mask (np.ndarray or str): Predicted mask array or path to .npy file
        ground_truth_path (str): Path to ground truth mask (.npy file)
        smooth (float): Smoothing factor for metric computation
        
    Returns:
        dict: Dictionary containing all computed metrics
        
    Raises:
        FileNotFoundError: If ground truth file doesn't exist
        
    Example:
        >>> metrics = evaluate_segmentation_from_paths(pred_mask, "ground_truth.npy")
        >>> print(f"Segmentation quality: Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}")
    """
    import os
    
    # Load ground truth mask
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    ground_truth = np.load(ground_truth_path)
    
    # Load predicted mask if it's a path
    if isinstance(predicted_mask, str):
        if not os.path.exists(predicted_mask):
            raise FileNotFoundError(f"Predicted mask file not found: {predicted_mask}")
        predicted_mask = np.load(predicted_mask)
    
    # Compute and return metrics
    return compute_segmentation_metrics(predicted_mask, ground_truth, smooth)


def print_segmentation_metrics(metrics: dict, title: str = "Segmentation Metrics") -> None:
    """
    Pretty print segmentation metrics.
    
    Args:
        metrics (dict): Dictionary of metrics from compute_segmentation_metrics
        title (str): Title for the metrics display
        
    Example:
        >>> metrics = compute_segmentation_metrics(pred, gt)
        >>> print_segmentation_metrics(metrics, "AP View Segmentation")
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"🎯 Dice Score:      {metrics['dice']:.4f}")
    print(f"🔗 IoU Score:       {metrics['iou']:.4f}")
    print(f"🎗️  Sensitivity:     {metrics['sensitivity']:.4f}")
    print(f"🛡️  Specificity:     {metrics['specificity']:.4f}")
    print(f"🎪 Precision:       {metrics['precision']:.4f}")
    print(f"✅ Accuracy:        {metrics['accuracy']:.4f}")
    print(f"{'='*50}")
    print(f"Confusion Matrix:")
    print(f"  TP: {metrics['tp']:6d}  |  FN: {metrics['fn']:6d}")
    print(f"  FP: {metrics['fp']:6d}  |  TN: {metrics['tn']:6d}")
    print(f"{'='*50}\n") 