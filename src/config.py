"""
Configuration management for DRR2-HUmap pipeline.

This module provides default configurations and utilities for loading
custom configurations from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: The best available device (MPS > CUDA > CPU)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for the DRR2-HUmap pipeline.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Device configuration
        "device": str(get_device()),
        
        # Segmentation configuration
        "segmentation": {
            "model_type": "attention_unet",
            "threshold": 0.5,
            "preprocessing": {
                "clahe_enabled": True,
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": (8, 8),
                "normalize": True,
                "normalize_range": (0, 1)
            },
            "postprocessing": {
                "morphological_operations": True,
                "remove_small_objects": True,
                "min_object_size": 100
            }
        },
        
        # CNN-PCA configuration
        "cnn_pca": {
            "model_type": "cnn_pca_regressor",
            "num_principal_components": 10,
            "dropout_rate": 0.3,
            "uncertainty_estimation": {
                "enabled": True,
                "mc_samples": 100,
                "method": "monte_carlo_dropout"
            }
        },
        
        # Image processing configuration
        "image_processing": {
            "target_size": [256, 64],  # [height, width]
            "interpolation": "linear",
            "crop_to_segmentation": True,
            "padding_mode": "constant",
            "padding_value": 0
        },
        
        # Visualization configuration
        "visualization": {
            "show_plots": True,
            "save_plots": False,
            "plot_format": "png",
            "plot_dpi": 300,
            "colormap": "gray",
            "figure_size": [12, 8]
        },
        
        # Evaluation configuration
        "evaluation": {
            "metrics": ["dice", "iou", "hausdorff", "mae", "rmse"],
            "save_metrics": True,
            "detailed_analysis": True
        },
        
        # I/O configuration
        "io": {
            "input_format": "npy",
            "output_format": "npy",
            "save_intermediate_results": False,
            "output_directory": "results",
            "create_subdirectories": True
        },
        
        # Logging configuration
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "save_to_file": False,
            "log_file": "drr2humap.log"
        }
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
    
    # Merge with default configuration
    default_config = get_default_config()
    merged_config = merge_configs(default_config, config)
    
    return merged_config


def merge_configs(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge custom configuration with default configuration.
    
    Args:
        default (Dict[str, Any]): Default configuration
        custom (Dict[str, Any]): Custom configuration to merge
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = default.copy()
    
    for key, value in custom.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        output_path (str): Path where to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["segmentation", "cnn_pca", "image_processing"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate device
    if "device" in config:
        try:
            torch.device(config["device"])
        except RuntimeError:
            raise ValueError(f"Invalid device specified: {config['device']}")
    
    # Validate target size
    target_size = config.get("image_processing", {}).get("target_size", [256, 64])
    if not isinstance(target_size, list) or len(target_size) != 2:
        raise ValueError("target_size must be a list of two integers [height, width]")
    
    # Validate threshold
    threshold = config.get("segmentation", {}).get("threshold", 0.5)
    if not 0 <= threshold <= 1:
        raise ValueError("segmentation threshold must be between 0 and 1")
    
    return True


class ConfigManager:
    """
    Configuration manager class for handling pipeline configurations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path (Optional[str]): Path to custom configuration file
        """
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = get_default_config()
        
        validate_config(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).
        
        Args:
            key (str): Configuration key (e.g., "segmentation.threshold")
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dots).
        
        Args:
            key (str): Configuration key (e.g., "segmentation.threshold")
            value (Any): Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path (str): Path where to save the configuration
        """
        save_config(self.config, output_path)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates (Dict[str, Any]): Dictionary of updates to apply
        """
        self.config = merge_configs(self.config, updates)
        validate_config(self.config) 