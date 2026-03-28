"""
Attention U-Net Architecture for Medical Image Segmentation

This module implements an Attention U-Net model specifically designed for segmenting
anatomical structures in medical images, particularly DRR (Digitally Reconstructed Radiographs).

The Attention U-Net incorporates attention mechanisms to focus on relevant features
and suppress irrelevant ones during the upsampling process.

References:
    Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.
    arXiv preprint arXiv:1804.03999.

Author: Maxime Huppe
Institution: Imperial College London
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionBlock(nn.Module):
    """
    Attention Block for focusing on relevant features.
    
    This block implements the attention mechanism that allows the model to focus
    on relevant activations while suppressing irrelevant ones. It takes gating
    signals from the decoder path and skip connections from the encoder path
    to generate attention coefficients.
    
    Args:
        F_g (int): Number of feature channels from gating signal (decoder)
        F_l (int): Number of feature channels from encoder skip connection  
        F_int (int): Number of intermediate feature channels for attention computation
        
    Example:
        >>> attention = AttentionBlock(F_g=512, F_l=256, F_int=256)
        >>> gating_signal = torch.randn(1, 512, 32, 32)
        >>> skip_connection = torch.randn(1, 256, 32, 32)
        >>> attended_features = attention(gating_signal, skip_connection)
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionBlock, self).__init__()
        
        # Gating signal processing (from decoder path)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection processing (from encoder path)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficient generation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention block.
        
        Args:
            g (torch.Tensor): Gating signal from decoder path, shape (B, F_g, H, W)
            x (torch.Tensor): Feature map from encoder path (skip connection), shape (B, F_l, H, W)
            
        Returns:
            torch.Tensor: Attention-weighted feature map, shape (B, F_l, H, W)
        """
        # Process gating signal and skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine and generate attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention weights to skip connection
        return x * psi


class ConvBlock(nn.Module):
    """
    Convolutional Block with Batch Normalization and ReLU activation.
    
    Standard building block consisting of two consecutive convolution operations,
    each followed by batch normalization and ReLU activation. This is the basic
    building block used throughout the U-Net architecture.
    
    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
        
    Example:
        >>> conv_block = ConvBlock(64, 128)
        >>> input_tensor = torch.randn(1, 64, 256, 256)
        >>> output = conv_block(input_tensor)  # Shape: (1, 128, 256, 256)
    """
    
    def __init__(self, in_ch: int, out_ch: int):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of convolutional block.
        
        Args:
            x (torch.Tensor): Input feature map, shape (B, in_ch, H, W)
            
        Returns:
            torch.Tensor: Processed feature map, shape (B, out_ch, H, W)
        """
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for Medical Image Segmentation.
    
    This implementation of Attention U-Net is specifically designed for segmenting
    anatomical structures in medical images. The attention mechanism helps the model
    focus on relevant features while suppressing noise and irrelevant information.
    
    Architecture Overview:
        - **Encoder**: 4 levels of downsampling with ConvBlocks (64→128→256→512 channels)
        - **Bottleneck**: Central processing layer (1024 channels)
        - **Decoder**: 4 levels of upsampling with attention-guided skip connections
        - **Output**: Single channel segmentation mask with sigmoid activation
    
    Key Features:
        - Attention gates at each decoder level for feature selection
        - Skip connections preserve fine-grained spatial information
        - Batch normalization for stable training
        - Sigmoid output for binary segmentation
    
    Args:
        in_ch (int, optional): Number of input channels. Defaults to 1 (grayscale).
        out_ch (int, optional): Number of output channels. Defaults to 1 (binary mask).
    
    Input Shape:
        (batch_size, in_ch, height, width)
        
    Output Shape:
        (batch_size, out_ch, height, width) with values in [0, 1]
        
    Example:
        >>> model = AttentionUNet(in_ch=1, out_ch=1)
        >>> input_image = torch.randn(1, 1, 512, 512)
        >>> segmentation_mask = model(input_image)
        >>> print(segmentation_mask.shape)  # torch.Size([1, 1, 512, 512])
    """
    
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super(AttentionUNet, self).__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # Pooling layer for downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        self.down1 = ConvBlock(in_ch, 64)      # Input -> 64 channels
        self.down2 = ConvBlock(64, 128)        # 64 -> 128 channels
        self.down3 = ConvBlock(128, 256)       # 128 -> 256 channels
        self.down4 = ConvBlock(256, 512)       # 256 -> 512 channels

        # Bottleneck
        self.middle = ConvBlock(512, 1024)     # 512 -> 1024 channels

        # Decoder (Expanding Path) with Attention
        self.att4 = AttentionBlock(F_g=1024, F_l=512, F_int=512)
        self.up4 = ConvBlock(1024 + 512, 512)  # Concatenated features -> 512

        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=256)
        self.up3 = ConvBlock(512 + 256, 256)   # Concatenated features -> 256

        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=128)
        self.up2 = ConvBlock(256 + 128, 128)   # Concatenated features -> 128

        self.att1 = AttentionBlock(F_g=128, F_l=64, F_int=64)
        self.up1 = ConvBlock(128 + 64, 64)     # Concatenated features -> 64

        # Final output layer
        self.final = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Attention U-Net.
        
        The forward pass consists of:
        1. Encoder path: Progressive downsampling and feature extraction
        2. Bottleneck: Deepest feature processing
        3. Decoder path: Progressive upsampling with attention-guided skip connections
        4. Final output: Sigmoid-activated segmentation mask
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, in_ch, H, W)
            
        Returns:
            torch.Tensor: Segmentation mask with sigmoid activation, shape (B, out_ch, H, W)
            
        Raises:
            RuntimeError: If input tensor has incorrect number of channels
        """
        if x.shape[1] != self.in_ch:
            raise RuntimeError(f"Expected {self.in_ch} input channels, got {x.shape[1]}")
        
        # Encoder path (contracting)
        d1 = self.down1(x)                    # Full resolution: (B, 64, H, W)
        d2 = self.down2(self.maxpool(d1))     # 1/2 resolution: (B, 128, H/2, W/2)
        d3 = self.down3(self.maxpool(d2))     # 1/4 resolution: (B, 256, H/4, W/4)
        d4 = self.down4(self.maxpool(d3))     # 1/8 resolution: (B, 512, H/8, W/8)

        # Bottleneck
        middle = self.middle(self.maxpool(d4)) # 1/16 resolution: (B, 1024, H/16, W/16)

        # Decoder path (expanding) with attention-guided skip connections
        # Level 4: 1/8 resolution
        g4 = F.interpolate(middle, size=d4.shape[2:], mode='bilinear', align_corners=True)
        a4 = self.att4(g=g4, x=d4)  # Apply attention to skip connection
        u4 = self.up4(torch.cat([g4, a4], dim=1))

        # Level 3: 1/4 resolution
        g3 = F.interpolate(u4, size=d3.shape[2:], mode='bilinear', align_corners=True)
        a3 = self.att3(g=g3, x=d3)  # Apply attention to skip connection
        u3 = self.up3(torch.cat([g3, a3], dim=1))

        # Level 2: 1/2 resolution
        g2 = F.interpolate(u3, size=d2.shape[2:], mode='bilinear', align_corners=True)
        a2 = self.att2(g=g2, x=d2)  # Apply attention to skip connection
        u2 = self.up2(torch.cat([g2, a2], dim=1))

        # Level 1: Full resolution
        g1 = F.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=True)
        a1 = self.att1(g=g1, x=d1)  # Apply attention to skip connection
        u1 = self.up1(torch.cat([g1, a1], dim=1))

        # Final segmentation output with sigmoid activation
        return torch.sigmoid(self.final(u1))

    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            dict: Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "AttentionUNet",
            "input_channels": self.in_ch,
            "output_channels": self.out_ch,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

