#!/usr/bin/env python3
"""PCA Net: Simple neural network operating in PCA coefficient space."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from .PCAReducer import PCAReducerMultiChannel


class PCANet(nn.Module):
    """
    PCA Net: Neural network that operates in PCA coefficient space.
    
    1. Encodes multi-channel input to PCA coefficients (using input PCA)
    2. Applies MLP transformation in coefficient space
    3. Decodes back to function space using output PCA
    
    Uses a single MLP instead of residual blocks.
    """
    
    def __init__(
        self,
        rb: PCAReducerMultiChannel,
        k_in: int,
        k_out: int,
        hidden: int = 256,
        depth: int = 3,
        dropout: float = 0.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)
        self.D_in = self.C_in * self.N * self.M  # Total input dimension
        self.D_out = self.C_out * self.N * self.M  # Output dimension
        
        # Validate k values
        max_k_in = rb.pca_in.n_components_
        max_k_out = rb.pca_out.n_components_
        if not (1 <= k_in <= max_k_in):
            raise ValueError(f"k_in={k_in} out of range [1, {max_k_in}]")
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")
        
        self.k_in = k_in
        self.k_out = k_out
        
        # Setup encoder: use input PCA
        enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
        enc_mean = rb.pca_in.mean_.astype(np.float32)
        
        # Setup decoder: use output PCA
        dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
        dec_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Register as buffers
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))  # (D_in, k_in)
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))  # (D_in,)
        self.register_buffer("dec_components", torch.from_numpy(dec_components))  # (k_out, D_out)
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))  # (D_out,)
        
        # MLP in coefficient space
        layers = []
        layers.append(nn.Linear(k_in, hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden, k_out))
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights(init_scale)
    
    def _init_weights(self, init_scale: float):
        """Initialize weights with small scale."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Project to PCA coefficient space."""
        x_centered = x_flat - self.enc_mean
        return x_centered @ self.enc_components_t  # (B, k_in)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from PCA coefficients."""
        return z @ self.dec_components + self.dec_mean  # (B, D_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, N, M)
        
        Returns:
            Output tensor of shape (B, C_out, N, M)
        """
        B = x.shape[0]
        
        # Flatten input
        x_flat = x.reshape(B, self.D_in)  # (B, D_in)
        
        # Encode to PCA coefficients
        z_in = self.encode(x_flat)  # (B, k_in)
        
        # Apply MLP in coefficient space
        z_out = self.mlp(z_in)  # (B, k_out)
        
        # Decode back to function space
        y_flat = self.decode(z_out)  # (B, D_out)
        
        # Reshape to spatial dimensions (D_out = C_out * N * M)
        y = y_flat.reshape(B, self.C_out, self.N, self.M)
        
        return y

