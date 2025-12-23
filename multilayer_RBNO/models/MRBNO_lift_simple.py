"""
MRBNO Lift Simple: Architecture using PCA basis instead of Fourier basis.

Uses PCA encoding/decoding instead of Fourier transforms. Block structure and skip/residual connections are the same as FNO2d.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PCASpectralConv2d(nn.Module):
    """
    PCA-based spectral convolution, analogous to FNO's SpectralConv2d.
    
    Instead of FFT → multiply in frequency domain → IFFT,
    this does: flatten → encode to PCA → multiply in PCA space → decode → reshape.
    
    The PCA basis is computed once over the whole dataset and shared across all blocks.
    """
    def __init__(self, in_channels, out_channels, n_modes: int, pca_components, pca_mean):
        """
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
            n_modes: Number of PCA modes to keep (analogous to modes_x * modes_y in FNO)
            pca_components: PCA components matrix (n_modes, H*W) - numpy array
            pca_mean: PCA mean vector (H*W,) - numpy array
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Use actual number of available components (min of requested and available)
        max_available_modes = pca_components.shape[0]
        self.n_modes = min(n_modes, max_available_modes)
        
        if n_modes > max_available_modes:
            print(f"Warning: Requested {n_modes} modes but only {max_available_modes} available. Using {self.n_modes}.")
        
        # Register PCA components and mean as buffers (non-trainable)
        # pca_components is (n_modes, H*W) - keep it in this form for easier einsum
        self.register_buffer('pca_components', torch.from_numpy(pca_components[:self.n_modes, :].astype(np.float32)))
        self.register_buffer('pca_mean', torch.from_numpy(pca_mean.astype(np.float32)))
        
        # Learnable weights for mixing in PCA space: (C_in, C_out, n_modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.n_modes) * scale
        )
    
    def forward(self, x):
        """
        x: (B, C_in, H, W) real
        returns: (B, C_out, H, W) real
        """
        B, C, H, W = x.shape
        spatial_dim = H * W
        
        # Flatten spatial dimensions: (B, C_in, H*W)
        x_flat = x.view(B, C, spatial_dim)
        
        # Encode to PCA coefficients: (B, C_in, n_modes)
        # Center data
        x_centered = x_flat - self.pca_mean.unsqueeze(0).unsqueeze(0)  # (B, C_in, H*W)
        # Project to PCA space
        x_pca = torch.einsum('bcd,md->bcm', x_centered, self.pca_components)  # (B, C_in, n_modes)
        
        # Mix in PCA space: (B, C_out, n_modes)
        out_pca = torch.einsum('bcm,com->bom', x_pca, self.weight)  # (B, C_out, n_modes)
        
        # Decode from PCA coefficients: (B, C_out, H*W)
        # pca_components is (n_modes, H*W), so we use it directly (transposed in einsum)
        out_flat = torch.einsum('bom,md->bod', out_pca, self.pca_components)  # (B, C_out, H*W)
        out_flat = out_flat + self.pca_mean.unsqueeze(0).unsqueeze(0)  # Add mean back
        
        # Reshape back to spatial dimensions: (B, C_out, H, W)
        x_out = out_flat.view(B, self.out_channels, H, W)
        return x_out


class MRBNOBlock2d(nn.Module):
    """
    MRBNO block with same structure as FNOBlock2d.
    
    Uses PCASpectralConv2d instead of SpectralConv2d. Skip/residual structure is the same as FNOBlock2d.
    """
    def __init__(self, width, n_modes, pca_components, pca_mean, norm="group", dropout_p=0.0, residual=True):
        super().__init__()
        self.spectral = PCASpectralConv2d(width, width, n_modes, pca_components, pca_mean)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()
        self.residual = residual

        if norm == "group":
            # Use adaptive number of groups: at most 8, but must divide width
            num_groups = min(8, width)
            # Find largest divisor of width that is <= 8
            for g in range(min(8, width), 0, -1):
                if width % g == 0:
                    num_groups = g
                    break
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=width)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(width, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(width)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        """
        x: (B, width, H, W)
        returns: (B, width, H, W)
        """
        # spectral(x) + local(x) → norm → act → dropout → residual
        y = self.spectral(x) + self.w(x)
        y = self.norm(y)
        y = self.act(y)              
        y = self.dropout(y)
        return x + y if self.residual else y


class MRBNOLiftSimple(nn.Module):
    """
    MRBNO Lift Simple: Architecture using PCA basis.
    
    Structure is the same as FNO2d, with PCA encoding/decoding replacing Fourier transforms.
    All blocks have the same structure, and skip/residual connections are the same as FNO2d.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Hidden dimension width
        n_modes: Number of PCA modes to keep (analogous to modes_x * modes_y in FNO)
        n_layers: Number of MRBNO blocks
        add_coords: Whether to add coordinate channels (same as FNO: concatenation)
        norm: Normalization type ("group", "instance", "batch", or None)
        dropout_p: Dropout probability
        residual: Whether to use residual connections
        pca_components: PCA components matrix (n_components, H*W) - numpy array
        pca_mean: PCA mean vector (H*W,) - numpy array
    """
    def __init__(self, in_channels=3, out_channels=1, width=32, n_modes=256, n_layers=3,
                 add_coords=False, norm="group", dropout_p=0.0, residual=True,
                 pca_components=None, pca_mean=None):
        super().__init__()
        
        if pca_components is None or pca_mean is None:
            raise ValueError("pca_components and pca_mean must be provided")
        
        self.width = width
        self.add_coords = add_coords

        # Lifting layer (same as FNO2d)
        lift_in = in_channels + (2 if add_coords else 0)
        self.fc0 = nn.Conv2d(lift_in, width, kernel_size=1)
        
        # Blocks (same structure as FNO2d)
        self.blocks = nn.ModuleList([
            MRBNOBlock2d(width, n_modes, pca_components, pca_mean, 
                        norm=norm, dropout_p=dropout_p, residual=residual)
            for _ in range(n_layers)
        ])
        
        # Final projection layers (same as FNO2d)
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def _concat_coords(self, x):
        """
        Coordinate concatenation (same as FNO2d).
        x: (B,C,H,W)
        returns: (B,C+2,H,W)
        """
        B, _, H, W = x.shape
        device = x.device
        ys = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x):
        """
        Forward pass (same structure as FNO2d.forward).
        x: (B, C_in, H, W)
        returns: (B, C_out, H, W)
        """
        if self.add_coords:
            x = self._concat_coords(x)
        x = self.fc0(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

