"""
Two-stage MRBNO training: Stage 1 in PCA space, Stage 2 in real space.

Stage 1: Train entirely in PCA space (loss in PCA coefficients).
Removes fc0 (real-space input) and fc2 (real-space output), keeps blocks.
Stage 2: Add back fc0/fc2, freeze blocks, fine-tune on real-space data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class PCASpectralConv2d(nn.Module):
    """
    PCA-based spectral convolution.
    """
    def __init__(self, in_channels, out_channels, n_modes: int, pca_components, pca_mean):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        max_available_modes = pca_components.shape[0]
        self.n_modes = min(n_modes, max_available_modes)
        
        if n_modes > max_available_modes:
            print(f"Warning: Requested {n_modes} modes but only {max_available_modes} available. Using {self.n_modes}.")
        
        self.register_buffer('pca_components', torch.from_numpy(pca_components[:self.n_modes, :].astype(np.float32)))
        self.register_buffer('pca_mean', torch.from_numpy(pca_mean.astype(np.float32)))
        
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.n_modes) * scale
        )
    
    def forward(self, x):
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        spatial_dim = H * W
        
        x_flat = x.view(B, C, spatial_dim)
        x_centered = x_flat - self.pca_mean.unsqueeze(0).unsqueeze(0)
        x_pca = torch.einsum('bcd,md->bcm', x_centered, self.pca_components)
        
        out_pca = torch.einsum('bcm,com->bom', x_pca, self.weight)
        
        out_flat = torch.einsum('bom,md->bod', out_pca, self.pca_components)
        out_flat = out_flat + self.pca_mean.unsqueeze(0).unsqueeze(0)
        
        x_out = out_flat.view(B, self.out_channels, H, W)
        return x_out


class MRBNOBlock2d(nn.Module):
    """
    MRBNO block with PCASpectralConv2d.
    """
    def __init__(self, width, n_modes, pca_components, pca_mean, norm="group", dropout_p=0.0, residual=True):
        super().__init__()
        self.spectral = PCASpectralConv2d(width, width, n_modes, pca_components, pca_mean)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()
        self.residual = residual

        if norm == "group":
            num_groups = min(8, width)
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
        y = self.spectral(x) + self.w(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return x + y if self.residual else y


class MRBNOStage1(nn.Module):
    """
    Stage 1: Train entirely in PCA space.
    
    Input/output are PCA coefficients. Removes fc0 and fc2 (real-space I/O).
    Blocks operate in latent space with spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, width, n_modes_in, n_modes_out, n_layers, H, W,
                 norm="group", dropout_p=0.0, residual=True,
                 pca_components_in=None, pca_mean_in=None, pca_components_out=None, pca_mean_out=None):
        super().__init__()
        
        if pca_components_in is None or pca_mean_in is None or pca_components_out is None or pca_mean_out is None:
            raise ValueError("All PCA components and means must be provided")
        
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        if pca_components_in.ndim == 3:
            max_modes_in = pca_components_in.shape[1]
        else:
            max_modes_in = pca_components_in.shape[0]
        if pca_components_out.ndim == 3:
            max_modes_out = pca_components_out.shape[1]
        else:
            max_modes_out = pca_components_out.shape[0]
        self.n_modes_in = min(n_modes_in, max_modes_in)
        self.n_modes_out = min(n_modes_out, max_modes_out)
        self.H = H
        self.W = W
        
        if pca_components_in.ndim == 3:
            self.register_buffer('pca_components_in', torch.from_numpy(pca_components_in[:, :self.n_modes_in, :].astype(np.float32)))
            self.register_buffer('pca_mean_in', torch.from_numpy(pca_mean_in.astype(np.float32)))
        else:
            self.register_buffer('pca_components_in', torch.from_numpy(pca_components_in[:self.n_modes_in, :].astype(np.float32)))
            self.register_buffer('pca_mean_in', torch.from_numpy(pca_mean_in.astype(np.float32)))
        
        if pca_components_out.ndim == 3:
            self.register_buffer('pca_components_out', torch.from_numpy(pca_components_out[:, :self.n_modes_out, :].astype(np.float32)))
            self.register_buffer('pca_mean_out', torch.from_numpy(pca_mean_out.astype(np.float32)))
        else:
            self.register_buffer('pca_components_out', torch.from_numpy(pca_components_out[:self.n_modes_out, :].astype(np.float32)))
            self.register_buffer('pca_mean_out', torch.from_numpy(pca_mean_out.astype(np.float32)))
        
        self.fc0_pca = nn.Conv2d(in_channels, width, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            MRBNOBlock2d(width, n_modes_out, pca_components_out, pca_mean_out, 
                        norm=norm, dropout_p=dropout_p, residual=residual)
            for _ in range(n_layers)
        ])
        
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2_pca = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()
        
    def _decode_pca_to_spatial(self, x_pca):
        """
        Decode PCA coefficients to spatial representation for blocks.
        x_pca: (B, in_channels, n_modes_in)
        returns: (B, in_channels, H, W)
        """
        B = x_pca.shape[0]
        
        x_flat_pca = x_pca.view(B, self.in_channels, self.n_modes_in)
        if self.pca_components_in.ndim == 3:
            x_flat_list = []
            for c in range(self.in_channels):
                x_flat_ch = torch.einsum('bm,md->bd', x_flat_pca[:, c, :], self.pca_components_in[c])
                x_flat_ch = x_flat_ch + self.pca_mean_in[c].unsqueeze(0)
                x_flat_list.append(x_flat_ch)
            x_flat = torch.stack(x_flat_list, dim=1)
        else:
            x_flat = torch.einsum('bcm,md->bcd', x_flat_pca, self.pca_components_in)
            x_flat = x_flat + self.pca_mean_in.unsqueeze(0).unsqueeze(0)
        x_spatial = x_flat.view(B, self.in_channels, self.H, self.W)
        
        return x_spatial
    
    def _encode_spatial_to_pca(self, x_spatial):
        """
        Encode spatial representation to PCA coefficients.
        x_spatial: (B, out_channels, H, W)
        returns: (B, out_channels, n_modes_out)
        """
        B = x_spatial.shape[0]
        
        x_flat = x_spatial.view(B, self.out_channels, self.H * self.W)
        if self.pca_components_out.ndim == 3:
            x_pca_list = []
            for c in range(self.out_channels):
                x_centered_ch = x_flat[:, c, :] - self.pca_mean_out[c].unsqueeze(0)
                x_pca_ch = torch.einsum('bd,md->bm', x_centered_ch, self.pca_components_out[c])
                x_pca_list.append(x_pca_ch)
            x_pca = torch.stack(x_pca_list, dim=1)
        else:
            x_centered = x_flat - self.pca_mean_out.unsqueeze(0).unsqueeze(0)
            x_pca = torch.einsum('bcd,md->bcm', x_centered, self.pca_components_out)
        
        return x_pca
        
    def forward(self, x_pca):
        """
        x_pca: (B, in_channels, n_modes_in) - PCA coefficients
        returns: (B, out_channels, n_modes_out) - PCA coefficients
        """
        x = self._decode_pca_to_spatial(x_pca)
        x = self.fc0_pca(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.act(self.fc1(x))
        x = self.fc2_pca(x)
        out_pca = self._encode_spatial_to_pca(x)
        
        return out_pca


class MRBNOStage2(nn.Module):
    """
    Stage 2: Add back real-space layers, freeze blocks.
    
    Uses pre-trained Stage 1 blocks (frozen) and adds fc0/fc2 for real-space I/O.
    """
    def __init__(self, stage1_model, in_channels, out_channels, width, add_coords=False, freeze_blocks=True):
        super().__init__()
        
        self.width = width
        self.add_coords = add_coords
        
        lift_in = in_channels + (2 if add_coords else 0)
        self.fc0 = nn.Conv2d(lift_in, width, kernel_size=1)
        
        self.blocks = stage1_model.blocks
        if freeze_blocks:
            for param in self.blocks.parameters():
                param.requires_grad = False
        
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()
        
    def _concat_coords(self, x):
        B, _, H, W = x.shape
        device = x.device
        ys = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xs, ys], dim=1)
        
    def forward(self, x):
        """
        x: (B, C_in, H, W) - real space
        returns: (B, C_out, H, W) - real space
        """
        if self.add_coords:
            x = self._concat_coords(x)
        x = self.fc0(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
