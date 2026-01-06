"""
MRBNO Lift Irregular: Architecture for arbitrary grids/geometries using PCA basis.

Uses PCA encoding/decoding on point clouds. Assumes constant mesh (same node positions
across all samples) but arbitrary node positions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PCASpectralConv1d(nn.Module):
    """
    PCA-based spectral convolution for point clouds.
    
    Instead of FFT → multiply in frequency domain → IFFT,
    this does: encode to PCA → multiply in PCA space → decode.
    
    The PCA basis is computed once over the whole dataset and shared across all blocks.
    """
    def __init__(self, in_channels, out_channels, n_modes: int, pca_components, pca_mean):
        """
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
            n_modes: Number of PCA modes to keep
            pca_components: PCA components matrix (n_modes, N) - numpy array
            pca_mean: PCA mean vector (N,) - numpy array
        """
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
        x: (B, C_in, N) - point cloud data
        returns: (B, C_out, N)
        """
        B, C, N = x.shape
        
        x_centered = x - self.pca_mean.unsqueeze(0).unsqueeze(0)  # (B, C_in, N)
        x_pca = torch.einsum('bcn,mn->bcm', x_centered, self.pca_components)  # (B, C_in, n_modes)
        
        out_pca = torch.einsum('bcm,com->bom', x_pca, self.weight)  # (B, C_out, n_modes)
        
        out_flat = torch.einsum('bom,mn->bon', out_pca, self.pca_components)  # (B, C_out, N)
        out_flat = out_flat + self.pca_mean.unsqueeze(0).unsqueeze(0)
        
        return out_flat


class MRBNOBlock1d(nn.Module):
    """
    MRBNO block for point clouds.
    
    Uses PCASpectralConv1d instead of SpectralConv2d. Skip/residual structure is the same.
    """
    def __init__(self, width, n_modes, pca_components, pca_mean, norm="group", dropout_p=0.0, residual=True):
        super().__init__()
        self.spectral = PCASpectralConv1d(width, width, n_modes, pca_components, pca_mean)
        self.w = nn.Conv1d(width, width, kernel_size=1)
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
            self.norm = nn.InstanceNorm1d(width, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(width)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        """
        x: (B, width, N)
        returns: (B, width, N)
        """
        y = self.spectral(x) + self.w(x)
        y = self.norm(y)
        y = self.act(y)              
        y = self.dropout(y)
        return x + y if self.residual else y


class MRBNOLiftIrregular(nn.Module):
    """
    MRBNO Lift Irregular: Architecture using PCA basis for arbitrary grids.
    
    Structure is similar to FNO2d/MRBNOLiftSimple, adapted for point clouds.
    Assumes constant mesh (same node positions for all samples).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Hidden dimension width
        n_modes: Number of PCA modes to keep
        n_layers: Number of MRBNO blocks
        add_coords: Whether to add coordinate channels
        norm: Normalization type ("group", "instance", "batch", or None)
        dropout_p: Dropout probability
        residual: Whether to use residual connections
        pca_components: PCA components matrix (n_components, N) - numpy array
        pca_mean: PCA mean vector (N,) - numpy array
        mesh_coords: Mesh coordinates (N, 2) or (N, 3) - numpy array
    """
    def __init__(self, in_channels=1, out_channels=1, width=32, n_modes=256, n_layers=3,
                 add_coords=False, norm="group", dropout_p=0.0, residual=True,
                 pca_components=None, pca_mean=None, mesh_coords=None):
        super().__init__()
        
        if pca_components is None or pca_mean is None:
            raise ValueError("pca_components and pca_mean must be provided")
        
        if add_coords and mesh_coords is None:
            raise ValueError("mesh_coords must be provided when add_coords=True")
        
        self.width = width
        self.add_coords = add_coords
        
        if mesh_coords is not None:
            mesh_coords = torch.from_numpy(mesh_coords.astype(np.float32))
            if mesh_coords.ndim == 2:
                coord_dim = mesh_coords.shape[1]
            else:
                coord_dim = 1
                mesh_coords = mesh_coords.unsqueeze(1)
            self.register_buffer('mesh_coords', mesh_coords)  # (N, coord_dim)
            self.coord_dim = coord_dim
        else:
            self.mesh_coords = None
            self.coord_dim = 0

        lift_in = in_channels + (self.coord_dim if add_coords else 0)
        self.fc0 = nn.Conv1d(lift_in, width, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            MRBNOBlock1d(width, n_modes, pca_components, pca_mean, 
                        norm=norm, dropout_p=dropout_p, residual=residual)
            for _ in range(n_layers)
        ])
        
        self.fc1 = nn.Conv1d(width, width, kernel_size=1)
        self.fc2 = nn.Conv1d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def _concat_coords(self, x):
        """
        Coordinate concatenation using actual mesh coordinates.
        x: (B, C, N)
        returns: (B, C+coord_dim, N)
        """
        B, _, N = x.shape
        coords = self.mesh_coords.T.unsqueeze(0).expand(B, -1, -1)  # (B, coord_dim, N)
        return torch.cat([x, coords], dim=1)

    def forward(self, x):
        """
        Forward pass.
        x: (B, C_in, N) - point cloud data
        returns: (B, C_out, N)
        """
        if self.add_coords:
            x = self._concat_coords(x)
        x = self.fc0(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
