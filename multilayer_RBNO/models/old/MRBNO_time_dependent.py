"""
Simplified Reduced Basis Neural Operator (MRBNO) with FNO-like architecture.
Mirrors FNO structure but replaces Fourier basis with PCA basis from a fitted PCAReducer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAConv2d(nn.Module):
    """
    PCA-based convolution layer analogous to SpectralConv2d in FNO.
    Uses the same PCA components for encoding and decoding.
    
    Encodes spatial dims to PCA coefficients, applies learned transformation,
    then decodes back to spatial domain.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 pca_components: np.ndarray, pca_mean: np.ndarray, k_modes: int):
        """
        Args:
            in_channels: Number of input channels (width)
            out_channels: Number of output channels (width)
            pca_components: (n_components, H*W) PCA basis vectors from rb.pca_*.components_
            pca_mean: (H*W,) PCA mean vector from rb.pca_*.mean_
            k_modes: Number of PCA modes to keep (analogous to Fourier modes)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_modes = min(k_modes, pca_components.shape[0])
        
        # Register PCA components and mean as buffers (not trained)
        components = pca_components[:self.k_modes, :].astype(np.float32)
        mean = pca_mean.astype(np.float32)
        
        self.register_buffer("components", torch.from_numpy(components))  # (k, H*W)
        self.register_buffer("mean", torch.from_numpy(mean))  # (H*W,)
        
        # Learned weights in PCA coefficient space: (C_in, C_out, k)
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.k_modes) * scale
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W) real
        returns: (B, C_out, H, W) real
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: (B, C, H*W)
        x_flat = x.view(B, C, H * W)
        
        # Center the data
        x_centered = x_flat - self.mean.unsqueeze(0).unsqueeze(0)  # (B, C, H*W)
        
        # Encode to PCA coefficients: (B, C, H*W) @ (H*W, k) -> (B, C, k)
        x_pca = torch.matmul(x_centered, self.components.T)
        
        # Transform in PCA space with channel mixing: (B, C_in, k) -> (B, C_out, k)
        out_pca = torch.einsum("bck,cok->bok", x_pca, self.weight)
        
        # Decode back to spatial domain: (B, C_out, k) @ (k, H*W) -> (B, C_out, H*W)
        out_flat = torch.matmul(out_pca, self.components)
        
        # Add mean back
        out_flat = out_flat + self.mean.unsqueeze(0).unsqueeze(0)
        
        # Reshape to spatial: (B, C_out, H, W)
        out = out_flat.view(B, self.out_channels, H, W)
        return out


class MRBNOBlock2d(nn.Module):
    """
    MRBNO block analogous to FNOBlock2d.
    Replaces spectral convolution with PCA-based convolution.
    """
    def __init__(self, width: int, pca_components: np.ndarray, pca_mean: np.ndarray, 
                 k_modes: int, norm: str = "group", dropout_p: float = 0.0, 
                 residual: bool = True):
        super().__init__()
        self.pca_conv = PCAConv2d(width, width, pca_components, pca_mean, k_modes)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()
        self.residual = residual
        
        if norm == "group":
            self.norm = nn.GroupNorm(num_groups=min(8, width), num_channels=width)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(width, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(width)
        else:
            self.norm = nn.Identity()
        
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pca_conv(x) + self.w(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return x + y if self.residual else y


class MRBNOTimeDependent(nn.Module):
    """
    Multi-Resolution Basis Neural Operator for 2D problems.
    Mirrors FNO2d architecture with PCA basis instead of Fourier basis.
    
    Args:
        rb: Fitted PCAReducerMultiChannel object containing pca_in/pca_out
        in_channels: Number of input channels (default: uses rb.n_channels)
        out_channels: Number of output channels (default: uses rb.out_channels or 1)
        width: Hidden dimension width
        k_modes: Number of PCA modes to keep (analogous to Fourier modes)
        n_layers: Number of MRBNO blocks
        add_coords: Whether to add coordinate channels
        norm: Normalization type ("group", "instance", "batch", or None)
        dropout_p: Dropout probability
        residual: Whether to use residual connections in blocks
        use_input_basis: If True, use pca_in; if False, use pca_out for the PCA basis
    """
    def __init__(
        self, 
        rb,
        in_channels: int = None,
        out_channels: int = None, 
        width: int = 32,
        k_modes: int = 64, 
        n_layers: int = 4,
        add_coords: bool = True, 
        norm: str = "group", 
        dropout_p: float = 0.0,
        residual: bool = True,
        use_input_basis: bool = True,
    ):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")
        
        # Extract dimensions from rb
        self.N, self.M = rb.shape_
        self.C_in = in_channels if in_channels is not None else rb.n_channels
        self.C_out = out_channels if out_channels is not None else getattr(rb, 'out_channels', 1)
        
        self.width = width
        self.add_coords = add_coords
        self.k_modes = k_modes
        self.use_input_basis = use_input_basis
        
        # Get PCA components and mean from rb
        if use_input_basis:
            pca_components = rb.pca_in.components_.astype(np.float32)
            pca_mean = rb.pca_in.mean_.astype(np.float32)
        else:
            pca_components = rb.pca_out.components_.astype(np.float32)
            pca_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Validate k_modes
        max_k = pca_components.shape[0]
        if k_modes > max_k:
            print(f"Warning: k_modes={k_modes} > available components={max_k}, using {max_k}")
            self.k_modes = max_k
        
        # Store for blocks
        self._pca_components = pca_components
        self._pca_mean = pca_mean
        
        # Lifting layer
        lift_in = self.C_in + (2 if add_coords else 0)
        self.fc0 = nn.Conv2d(lift_in, width, kernel_size=1)
        
        # MRBNO blocks (all share the same PCA basis)
        self.blocks = nn.ModuleList([
            MRBNOBlock2d(
                width, 
                pca_components, 
                pca_mean, 
                self.k_modes, 
                norm=norm,
                dropout_p=dropout_p, 
                residual=residual
            )
            for _ in range(n_layers)
        ])
        
        # Projection layers
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, self.C_out, kernel_size=1)
        self.act = nn.GELU()
    
    def _concat_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Add normalized coordinate channels."""
        B, _, H, W = x.shape
        device = x.device
        dtype = x.dtype
        ys = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xs, ys], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H, W)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, H, W), got {x.shape}")
        
        if self.add_coords:
            x = self._concat_coords(x)
        
        x = self.fc0(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x