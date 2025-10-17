"""
Multilayer Reduced Basis Neural Operator (MRBNO) for Multi-Channel Inputs
Handles 2D fields with multiple input channels
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.decomposition import PCA
from MRBNO_model import PCAReducer, ResidualBlock, ScalarGate


class PCAReducerMultiChannel:
    """
    PCA-based reduced-basis helper for multi-channel (num_samples, C_in, N, N) inputs
    and single-channel (num_samples, N, N) outputs.
    """
    
    def __init__(self, svd_solver='auto', whiten=False, random_state=None):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.pca_in: Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.n_channels = None
        self.shape_ = None  # (N, N)

    def fit(self, samples: np.ndarray, solution: np.ndarray):
        """
        samples: (num_samples, C_in, N, N) - multi-channel input
        solution: (num_samples, N, N) - single-channel output
        """
        if samples.ndim == 4:
            num_samples, C_in, N, M = samples.shape
            self.n_channels = C_in
            self.shape_ = (N, M)
            # Flatten channels into spatial dimensions: (num_samples, C_in*N*N)
            X_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Expected (num_samples, C_in, N, N), got {samples.shape}")
        
        # Output: standard flattening
        assert solution.ndim == 3, f"Expected (num_samples, N, N), got {solution.shape}"
        Y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)
        
        self.pca_in = PCA(n_components=None, svd_solver=self.svd_solver,
                          whiten=self.whiten, random_state=self.random_state).fit(X_in)
        self.pca_out = PCA(n_components=None, svd_solver=self.svd_solver,
                           whiten=self.whiten, random_state=self.random_state).fit(Y_out)
        
        return self

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99) -> tuple:
        """Return (k_in, k_out) meeting given variance cutoffs."""
        assert self.pca_in and self.pca_out
        k_in  = self._k_from_cutoff(self.pca_in.explained_variance_ratio_,  j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out
    
    @property
    def input_components(self) -> np.ndarray:
        """(n_in_components, D) principal directions for inputs."""
        assert self.pca_in
        return self.pca_in.components_

    @property
    def output_components(self) -> np.ndarray:
        """(n_out_components, D) principal directions for outputs."""
        assert self.pca_out
        return self.pca_out.components_


class ReducedBasisResNetMultiChannel(nn.Module):
    """
    Core MRBNO block that handles multi-channel inputs.
    
    1. Encodes multi-channel input to PCA coefficients
    2. Applies ResNet transformation in coefficient space
    3. Decodes back to function space using output basis
    """
    
    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, k_out: int, 
                 hidden: int = 256, depth: int = 4, dropout: float = 0.0,
                 use_input_basis_for_encoder: bool = False):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.D_in = self.C_in * self.N * self.M  # Total input dimension
        self.D_out = self.N * self.M  # Output dimension
        
        # Validate k values
        max_k_in = rb.pca_in.n_components_
        max_k_out = rb.pca_out.n_components_
        if not (1 <= k_in <= max_k_in):
            raise ValueError(f"k_in={k_in} out of range [1, {max_k_in}]")
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")
        
        self.use_input_basis = use_input_basis_for_encoder
        self.k_in = k_in
        self.k_out = k_out
        
        # Setup encoder components
        if use_input_basis_for_encoder:
            self.encoder_dim = k_in
            enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            enc_mean = rb.pca_in.mean_.astype(np.float32)
        else:
            self.encoder_dim = k_out
            # Pad output PCA to match input dimensions
            enc_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
            # Pad to match input dimension
            if enc_components.shape[1] < self.D_in:
                padding = np.zeros((k_out, self.D_in - enc_components.shape[1]), dtype=np.float32)
                enc_components = np.concatenate([enc_components, padding], axis=1)
            enc_mean = rb.pca_out.mean_.astype(np.float32)
            # Pad mean as well
            if enc_mean.shape[0] < self.D_in:
                enc_mean = np.pad(enc_mean, (0, self.D_in - enc_mean.shape[0]), 'constant')
        
        # Always use output PCA for decoding
        dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
        dec_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Register as buffers
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))  # (D_in, encoder_dim)
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))  # (D_in,)
        self.register_buffer("dec_components", torch.from_numpy(dec_components))  # (k_out, D_out)
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))  # (D_out,)
        
        # ResNet in coefficient space
        self.in_proj = nn.Linear(self.encoder_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden, self.k_out)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Project to PCA coefficient space."""
        x_centered = x_flat - self.enc_mean
        return x_centered @ self.enc_components_t  # (B, encoder_dim)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from PCA coefficients."""
        return z @ self.dec_components + self.dec_mean  # (B, D_out)
    
    def forward(self, x):
        """
        x: (B, C_in, N, M)
        returns: (B, 1, N, M)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        
        B, C, H, W = x.shape
        if C != self.C_in or H != self.N or W != self.M:
            raise ValueError(f"Expected (B, {self.C_in}, {self.N}, {self.M}), got {x.shape}")
        
        # Flatten all channels
        x_flat = x.reshape(B, self.D_in)
        
        # Encode to coefficients
        z = self.encode(x_flat)  # (B, encoder_dim)
        
        # Transform in coefficient space
        h = F.gelu(self.in_proj(z))  # (B, hidden)
        for block in self.blocks:
            h = block(h)  # (B, hidden)
        z_out = self.out_proj(h)  # (B, k_out)
        
        # Decode back to function space
        y_flat = self.decode(z_out)  # (B, D_out)
        y = y_flat.reshape(B, 1, self.N, self.M)
        
        return y


class MultiLayerReducedBasisResNetMultiChannel(nn.Module):
    """
    Stacks multiple ReducedBasisResNet blocks with residual connections.
    Handles multi-channel inputs (C_in, N, N) and outputs (1, N, N).
    """
    
    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, k_out: int,
                 layers: int = 4, hidden: int = 256, depth: int = 4,
                 dropout: float = 0.0, init_scale: float = 0.1):
        super().__init__()
        
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.layers = layers
        
        # First block: input basis -> output basis
        self.block0 = ReducedBasisResNetMultiChannel(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, depth=depth, dropout=dropout,
            use_input_basis_for_encoder=True
        )
        self.gate0 = ScalarGate(init=init_scale)
        
        # Subsequent blocks: output basis -> output basis
        # These need special handling since output is single channel
        if layers > 0:
            # Create a modified PCAReducer for subsequent blocks
            rb_single = PCAReducerMultiChannel()
            rb_single.pca_in = rb.pca_out  # Use output PCA for both
            rb_single.pca_out = rb.pca_out
            rb_single.n_channels = 1  # Single channel
            rb_single.shape_ = rb.shape_
            
            self.blocks = nn.ModuleList([
                ReducedBasisResNetMultiChannel(
                    rb_single, k_in=k_out, k_out=k_out,
                    hidden=hidden, depth=depth, dropout=dropout,
                    use_input_basis_for_encoder=False
                ) for _ in range(layers)
            ])
            
            self.gates = nn.ModuleList([ScalarGate(init=init_scale) for _ in range(layers)])
        else:
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()
        
        self.act = nn.GELU()
    
    def forward(self, x):
        """
        x: (B, C_in, N, M) multi-channel input
        returns: (B, 1, N, M) single-channel output
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        
        # First block with multi-channel input
        y = self.gate0(self.block0(x))  # (B, 1, N, M)
        
        # Subsequent blocks with single-channel input
        for block, gate in zip(self.blocks, self.gates):
            y = self.act(y)  # Nonlinearity between blocks
            y = y + gate(block(y))  # Residual connection
        
        return y

