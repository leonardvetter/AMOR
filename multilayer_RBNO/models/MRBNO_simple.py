"""
Multilayer Reduced Basis Neural Operator (MRBNO) for Multi-Channel Inputs and Outputs
Handles 2D fields with multiple input channels
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.decomposition import PCA
from MRBNO_model import PCAReducer, ResidualBlock, ScalarGate
from MRBNO_model_multichannel import PCAReducerMultiChannel


""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MRBNO Simple Model for Multi-Channel Inputs
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MRBNOSimple(nn.Module):
    """
    Multilayer reduced-basis model with flexible channel counts.

    - Block 0 operates in the PCA input basis (same channel width as the training data)
    - Block 1 converts from the PCA input basis to the PCA output basis
    - Blocks 2+ apply residual updates within the PCA output basis

    Skip connections:
    - Global skip: direct path from PCA input basis to PCA output basis
    - Block 0 → Block 1: Block 0 output converted to the output basis and added to Block 1
    - Blocks 2+: standard residual additions in the output basis
    """
    
    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, k_out: int,
                 layers: int = 7, hidden: int = 1024,
                 dropout: float = 0.0, init_scale: float = 0.1, 
                 use_skip_connections: bool = True,
                 latent_batch_norm: bool = False):
        super().__init__()
        
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out_basis = getattr(rb, 'out_channels', 1)
        self.layers = layers
        self.use_skip_connections = use_skip_connections
        self.latent_batch_norm = latent_batch_norm
        
        # Block 0: stays in the PCA input basis with the original input channel width
        self.block0 = ReducedBasisResNetMultiChannel(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True,
            latent_batch_norm=latent_batch_norm,
        )
        self.gate0 = ScalarGate(init=init_scale)

        
        # Block 1: converts from the PCA input basis to the PCA output basis
        self.block1 = ReducedBasisResNetMultiChannel(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False,  # Switch to output basis
            latent_batch_norm=latent_batch_norm,
        )
        self.gate1 = ScalarGate(init=init_scale)
        
        # Global skip connection: map PCA input basis directly to the output basis
        self.global_skip_proj = nn.Conv2d(self.C_in, self.C_out_basis, 1)
        self.global_skip_gate = ScalarGate(init=init_scale)
        
        # Skip connection from Block 0 to Block 1 is evaluated in the PCA output basis
        # Residual blocks operate entirely in that PCA output basis
        if layers > 1:
            self.blocks = nn.ModuleList([
                ReducedBasisBlockSingleChannel(
                    rb, k_out=k_out, hidden=hidden, dropout=dropout,
                    latent_batch_norm=latent_batch_norm,
                ) for _ in range(layers - 1)
            ])
            
            self.gates = nn.ModuleList([ScalarGate(init=init_scale) for _ in range(layers - 1)])
        else:
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()
        
        self.output_proj = nn.Conv2d(self.C_out_basis, self.C_out_basis, 1) # Changed to C_out_basis
        self.act = nn.GELU()
    
    def forward(self, x):
        """
        x: (B, C_in, N, M) tensor in the PCA input basis (C_in configurable)
        returns: (B, C_out, N, M) tensor in the PCA output basis
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        if x.shape[1] != self.C_in: # Changed to self.C_in
            raise ValueError(f"Expected {self.C_in} input channels, got {x.shape[1]}")
        
        # Global skip connection: project input directly to output basis
        global_skip = self.global_skip_gate(self.global_skip_proj(x))  # (B, C_out_basis, N, M)
        
        # Block 0: Work in the PCA input basis using its channel width
        y0_multi = self.gate0(self.block0(x))  # (B, C_in, N, M)
        y0_multi = self.act(y0_multi)
        
        if self.use_skip_connections:
            y0_multi = y0_multi + x  # Skip connection: Input → Block 0
        
        # Block 1: Convert from the PCA input basis to the PCA output basis
        y1 = self.gate1(self.block1(y0_multi))  # (B, C_out_basis, N, M)
        y1 = self.act(y1)

        # Project Block 0 output to output basis for the residual into Block 1
        y0_proj = self.global_skip_proj(y0_multi)  # (B, C_out_basis, N, M)
        
        if self.use_skip_connections:
            y = y1 + y0_proj
        else:
            y = y1
        
        y = y + global_skip
        
        for block, gate in zip(self.blocks, self.gates):
            y = self.act(y)
            residual = gate(block(y))  # (B, C_out_basis, N, M)
            if self.use_skip_connections:
                y = y + residual
            else:
                y = residual
        
        return y


""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Single-Channel Reduced Basis Block
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class ReducedBasisBlockSingleChannel(nn.Module):
    """
    Residual block operating entirely in the PCA output basis.
    Used for subsequent updates after the model has switched bases.
    """
    
    def __init__(self, rb: PCAReducerMultiChannel, k_out: int, 
                 hidden: int = 256, dropout: float = 0.0,
                 latent_batch_norm: bool = False):
        super().__init__()
        
        if rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        self.D_out = self.N * self.M  # Spatial dimension per channel
        self.C_out = getattr(rb, 'out_channels', 1)
        self.k_out = k_out
        self.latent_batch_norm = latent_batch_norm
        
        max_k_out = rb.pca_out.n_components_
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")
        
        dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
        dec_mean = rb.pca_out.mean_.astype(np.float32)
        
        self.register_buffer("dec_components", torch.from_numpy(dec_components))
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.in_proj = nn.Linear(k_out, hidden)
        self.bn = nn.BatchNorm1d(hidden) if latent_batch_norm else nn.Identity()
        self.out_proj = nn.Linear(hidden, k_out)
        self.latent_net = nn.Sequential(
            self.in_proj,
            self.bn,
            self.dropout,
            self.out_proj
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Project to the PCA output basis coefficients."""
        x_centered = x_flat - self.dec_mean
        return x_centered @ self.dec_components.T  # (B, k_out)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from PCA output basis coefficients."""
        return z @ self.dec_components + self.dec_mean  # (B, D_out)
    
    def forward(self, x):
        """
        x: (B, C_out, N, M) input in output PCA basis
        returns: (B, C_out, N, M)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_out, N, M), got {x.shape}")
        B, C, H, W = x.shape
        if C != self.C_out or H != self.N or W != self.M:
            raise ValueError(f"Expected (B, {self.C_out}, {self.N}, {self.M}), got {x.shape}")
        
        x_flat = x.reshape(B, -1)
        # Encode to coefficients (flatten per channel then project)
        z = self.encode(x_flat)  # (B, k_out)

        z_out = self.latent_net(z)
        y_flat = self.decode(z_out)
        y = y_flat.reshape(B, self.C_out, self.N, self.M)
        
        return y
    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Multi-Channel Reduced Basis Block
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class ReducedBasisResNetMultiChannel(nn.Module):
    """
    Core MRBNO block that handles tensors prepared in the PCA input basis.
    
    1. Encodes to PCA coefficients
    2. Applies a latent transformation
    3. Decodes back using either the input or output basis
    """
    
    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, k_out: int, 
                 hidden: int = 256, dropout: float = 0.0,
                 use_input_basis_for_encoder: bool = False, use_input_basis_for_decoder: bool = False,
                 latent_batch_norm: bool = False):
        super().__init__()
        
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)
        self.D_in = self.C_in * self.N * self.M
        self.D_out = self.C_out * self.N * self.M
        self.latent_batch_norm = latent_batch_norm

        if use_input_basis_for_encoder and use_input_basis_for_decoder:
            self.stay_multi_channel = True
            self.output_dim = self.D_in
        else:
            self.stay_multi_channel = False
            self.output_dim = self.D_out

        max_k_in = rb.pca_in.n_components_
        max_k_out = rb.pca_out.n_components_
        if not (1 <= k_in <= max_k_in):
            raise ValueError(f"k_in={k_in} out of range [1, {max_k_in}]")
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")
        
        self.use_input_basis = use_input_basis_for_encoder
        self.k_in = k_in
        self.k_out = k_out

        if use_input_basis_for_encoder:
            self.encoder_dim = k_in
            enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            enc_mean = rb.pca_in.mean_.astype(np.float32)
        else:
            self.encoder_dim = k_out
            enc_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
            if enc_components.shape[1] < self.D_in:
                padding = np.zeros((k_out, self.D_in - enc_components.shape[1]), dtype=np.float32)
                enc_components = np.concatenate([enc_components, padding], axis=1)
            enc_mean = rb.pca_out.mean_.astype(np.float32)
            if enc_mean.shape[0] < self.D_in:
                enc_mean = np.pad(enc_mean, (0, self.D_in - enc_mean.shape[0]), 'constant')

        if use_input_basis_for_decoder:
            self.decoder_dim = k_in
            dec_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            dec_mean = rb.pca_in.mean_.astype(np.float32)
            self.output_channels = self.C_in
        else:
            self.decoder_dim = k_out
            dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
            dec_mean = rb.pca_out.mean_.astype(np.float32)
            self.output_channels = self.C_out
        
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))
        self.register_buffer("dec_components", torch.from_numpy(dec_components))
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.in_proj = nn.Linear(self.encoder_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden) if latent_batch_norm else nn.Identity()
        self.out_proj = nn.Linear(hidden, self.decoder_dim)
        self.latent_net = nn.Sequential(
            self.in_proj,
            self.bn,
            self.dropout,
            self.out_proj
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        x_centered = x_flat - self.enc_mean
        return x_centered @ self.enc_components_t
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.dec_components + self.dec_mean
    
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        B, C, H, W = x.shape
        if C != self.C_in or H != self.N or W != self.M:
            raise ValueError(f"Expected (B, {self.C_in}, {self.N}, {self.M}), got {x.shape}")
        
        x_flat = x.reshape(B, self.D_in)
        z = self.encode(x_flat)
        z_out = self.latent_net(z)
        y_flat = self.decode(z_out)
        if self.stay_multi_channel:
            y = y_flat.reshape(B, self.C_in, self.N, self.M)
        else:
            y = y_flat.reshape(B, self.C_out, self.N, self.M)
        return y

""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# PCA-based Reduced Basis Helper for Multi-Channel Inputs
""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class PCAReducerMultiChannel:
    """
    PCA-based reduced-basis helper for tensors with arbitrary channel counts
    (num_samples, C_in, N, N) → (num_samples, C_out, N, N) with optional boundary-aware weighting.
    """
    
    def __init__(self, svd_solver='auto', whiten=False, random_state=None, 
                 bc_weight=1.0, boundary_threshold=0.1, use_boundary_weighting=False):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.bc_weight = bc_weight  # Weight for boundary samples
        self.boundary_threshold = boundary_threshold  # Distance threshold for boundary
        self.use_boundary_weighting = use_boundary_weighting  # Enable/disable boundary weighting
        self.pca_in: Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.n_channels = None       # input channels
        self.out_channels = None     # output channels
        self.shape_ = None  # (N, N)
        self.output_shape_ = None  # (C_out, N, N)

    def fit(self, samples: np.ndarray, solution: np.ndarray):
        """
        samples: (num_samples, C_in, N, N) – inputs in channel-first format
        solution: (num_samples, C_out, N, N) – outputs in channel-first format
        """
        if samples.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_in, N, N), got {samples.shape}")
        num_samples, C_in, N, M = samples.shape
        self.n_channels = C_in
        self.shape_ = (N, M)
        X_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
        
        if solution.ndim == 3:
            solution = solution[:, None, :, :]
        if solution.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_out, N, N) or (num_samples, N, N), got {solution.shape}")
        _, C_out, N_out, M_out = solution.shape
        if (N_out, M_out) != (N, M):
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = C_out
        self.output_shape_ = (C_out, N, M)
        Y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)
        
        if self.use_boundary_weighting:
            boundary_weights = self._compute_boundary_weights(N, M)
            Y_weighted = Y_out * boundary_weights.reshape(1, -1)
        else:
            Y_weighted = Y_out
        
        self.pca_in = PCA(n_components=None, svd_solver=self.svd_solver,
                          whiten=self.whiten, random_state=self.random_state).fit(X_in)
        self.pca_out = PCA(n_components=None, svd_solver=self.svd_solver,
                           whiten=self.whiten, random_state=self.random_state).fit(Y_weighted)
        
        return self
    
    def _compute_boundary_weights(self, N: int, M: int) -> np.ndarray:
        """Compute weights for boundary condition awareness."""
        weights = np.ones((N, M), dtype=np.float32)
        
        # Create coordinate grids
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, M)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Compute distance to boundaries
        dist_to_boundary = np.minimum(
            np.minimum(X, 1-X),  # Distance to left/right boundaries
            np.minimum(Y, 1-Y)   # Distance to top/bottom boundaries
        )
        
        # Apply boundary weighting
        boundary_mask = dist_to_boundary < self.boundary_threshold
        weights[boundary_mask] = self.bc_weight
        
        return weights.flatten()

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


    



