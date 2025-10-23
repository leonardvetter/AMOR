"""
Multi-Channel MRBNO v2: Stays in input basis for block 0, switches to output basis in block 1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.decomposition import PCA
from MRBNO_model import PCAReducer, ResidualBlock, ScalarGate
from MRBNO_model_multichannel import PCAReducerMultiChannel, ReducedBasisResNetMultiChannel, ReducedBasisResNetSingleChannel


class ReducedBasisResNetMultiChannelInputBasis(nn.Module):
    """
    MRBNO block that stays in input basis and outputs multi-channel.
    This is different from ReducedBasisResNetMultiChannel which always outputs 1-channel.
    """
    
    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, 
                 hidden: int = 256, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.D_in = self.C_in * self.N * self.M  # Total input dimension
        
        # Validate k values
        max_k_in = rb.pca_in.n_components_
        if not (1 <= k_in <= max_k_in):
            raise ValueError(f"k_in={k_in} out of range [1, {max_k_in}]")
        
        self.k_in = k_in
        
        # Setup encoder: input basis
        self.encoder_dim = k_in
        enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
        enc_mean = rb.pca_in.mean_.astype(np.float32)
        
        # Setup decoder: input basis (stays in input basis)
        self.decoder_dim = k_in
        dec_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
        dec_mean = rb.pca_in.mean_.astype(np.float32)
        
        # Register as buffers
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))  # (D_in, encoder_dim)
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))  # (D_in,)
        self.register_buffer("dec_components", torch.from_numpy(dec_components))  # (k_in, D_in)
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))  # (D_in,)
        
        # ResNet in coefficient space
        self.in_proj = nn.Linear(self.encoder_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden, self.decoder_dim)
        
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
        return z @ self.dec_components + self.dec_mean  # (B, D_in)
    
    def forward(self, x):
        """
        x: (B, C_in, N, M)
        returns: (B, C_in, N, M) - stays multi-channel
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
        z_out = self.out_proj(h)  # (B, k_in)
        
        # Decode back to function space
        y_flat = self.decode(z_out)  # (B, D_in)
        y = y_flat.reshape(B, self.C_in, self.N, self.M)  # Keep multi-channel
        
        return y


class MultiChannelMRBNOv2(nn.Module):
    """
    Multi-Channel MRBNO v2: Proper multi-channel flow with skip connections around every block
    
    Architecture:
    - Block 0: Input basis -> Input basis (stays in input basis, outputs multi-channel)
    - Block 1: Multi-channel -> Single-channel (transitions from input to output basis) + skip from Block 0
    - Blocks 2+: Single-channel -> Single-channel (FNO-style residuals in output basis)
    - blockLearnable (optional): Single-channel -> Single-channel with learnable basis and orthogonality penalty
    
    Skip Connections:
    - Global skip: Direct path from 2-channel input to 1-channel output
    - Block 0 → Block 1: Skip connection from Block 0 output to Block 1
    - Blocks 2+: FNO-style residuals (y = y + gate(block(y)))
    
    This allows the model to learn complex patterns in the input basis
    before transitioning to the output basis, with skip connections preserving
    information flow throughout the network.
    """
    
    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, k_out: int,
                 layers: int = 4, hidden: int = 256, depth: int = 4,
                 dropout: float = 0.0, init_scale: float = 0.1, 
                 use_skip_connections: bool = True, block0_dropout: float = 0.0 ,add_learnable_basis_block: bool = False,ortho_penalty_weight: float = 1e-3):
        super().__init__()
        
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.layers = layers
        self.use_skip_connections = use_skip_connections
        self.add_learnable_basis_block = add_learnable_basis_block
        self.ortho_penalty_weight = ortho_penalty_weight


        # Block 0: Input basis -> Input basis (stays in input basis, outputs multi-channel)
        self.block0 = ReducedBasisResNetMultiChannelInputBasis(
            rb, k_in=k_in,  # Stay in input basis
            hidden=hidden, depth=depth, dropout=block0_dropout
        )
        self.gate0 = ScalarGate(init=init_scale)
        
        # Block 1: Multi-channel -> Single-channel (transitions from input to output basis)
        self.block1 = ReducedBasisResNetMultiChannel(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, depth=depth, dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False  # Switch to output basis
        )
        self.gate1 = ScalarGate(init=init_scale)

        if self.add_learnable_basis_block:
            self.blockLearnable = ReducedBasisResNetSingleChannelLearnable(
                rb, k_out=k_out,
                hidden=hidden, depth=depth, dropout=dropout,
                ortho_penalty_weight=1e-3
            )
            self.gateLearnable = ScalarGate(init=init_scale)
        
        # Global skip connection: project 2-channel input directly to 1-channel output
        self.global_skip_proj = nn.Conv2d(self.C_in, 1, 1)
        self.global_skip_gate = ScalarGate(init=init_scale)
        
        # Subsequent blocks: Single-channel -> Single-channel (FNO-style residuals)
        if layers > 1:
            self.blocks = nn.ModuleList([
                ReducedBasisResNetSingleChannel(
                    rb, k_out=k_out, hidden=hidden, depth=depth, dropout=dropout
                ) for _ in range(layers - 1)  # layers-1 because we have block0 and block1
            ])
            
            self.gates = nn.ModuleList([ScalarGate(init=init_scale) for _ in range(layers - 1)])
        else:
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()
        
        self.act = nn.GELU()
    
    def forward(self, x,return_penalty: Optional[bool]=False):
        """
        x: (B, C_in, N, M) multi-channel input
        returns: (B, 1, N, M) single-channel output
        
        Architecture with skip connections around every block:
        - Global skip: Project 2-channel input directly to 1-channel output
        - Block 0: Transform 2-channel input to 2-channel output (input->input basis)
        - Block 1: Transform 2-channel input to 1-channel output (input->output basis) + skip from Block 0
        - Blocks 2+: FNO-style residuals in output basis (y = y + gate(block(y)))
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        
        # Global skip connection: project input directly to output space
        global_skip = self.global_skip_gate(self.global_skip_proj(x))  # (B, 1, N, M)
        
        # Block 0: Transform 2-channel input to 2-channel output (stays in input basis)
        y0_multi = self.gate0(self.block0(x))  # (B, 2, N, M) - stays multi-channel
        y0_multi = self.act(y0_multi)
        
        # Project Block 0's multi-channel output to single-channel for skip connection
        y0_proj = self.global_skip_proj(y0_multi)  # (B, 2, N, M) -> (B, 1, N, M)
        
        # Block 1: Transform 2-channel input to 1-channel output (switches to output basis)
        y1 = self.gate1(self.block1(x))  # (B, 1, N, M)
        y1 = self.act(y1)
        
        # Add skip connection from Block 0 to Block 1
        if self.use_skip_connections:
            y = y1 + y0_proj  # Skip connection: Block 0 → Block 1
        else:
            y = y1
        
        # Add global skip to Block 1 output
        y = y + global_skip
        
        # Subsequent blocks: FNO-style residuals in output basis
        for i, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            y = self.act(y)  # Nonlinearity between blocks
            residual = gate(block(y))  # (B, 1, N, M)
            
            if self.use_skip_connections:
                y = y + residual  # FNO-style residual: y = y + f(y)
            else:
                y = residual

        if self.add_learnable_basis_block:
            y = self.act(y)
            residual, ortho_penalty = self.blockLearnable(y, return_penalty=True)

            return y + self.gateLearnable(residual), ortho_penalty
        else:
            return y



"""
Learnable basis block with orthogonalization penalty
"""

class ReducedBasisResNetSingleChannelLearnable(nn.Module):
    """
    Simplified ResNet block for single-channel inputs with learnable basis.
    The basis is initialized from output PCA but is learnable during training.
    Includes orthogonality penalty to maintain approximate orthogonality.
    """
    
    def __init__(self, rb, k_out: int, 
                 hidden: int = 256, depth: int = 4, dropout: float = 0.0,
                 ortho_penalty_weight: float = 1e-3):
        super().__init__()
        
        if rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        self.D_out = self.N * self.M  # Output dimension
        self.k_out = k_out
        self.ortho_penalty_weight = ortho_penalty_weight
        
        # Validate k_out
        max_k_out = rb.pca_out.n_components_
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")
        
        # Initialize from output PCA
        dec_components_init = rb.pca_out.components_[:k_out, :].astype(np.float32)
        dec_mean_init = rb.pca_out.mean_.astype(np.float32)
        
        # Make basis learnable as parameters instead of buffers
        self.enc_basis = nn.Parameter(torch.from_numpy(dec_components_init.copy()))  # (k_out, D_out)
        self.dec_basis = nn.Parameter(torch.from_numpy(dec_components_init.copy()))  # (k_out, D_out)
        
        # Make mean learnable as well (optional - you can keep it fixed if preferred)
        self.enc_mean = nn.Parameter(torch.from_numpy(dec_mean_init.copy()))  # (D_out,)
        self.dec_mean = nn.Parameter(torch.from_numpy(dec_mean_init.copy()))  # (D_out,)
        
        # Alternative: Keep means fixed
        # self.register_buffer("enc_mean", torch.from_numpy(dec_mean_init))
        # self.register_buffer("dec_mean", torch.from_numpy(dec_mean_init))
        
        # ResNet in coefficient space
        self.in_proj = nn.Linear(k_out, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden, k_out)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Project to coefficient space using learnable encoder basis."""
        x_centered = x_flat - self.enc_mean
        return x_centered @ self.enc_basis.T  # (B, k_out)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from coefficients using learnable decoder basis."""
        return z @ self.dec_basis + self.dec_mean  # (B, D_out)
    
    def compute_orthogonality_penalty(self):
        """
        Compute penalty for deviation from orthogonality.
        For orthonormal basis B, we want B @ B.T = I
        """
        # Encoder basis orthogonality
        enc_gram = self.enc_basis @ self.enc_basis.T  # (k_out, k_out)
        enc_eye = torch.eye(self.k_out, device=enc_gram.device)
        enc_penalty = torch.norm(enc_gram - enc_eye, p='fro')**2
        
        # Decoder basis orthogonality
        dec_gram = self.dec_basis @ self.dec_basis.T  # (k_out, k_out)
        dec_eye = torch.eye(self.k_out, device=dec_gram.device)
        dec_penalty = torch.norm(dec_gram - dec_eye, p='fro')**2
        
        # Optional: Also penalize deviation from unit norm for each basis vector
        enc_norm_penalty = torch.sum((torch.norm(self.enc_basis, dim=1) - 1)**2)
        dec_norm_penalty = torch.sum((torch.norm(self.dec_basis, dim=1) - 1)**2)
        
        total_penalty = enc_penalty + dec_penalty + 0.1 * (enc_norm_penalty + dec_norm_penalty)
        
        return total_penalty
    
    def project_to_orthonormal(self):
        """
        Optional: Project bases to orthonormal manifold using SVD.
        Can be called periodically during training or after training.
        """
        with torch.no_grad():
            # Project encoder basis
            U, S, Vt = torch.linalg.svd(self.enc_basis, full_matrices=False)
            self.enc_basis.data = U @ Vt
            
            # Project decoder basis
            U, S, Vt = torch.linalg.svd(self.dec_basis, full_matrices=False)
            self.dec_basis.data = U @ Vt
    
    def forward(self, x, return_penalty=False):
        """
        x: (B, 1, N, M) single-channel input
        returns: 
            - (B, 1, N, M) single-channel output
            - optionally also returns orthogonality penalty if return_penalty=True
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, 1, N, M), got {x.shape}")
        
        B, C, H, W = x.shape
        if C != 1 or H != self.N or W != self.M:
            raise ValueError(f"Expected (B, 1, {self.N}, {self.M}), got {x.shape}")
        
        # Flatten
        x_flat = x.reshape(B, self.D_out)
        
        # Encode to coefficients
        z = self.encode(x_flat)  # (B, k_out)
        
        # Transform in coefficient space
        h = F.gelu(self.in_proj(z))  # (B, hidden)
        for block in self.blocks:
            h = block(h)  # (B, hidden)
        z_out = self.out_proj(h)  # (B, k_out)
        
        # Decode back to function space
        y_flat = self.decode(z_out)  # (B, D_out)
        y = y_flat.reshape(B, 1, self.N, self.M)
        
        if return_penalty:
            penalty = self.compute_orthogonality_penalty()
            return y, penalty
        return y
    
    def get_basis_info(self):
        """Get information about the learned bases."""
        with torch.no_grad():
            # Check orthogonality
            enc_gram = self.enc_basis @ self.enc_basis.T
            dec_gram = self.dec_basis @ self.dec_basis.T
            enc_eye = torch.eye(self.k_out, device=enc_gram.device)
            dec_eye = torch.eye(self.k_out, device=dec_gram.device)
            
            info = {
                'encoder_ortho_error': torch.norm(enc_gram - enc_eye).item(),
                'decoder_ortho_error': torch.norm(dec_gram - dec_eye).item(),
                'encoder_mean_norm': torch.norm(self.enc_mean).item(),
                'decoder_mean_norm': torch.norm(self.dec_mean).item(),
                'encoder_basis_norm': torch.norm(self.enc_basis, dim=1).mean().item(),
                'decoder_basis_norm': torch.norm(self.dec_basis, dim=1).mean().item(),
            }
        return info








