

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ReducedBasisBlockMultiChannel(nn.Module):
    """
    Core MRBNO block that handles multi-channel (width) inputs after lifting.
    
    1. Encodes multi-channel input to PCA coefficients (channel-wise)
    2. Applies transformation in coefficient space
    3. Decodes back to function space
    """
    
    def __init__(self, width: int, rb, k_in: int, k_out: int,
                 dropout: float = 0.0,
                 use_input_basis_for_encoder: bool = False, 
                 use_input_basis_for_decoder: bool = False):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.D_in = self.C_in * self.N * self.M  # Total input dimension
        self.D_out = self.N * self.M  # Output dimension (single channel)
        self.width = width

        # Determine output dimension based on decoder choice
        if use_input_basis_for_decoder:
            self.output_dim = self.D_in  
        else:
            self.output_dim = self.D_out 

        # Validate k values
        max_k_in = rb.pca_in.n_components_
        max_k_out = rb.pca_out.n_components_
        if not (1 <= k_in <= max_k_in):
            raise ValueError(f"k_in={k_in} out of range [1, {max_k_in}]")
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")
        
        self.use_input_basis_for_encoder = use_input_basis_for_encoder
        self.use_input_basis_for_decoder = use_input_basis_for_decoder
        self.k_in = k_in
        self.k_out = k_out

        # Setup encoder components
        if use_input_basis_for_encoder:
            self.encoder_dim = k_in
            enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            enc_mean = rb.pca_in.mean_.astype(np.float32)
        else:
            self.encoder_dim = k_out
            enc_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
            enc_mean = rb.pca_out.mean_.astype(np.float32)
            # Pad if necessary
            if enc_components.shape[1] < self.D_in:
                padding = np.zeros((k_out, self.D_in - enc_components.shape[1]), dtype=np.float32)
                enc_components = np.concatenate([enc_components, padding], axis=1)
            if enc_mean.shape[0] < self.D_in:
                enc_mean = np.pad(enc_mean, (0, self.D_in - enc_mean.shape[0]), 'constant')

        # Setup decoder components
        if use_input_basis_for_decoder:
            self.decoder_dim = k_in
            dec_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            dec_mean = rb.pca_in.mean_.astype(np.float32)
        else:
            self.decoder_dim = k_out
            dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
            dec_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Register as buffers
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))
        self.register_buffer("dec_components", torch.from_numpy(dec_components))
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))
        
        # Transformation in latent space
        scale = 1.0 / (self.width * self.width)
        self.weight = nn.Parameter(
            torch.randn(self.width, self.width, self.encoder_dim, self.decoder_dim) * scale
        )
        self.w = nn.Conv1d(self.width, self.width, kernel_size=1)
        
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to PCA coefficient space (channel-wise).
        x: (B, width, D) where D can be D_in or D_out depending on the input
        returns: (B, width, encoder_dim)
        """
        B, width, D = x.shape
        
        # Process all channels at once by reshaping
        x_reshaped = x.reshape(B * width, D)  # (B*width, D)
        
        # Determine which encoding path to use based on input dimension
        if self.use_input_basis_for_encoder:
            # We're using input basis for encoding
            if D == self.D_in:
                # Input is in multi-channel space (D_in)
                x_centered = x_reshaped - self.enc_mean
                z_reshaped = x_centered @ self.enc_components_t  # (B*width, k_in)
            elif D == self.D_out:
                # Input is in single-channel space but we need to use input basis
                # This shouldn't happen in normal flow, but handle it
                raise ValueError(f"Encoder expects D_in={self.D_in} but got D_out={self.D_out}")
            else:
                raise ValueError(f"Unexpected input dimension {D}")
        else:
            # We're using output basis for encoding
            if D == self.D_out:
                # Input is in single-channel space (expected)
                x_centered = x_reshaped - self.enc_mean[:D]
                z_reshaped = x_centered @ self.enc_components_t[:D, :]  # (B*width, k_out)
            elif D == self.D_in:
                # Input is in multi-channel space but we're using output basis
                # Need to handle dimension mismatch
                # Take only the first D_out dimensions if possible
                if self.enc_components_t.shape[0] == self.D_out:
                    # Can't directly apply output basis to input space
                    # This block configuration shouldn't occur
                    raise ValueError(f"Cannot apply output basis encoder to input space data")
                else:
                    # Use padded components
                    x_centered = x_reshaped - self.enc_mean
                    z_reshaped = x_centered @ self.enc_components_t
            else:
                raise ValueError(f"Unexpected input dimension {D}")
        
        # Reshape back to separate batch and width
        z = z_reshaped.reshape(B, width, -1)  # (B, width, encoder_dim)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from PCA coefficients (channel-wise).
        z: (B, width, decoder_dim)
        returns: (B, width, output_dim)
        """
        B, width, decoder_dim = z.shape
        
        # Process all channels at once
        z_reshaped = z.reshape(B * width, decoder_dim)  # (B*width, decoder_dim)
        
        # Apply PCA decoding
        if self.use_input_basis_for_decoder:
            # Decode to multi-channel space
            y_reshaped = z_reshaped @ self.dec_components + self.dec_mean  # (B*width, D_in)
            output_dim = self.D_in
        else:
            # Decode to single-channel space
            y_reshaped = z_reshaped @ self.dec_components + self.dec_mean  # (B*width, D_out)
            output_dim = self.D_out
        
        # Reshape back
        y = y_reshaped.reshape(B, width, output_dim)  # (B, width, output_dim)
        
        return y
    
    def forward(self, x):
        """
        x: (B, width, D_in) or (B, width, D_out) depending on the block
        returns: (B, width, output_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, width, D), got {x.shape}")

        B, width, D = x.shape

        # Encode to coefficients
        z = self.encode(x)  # (B, width, encoder_dim)
        
        # Transform in coefficient space
        z_out = torch.einsum("bwi,woij->boj", z, self.weight)  # (B, width, decoder_dim)
        
        # Apply 1D convolution in width dimension
        z_out = z_out + self.w(z_out)  # (B, width, decoder_dim)
        
        z_out = self.dropout(z_out)
        
        # Decode back to function space
        y = self.decode(z_out)  # (B, width, output_dim)
        
        return y



class ScalarGate(nn.Module):
    """Learnable scalar gate."""
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init))
    
    def forward(self, x):
        return x * self.alpha


class MRBNOLiftV2(nn.Module):
    """
     MRBNO with proper multi-channel handling after lifting.
    """
    
    def __init__(self, width: int = 8, rb=None, k_in: int = 50, k_out: int = 50,
                 layers: int = 7, 
                 dropout: float = 0.0, init_scale: float = 0.1, 
                 use_skip_connections: bool = True):
        super().__init__()
        
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = 1  # Single-channel output TODO: adapt such that C_out is read from rb if needed
        self.layers = layers
        self.use_skip_connections = use_skip_connections

        self.width = width

        self.D_in = self.C_in * self.N * self.M
        self.D_out = self.C_out * self.N * self.M

        # Lifting layer: from (B, D_in) to (B, width, D_in)
        self.fc0 = nn.Conv1d(1, width, kernel_size=1)
        
        # Block 0: Input basis -> Input basis (stays in input basis)
        # Note: When staying in input basis, decoder_dim should also be k_in
        self.block0 = ReducedBasisBlockMultiChannel(
            width, rb, k_in=k_in, k_out=k_in,  # Both k_in to stay in input space
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True,  # Stay in input basis
        )
        self.gate0 = ScalarGate(init=init_scale)
        
        # Block 1: Input basis encoder -> Output basis decoder (transition)
        # This block takes D_in input and produces D_out output
        self.block1 = ReducedBasisBlockMultiChannel(
            width, rb, k_in=k_in, k_out=k_out,
            dropout=dropout,
            use_input_basis_for_encoder=True,   # Encoder uses input basis (expects D_in)
            use_input_basis_for_decoder=False,  # Decoder uses output basis (produces D_out)
        )
        self.gate1 = ScalarGate(init=init_scale)
        
        # Global skip connection
        self.global_skip_proj = nn.Conv2d(self.C_in, self.C_out, kernel_size=1)
        self.global_skip_gate = ScalarGate(init=init_scale)
        
        # Subsequent blocks: Output basis -> Output basis
        if layers > 1:
            self.blocks = nn.ModuleList([
                ReducedBasisBlockMultiChannel(
            width, rb, k_in=k_in, k_out=k_out,
            dropout=dropout,
            use_input_basis_for_encoder=False,   # Encoder uses output basis (expects D_out)
            use_input_basis_for_decoder=False,  # Decoder uses output basis (produces D_out)
        )    for _ in range(layers - 1)
            ])
            self.gates = nn.ModuleList([ScalarGate(init=init_scale) for _ in range(layers - 1)])
        else:
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()

        # Final projection layers
        self.fc1 = nn.Conv1d(width, width, kernel_size=1)
        self.fc2 = nn.Conv1d(width, self.C_out, kernel_size=1)
        
        self.act = nn.GELU()
    
    def forward(self, x):
        """
        x: (B, C_in, N, M) multi-channel input
        returns: (B, 1, N, M) single-channel output
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        
        B, C, H, W = x.shape
        
        # Global skip connection (before lifting)
        global_skip = self.global_skip_gate(self.global_skip_proj(x))  # (B, 1, N, M)
        
        # Reshape for PCA processing
        x_input = x.reshape(B, self.D_in)  # (B, D_in)
        x_input = x_input.unsqueeze(1)  # (B, 1, D_in) for Conv1d
        # Lift to width channels
        x = self.fc0(x_input)  # (B, width, D_in)
        
        
        # Block 0: Stays in input basis (multi-channel space)
        # Input: (B, width, D_in), Output: (B, width, D_in)
        y0 = self.block0(x_input)  # (B, width, D_in)
        y0 = self.gate0(y0)
        y0 = self.act(y0)
        
        if self.use_skip_connections:
            y = y0 + x_input  # Skip connection in input space (both are D_in)
        else:
            y = y0
        
        # Block 1: Transition from input basis to output basis
        # Input: (B, width, D_in), Output: (B, width, D_out)
        y1 = self.block1(y)  # Takes D_in, outputs D_out
        y1 = self.gate1(y1)
        y1 = self.act(y1)
        
        # Now we're in output basis space (single-channel)
        y = y1  # (B, width, D_out)
        
        # Subsequent blocks in output basis
        # Input: (B, width, D_out), Output: (B, width, D_out)
        for i, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            residual = gate(block(y))  # (B, width, D_out)
            
            if self.use_skip_connections:
                y = y + residual  # Both are D_out
            else:
                y = residual
            y = self.act(y)
        
        # Final projections
        # y shape: (B, width, D_out)
        y = self.act(self.fc1(y))  # Still (B, width, D_out)
        
        # Project width to 1 channel
        # Transpose to (B, D_out, width) for Conv1d, then back
        #y = y.transpose(1, 2)  # (B, D_out, width)
        y = self.fc2(y)  # (B, D_out, 1)
        y = y.transpose(1, 2)  # (B, 1, D_out)
        
        # Reshape back to spatial dimensions
        y = y.reshape(B, 1, self.N, self.M)  # (B, 1, N, M)
        
        # Add global skip
        y = y + global_skip
        
        return y