"""
MRBNO Lift v3 with Convolution: Uses finite-kernel convolution instead of full einsum.

This version replaces the O(width²) einsum transformation with a finite-kernel
convolution for more parameter-efficient channel mixing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import ScalarGate from v3
from MRBNO_lift_v3 import ScalarGate


class ReducedBasisBlockMultiChannelV3Conv(nn.Module):
    """
    Multi-channel reduced-basis block with convolution instead of einsum.
    
    Uses finite-kernel 1D convolution along the width dimension instead of
    einsum("bwi,woij->boj", z, weight) transformation.
    """

    def __init__(
        self,
        width: int,
        rb,
        k_in: int,
        k_out: int,
        dropout: float = 0.0,
        use_input_basis_for_encoder: bool = False,
        use_input_basis_for_decoder: bool = False,
        use_batch_norm: bool = False,
        conv_kernel_size: int = 3,
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
        self.D_in = self.C_in * self.N * self.M
        self.D_out = self.C_out * self.N * self.M
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

        # Setup encoder components (same as V3)
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
                enc_mean = np.pad(enc_mean, (0, self.D_in - enc_mean.shape[0]), "constant")

        # Setup decoder components (same as V3)
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

        # Uses finite-kernel convolution instead of einsum for channel mixing
        # Convolution operates along width dimension: (B, width, decoder_dim) -> (B, width, decoder_dim)
        # Projects from encoder_dim to decoder_dim, then mixes width channels via conv
        self.proj = nn.Linear(self.encoder_dim, self.decoder_dim)
        # Conv1d operates on (B, decoder_dim, width): mixes width channels for each decoder_dim feature
        # groups=decoder_dim means each feature dimension mixes independently
        self.conv = nn.Conv1d(
            in_channels=self.decoder_dim,
            out_channels=self.decoder_dim,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,  # Same padding
            groups=self.decoder_dim  # Each decoder_dim feature mixes width channels independently
        )
        
        # Optional residual connection (mixes width channels for each decoder_dim feature)
        self.w = nn.Conv1d(self.decoder_dim, self.decoder_dim, kernel_size=1, groups=self.decoder_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(self.decoder_dim)
        else:
            self.bn = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers and convs."""
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
        """Same as V3."""
        B, width, D = x.shape

        x_reshaped = x.reshape(B * width, D)

        if self.use_input_basis_for_encoder:
            if D == self.D_in:
                x_centered = x_reshaped - self.enc_mean
                z_reshaped = x_centered @ self.enc_components_t
            elif D == self.D_out:
                raise ValueError(f"Encoder expects D_in={self.D_in} but got D_out={self.D_out}")
            else:
                raise ValueError(f"Unexpected input dimension {D}")
        else:
            if D == self.D_out:
                x_centered = x_reshaped - self.enc_mean[:D]
                z_reshaped = x_centered @ self.enc_components_t[:D, :]
            elif D == self.D_in:
                if self.enc_components_t.shape[0] == self.D_out:
                    raise ValueError("Cannot apply output basis encoder to input space data")
                else:
                    x_centered = x_reshaped - self.enc_mean
                    z_reshaped = x_centered @ self.enc_components_t
            else:
                raise ValueError(f"Unexpected input dimension {D}")

        z = z_reshaped.reshape(B, width, -1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Same as V3."""
        B, width, decoder_dim = z.shape

        z_reshaped = z.reshape(B * width, decoder_dim)

        if self.use_input_basis_for_decoder:
            y_reshaped = z_reshaped @ self.dec_components + self.dec_mean
            output_dim = self.D_in
        else:
            y_reshaped = z_reshaped @ self.dec_components + self.dec_mean
            output_dim = self.D_out

        y = y_reshaped.reshape(B, width, output_dim)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, width, D_in) or (B, width, D_out) depending on the block
        returns: (B, width, output_dim)
        
        Keeps tensors in convolution format (B, decoder_dim, width) for
        convolution, residual, and batch norm operations, then transposes
        to (B, width, decoder_dim) before decode.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, width, D), got {x.shape}")

        # Encode to coefficients
        z = self.encode(x)  # (B, width, encoder_dim)

        # Project to decoder dimension
        z_proj = self.proj(z)  # (B, width, decoder_dim)
        
        # Transpose to convolution format: (B, width, decoder_dim) -> (B, decoder_dim, width)
        z_conv_format = z_proj.transpose(1, 2).contiguous()  # (B, decoder_dim, width)
        
        # Apply convolution along width dimension
        # Conv mixes width channels for each decoder_dim feature independently
        z_conv = self.conv(z_conv_format)  # (B, decoder_dim, width)
        
        # Optional residual connection (mixes width channels)
        # Residual operates in (B, decoder_dim, width) format
        z_residual = self.w(z_conv)  # (B, decoder_dim, width)
        z_out = z_conv + z_residual  # (B, decoder_dim, width)

        # Optional batch norm in (B, decoder_dim, width) format
        z_out = self.bn(z_out)  # (B, decoder_dim, width)

        # Dropout
        z_out = self.dropout(z_out)  # (B, decoder_dim, width)

        # Transpose to decode format: (B, decoder_dim, width) -> (B, width, decoder_dim)
        z_out = z_out.transpose(1, 2).contiguous()  # (B, width, decoder_dim)

        # Decode back to function space
        y = self.decode(z_out)  # (B, width, output_dim)
        return y


class MRBNOLiftV3Conv(nn.Module):
    """
    MRBNO lift v3 with convolution instead of einsum.
    
    Identical to MRBNOLiftV3 except that blocks use finite-kernel convolution
    for channel mixing instead of full einsum transformation.
    """

    def __init__(
        self,
        width: int = 8,
        rb=None,
        k_in: int = 50,
        k_out: int = 50,
        layers: int = 7,
        dropout: float = 0.0,
        init_scale: float = 0.1,
        use_skip_connections: bool = True,
        use_batch_norm: bool = False,
        use_coordinates: bool = False,
        conv_kernel_size: int = 3,
    ):
        super().__init__()

        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted")

        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)
        self.layers = layers
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.use_coordinates = use_coordinates

        self.width = width

        self.D_in = self.C_in * self.N * self.M
        self.D_out = self.C_out * self.N * self.M

        # Optional coordinate embedding
        if self.use_coordinates:
            self.coord_embed = nn.Conv2d(2, self.C_in, kernel_size=1)
        else:
            self.coord_embed = None

        # Lifting layer
        self.fc0 = nn.Conv1d(1, width, kernel_size=1)

        # Block 0: Input basis -> Input basis
        self.block0 = ReducedBasisBlockMultiChannelV3Conv(
            width,
            rb,
            k_in=k_in,
            k_out=k_in,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True,
            use_batch_norm=use_batch_norm,
            conv_kernel_size=conv_kernel_size,
        )
        self.gate0 = ScalarGate(init=init_scale)

        # Block 1: Input basis encoder -> Output basis decoder
        self.block1 = ReducedBasisBlockMultiChannelV3Conv(
            width,
            rb,
            k_in=k_in,
            k_out=k_out,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False,
            use_batch_norm=use_batch_norm,
            conv_kernel_size=conv_kernel_size,
        )
        self.gate1 = ScalarGate(init=init_scale)

        # Global skip connection
        self.global_skip_proj = nn.Conv2d(self.C_in, self.C_out, kernel_size=1)
        self.global_skip_gate = ScalarGate(init=init_scale)

        # Subsequent blocks: Output basis -> Output basis
        if layers > 1:
            self.blocks = nn.ModuleList(
                [
                    ReducedBasisBlockMultiChannelV3Conv(
                        width,
                        rb,
                        k_in=k_in,
                        k_out=k_out,
                        dropout=dropout,
                        use_input_basis_for_encoder=False,
                        use_input_basis_for_decoder=False,
                        use_batch_norm=use_batch_norm,
                        conv_kernel_size=conv_kernel_size,
                    )
                    for _ in range(layers - 1)
                ]
            )
            self.gates = nn.ModuleList([ScalarGate(init=init_scale) for _ in range(layers - 1)])
        else:
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()

        # Final projection layers
        self.fc1 = nn.Conv1d(width, width, kernel_size=1)
        self.fc2 = nn.Conv1d(width, self.C_out, kernel_size=1)

        self.act = nn.GELU()

    def _coord_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_coordinates:
            return None
        B, _, H, W = x.shape
        xs = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(
            B, 1, H, W
        )
        ys = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(
            B, 1, H, W
        )
        return torch.cat([xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same as V3."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")

        B, C, H, W = x.shape

        # Optional coordinate embedding
        if self.use_coordinates and self.coord_embed is not None:
            coords = self._coord_maps(x)
            coord_feat = self.coord_embed(coords)
            x_in = x + coord_feat
        else:
            x_in = x

        # Global skip connection
        global_skip = self.global_skip_gate(self.global_skip_proj(x_in))

        # Reshape for PCA processing
        x_flat = x_in.reshape(B, self.D_in)
        x_seq = x_flat.unsqueeze(1)  # (B, 1, D_in)

        # Lift to width channels
        x_lifted = self.fc0(x_seq)  # (B, width, D_in)

        # Block 0: stays in input basis (use x_lifted to preserve width dimension)
        y0 = self.block0(x_lifted)
        y0 = self.gate0(y0)
        y0 = self.act(y0)

        if self.use_skip_connections:
            y = y0 + x_lifted
        else:
            y = y0

        # Block 1: transition from input basis to output basis
        y1 = self.block1(y)
        y1 = self.gate1(y1)
        y1 = self.act(y1)

        y = y1

        # Subsequent blocks in output basis
        for block, gate in zip(self.blocks, self.gates):
            residual = gate(block(y))
            if self.use_skip_connections:
                y = y + residual
            else:
                y = residual
            y = self.act(y)

        # Final projections
        y = self.act(self.fc1(y))

        # Project width to C_out channels
        y = self.fc2(y)

        # Reshape back to spatial dimensions
        if self.C_out == 1:
            y = y.reshape(B, self.C_out, self.N, self.M)
        else:
            y_list = []
            for i in range(self.C_out):
                start_idx = i * self.N * self.M
                end_idx = (i + 1) * self.N * self.M
                channel_data = y[:, i, start_idx:end_idx].reshape(B, self.N, self.M)
                y_list.append(channel_data)
            y = torch.stack(y_list, dim=1)

        # Add global skip
        y = y + global_skip

        return y

