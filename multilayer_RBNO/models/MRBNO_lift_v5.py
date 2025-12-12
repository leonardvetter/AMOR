import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReducedBasisBlockMultiChannelV5(nn.Module):
    """
    Multi-channel reduced-basis block with optional batch norm in the lifted width dimension.

    This is a minimally modified version of the v2 block. When batch norm is disabled,
    the forward pass is algebraically identical to the original v2 implementation.
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
    ):
        super().__init__()

        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")

        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)  # Support arbitrary output channels
        self.D_in = self.C_in * self.N * self.M  # Total input dimension
        self.D_out = self.C_out * self.N * self.M  # Output dimension (multi-channel support)
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
                enc_mean = np.pad(enc_mean, (0, self.D_in - enc_mean.shape[0]), "constant")

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
        if self.D_in == self.output_dim:
            self.w = nn.Conv1d(self.width, self.width, kernel_size=1)
        else:
            self.w = nn.Conv1d(self.width*self.C_in, self.width*self.C_out, kernel_size=1)  #In block 1 the local conv needs to map multi-channel input to multi-channel output

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            # Normalise across the decoder_dim dimension; keep width as sequence length
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
        """
        Project to PCA coefficient space (channel-wise).
        x: (B, width, D) where D can be D_in or D_out depending on the input
        returns: (B, width, encoder_dim)
        """
        B, width, D = x.shape

        x_reshaped = x.reshape(B * width, D)  # (B*width, D)

        if self.use_input_basis_for_encoder:
            if D == self.D_in:
                x_centered = x_reshaped - self.enc_mean
                z_reshaped = x_centered @ self.enc_components_t  # (B*width, k_in)
            elif D == self.D_out:
                raise ValueError(f"Encoder expects D_in={self.D_in} but got D_out={self.D_out}")
            else:
                raise ValueError(f"Unexpected input dimension {D}")
        else:
            if D == self.D_out:
                x_centered = x_reshaped - self.enc_mean[:D]
                z_reshaped = x_centered @ self.enc_components_t[:D, :]  # (B*width, k_out)
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
        """
        Reconstruct from PCA coefficients (channel-wise).
        z: (B, width, decoder_dim)
        returns: (B, width, output_dim)
        """
        B, width, decoder_dim = z.shape

        z_reshaped = z.reshape(B * width, decoder_dim)

        if self.use_input_basis_for_decoder:
            y_reshaped = z_reshaped @ self.dec_components + self.dec_mean  # (B*width, D_in)
            output_dim = self.D_in
        else:
            y_reshaped = z_reshaped @ self.dec_components + self.dec_mean  # (B*width, D_out)
            output_dim = self.D_out

        y = y_reshaped.reshape(B, width, output_dim)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, width, D_in) or (B, width, D_out) depending on the block
        returns: (B, width, output_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, width, D), got {x.shape}")
        
        # Encode to coefficients
        z = self.encode(x)  # (B, width, encoder_dim)

        # Transform in coefficient space
        z_out = torch.einsum("bwi,woij->boj", z, self.weight)  # (B, width, decoder_dim)

        # Apply 1D convolution in width dimension
        #z_out = z_out + self.w(z_out)  # (B, width, decoder_dim)

        # Optional batch norm across decoder feature dimension (keep algebra identical when disabled)
        z_out = z_out.transpose(1, 2)  # (B, decoder_dim, width)
        z_out = self.bn(z_out)
        z_out = z_out.transpose(1, 2)  # (B, width, decoder_dim)

        z_out = self.dropout(z_out)

        # Decode back to function space
        y = self.decode(z_out)  # (B, width, output_dim)

        # Local 1D convolutional skip connection in width dimension depending on input/output dims
        # Check actual input dimension, not self.D_in (which is fixed at init)
        actual_input_dim = x.shape[2]
        
        if actual_input_dim == self.output_dim:
            # For blocks where input and output dimensions match (subsequent blocks after block1)
            # self.w should be Conv1d(width, width, 1) - check if we can use it directly
            if self.w.weight.shape[0] == self.width and self.w.weight.shape[1] == self.width:
                y = y + self.w(x)  # (B, width, output_dim)
            else:
                # self.w was initialized for else branch, skip the skip connection
                # This happens when block init had D_in != output_dim but runtime input matches output_dim
                pass
        else: 
            # For block 1 (transition block) where D_in != D_out
            B = x.shape[0]
            # x is (B, width, actual_input_dim) where actual_input_dim = C_in * N * M
            # Compute N*M from actual_input_dim
            N_M = actual_input_dim // self.C_in  # Should equal N * M
            # Reshape to (B, width, C_in, N*M) then (B, width*C_in, N*M)
            x_reshaped = x.view(B, self.width, self.C_in, N_M)  # (B, width, C_in, N*M)
            x_reshaped = x_reshaped.view(B, self.width * self.C_in, N_M)  # (B, width*C_in, N*M)
            
            # Conv1d with kernel_size=1: (B, width*C_out, N*M)
            # Check if self.w matches this shape
            if self.w.weight.shape[0] == self.width * self.C_out and self.w.weight.shape[1] == self.width * self.C_in:
                out = self.w(x_reshaped)
                # Reshape back to (B, width, C_out * N * M) = (B, width, D_out)
                out = out.view(B, self.width, self.C_out, N_M)  # (B, width, C_out, N*M)
                out = out.view(B, self.width, self.C_out * N_M)  # (B, width, C_out*N*M)
                y = y + out  # Add skip connection
            # else: self.w doesn't match, skip the skip connection
        return y


class ScalarGate(nn.Module):
    """Learnable scalar gate."""

    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha


class MRBNOLiftV5(nn.Module):
    """
    MRBNO lift v3: lifted multi-channel MRBNO with optional batch norm and coordinate inputs.

    - When ``use_batch_norm=False`` and ``use_coordinates=False``, the computations reduce to
      the original MRBNOLiftV2 behaviour.
    - Coordinates, when enabled, are injected via a 1×1 conv into the input channels prior
      to lifting, analogous in spirit to FNO's coordinate concatenation while keeping the
      PCA dimensions unchanged.
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
    ):
        super().__init__()

        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted")

        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)  # Support arbitrary output channels
        self.layers = layers
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.use_coordinates = use_coordinates

        self.width = width

        self.D_in = self.C_in * self.N * self.M
        self.D_out = self.C_out * self.N * self.M

        # Optional coordinate embedding: (x,y) -> C_in channels
        if self.use_coordinates:
            self.coord_embed = nn.Conv2d(2, self.C_in, kernel_size=1)
        else:
            self.coord_embed = None

        # Lifting layer: from (B, D_in) to (B, width, D_in)
        self.fc0 = nn.Conv1d(1, width, kernel_size=1)

        # Block 0: Input basis -> Input basis (stays in input basis)
        self.block0 = ReducedBasisBlockMultiChannelV5(
            width,
            rb,
            k_in=k_in,
            k_out=k_in,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True,
            use_batch_norm=use_batch_norm,
        )
        self.gate0 = ScalarGate(init=init_scale)

        # Block 1: Input basis encoder -> Output basis decoder (transition)
        self.block1 = ReducedBasisBlockMultiChannelV5(
            width,
            rb,
            k_in=k_in,
            k_out=k_out,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False,
            use_batch_norm=use_batch_norm,
        )
        self.gate1 = ScalarGate(init=init_scale)

        # Global skip connection
        self.global_skip_proj = nn.Conv2d(self.C_in, self.C_out, kernel_size=1)
        self.global_skip_gate = ScalarGate(init=init_scale)

        # Subsequent blocks: Output basis -> Output basis
        if layers > 1:
            self.blocks = nn.ModuleList(
                [
                    ReducedBasisBlockMultiChannelV5(
                        width,
                        rb,
                        k_in=k_in,
                        k_out=k_out,
                        dropout=dropout,
                        use_input_basis_for_encoder=False,
                        use_input_basis_for_decoder=False,
                        use_batch_norm=use_batch_norm,
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
        """
        x: (B, C_in, N, M) multi-channel input
        returns: (B, C_out, N, M) output (C_out can be 1 or more)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")

        B, C, H, W = x.shape

        # Optional coordinate embedding into input channels
        if self.use_coordinates and self.coord_embed is not None:
            coords = self._coord_maps(x)
            coord_feat = self.coord_embed(coords)
            x_in = x + coord_feat
        else:
            x_in = x

        # Global skip connection (before lifting)
        global_skip = self.global_skip_gate(self.global_skip_proj(x_in))  # (B, 1, N, M)

        # Reshape for PCA processing
        x_flat = x_in.reshape(B, self.D_in)  # (B, D_in)
        x_seq = x_flat.unsqueeze(1)  # (B, 1, D_in) for Conv1d

        # Lift to width channels
        x_lifted = self.fc0(x_seq)  # (B, width, D_in)

        # Block 0: stays in input basis (multi-channel space)
        y0 = self.block0(x_lifted)  # uses D_in
        y0 = self.gate0(y0)
        y0 = self.act(y0)

        if self.use_skip_connections:
            y = y0 + x_seq
        else:
            y = y0

        # Block 1: transition from input basis to output basis
        y1 = self.block1(y)  # (B, width, D_out)
        y1 = self.gate1(y1)
        y1 = self.act(y1)

        # Now we're in output basis space (single-channel)
        y = y1  # (B, width, D_out)

        # Subsequent blocks in output basis
        for block, gate in zip(self.blocks, self.gates):
            residual = gate(block(y))  # (B, width, D_out)
            if self.use_skip_connections:
                y = y + residual
            else:
                y = residual
            y = self.act(y)

        # Final projections
        y = self.act(self.fc1(y))  # (B, width, D_out)

        # Project width to C_out channels
        y = self.fc2(y)  # (B, C_out, D_out) where D_out = C_out * N * M
        
        # Reshape back to spatial dimensions
        # The PCA flattens as (C_out, N, M) -> (C_out * N * M) where channels are flattened sequentially
        # So channel 0 pixels [0 : N*M], channel 1 pixels [N*M : 2*N*M], etc.
        if self.C_out == 1:
            # Simple case: D_out = N * M, just reshape
            y = y.reshape(B, self.C_out, self.N, self.M)  # (B, 1, N, M)
        else:
            # Multi-channel case: extract each channel's segment from the flattened representation
            # After fc2: (B, C_out, C_out * N * M)
            # The PCA flattens channels sequentially: channel 0 [0:N*M], channel 1 [N*M:2*N*M], etc.
            # For output channel i, extract segment [i*N*M : (i+1)*N*M] from dimension 2
            y_list = []
            for i in range(self.C_out):
                start_idx = i * self.N * self.M
                end_idx = (i + 1) * self.N * self.M
                # Take the segment corresponding to channel i from any of the C_out output channels
                # We use the i-th output channel's data: y[:, i, start_idx:end_idx]
                channel_data = y[:, i, start_idx:end_idx].reshape(B, self.N, self.M)
                y_list.append(channel_data)
            y = torch.stack(y_list, dim=1)  # (B, C_out, N, M)

        # Add global skip
        y = y + global_skip

        return y