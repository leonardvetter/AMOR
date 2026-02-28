"""
Multilayer Reduced Basis Neural Operator (MRBNO)

A neural operator architecture that learns mappings between function spaces using
data-driven PCA bases instead of fixed Fourier bases. This approach adapts the
spectral representation to the specific problem structure, potentially offering
improved efficiency for problems with non-periodic or localized features.

Architecture Overview:
    Input (B, C_in, N, M)
        ↓ flatten + lift
    (B, width, C_in * N * M)
        ↓ block0: input basis → input basis
    (B, width, C_in * N * M)
        ↓ block1: input basis → output basis (transition)
    (B, width, C_out * N * M)
        ↓ subsequent blocks: output basis → output basis
    (B, width, C_out * N * M)
        ↓ project + reshape
    Output (B, C_out, N, M)

Reference:
    [...]

Authors:
    [...]

"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ReducedBasisBlock(nn.Module):
    """
    Core building block of the MRBNO architecture.
    
    Performs the following operations:
        1. Encode: Project input to PCA coefficient space
        2. Transform: Apply learned linear transformation in coefficient space
        3. Decode: Reconstruct output from transformed coefficients
        4. Skip connection: Add local convolution bypass
    
    This block is analogous to a spectral convolution layer in FNO, but uses
    data-driven PCA bases instead of Fourier bases.
    
    Args:
        width: Number of lifted channels (hidden dimension)
        rb: Fitted PCAReducerMultiChannel object containing pca_in and pca_out
        k_in: Number of input PCA modes in input basis
        k_out: Number of output PCA modes in output basis
        dropout: Dropout probability (default: 0.0)
        use_input_basis_for_encoder: If True, use input PCA basis for encoding
        use_input_basis_for_decoder: If True, use input PCA basis for decoding
        use_batch_norm: If True, apply batch normalization in latent space
        add_constant_mode: If True, append constant basis vector to PCA modes
        use_factorized_weights: If True or 'auto', use factorized weight representation
            for reduced parameter count (note: not algebraically equivalent,
            but less paramters). If 'auto', automatically uses factorized weights
            when encoder_dim × decoder_dim > use_factorized_weights_auto_threshold
        use_factorized_weights_auto_threshold: Threshold for auto mode
            (default: 10000, i.e., k_in × k_out > 10000). If 'auto', automatically uses factorized weights
            when encoder_dim × decoder_dim > use_factorized_weights_auto_threshold
        use_factorized_weights_auto_threshold: Threshold for auto mode
            (default: 10000, i.e., k_in × k_out > 10000)
    
    Shape:
        - Input: (B, width, input_dim) where input_dim depends on encoder basis
        - Output: (B, width, output_dim) where output_dim depends on decoder basis
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
        add_constant_mode: bool = False,
        use_factorized_weights: bool = False,
        use_factorized_weights_auto_threshold: int = 10000,
    ):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted before creating blocks")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")

        # Store spatial and channel dimensions
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)
        
        # Compute flattened dimensions
        self.D_in = self.C_in * self.N * self.M
        self.D_out = self.C_out * self.N * self.M
        
        # Store configuration
        self.width = width
        self.add_constant_mode = add_constant_mode
        # Handle auto mode for factorized weights (will be resolved after dimensions are known)
        if use_factorized_weights == 'auto':
            self.use_factorized_weights = 'auto'
            self.use_factorized_weights_auto_threshold = use_factorized_weights_auto_threshold
        else:
            self.use_factorized_weights = bool(use_factorized_weights)
            self.use_factorized_weights_auto_threshold = use_factorized_weights_auto_threshold
        self.use_input_basis_for_encoder = use_input_basis_for_encoder
        self.use_input_basis_for_decoder = use_input_basis_for_decoder
        self.use_batch_norm = use_batch_norm
        self.k_in = k_in
        self.k_out = k_out

        # Determine input/output dimensions based on basis choices
        if use_input_basis_for_encoder:
            self.input_dim = self.D_in
            self.input_channels = self.C_in
        else:
            self.input_dim = self.D_out
            self.input_channels = self.C_out

        if use_input_basis_for_decoder:
            self.output_dim = self.D_in
        else:
            self.output_dim = self.D_out

        # Validate k values against available PCA components
        self._validate_k_values(rb, k_in, k_out)

        # Setup PCA components for encoder and decoder
        enc_components, enc_mean = self._setup_encoder_components(rb)
        dec_components, dec_mean = self._setup_decoder_components(rb)

        # Optionally add constant mode to capture mean/DC component
        if self.add_constant_mode:
            enc_components, dec_components = self._add_constant_mode(
                enc_components, dec_components
            )

        # Store dimensions after potential constant mode addition
        self.encoder_dim = enc_components.shape[0]
        self.decoder_dim = dec_components.shape[0]
        
        # Handle auto mode for factorized weights (now that dimensions are known)
        if self.use_factorized_weights == 'auto':
            should_use_factorized = (self.encoder_dim * self.decoder_dim) > self.use_factorized_weights_auto_threshold
            self.use_factorized_weights = should_use_factorized

        # Register PCA components as buffers (not trainable)
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))
        self.register_buffer("dec_components", torch.from_numpy(dec_components))
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))

        # Initialize learnable transformation weights
        self._init_transformation_weights()

        # Skip connection with 1x1 convolution
        self._init_skip_connection()

        # Regularization layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bn = nn.BatchNorm1d(self.decoder_dim) if use_batch_norm else nn.Identity()

        # Apply weight initialization
        self._init_weights()

    def _validate_k_values(self, rb, k_in: int, k_out: int) -> None:
        """Validate that k values are within valid ranges."""
        max_k_in = rb.pca_in.n_components_
        max_k_out = rb.pca_out.n_components_
        
        if not (1 <= k_in <= max_k_in):
            raise ValueError(f"k_in={k_in} out of range [1, {max_k_in}]")
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")

    def _setup_encoder_components(self, rb) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and prepare encoder PCA components."""
        if self.use_input_basis_for_encoder:
            components = rb.pca_in.components_[:self.k_in, :].astype(np.float32)
            mean = rb.pca_in.mean_.astype(np.float32)
        else:
            components = rb.pca_out.components_[:self.k_out, :].astype(np.float32)
            mean = rb.pca_out.mean_.astype(np.float32)
            
            # Pad components if output dimension is smaller than input
            if components.shape[1] < self.D_in:
                padding = np.zeros(
                    (components.shape[0], self.D_in - components.shape[1]),
                    dtype=np.float32
                )
                components = np.concatenate([components, padding], axis=1)
            if mean.shape[0] < self.D_in:
                mean = np.pad(mean, (0, self.D_in - mean.shape[0]), mode="constant")
        
        return components, mean

    def _setup_decoder_components(self, rb) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and prepare decoder PCA components."""
        if self.use_input_basis_for_decoder:
            components = rb.pca_in.components_[:self.k_in, :].astype(np.float32)
            mean = rb.pca_in.mean_.astype(np.float32)
        else:
            components = rb.pca_out.components_[:self.k_out, :].astype(np.float32)
            mean = rb.pca_out.mean_.astype(np.float32)
        
        return components, mean

    def _add_constant_mode(
        self, 
        enc_components: np.ndarray, 
        dec_components: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Append normalized constant vector to basis.
        
        This helps preserve mean values through the transformation,
        analogous to the DC component in Fourier representations.
        """
        const_scale = 1.0 / (self.N * self.M)
        
        enc_const = np.ones((1, enc_components.shape[1]), dtype=np.float32) * const_scale
        enc_components = np.concatenate([enc_components, enc_const], axis=0)
        
        dec_const = np.ones((1, dec_components.shape[1]), dtype=np.float32) * const_scale
        dec_components = np.concatenate([dec_components, dec_const], axis=0)
        
        return enc_components, dec_components

    def _init_transformation_weights(self) -> None:
        """Initialize learnable weights for latent space transformation."""
        # Handle auto mode: determine if factorized weights should be used
        if self.use_factorized_weights == 'auto':
            # Use factorized if encoder_dim × decoder_dim exceeds threshold
            should_use_factorized = (self.encoder_dim * self.decoder_dim) > self.use_factorized_weights_auto_threshold
            self.use_factorized_weights = should_use_factorized
        
        if not self.use_factorized_weights:
            # Full 4D weight tensor: mixes width channels and PCA modes
            scale = 1.0 / (self.width ** 2)
            self.weight = nn.Parameter(
                torch.randn(
                    self.width, self.width, 
                    self.encoder_dim, self.decoder_dim
                ) * scale
            )
        else:
            # Factorized representation: reduces parameters but changes expressivity
            self.width_weights = nn.Parameter(
                torch.randn(self.encoder_dim, self.width, self.width) 
                * (1.0 / self.width)
            )
            self.mode_weight = nn.Parameter(
                torch.randn(self.width, self.encoder_dim, self.decoder_dim) 
                * (1.0 / (self.width * self.encoder_dim))
            )

    def _init_skip_connection(self) -> None:
        """Initialize 1x1 convolution for skip connection."""
        if self.input_dim == self.output_dim:
            # Same dimension: simple width-to-width convolution
            self.skip_conv = nn.Conv1d(self.width, self.width, kernel_size=1)
        else:
            # Dimension change: need to handle channel transition
            self.skip_conv = nn.Conv1d(
                self.width * self.input_channels,
                self.width * self.C_out,
                kernel_size=1
            )

    def _init_weights(self) -> None:
        """Apply Xavier initialization to linear layers and convolutions."""
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
        Project input to PCA coefficient space.
        
        Args:
            x: Input tensor of shape (B, width, D)
            
        Returns:
            Tensor of shape (B, width, encoder_dim) containing PCA coefficients
        """
        B, width, D = x.shape
        
        # Validate input dimension
        if D != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {D}. "
                f"Check basis configuration for this block."
            )
        
        # Flatten batch and width dimensions for matrix multiplication
        x_flat = x.reshape(B * width, D)
        
        # Center data and project onto PCA basis
        x_centered = x_flat - self.enc_mean[:D]
        z_flat = x_centered @ self.enc_components_t[:D, :]
        
        return z_flat.reshape(B, width, -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from PCA coefficients.
        
        Args:
            z: Coefficient tensor of shape (B, width, decoder_dim)
            
        Returns:
            Reconstructed tensor of shape (B, width, output_dim)
        """
        B, width, _ = z.shape
        
        # Flatten for matrix multiplication
        z_flat = z.reshape(B * width, self.decoder_dim)
        
        # Reconstruct: linear combination of basis vectors plus mean
        y_flat = z_flat @ self.dec_components + self.dec_mean
        
        return y_flat.reshape(B, width, self.output_dim)

    def _transform_coefficients(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply learned transformation in coefficient space.
        
        Args:
            z: Input coefficients of shape (B, width, encoder_dim)
            
        Returns:
            Transformed coefficients of shape (B, width, decoder_dim)
        """
        if not self.use_factorized_weights:
            # Full transformation: einsum contracts over input width and encoder dims
            return torch.einsum("bwi,woij->boj", z, self.weight)
        else:
            # Factorized transformation: width mixing then mode mixing
            z_t = z.transpose(1, 2)  # (B, encoder_dim, width)
            z_width_t = torch.einsum("bei,eij->bej", z_t, self.width_weights)
            z_width = z_width_t.transpose(1, 2)  # (B, width, encoder_dim)
            return torch.einsum("boi,oij->boj", z_width, self.mode_weight)

    def _apply_skip_connection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute skip connection with 1x1 convolution.
        
        Args:
            x: Input tensor of shape (B, width, input_dim)
            
        Returns:
            Skip tensor of shape (B, width, output_dim)
        """
        B = x.shape[0]
        
        if self.input_dim == self.output_dim:
            # Same dimensions: apply convolution directly
            return self.skip_conv(x)
        else:
            # Dimension transition: reshape to separate spatial and channel dims
            # (B, width, C_in * N * M) -> (B, width, C_in, N * M)
            x_reshape = x.view(B, self.width, self.input_channels, -1)
            # (B, width, C_in, N * M) -> (B, width * C_in, N * M)
            x_reshape = x_reshape.view(B, self.width * self.input_channels, -1)
            
            # Apply convolution: (B, width * C_in, N * M) -> (B, width * C_out, N * M)
            out = self.skip_conv(x_reshape)
            
            # Reshape back: (B, width * C_out, N * M) -> (B, width, C_out * N * M)
            out = out.view(B, self.width, self.C_out, -1)
            return out.view(B, self.width, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the reduced basis block.
        
        Args:
            x: Input tensor of shape (B, width, input_dim)
            
        Returns:
            Output tensor of shape (B, width, output_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (B, width, D), got shape {x.shape}")
        
        # 1. Encode to PCA coefficient space
        z = self.encode(x)
        
        # 2. Transform in coefficient space
        z_out = self._transform_coefficients(z)
        
        # 3. Apply batch normalization (if enabled)
        z_out = z_out.transpose(1, 2)  # (B, decoder_dim, width) for BatchNorm1d
        z_out = self.bn(z_out)
        z_out = z_out.transpose(1, 2)  # (B, width, decoder_dim)
        
        # 4. Apply dropout
        z_out = self.dropout(z_out)
        
        # 5. Decode back to function space
        y = self.decode(z_out)
        
        # 6. Add skip connection
        y = y + self._apply_skip_connection(x)
        
        return y


class ScalarGate(nn.Module):
    """
    Learnable scalar gate for residual connections.
    
    Multiplies input by a learnable scalar, allowing the network to
    adaptively weight residual contributions during training.
    
    Args:
        init: Initial value of the scalar gate (default: 0.1)
    """

    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha


class MRBNO(nn.Module):
    """
    Multi-Resolution Basis Neural Operator.
    
    A neural operator that learns mappings between function spaces using
    data-driven PCA bases. The architecture consists of:
    
        1. Coordinate embedding (optional): Injects spatial position information
        2. Lifting layer: Projects input to higher-dimensional feature space
        3. Reduced basis blocks: Transform features using learned PCA representations
        4. Projection layers: Map back to output function space
        5. Global skip connection: Direct path from input to output
    
    The key innovation is using PCA bases fitted on training data instead of
    fixed Fourier bases, allowing the representation to adapt to problem structure.
    
    Args:
        width: Number of hidden channels in lifted representation
        rb: Fitted PCAReducerMultiChannel containing input/output PCA bases
        k_in: Number of input PCA modes to retain
        k_out: Number of output PCA modes to retain
        layers: Number of reduced basis blocks (excluding transition blocks)
        dropout: Dropout probability for regularization
        init_scale: Initial value for scalar gates
        use_skip_connections: If True, add residual connections within blocks
        use_batch_norm: If True, apply batch normalization in latent space
        use_coordinates: If True, inject coordinate features into input
        add_constant_mode: If True, append constant basis vector to PCA modes
        use_factorized_weights: If True or 'auto', use factorized weight representation.
            If 'auto', automatically uses factorized when k_in × k_out > threshold
    
    Shape:
        - Input: (B, C_in, N, M) where C_in is number of input channels
        - Output: (B, C_out, N, M) where C_out is number of output channels
    
    Example:
        >>> from MRBNO_simple import PCAReducerMultiChannel
        >>> # Fit PCA on training data
        >>> rb = PCAReducerMultiChannel().fit(X_train, Y_train)
        >>> k_in, k_out = rb.k_from_cutoff(j_in=0.999, j_out=0.999)
        >>> # Create model
        >>> model = MRBNO(
        ...     width=24,
        ...     rb=rb,
        ...     k_in=k_in,
        ...     k_out=k_out,
        ...     layers=4,
        ... )
        >>> # Forward pass
        >>> y = model(x)  # x: (B, C_in, N, M) -> y: (B, C_out, N, M)
    """

    def __init__(
        self,
        width: int = 24,
        rb=None,
        k_in: int = 50,
        k_out: int = 50,
        layers: int = 4,
        dropout: float = 0.0,
        init_scale: float = 0.1,
        use_skip_connections: bool = True,
        use_batch_norm: bool = True,
        use_coordinates: bool = False,
        add_constant_mode: bool = False,
        use_factorized_weights: bool = False,
        use_factorized_weights_auto_threshold: int = 10000,
    ):
        super().__init__()
        
        # Validate PCAReducer
        if rb is None:
            raise ValueError("rb (PCAReducer) must be provided")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted before creating model")

        # Store dimensions
        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)
        self.D_in = self.C_in * self.N * self.M
        self.D_out = self.C_out * self.N * self.M

        # Store configuration
        self.width = width
        self.num_layers = layers
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.use_coordinates = use_coordinates
        self.add_constant_mode = add_constant_mode
        # Handle auto mode for factorized weights
        if use_factorized_weights == 'auto':
            self.use_factorized_weights = 'auto'
            self.use_factorized_weights_auto_threshold = use_factorized_weights_auto_threshold
        else:
            self.use_factorized_weights = bool(use_factorized_weights)
            self.use_factorized_weights_auto_threshold = use_factorized_weights_auto_threshold

        # Optional coordinate embedding
        if self.use_coordinates:
            self.coord_embed = nn.Conv2d(2, self.C_in, kernel_size=1)
        else:
            self.coord_embed = None

        # Lifting layer: (B, 1, D_in) -> (B, width, D_in)
        self.lift = nn.Conv1d(1, width, kernel_size=1)

        # Block 0: Input basis -> Input basis
        self.block0 = ReducedBasisBlock(
            width=width,
            rb=rb,
            k_in=k_in,
            k_out=k_in,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True,
            use_batch_norm=use_batch_norm,
            add_constant_mode=add_constant_mode,
            use_factorized_weights=use_factorized_weights,
        )
        self.gate0 = ScalarGate(init=init_scale)

        # Block 1: Input basis -> Output basis (transition)
        self.block1 = ReducedBasisBlock(
            width=width,
            rb=rb,
            k_in=k_in,
            k_out=k_out,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False,
            use_batch_norm=use_batch_norm,
            add_constant_mode=add_constant_mode,
            use_factorized_weights=use_factorized_weights,
        )
        self.gate1 = ScalarGate(init=init_scale)

        # Global skip connection: direct path from input to output
        self.global_skip_proj = nn.Conv2d(self.C_in, self.C_out, kernel_size=1)
        self.global_skip_gate = ScalarGate(init=init_scale)

        # Subsequent blocks: Output basis -> Output basis
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        
        for _ in range(max(0, layers - 1)):
            self.blocks.append(
                ReducedBasisBlock(
                    width=width,
                    rb=rb,
                    k_in=k_in,
                    k_out=k_out,
                    dropout=dropout,
                    use_input_basis_for_encoder=False,
                    use_input_basis_for_decoder=False,
                    use_batch_norm=use_batch_norm,
                    add_constant_mode=add_constant_mode,
                    use_factorized_weights=use_factorized_weights,
                )
            )
            self.gates.append(ScalarGate(init=init_scale))

        # Projection layers
        self.proj1 = nn.Conv1d(width, width, kernel_size=1)
        self.proj2 = nn.Conv1d(width, 1, kernel_size=1)
        
        # Activation function
        self.act = nn.GELU()

    def _get_coordinate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate normalized coordinate grid.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Coordinate tensor of shape (B, 2, H, W) with x and y coordinates
        """
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype
        
        # Create normalized coordinate grids [0, 1]
        xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
        
        # Expand to batch size
        xs = xs.view(1, 1, 1, W).expand(B, 1, H, W)
        ys = ys.view(1, 1, H, 1).expand(B, 1, H, W)
        
        return torch.cat([xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MRBNO.

        Args:
            x: Input tensor of shape (B, C_in, N, M) for regular grids or (B, C_in, N) for irregular grids

        Returns:
            Output tensor of shape (B, C_out, N, M) for regular grids or (B, C_out, N) for irregular grids
        """
        # Handle both 3D (irregular grid) and 4D (regular grid) inputs
        is_irregular = x.ndim == 3
        if is_irregular:
            x = x.unsqueeze(-1)  # (B, C, N) -> (B, C, N, 1)

        if x.ndim != 4:
            raise ValueError(f"Expected 3D (B, C, N) or 4D (B, C, N, M) input, got shape {x.shape}")

        B, C, H, W = x.shape

        # Validate input dimensions
        if C != self.C_in:
            raise ValueError(f"Expected {self.C_in} input channels, got {C}")
        if H != self.N or W != self.M:
            raise ValueError(f"Expected spatial dims ({self.N}, {self.M}), got ({H}, {W})")

        # Optional: inject coordinate information (only for regular grids)
        if self.use_coordinates and self.coord_embed is not None and not is_irregular:
            coords = self._get_coordinate_features(x)
            x = x + self.coord_embed(coords)

        # Compute global skip connection (before lifting)
        global_skip = self.global_skip_gate(self.global_skip_proj(x))

        # Flatten spatial dimensions and add sequence dimension
        x_flat = x.reshape(B, self.D_in)
        x_seq = x_flat.unsqueeze(1)  # (B, 1, D_in)

        # Lift to higher-dimensional feature space
        h = self.lift(x_seq)  # (B, width, D_in)

        # Block 0: process in input basis space
        h0 = self.act(self.gate0(self.block0(h)))

        if self.use_skip_connections:
            h = h0 + h
        else:
            h = h0

        # Block 1: transition from input to output basis
        h = self.act(self.gate1(self.block1(h)))  # (B, width, D_out)

        # Subsequent blocks: process in output basis space
        for block, gate in zip(self.blocks, self.gates):
            residual = self.act(gate(block(h)))

            if self.use_skip_connections:
                h = h + residual
            else:
                h = residual

        # Project back to output space
        h = self.act(self.proj1(h))
        h = self.proj2(h)  # (B, 1, D_out)

        # Reshape to spatial dimensions
        y = h.reshape(B, self.C_out, self.N, self.M)

        # Add global skip connection
        y = y + global_skip

        # Return in same format as input (3D for irregular, 4D for regular)
        if is_irregular:
            y = y.squeeze(-1)  # (B, C_out, N, 1) -> (B, C_out, N)

        return y

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return string representation with key hyperparameters."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  spatial_dims=({self.N}, {self.M}),\n"
            f"  channels=(in={self.C_in}, out={self.C_out}),\n"
            f"  width={self.width},\n"
            f"  layers={self.num_layers},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )
