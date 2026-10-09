"""
Adaptive Modal Operator Refinement (AMOR)

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


def _rb_spatial_metadata(rb) -> Tuple[Tuple[int, ...], int, int]:
    """(spatial_shape, spatial_size, spatial_ndim) from PCAReducer ``shape_``."""
    sh = tuple(int(x) for x in rb.shape_)
    if len(sh) < 2:
        raise ValueError(f"rb.shape_ must have length >= 2, got {sh}")
    size = 1
    for x in sh:
        size *= x
    return sh, size, len(sh)


from .bases.learnable import (
    CayleyOrthonormalBasis,
    LearnableOrthonormalBasis,
    QROrthonormalBasis,
    SpanFrozenCayleyBasis,
    make_learnable_basis,
)
TrainableOrthonormalBasis = QROrthonormalBasis


class SharedFixedPCABasis(nn.Module):
    """One global copy of each fixed PCA basis (input/output), shared by all blocks.

    Stores decoder Φ as (k, D) plus mean (D,). When a dual encoder E is present
    (``encoder_components_``), encode uses E and decode uses Φ; otherwise encode
    and decode share Φ (Euclidean / M-adjoint handled outside).
    """

    def __init__(self, rb, k_in: int, k_out: int):
        super().__init__()
        k_in = int(min(k_in, rb.pca_in.components_.shape[0]))
        k_out = int(min(k_out, rb.pca_out.components_.shape[0]))
        self.register_buffer("in_comp", torch.from_numpy(rb.pca_in.components_[:k_in].astype(np.float32)))
        self.register_buffer("in_mean", torch.from_numpy(rb.pca_in.mean_.astype(np.float32)))
        self.register_buffer("out_comp", torch.from_numpy(rb.pca_out.components_[:k_out].astype(np.float32)))
        self.register_buffer("out_mean", torch.from_numpy(rb.pca_out.mean_.astype(np.float32)))
        enc_in = getattr(rb.pca_in, "encoder_components_", None)
        enc_out = getattr(rb.pca_out, "encoder_components_", None)
        if enc_in is not None:
            self.register_buffer(
                "in_enc", torch.from_numpy(np.asarray(enc_in[:k_in], dtype=np.float32))
            )
        else:
            self.in_enc = None
        if enc_out is not None:
            self.register_buffer(
                "out_enc", torch.from_numpy(np.asarray(enc_out[:k_out], dtype=np.float32))
            )
        else:
            self.out_enc = None


class ReducedBasisBlock(nn.Module):
    """
    Core building block of the AMOR architecture.
    
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
        use_factorized_weights="auto",
        use_factorized_weights_auto_threshold: int = 10000,
        shared_enc_basis: Optional[LearnableOrthonormalBasis] = None,
        shared_dec_basis: Optional[LearnableOrthonormalBasis] = None,
        shared_fixed_basis: Optional["SharedFixedPCABasis"] = None,
        mass: bool = False,
    ):
        super().__init__()

        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted before creating blocks")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")

        # Store spatial and channel dimensions
        self.spatial_shape, self.spatial_size, self.spatial_ndim = _rb_spatial_metadata(rb)
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)
        self.mass = bool(mass)

        # Compute flattened dimensions
        self.D_in = self.C_in * self.spatial_size
        self.D_out = self.C_out * self.spatial_size
        # M-weighted encode needs the flat D-dim layout with one weight per DOF,
        # so the single-channel spatial shortcut does not apply.
        self.spatial_field = (
            not self.mass
            and int(rb.pca_in.components_.shape[1]) == self.spatial_size
            and int(rb.pca_out.components_.shape[1]) == self.spatial_size
        )
        
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
        self.shared_enc_basis = shared_enc_basis
        self.shared_dec_basis = shared_dec_basis
        self.trainable_basis = shared_enc_basis is not None or shared_dec_basis is not None

        # Determine input/output dimensions based on basis choices
        if self.spatial_field:
            self.input_dim = self.spatial_size
            self.output_dim = self.spatial_size
            self.input_channels = 1
        elif use_input_basis_for_encoder:
            self.input_dim = self.D_in
            self.input_channels = self.C_in
        else:
            self.input_dim = self.D_out
            self.input_channels = self.C_out

        if self.spatial_field:
            pass
        elif use_input_basis_for_decoder:
            self.output_dim = self.D_in
        else:
            self.output_dim = self.D_out

        # Validate k values against available PCA components
        self._validate_k_values(rb, k_in, k_out)

        # Shared fixed bases: reference one global copy per PCA side instead of
        # per-block buffers (~5x less GPU memory across the 5 blocks). Stored via
        # object.__setattr__ so nn.Module does not register it as a duplicate
        # submodule under every block (the holder is owned once by the parent AMOR).
        object.__setattr__(self, "_shared_fixed_basis", shared_fixed_basis)
        self.use_shared_fixed_basis = (
            shared_fixed_basis is not None
            and not self.trainable_basis
            and not add_constant_mode
        )

        object.__setattr__(self, "_has_dual_encoder", False)
        if self.use_shared_fixed_basis:
            # No per-block buffers; dims follow directly from k and basis choice.
            self.encoder_dim = self.k_in if use_input_basis_for_encoder else self.k_out
            self.decoder_dim = self.k_in if use_input_basis_for_decoder else self.k_out
            # Shared path prefers E when registered on SharedFixedPCABasis.
            h = shared_fixed_basis
            has_e = (
                (use_input_basis_for_encoder and getattr(h, "in_enc", None) is not None)
                or ((not use_input_basis_for_encoder) and getattr(h, "out_enc", None) is not None)
            )
            object.__setattr__(self, "_has_dual_encoder", bool(has_e))
        else:
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

        if self.use_shared_fixed_basis:
            pass
        elif not self.trainable_basis:
            # Only the k used modes are stored; buffers move with the module (GPU when model on GPU).
            self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))
            self.register_buffer("enc_mean", torch.from_numpy(enc_mean))
            self.register_buffer("dec_components", torch.from_numpy(dec_components))
            self.register_buffer("dec_mean", torch.from_numpy(dec_mean))
        else:
            if self.use_input_basis_for_encoder and self.shared_enc_basis is None:
                raise ValueError("trainable input encoder requires shared_enc_basis")
            if not self.use_input_basis_for_encoder and self.shared_dec_basis is None:
                raise ValueError("trainable output encoder requires shared_dec_basis")
            if self.use_input_basis_for_decoder and self.shared_enc_basis is None:
                raise ValueError("trainable input decoder requires shared_enc_basis")
            if not self.use_input_basis_for_decoder and self.shared_dec_basis is None:
                raise ValueError("trainable output decoder requires shared_dec_basis")

        self._enc_mass_sp = None  # scipy.sparse CSR for general M (CPU encode path)
        if self.mass:
            if not hasattr(rb, "mass_matrix"):
                raise ValueError(
                    "mass=True requires rb.mass_matrix() (consistent sparse P1); "
                    "diagonal-only mass is not supported"
                )
            if not self.trainable_basis:
                M = rb.mass_matrix(use_input_basis=self.use_input_basis_for_encoder)
                if M is None:
                    raise RuntimeError("rb.mass_matrix() returned None")
                object.__setattr__(self, "_enc_mass_sp", M)
                # Row-sum buffer kept for diagnostics / legacy ckpt keys only.
                enc_mass = np.asarray(M.sum(axis=1)).reshape(-1)
                self.register_buffer(
                    "enc_mass_diag", torch.from_numpy(enc_mass[: self.input_dim].astype(np.float32))
                )

        # Handle auto mode for factorized weights (now that dimensions are known)
        if self.use_factorized_weights == 'auto':
            should_use_factorized = (self.encoder_dim * self.decoder_dim) > self.use_factorized_weights_auto_threshold
            self.use_factorized_weights = should_use_factorized

        # Initialize learnable transformation weights
        self._init_transformation_weights()

        # Skip connection with 1x1 convolution
        self._init_skip_connection()

        # Regularization layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bn = nn.BatchNorm1d(self.decoder_dim) if use_batch_norm else nn.Identity()

        # Apply weight initialization
        self._init_weights()

    def to(self, *args, **kwargs):
        """Move module and fixed PCA bases together (encode/decode GEMM on-device)."""
        return super().to(*args, **kwargs)

    def _validate_k_values(self, rb, k_in: int, k_out: int) -> None:
        """Validate k against the PCA side each encoder/decoder actually uses."""
        max_k_in = rb.pca_in.n_components_
        max_k_out = rb.pca_out.n_components_
        enc_max = max_k_in if self.use_input_basis_for_encoder else max_k_out
        dec_max = max_k_in if self.use_input_basis_for_decoder else max_k_out
        if not (1 <= k_in <= enc_max):
            raise ValueError(f"k_in={k_in} out of range [1, {enc_max}] for encoder basis")
        if not (1 <= k_out <= dec_max):
            raise ValueError(f"k_out={k_out} out of range [1, {dec_max}] for decoder basis")

    def _setup_encoder_components(self, rb) -> Tuple[np.ndarray, np.ndarray]:
        """Encoder rows: prefer dual ``encoder_components_`` (DIS / mesh-transfer E).

        When E is present, encode is z = E(x−μ) (never also multiply by M).
        When mass=True and E is absent, store Φ and apply sparse M in ``encode``.
        """
        if self.use_input_basis_for_encoder:
            pca = rb.pca_in
            k = self.k_in
        else:
            pca = rb.pca_out
            k = self.k_out
        enc = getattr(pca, "encoder_components_", None)
        use_dual = enc is not None
        object.__setattr__(self, "_has_dual_encoder", bool(use_dual))
        if use_dual:
            components = np.asarray(enc[:k], dtype=np.float32)
        else:
            components = np.asarray(pca.components_[:k], dtype=np.float32)
        mean = np.asarray(pca.mean_, dtype=np.float32)
        if components.shape[1] < self.input_dim:
            padding = np.zeros(
                (components.shape[0], self.input_dim - components.shape[1]),
                dtype=np.float32,
            )
            components = np.concatenate([components, padding], axis=1)
        if mean.shape[0] < self.input_dim:
            mean = np.pad(mean, (0, self.input_dim - mean.shape[0]), mode="constant")
        return components, mean

    def _setup_decoder_components(self, rb) -> Tuple[np.ndarray, np.ndarray]:
        """Decoder rows = physical modes Φ (``components_``)."""
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
        Append the Euclidean unit-norm constant mode
        e_1 = n^{-1/2} 1 to the encoder and decoder bases.
        """
        enc_n = max(int(enc_components.shape[1]), 1)
        dec_n = max(int(dec_components.shape[1]), 1)
        enc_const = np.full((1, enc_n), 1.0 / np.sqrt(enc_n), dtype=np.float32)
        dec_const = np.full((1, dec_n), 1.0 / np.sqrt(dec_n), dtype=np.float32)
        enc_components = np.concatenate([enc_components, enc_const], axis=0)
        dec_components = np.concatenate([dec_components, dec_const], axis=0)
        return enc_components, dec_components

    def _init_transformation_weights(self) -> None:
        """Initialize learnable weights for latent space transformation."""
        # Auto mode: factorized weights when encoder_dim * decoder_dim > threshold
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

        if self.trainable_basis:
            basis = self.shared_enc_basis if self.use_input_basis_for_encoder else self.shared_dec_basis
            z_flat = basis.encode(x_flat, D)
        elif self.use_shared_fixed_basis:
            # Shared global basis (k, D); dual / M-adjoint E preferred when present.
            h = self._shared_fixed_basis
            if self.use_input_basis_for_encoder:
                enc = h.in_enc if getattr(h, "in_enc", None) is not None else h.in_comp
                mean = h.in_mean
            else:
                enc = h.out_enc if getattr(h, "out_enc", None) is not None else h.out_comp
                mean = h.out_mean
            z_flat = torch.matmul(x_flat - mean[:D].unsqueeze(0), enc[:, :D].transpose(0, 1))
        elif getattr(self, "_has_dual_encoder", False):
            # Dual / M-adjoint rows already bake the metric: z = E (x − μ).
            enc = self.enc_components_t[:D, :]
            enc_mean = self.enc_mean[:D]
            z_flat = torch.matmul(x_flat - enc_mean.unsqueeze(0), enc)
        elif self.mass:
            # M-weighted encode: z = Φᵀ M (x − μ). Requires sparse mass_matrix
            # (consistent P1); no diagonal-M assumption. Φ stored in enc_components.
            x_centered = x_flat - self.enc_mean[:D]
            enc = self.enc_components_t[:D, :]  # (D, k) = Φᵀ
            M = getattr(self, "_enc_mass_sp", None)
            if M is None:
                raise RuntimeError(
                    "mass=True encode requires sparse mass_matrix on the reducer "
                    "(consistent P1); diagonal-only fallback removed"
                )
            xc = x_centered.detach().cpu().numpy()
            if M.shape[0] != D or M.shape[1] != D:
                M = M[:D, :D]
            Mx = M @ xc.T  # (D, B)
            z_np = (enc.detach().cpu().numpy().T @ Mx).T  # (B, k)
            z_flat = torch.from_numpy(np.asarray(z_np, dtype=np.float32)).to(
                device=x_flat.device, dtype=x_flat.dtype
            )
        else:
            # On-device GEMM: bases move with the module (GPU when model on GPU).
            enc = self.enc_components_t[:D, :]
            enc_mean = self.enc_mean[:D]
            z_flat = torch.matmul(x_flat - enc_mean.unsqueeze(0), enc)
        
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

        if self.trainable_basis:
            basis = self.shared_enc_basis if self.use_input_basis_for_decoder else self.shared_dec_basis
            y_flat = basis.decode(z_flat, self.output_dim)
        elif self.use_shared_fixed_basis:
            # Shared global basis (k, D); decode = z @ components + mean on-device.
            h = self._shared_fixed_basis
            comp, mean = (h.in_comp, h.in_mean) if self.use_input_basis_for_decoder else (h.out_comp, h.out_mean)
            out_dim = self.output_dim
            y_flat = torch.matmul(z_flat, comp[:, :out_dim]) + mean[:out_dim]
        else:
            # On-device GEMM: bases move with the module (GPU when model on GPU).
            y_flat = torch.matmul(z_flat, self.dec_components) + self.dec_mean
        
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


class AMOR(nn.Module):
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
        layers: Total number of nonlocal blocks, including the input→output
            transition. Requires ``layers >= 2``: block0 (input→input), block1
            (transition), then ``layers - 2`` output→output residual blocks.
            ``uniform_basis_blocks=True`` builds exactly ``layers`` blocks on the
            output basis and omits the transition.
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
        >>> from AMOR_simple import PCAReducerMultiChannel
        >>> # Fit PCA on training data
        >>> rb = PCAReducerMultiChannel().fit(X_train, Y_train)
        >>> k_in, k_out = rb.k_from_cutoff(j_in=0.999, j_out=0.999)
        >>> # Create model
        >>> model = AMOR(
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
        use_batch_norm: bool = False,
        use_coordinates: bool = False,
        add_constant_mode: bool = False,
        use_factorized_weights="auto",
        use_factorized_weights_auto_threshold: int = 10000,
        trainable_basis: bool = False,
        trainable_basis_param: str = "qr",
        uniform_basis_blocks: bool = False,
        share_bases: bool = True,
        mass: Optional[bool] = None,
        use_film_conditioning: bool = False,
        film_hidden: int = 64,
        film_in_dim: int = 1,
    ):
        super().__init__()
        
        # Validate PCAReducer
        if rb is None:
            raise ValueError("rb (PCAReducer) must be provided")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted before creating model")

        if uniform_basis_blocks and trainable_basis:
            raise ValueError("uniform_basis_blocks not implemented with trainable_basis")

        # M-weighted encode when the reducer carries a sparse mass matrix (FEM);
        # pass mass explicitly to override the reducer-based default.
        self.mass = bool(hasattr(rb, "mass_matrix")) if mass is None else bool(mass)
        if self.mass and not hasattr(rb, "mass_matrix"):
            raise ValueError("mass=True requires PCAReducerMultiChannelMass (sparse M)")

        # Store dimensions
        self.spatial_shape, self.spatial_size, self.spatial_ndim = _rb_spatial_metadata(rb)
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, 'out_channels', 1)
        self.D_in = self.C_in * self.spatial_size
        self.D_out = self.C_out * self.spatial_size
        self.spatial_field = (
            not self.mass
            and int(rb.pca_in.components_.shape[1]) == self.spatial_size
            and int(rb.pca_out.components_.shape[1]) == self.spatial_size
        )
        self.trainable_basis = trainable_basis
        self.trainable_basis_param = trainable_basis_param
        self.uniform_basis_blocks = uniform_basis_blocks

        shared_enc_basis = None
        shared_dec_basis = None
        if trainable_basis:
            if add_constant_mode:
                raise ValueError("trainable_basis incompatible with add_constant_mode")
            enc_init = rb.pca_in.components_[:k_in].astype(np.float32)
            dec_init = rb.pca_out.components_[:k_out].astype(np.float32)
            basis_dim_in = self.spatial_size if self.spatial_field else self.D_in
            basis_dim_out = self.spatial_size if self.spatial_field else self.D_out
            enc_mass = rb.encoder_mass_diag(use_input_basis=True, dim=basis_dim_in) if self.mass else None
            dec_mass = rb.encoder_mass_diag(use_input_basis=False, dim=basis_dim_out) if self.mass else None
            enc_M = rb.mass_matrix(use_input_basis=True) if self.mass and hasattr(rb, "mass_matrix") else None
            dec_M = rb.mass_matrix(use_input_basis=False) if self.mass and hasattr(rb, "mass_matrix") else None
            shared_enc_basis = make_learnable_basis(
                trainable_basis_param,
                basis_dim_in,
                k_in,
                enc_init,
                rb.pca_in.mean_.astype(np.float32),
                enc_mass,
                mass_matrix=enc_M,
            )
            shared_dec_basis = make_learnable_basis(
                trainable_basis_param,
                basis_dim_out,
                k_out,
                dec_init,
                rb.pca_out.mean_.astype(np.float32),
                dec_mass,
                mass_matrix=dec_M,
            )
            # The blocks own these modules; the top-level reference only reproduces
            # the Euclidean checkpoint layout, which also stores them at model level.
            if not self.mass:
                self.shared_enc_basis = shared_enc_basis
                self.shared_dec_basis = shared_dec_basis

        # Single shared copy of the fixed PCA/DIS bases (owned by the parent,
        # referenced by every block including the transition) — avoids O(L)
        # duplicated buffers. Default on for fixed bases; off when trainable.
        self.shared_fixed_basis = None
        if share_bases and not trainable_basis and not add_constant_mode:
            self.shared_fixed_basis = SharedFixedPCABasis(rb, k_in, k_out)
        self.share_bases = bool(self.shared_fixed_basis is not None)

        # Dual-encoder DIS stores E / Φ on the reducer; no input whitening.
        self._whiten_input = False

        # Store configuration
        self.width = width
        self.k_in = int(k_in)
        self.k_out = int(k_out)
        self.num_layers = layers
        # Alias spatial dims for mesh-generalization callers (model.N / model.M).
        if self.spatial_ndim >= 2:
            self.N = int(self.spatial_shape[0])
            self.M = int(self.spatial_shape[1])
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.use_coordinates = use_coordinates
        self.use_film_conditioning = use_film_conditioning
        # FiLM modulates the lifted (B, width, D) latent, and M-weighted encode works
        # on the flat D-dim layout, so both use the flat path. The two paths are
        # numerically equivalent; they differ only in the rank of the skip weights.
        self._native_grid_forward = not use_film_conditioning and not self.mass and (
            self.spatial_ndim >= 3
            or (self.spatial_ndim == 2 and not self.use_coordinates)
        )
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

        if self.spatial_field:
            if self.spatial_ndim == 3:
                self.lift = nn.Conv3d(self.C_in, width, kernel_size=1)
                self.proj1 = nn.Conv3d(width, width, kernel_size=1)
                self.proj2 = nn.Conv3d(width, self.C_out, kernel_size=1)
            elif self.spatial_ndim == 2 and not self.use_coordinates:
                self.lift = nn.Conv2d(self.C_in, width, kernel_size=1)
                self.proj1 = nn.Conv2d(width, width, kernel_size=1)
                self.proj2 = nn.Conv2d(width, self.C_out, kernel_size=1)
            else:
                self.lift = nn.Conv1d(self.C_in, width, kernel_size=1)
                self.proj1 = nn.Conv1d(width, width, kernel_size=1)
                self.proj2 = nn.Conv1d(width, self.C_out, kernel_size=1)
        else:
            self.lift = nn.Conv1d(1, width, kernel_size=1)
            self.proj1 = nn.Conv1d(width, width, kernel_size=1)
            self.proj2 = nn.Conv1d(width, 1, kernel_size=1)

        block_kw = dict(
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            add_constant_mode=add_constant_mode,
            use_factorized_weights=use_factorized_weights,
            shared_enc_basis=shared_enc_basis,
            shared_dec_basis=shared_dec_basis,
            shared_fixed_basis=self.shared_fixed_basis,
            mass=self.mass,
        )

        if uniform_basis_blocks:
            # All layers in one PCA side (pca_out): no input→output basis transition.
            self.block0 = None
            self.gate0 = None
            self.block1 = None
            self.gate1 = None
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()
            for _ in range(layers):
                self.blocks.append(
                    ReducedBasisBlock(
                        width=width,
                        rb=rb,
                        k_in=k_out,
                        k_out=k_out,
                        use_input_basis_for_encoder=False,
                        use_input_basis_for_decoder=False,
                        **block_kw,
                    )
                )
                self.gates.append(ScalarGate(init=init_scale))
        else:
            if layers < 2:
                raise ValueError(
                    f"layers must be >= 2 (total nonlocal blocks including transition), got {layers}"
                )
            # Block 0: Input basis -> Input basis
            self.block0 = ReducedBasisBlock(
                width=width,
                rb=rb,
                k_in=k_in,
                k_out=k_in,
                use_input_basis_for_encoder=True,
                use_input_basis_for_decoder=True,
                **block_kw,
            )
            self.gate0 = ScalarGate(init=init_scale)

            # Block 1: Input basis -> Output basis (transition; counted in ``layers``)
            self.block1 = ReducedBasisBlock(
                width=width,
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                use_input_basis_for_encoder=True,
                use_input_basis_for_decoder=False,
                **block_kw,
            )
            self.gate1 = ScalarGate(init=init_scale)

            # Subsequent blocks: Output basis -> Output basis
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()

            for _ in range(layers - 2):
                self.blocks.append(
                    ReducedBasisBlock(
                        width=width,
                        rb=rb,
                        k_in=k_out,
                        k_out=k_out,
                        use_input_basis_for_encoder=False,
                        use_input_basis_for_decoder=False,
                        **block_kw,
                    )
                )
                self.gates.append(ScalarGate(init=init_scale))

        # Global skip: Conv1d on flattened native grids; Conv2d only for legacy 2D+coords
        if self._native_grid_forward:
            self.global_skip_proj = nn.Conv1d(self.C_in, self.C_out, kernel_size=1)
        else:
            self.global_skip_proj = nn.Conv2d(self.C_in, self.C_out, kernel_size=1)
        self.global_skip_gate = ScalarGate(init=init_scale)

        # Activation function
        self.act = nn.GELU()

        # FiLM: per-sample (gamma, beta) for each lifted channel
        if use_film_conditioning:
            self.film_mlp = nn.Sequential(
                nn.Linear(int(film_in_dim), film_hidden),
                nn.GELU(),
                nn.Linear(film_hidden, 2 * width),
            )
        else:
            self.film_mlp = None

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def _apply_film(self, h: torch.Tensor, film_cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.film_mlp is None:
            return h
        if film_cond is None:
            raise ValueError("use_film_conditioning=True requires film_cond input")
        cond = film_cond.view(film_cond.shape[0], -1).to(h.dtype)
        gamma, beta = torch.chunk(self.film_mlp(cond), 2, dim=1)
        return (1.0 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)

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

    def _apply_global_skip(self, x: torch.Tensor) -> torch.Tensor:
        if self._native_grid_forward:
            b = x.shape[0]
            flat = x.reshape(b, self.C_in, self.spatial_size)
            skip = self.global_skip_proj(flat)
            return self.global_skip_gate(
                skip.reshape(b, self.C_out, *self.spatial_shape)
            )
        return self.global_skip_gate(self.global_skip_proj(x))

    def _forward_native_grid(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, *spatial_shape) for native ND grids (2D/3D/... without coord embed)."""
        expected_ndim = 2 + self.spatial_ndim
        if x.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim}D input (B, C, *{self.spatial_shape}), got {x.shape}"
            )
        b, c = x.shape[:2]
        spatial = tuple(x.shape[2:])
        if c != self.C_in:
            raise ValueError(f"Expected {self.C_in} input channels, got {c}")
        if spatial != self.spatial_shape:
            raise ValueError(f"Expected spatial dims {self.spatial_shape}, got {spatial}")
        if self.use_coordinates:
            raise NotImplementedError(
                f"use_coordinates not supported for native {self.spatial_ndim}D grids"
            )
        global_skip = self._apply_global_skip(x)
        x_flat = x.reshape(b, self.D_in)
        if self.spatial_field:
            xw = x_flat.reshape(b, self.C_in, *self.spatial_shape)
            h = self.lift(xw)
            h = h.reshape(b, self.width, self.spatial_size)
            h = self._forward_blocks(h)
            h = h.reshape(b, self.width, *self.spatial_shape)
            h = self.act(self.proj1(h))
            return self.proj2(h) + global_skip
        h = self.lift(x_flat.unsqueeze(1))
        h = self._forward_blocks(h)
        h = self.act(self.proj1(h))
        h = self.proj2(h)
        return h.reshape(b, self.C_out, *self.spatial_shape) + global_skip

    def forward(self, x: torch.Tensor, film_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the AMOR.

        Args:
            x: (B, C_in, *spatial) for native grids (2D/3D/ND), or (B, C_in, N) irregular 1D
            film_cond: (B, film_in_dim) conditioning, required when use_film_conditioning

        Returns:
            Output in the same layout as input.
        """
        if self._native_grid_forward:
            return self._forward_native_grid(x)

        # Legacy 2D+coords: (B, C, N, M) or irregular (B, C, N)
        is_irregular = x.ndim == 3
        if is_irregular:
            x = x.unsqueeze(-1)  # (B, C, N) -> (B, C, N, 1)

        if x.ndim != 4:
            raise ValueError(f"Expected 3D (B, C, N) or 4D (B, C, N, M) input, got shape {x.shape}")

        B, C, H, W = x.shape
        n, m = self.spatial_shape

        # Validate input dimensions
        if C != self.C_in:
            raise ValueError(f"Expected {self.C_in} input channels, got {C}")
        if H != n or W != m:
            raise ValueError(f"Expected spatial dims ({n}, {m}), got ({H}, {W})")

        # Optional: inject coordinate information (only for regular grids)
        if self.use_coordinates and self.coord_embed is not None and not is_irregular:
            coords = self._get_coordinate_features(x)
            x = x + self.coord_embed(coords)

        # Compute global skip connection (before lifting)
        global_skip = self._apply_global_skip(x)

        x_flat = x.reshape(B, self.D_in)
        if self.spatial_field:
            if is_irregular:
                xw = x_flat.reshape(B, self.C_in, n)
                h = self.lift(xw)
                h = self._forward_blocks(h)
                h = self.act(self.proj1(h))
                y = self.proj2(h).unsqueeze(-1)
            else:
                xw = x_flat.reshape(B, self.C_in, n, m)
                h = self.lift(xw)
                h = h.reshape(B, self.width, self.spatial_size)
                h = self._forward_blocks(h)
                h = h.reshape(B, self.width, n, m)
                h = self.act(self.proj1(h))
                y = self.proj2(h)
        else:
            x_seq = x_flat.unsqueeze(1)
            h = self.lift(x_seq)
            h = self._apply_film(h, film_cond)
            h = self._forward_blocks(h, film_cond)
            h = self.act(self.proj1(h))
            h = self._apply_film(h, film_cond)
            h = self.proj2(h)
            y = h.reshape(B, self.C_out, n, m)

        y = y + global_skip

        if is_irregular:
            y = y.squeeze(-1)

        return y

    def _forward_blocks(
        self, h: torch.Tensor, film_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.uniform_basis_blocks:
            for block, gate in zip(self.blocks, self.gates):
                residual = self._apply_film(self.act(gate(block(h))), film_cond)
                if self.use_skip_connections:
                    h = h + residual
                else:
                    h = residual
        else:
            h0 = self._apply_film(self.act(self.gate0(self.block0(h))), film_cond)
            if self.use_skip_connections:
                h = h0 + h
            else:
                h = h0
            h = self._apply_film(self.act(self.gate1(self.block1(h))), film_cond)
            for block, gate in zip(self.blocks, self.gates):
                residual = self._apply_film(self.act(gate(block(h))), film_cond)
                if self.use_skip_connections:
                    h = h + residual
                else:
                    h = residual
        return h

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_nonlocal_blocks(self) -> int:
        """Instantiated nonlocal blocks, including the input→output transition."""
        n = int(self.block0 is not None) + int(self.block1 is not None)
        return n + len(self.blocks)

    def __repr__(self) -> str:
        """Return string representation with key hyperparameters."""
        spatial_repr = str(self.spatial_shape)
        return (
            f"{self.__class__.__name__}(\n"
            f"  spatial_dims={spatial_repr},\n"
            f"  channels=(in={self.C_in}, out={self.C_out}),\n"
            f"  width={self.width},\n"
            f"  layers={self.num_layers},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )

# =============================================================================
# Mass-weighted AMOR (rib meshes). Euclidean AMOR is above.
# =============================================================================

def make_learnable_basis_mass(
    param: str,
    dim: int,
    k: int,
    mass_diag: np.ndarray,
    init_components: Optional[np.ndarray] = None,
    init_mean: Optional[np.ndarray] = None,
) -> LearnableOrthonormalBasis:
    """M-weighted learnable basis (bases/learnable.py with mass weights)."""
    return make_learnable_basis(param, dim, k, init_components, init_mean, mass_diag)


class ReducedBasisBlockMass(ReducedBasisBlock):
    """ReducedBasisBlock with M-weighted encode (mass=True)."""

    def __init__(self, *args, **kwargs):
        kwargs["mass"] = True
        super().__init__(*args, **kwargs)


ScalarGateMass = ScalarGate


class AMORMass(AMOR):
    """AMOR with M-weighted encode (mass=True)."""

    def __init__(self, *args, **kwargs):
        kwargs["mass"] = True
        super().__init__(*args, **kwargs)

