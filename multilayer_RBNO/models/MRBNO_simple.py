#!/usr/bin/env python3
"""Multilayer Reduced Basis Neural Operator with optional coordinate conditioning."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from sklearn.decomposition import PCA

from MRBNO_model import ScalarGate


class MRBNOSimple(nn.Module):
    """
    MRBNO Simple variant with optional (x, y) coordinate conditioning.

    When ``use_coordinates`` is True, normalized coordinate maps are projected with learnable 1×1
    convolutions and added to:
      * the inputs to Block 0 and Block 1,
      * the global skip and Block0→Block1 skip projections,
      * every residual update in the PCA output stack,
      * the final output bias.

    Disabling coordinates reproduces the vanilla MRBNO Simple architecture.
    """

    def __init__(
        self,
        rb: PCAReducerMultiChannel,
        k_in: int,
        k_out: int,
        layers: int = 7,
        hidden: int = 1024,
        dropout: float = 0.0,
        init_scale: float = 0.1,
        use_skip_connections: bool = True,
        latent_batch_norm: bool = False,
        use_coordinates: bool = True,
    ):
        super().__init__()

        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer must be fitted")

        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out_basis = getattr(rb, 'out_channels', 1)
        self.layers = layers
        self.use_skip_connections = use_skip_connections
        self.latent_batch_norm = latent_batch_norm
        self.use_coordinates = use_coordinates

        coord_channels = 2 if use_coordinates else 0

        if self.use_coordinates:
            self.coord_input_proj = nn.Conv2d(2, self.C_in, 1)
            self.coord_block1_proj = nn.Conv2d(2, self.C_in, 1)
            self.coord_output_proj = nn.Conv2d(2, self.C_out_basis, 1)
        else:
            self.coord_input_proj = None
            self.coord_block1_proj = None
            self.coord_output_proj = None

        # Block 0 (input basis)
        self.block0 = ReducedBasisResNetMultiChannel(
            rb,
            k_in=k_in,
            k_out=k_out,
            hidden=hidden,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True,
            latent_batch_norm=latent_batch_norm,
        )
        self.gate0 = ScalarGate(init=init_scale)

        # Block 1 (input → output basis)
        self.block1 = ReducedBasisResNetMultiChannel(
            rb,
            k_in=k_in,
            k_out=k_out,
            hidden=hidden,
            dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False,
            latent_batch_norm=latent_batch_norm,
        )
        self.gate1 = ScalarGate(init=init_scale)

        # Global skip projection (optionally augmented with coordinates)
        self.global_skip_proj = nn.Conv2d(self.C_in + coord_channels, self.C_out_basis, 1)
        self.global_skip_gate = ScalarGate(init=init_scale)

        if self.use_coordinates:
            self.block0_skip_proj = nn.Conv2d(self.C_in + coord_channels, self.C_out_basis, 1)
        else:
            self.block0_skip_proj = None

        if layers > 1:
            self.blocks = nn.ModuleList([
                ReducedBasisBlockSingleChannel(
                    rb,
                    k_out=k_out,
                    hidden=hidden,
                    dropout=dropout,
                    latent_batch_norm=latent_batch_norm,
                ) for _ in range(layers - 1)
            ])
            self.gates = nn.ModuleList([ScalarGate(init=init_scale) for _ in range(layers - 1)])
            if self.use_coordinates:
                self.residual_coord_projs = nn.ModuleList([
                    nn.Conv2d(2, self.C_out_basis, 1) for _ in range(layers - 1)
                ])
            else:
                self.residual_coord_projs = nn.ModuleList()
        else:
            self.blocks = nn.ModuleList()
            self.gates = nn.ModuleList()
            self.residual_coord_projs = nn.ModuleList()

        self.output_proj = nn.Conv2d(self.C_out_basis, self.C_out_basis, 1)
        self.act = nn.GELU()

    def _coord_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_coordinates:
            return None
        B, _, H, W = x.shape
        xs = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        return torch.cat([xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        if x.shape[1] != self.C_in:
            raise ValueError(f"Expected {self.C_in} input channels, got {x.shape[1]}")

        coords = self._coord_maps(x)
        if coords is not None:
            x_with_coords = torch.cat([x, coords], dim=1)
        else:
            x_with_coords = x

        global_skip = self.global_skip_gate(self.global_skip_proj(x_with_coords))

        if coords is not None and self.coord_input_proj is not None:
            x_for_block0 = x + self.coord_input_proj(coords)
        else:
            x_for_block0 = x

        y0 = self.gate0(self.block0(x_for_block0))
        y0 = self.act(y0)
        if self.use_skip_connections:
            y0 = y0 + x

        if coords is not None and self.coord_block1_proj is not None:
            y0_for_block1 = y0 + self.coord_block1_proj(coords)
        else:
            y0_for_block1 = y0

        if coords is not None:
            block0_skip_input = torch.cat([y0, coords], dim=1)
        else:
            block0_skip_input = y0

        y1 = self.gate1(self.block1(y0_for_block1))
        y1 = self.act(y1)

        if self.use_skip_connections:
            if self.block0_skip_proj is not None:
                y0_proj = self.block0_skip_proj(block0_skip_input)
            else:
                y0_proj = self.global_skip_proj(block0_skip_input)
            y = y1 + y0_proj
        else:
            y = y1

        y = y + global_skip
        if coords is not None and self.coord_output_proj is not None:
            y = y + self.coord_output_proj(coords)

        for idx, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            y = self.act(y)
            if coords is not None and self.use_coordinates and len(self.residual_coord_projs) > idx:
                y = y + self.residual_coord_projs[idx](coords)
            residual = gate(block(y))
            if self.use_skip_connections:
                y = y + residual
            else:
                y = residual

        return y


########################################################
# Reduced-basis building blocks reused by MRBNOSimple  #
########################################################

# Single-Channel Reduced Basis Block
class ReducedBasisBlockSingleChannel(nn.Module):
    """Residual block operating entirely in the PCA output basis."""

    def __init__(
        self,
        rb: "PCAReducerMultiChannel",
        k_out: int,
        hidden: int = 256,
        dropout: float = 0.0,
        latent_batch_norm: bool = False,
    ):
        super().__init__()

        if rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted")

        self.N, self.M = rb.shape_
        self.D_out = self.N * self.M
        self.C_out = getattr(rb, "out_channels", 1)
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
            self.out_proj,
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        x_centered = x_flat - self.dec_mean
        return x_centered @ self.dec_components.T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.dec_components + self.dec_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_out, N, M), got {x.shape}")
        batch, channels, height, width = x.shape
        if channels != self.C_out or height != self.N or width != self.M:
            raise ValueError(
                f"Expected (B, {self.C_out}, {self.N}, {self.M}), got {x.shape}"
            )

        x_flat = x.reshape(batch, -1)
        z = self.encode(x_flat)
        z_out = self.latent_net(z)
        y_flat = self.decode(z_out)
        return y_flat.reshape(batch, self.C_out, self.N, self.M)


# Multi-Channel Reduced Basis Block
class ReducedBasisResNetMultiChannel(nn.Module):
    """Core MRBNO block that handles tensors prepared in the PCA input basis."""

    def __init__(
        self,
        rb: "PCAReducerMultiChannel",
        k_in: int,
        k_out: int,
        hidden: int = 256,
        dropout: float = 0.0,
        use_input_basis_for_encoder: bool = False,
        use_input_basis_for_decoder: bool = False,
        latent_batch_norm: bool = False,
    ):
        super().__init__()

        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None:
            raise ValueError("PCAReducer not properly initialized")

        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = getattr(rb, "out_channels", 1)
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
                enc_mean = np.pad(enc_mean, (0, self.D_in - enc_mean.shape[0]), "constant")

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
            self.out_proj,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        x_centered = x_flat - self.enc_mean
        return x_centered @ self.enc_components_t

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.dec_components + self.dec_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_in, N, M), got {x.shape}")
        batch, channels, height, width = x.shape
        if channels != self.C_in or height != self.N or width != self.M:
            raise ValueError(
                f"Expected (B, {self.C_in}, {self.N}, {self.M}), got {x.shape}"
            )

        x_flat = x.reshape(batch, self.D_in)
        z = self.encode(x_flat)
        z_out = self.latent_net(z)
        y_flat = self.decode(z_out)
        if self.stay_multi_channel:
            return y_flat.reshape(batch, self.C_in, self.N, self.M)
        return y_flat.reshape(batch, self.C_out, self.N, self.M)


# PCA-based Reduced Basis Helper for Multi-Channel Inputs
class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors with optional boundary weighting."""

    def __init__(
        self,
        svd_solver: str = "auto",
        whiten: bool = False,
        random_state: Optional[int] = None,
        bc_weight: float = 1.0,
        boundary_threshold: float = 0.1,
        use_boundary_weighting: bool = False,
    ):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.bc_weight = bc_weight
        self.boundary_threshold = boundary_threshold
        self.use_boundary_weighting = use_boundary_weighting
        self.pca_in: Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.n_channels: Optional[int] = None
        self.out_channels: Optional[int] = None
        self.shape_: Optional[tuple[int, int]] = None
        self.output_shape_: Optional[tuple[int, int, int]] = None

    def fit(self, samples: np.ndarray, solution: np.ndarray) -> "PCAReducerMultiChannel":
        if samples.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_in, N, N), got {samples.shape}")
        num_samples, c_in, n, m = samples.shape
        self.n_channels = c_in
        self.shape_ = (n, m)
        x_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)

        if solution.ndim == 3:
            solution = solution[:, None, :, :]
        if solution.ndim != 4:
            raise ValueError(
                f"Expected (num_samples, C_out, N, N) or (num_samples, N, N), got {solution.shape}"
            )
        _, c_out, n_out, m_out = solution.shape
        if (n_out, m_out) != (n, m):
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        self.output_shape_ = (c_out, n, m)
        y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

        if self.use_boundary_weighting:
            boundary_weights = self._compute_boundary_weights(n, m)
            y_weighted = y_out * boundary_weights.reshape(1, -1)
        else:
            y_weighted = y_out

        self.pca_in = PCA(
            n_components=None,
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(x_in)
        self.pca_out = PCA(
            n_components=None,
            svd_solver=self.svd_solver,
            whiten=self.whiten,
            random_state=self.random_state,
        ).fit(y_weighted)

        return self

    def _compute_boundary_weights(self, n: int, m: int) -> np.ndarray:
        weights = np.ones((n, m), dtype=np.float32)
        x_coords = np.linspace(0, 1, n)
        y_coords = np.linspace(0, 1, m)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="ij")
        dist_to_boundary = np.minimum(
            np.minimum(grid_x, 1 - grid_x),
            np.minimum(grid_y, 1 - grid_y),
        )
        boundary_mask = dist_to_boundary < self.boundary_threshold
        weights[boundary_mask] = self.bc_weight
        return weights.flatten()

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99) -> tuple[int, int]:
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before requesting cutoffs")
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out

    @property
    def input_components(self) -> np.ndarray:
        if self.pca_in is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_in.components_

    @property
    def output_components(self) -> np.ndarray:
        if self.pca_out is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_out.components_
