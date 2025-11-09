import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from sklearn.decomposition import PCA


class ScalarGate(nn.Module):
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init), dtype=torch.float32))

    def forward(self, x):
        return x * self.alpha


class PCAReducerMultiChannel:
    def __init__(self, svd_solver='auto', whiten=False, random_state=None,
                 bc_weight=1.0, boundary_threshold=0.1, use_boundary_weighting=False):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.bc_weight = bc_weight
        self.boundary_threshold = boundary_threshold
        self.use_boundary_weighting = use_boundary_weighting
        self.pca_in: Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.n_channels = None
        self.out_channels = None
        self.shape_ = None

    def fit(self, samples: np.ndarray, solution: np.ndarray):
        if samples.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_in, N, N), got {samples.shape}")
        num_samples, C_in, N, M = samples.shape
        self.n_channels = C_in
        self.shape_ = (N, M)
        X_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)

        if solution.ndim == 3:
            solution = solution[:, None, :, :]
        if solution.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_out, N, N), got {solution.shape}")
        _, C_out, N_out, M_out = solution.shape
        if (N_out, M_out) != (N, M):
            raise ValueError("Input/output spatial dimensions must match")
        self.out_channels = C_out
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
        weights = np.ones((N, M), dtype=np.float32)
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, M)
        X, Y = np.meshgrid(x, y, indexing='ij')
        dist_to_boundary = np.minimum(
            np.minimum(X, 1 - X),
            np.minimum(Y, 1 - Y)
        )
        boundary_mask = dist_to_boundary < self.boundary_threshold
        weights[boundary_mask] = self.bc_weight
        return weights.flatten()

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99) -> tuple:
        assert self.pca_in and self.pca_out
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out


class ReducedBasisResNetMultiChannel(nn.Module):
    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, k_out: int,
                 hidden: int = 256, dropout: float = 0.0,
                 use_input_basis_for_encoder: bool = False,
                 use_input_basis_for_decoder: bool = False,
                 latent_batch_norm: bool = False):
        super().__init__()

        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None or rb.n_channels is None or rb.out_channels is None:
            raise ValueError("PCAReducer not properly initialized")

        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = rb.out_channels
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

        self.encoder_dim = k_in if use_input_basis_for_encoder else k_out
        if use_input_basis_for_encoder:
            enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            enc_mean = rb.pca_in.mean_.astype(np.float32)
        else:
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

        self.in_proj = nn.Linear(self.encoder_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden) if latent_batch_norm else nn.Identity()
        self.out_proj = nn.Linear(hidden, self.decoder_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
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


class ReducedBasisBlockOutputBasis(nn.Module):
    def __init__(self, rb: PCAReducerMultiChannel, k_out: int,
                 hidden: int = 256, dropout: float = 0.0,
                 latent_batch_norm: bool = False):
        super().__init__()

        if rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted")

        self.N, self.M = rb.shape_
        self.C_out = rb.out_channels
        self.D = self.C_out * self.N * self.M
        self.latent_batch_norm = latent_batch_norm

        max_k_out = rb.pca_out.n_components_
        if not (1 <= k_out <= max_k_out):
            raise ValueError(f"k_out={k_out} out of range [1, {max_k_out}]")

        dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
        dec_mean = rb.pca_out.mean_.astype(np.float32)
        self.register_buffer("dec_components", torch.from_numpy(dec_components))
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))

        self.in_proj = nn.Linear(k_out, hidden)
        self.bn = nn.BatchNorm1d(hidden) if latent_batch_norm else nn.Identity()
        self.out_proj = nn.Linear(hidden, k_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
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
        return (x_flat - self.dec_mean) @ self.dec_components.T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.dec_components + self.dec_mean

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C_out, N, M), got {x.shape}")
        B, C, H, W = x.shape
        if C != self.C_out or H != self.N or W != self.M:
            raise ValueError(f"Expected (B, {self.C_out}, {self.N}, {self.M}), got {x.shape}")

        x_flat = x.reshape(B, self.D)
        z = self.encode(x_flat)
        z_out = self.latent_net(z)
        y_flat = self.decode(z_out)
        return y_flat.reshape(B, self.C_out, self.N, self.M)


class MRBNOSimpleV2(nn.Module):
    """
    MRBNO Simple variant with optional FNO-style linear skip connections prior to activation.
    """

    def __init__(self, rb: PCAReducerMultiChannel, k_in: int, k_out: int,
                 layers: int = 7, hidden: int = 1024,
                 dropout: float = 0.0, init_scale: float = 0.1,
                 use_skip_connections: bool = True,
                 latent_batch_norm: bool = False,
                 use_linear_skips: bool = True):
        super().__init__()

        if rb.shape_ is None or rb.n_channels is None or rb.out_channels is None:
            raise ValueError("PCAReducer must be fitted")

        self.N, self.M = rb.shape_
        self.C_in = rb.n_channels
        self.C_out = rb.out_channels
        self.layers = layers
        self.use_skip_connections = use_skip_connections
        self.latent_batch_norm = latent_batch_norm
        self.use_linear_skips = use_linear_skips

        self.block0 = ReducedBasisResNetMultiChannel(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True,
            latent_batch_norm=latent_batch_norm,
        )
        self.block0_skip = nn.Conv2d(self.C_in, self.C_in, 1) if use_linear_skips else None
        self.gate0 = ScalarGate(init=init_scale)

        self.block1 = ReducedBasisResNetMultiChannel(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, dropout=dropout,
            use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False,
            latent_batch_norm=latent_batch_norm,
        )
        self.block1_skip = nn.Conv2d(self.C_in, self.C_out, 1) if use_linear_skips else None
        self.gate1 = ScalarGate(init=init_scale)

        self.global_skip_proj = nn.Conv2d(self.C_in, self.C_out, 1)
        self.global_skip_gate = ScalarGate(init=init_scale)

        if layers > 1:
            self.blocks = nn.ModuleList([
                ReducedBasisBlockOutputBasis(
                    rb, k_out=k_out, hidden=hidden, dropout=dropout,
                    latent_batch_norm=latent_batch_norm,
                ) for _ in range(layers - 1)
            ])
            if use_linear_skips:
                self.linear_skips = nn.ModuleList([
                    nn.Conv2d(self.C_out, self.C_out, 1) for _ in range(layers - 1)
                ])
            else:
                self.linear_skips = nn.ModuleList()
            self.gates = nn.ModuleList([ScalarGate(init=init_scale) for _ in range(layers - 1)])
        else:
            self.blocks = nn.ModuleList()
            self.linear_skips = nn.ModuleList()
            self.gates = nn.ModuleList()

        self.act = nn.GELU()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, {self.C_in}, N, M), got {x.shape}")
        if x.shape[1] != self.C_in:
            raise ValueError(f"Expected {self.C_in} input channels, got {x.shape[1]}")

        global_skip = self.global_skip_gate(self.global_skip_proj(x))

        block0_out = self.block0(x)
        if self.block0_skip is not None and self.use_linear_skips:
            block0_skip = self.block0_skip(x)
            y0_pre = block0_out + block0_skip
        else:
            y0_pre = block0_out
        y0 = self.gate0(y0_pre)
        y0 = self.act(y0)
        if self.use_skip_connections:
            y0 = y0 + x

        block1_out = self.block1(y0)
        if self.block1_skip is not None and self.use_linear_skips:
            block1_skip = self.block1_skip(y0)
            y1_pre = block1_out + block1_skip
        else:
            y1_pre = block1_out
        y = self.gate1(y1_pre)
        y = self.act(y)
        y0_proj = self.global_skip_proj(y0)
        if self.use_skip_connections:
            y = y + y0_proj
        y = y + global_skip

        for idx, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            block_out = block(y)
            if self.use_linear_skips and len(self.linear_skips) > 0:
                skip_out = self.linear_skips[idx](y)
                pre_act = block_out + skip_out
            else:
                pre_act = block_out
            pre_act = gate(pre_act)
            pre_act = self.act(pre_act)
            if self.use_skip_connections:
                y = y + pre_act
            else:
                y = pre_act

        return y

