import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ============= PCAReducer Class =============
@dataclass
class PCAConfig:
    svd_solver: str = 'auto'
    whiten: bool = False
    random_state: Optional[int] = None

class PCAReducer:
    """
    PCA-based reduced-basis helper for (num_samples, N, N) fields.
    - Fits separate PCA models for inputs and outputs.
    - Projects to first k components or components covering a variance cutoff j.
    - Reconstructs outputs from reduced coefficients.
    """
    def __init__(self, cfg: PCAConfig = PCAConfig()):
        self.cfg = cfg
        self.pca_in:  Optional[PCA] = None
        self.pca_out: Optional[PCA] = None
        self.shape_ = None  # (N, N)

    def fit(self, samples: np.ndarray, solution: np.ndarray):
        """
        samples, solution: (num_samples, N, N).
        """
        assert samples.shape == solution.shape and samples.ndim == 3, "Use (num_samples, N, N)"
        num_samples, N, M = samples.shape
        self.shape_ = (N, M)

        X_in  = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
        Y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

        self.pca_in  = PCA(n_components=None, svd_solver=self.cfg.svd_solver,
                           whiten=self.cfg.whiten, random_state=self.cfg.random_state).fit(X_in)
        self.pca_out = PCA(n_components=None, svd_solver=self.cfg.svd_solver,
                           whiten=self.cfg.whiten, random_state=self.cfg.random_state).fit(Y_out)
        return self

    @staticmethod
    def _k_from_cutoff(evr: np.ndarray, cutoff: float) -> int:
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)

    def k_from_cutoff(self, j_in: float = 0.99, j_out: float = 0.99) -> tuple[int, int]:
        """Return (k_in, k_out) meeting given variance cutoffs."""
        assert self.pca_in and self.pca_out
        k_in  = self._k_from_cutoff(self.pca_in.explained_variance_ratio_,  j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out

    def transform_inputs(self, samples: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Return input PCA coefficients (num_samples, k)."""
        assert self.pca_in
        X = samples.reshape(samples.shape[0], -1).astype(np.float32, copy=False)
        Z = self.pca_in.transform(X)
        return Z[:, :k] if k is not None else Z

    def transform_outputs(self, solution: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Return output PCA coefficients (num_samples, k)."""
        assert self.pca_out
        Y = solution.reshape(solution.shape[0], -1).astype(np.float32, copy=False)
        Z = self.pca_out.transform(Y)
        return Z[:, :k] if k is not None else Z

    def inverse_outputs(self, coeffs: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct outputs (num_samples, N, N) from output PCA coeffs.
        If k is provided, only the first k components are used.
        """
        assert self.pca_out and self.shape_ is not None
        if k is None:
            Y_flat = self.pca_out.inverse_transform(coeffs)
        else:
            comps = self.pca_out.components_[:k, :]
            mean  = self.pca_out.mean_
            Y_flat = coeffs @ comps + mean
        return Y_flat.reshape((coeffs.shape[0], *self.shape_))

    def plot_explained_variance(self):
        """Plot explained variance ratios for input and output PCA."""
        assert self.pca_in and self.pca_out
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(self.pca_in.explained_variance_ratio_, marker='o', linewidth=1)
        axes[0].set_title("Input PCA — explained variance ratio")
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("EVR")
        axes[0].grid(True)
        axes[1].plot(self.pca_out.explained_variance_ratio_, marker='o', linewidth=1)
        axes[1].set_title("Output PCA — explained variance ratio")
        axes[1].set_xlabel("Component")
        axes[1].set_ylabel("EVR")
        axes[1].grid(True)
        plt.show()

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


# ============= Neural Network Architecture =============

class ScalarGate(nn.Module):
    """Learnable scalar multiplier for residual connection."""
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init), dtype=torch.float32))
    
    def forward(self, x):
        return x * self.alpha


class ResidualBlock(nn.Module):
    """Standard residual block with pre-norm."""
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x):
        y = self.lin2(self.act(self.lin1(self.norm(x))))
        y = self.dropout(y)
        return x + y


class ReducedBasisResNet(nn.Module):
    """
    Core building block that:
    1. Encodes input to PCA coefficients (using input or output basis)
    2. Applies ResNet transformation in coefficient space
    3. Decodes back to function space using output basis
    """
    def __init__(self, rb: PCAReducer, k_in: int, k_out: int, 
                 hidden: int = 256, depth: int = 4, dropout: float = 0.0,
                 use_input_basis_for_encoder: bool = False):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None:
            raise ValueError("PCAReducer.shape_ is None")
        
        self.N, self.M = rb.shape_
        self.D = self.N * self.M
        
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
            # Use input PCA for encoding
            self.encoder_dim = k_in
            enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            enc_mean = rb.pca_in.mean_.astype(np.float32)
        else:
            # Use output PCA for encoding
            self.encoder_dim = k_out
            enc_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
            enc_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Always use output PCA for decoding
        dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
        dec_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Register as buffers
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))  # (D, encoder_dim)
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))  # (D,)
        self.register_buffer("dec_components", torch.from_numpy(dec_components))  # (k_out, D)
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))  # (D,)
        
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
        return z @ self.dec_components + self.dec_mean  # (B, D)
    
    def forward(self, x):
        # Handle input shapes
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0, :, :]
        if x.ndim != 3 or x.shape[1:] != (self.N, self.M):
            raise ValueError(f"Expected (B, {self.N}, {self.M}), got {x.shape}")
        
        B = x.shape[0]
        x_flat = x.reshape(B, self.D)
        
        # Encode to coefficients
        z = self.encode(x_flat)  # (B, encoder_dim)
        
        # Transform in coefficient space
        h = F.gelu(self.in_proj(z))  # (B, hidden)
        for block in self.blocks:
            h = block(h)  # (B, hidden)
        z_out = self.out_proj(h)  # (B, k_out)
        
        # Decode back to function space
        y_flat = self.decode(z_out)  # (B, D)
        y = y_flat.reshape(B, 1, self.N, self.M)
        
        return y


class MultiLayerReducedBasisResNet(nn.Module):
    """
    Stacks multiple ReducedBasisResNet blocks with residual connections.
    
    Architecture:
    - Block 0: Uses input PCA basis for encoding, output basis for decoding
    - Blocks 1+: Use output PCA basis for both encoding and decoding
    - Each block has a residual connection with learnable scalar gate
    - Nonlinearity applied between blocks
    """
    def __init__(self, rb: PCAReducer, k_in: int, k_out: int,
                 layers: int = 4, hidden: int = 256, depth: int = 4,
                 dropout: float = 0.0, share_weights: bool = False,
                 init_scale: float = 0.1):
        super().__init__()
        
        if rb.shape_ is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        self.layers = layers
        
        # First block: input basis -> output basis (use_input_basis_for_encoder=True)
        self.block0 = ReducedBasisResNet(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, depth=depth, dropout=dropout,
            use_input_basis_for_encoder=True
        )
        self.gate0 = ScalarGate(init=init_scale)
        
        # Subsequent blocks: output basis -> output basis
        if layers > 0:
            if share_weights:
                # Share weights across all layers
                core = ReducedBasisResNet(
                    rb, k_in=k_out, k_out=k_out,
                    hidden=hidden, depth=depth, dropout=dropout,
                    use_input_basis_for_encoder=False
                )
                self.blocks = nn.ModuleList([core for _ in range(layers)])
            else:
                # Independent weights for each layer
                self.blocks = nn.ModuleList([
                    ReducedBasisResNet(
                        rb, k_in=k_out, k_out=k_out,
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
        # Ensure correct input shape
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected (B, 1, N, M), got {x.shape}")
        
        # First block with input basis encoding
        y = x + self.gate0(self.block0(x))
        
        # Subsequent blocks with output basis encoding
        for block, gate in zip(self.blocks, self.gates):
            y = self.act(y)  # Nonlinearity between blocks
            y = y + gate(block(y))  # Residual connection
        
        return y
    






class ReducedBasisResNetV2(nn.Module):
    """
    Improved version with better initialization and stability.
    """
    def __init__(self, rb, k_in: int, k_out: int, 
                 hidden: int = 256, depth: int = 4, dropout: float = 0.0,
                 use_input_basis_for_encoder: bool = False):
        super().__init__()
        
        # Validate PCAReducer
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted first")
        if rb.shape_ is None:
            raise ValueError("PCAReducer.shape_ is None")
        
        self.N, self.M = rb.shape_
        self.D = self.N * self.M
        
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
            # Use input PCA for encoding
            self.encoder_dim = k_in
            enc_components = rb.pca_in.components_[:k_in, :].astype(np.float32)
            enc_mean = rb.pca_in.mean_.astype(np.float32)
        else:
            # Use output PCA for encoding
            self.encoder_dim = k_out
            enc_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
            enc_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Always use output PCA for decoding
        dec_components = rb.pca_out.components_[:k_out, :].astype(np.float32)
        dec_mean = rb.pca_out.mean_.astype(np.float32)
        
        # Register as buffers
        self.register_buffer("enc_components_t", torch.from_numpy(enc_components.T))
        self.register_buffer("enc_mean", torch.from_numpy(enc_mean))
        self.register_buffer("dec_components", torch.from_numpy(dec_components))
        self.register_buffer("dec_mean", torch.from_numpy(dec_mean))
        
        # Improved network architecture with better initialization
        self.network = nn.ModuleList()
        
        # Input projection with careful initialization
        self.in_proj = nn.Linear(self.encoder_dim, hidden)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.1)  # Very small gain
        nn.init.zeros_(self.in_proj.bias)
        
        # Residual blocks
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        
        # Output projection - initialize to near-identity transformation
        self.out_proj = nn.Linear(hidden, self.k_out)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)  # Start with zeros
        nn.init.xavier_uniform_(self.out_proj.bias, gain=0.1)
        
        # Skip connection from input to output if dimensions match
        self.has_skip = (self.encoder_dim == self.k_out)
        if self.has_skip:
            self.skip_scale = nn.Parameter(torch.tensor(1.0))  # Start with identity
    
    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Project to PCA coefficient space."""
        x_centered = x_flat - self.enc_mean
        return x_centered @ self.enc_components_t
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from PCA coefficients."""
        return z @ self.dec_components + self.dec_mean
    
    def forward(self, x):
        # Handle input shapes
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0, :, :]
        if x.ndim != 3 or x.shape[1:] != (self.N, self.M):
            raise ValueError(f"Expected (B, {self.N}, {self.M}), got {x.shape}")
        
        B = x.shape[0]
        x_flat = x.reshape(B, self.D)
        
        # Encode to coefficients
        z = self.encode(x_flat)  # (B, encoder_dim)
        
        # Transform with careful scaling
        h = 0.1 * F.gelu(self.in_proj(z))  # Scale down activation
        for block in self.blocks:
            h = block(h)
        h_out = self.out_proj(h)  # (B, k_out)
        
        # Add skip connection if dimensions match
        if self.has_skip:
            z_out = self.skip_scale * z + h_out
        else:
            # If dimensions don't match, we need to be more careful
            if self.encoder_dim < self.k_out:
                # Pad zeros
                z_padded = F.pad(z, (0, self.k_out - self.encoder_dim))
                z_out = z_padded + h_out
            else:
                # Use only the network output
                z_out = z[:, :self.k_out] + h_out
        
        # Decode back to function space
        y_flat = self.decode(z_out)
        y = y_flat.reshape(B, 1, self.N, self.M)
        
        return y


class SimplifiedMultiLayerRBNet(nn.Module):
    """
    Simplified architecture that starts closer to identity mapping.
    """
    def __init__(self, rb, k_in: int, k_out: int,
                 layers: int = 3, hidden: int = 128, depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        
        if rb.shape_ is None:
            raise ValueError("PCAReducer must be fitted")
        
        self.N, self.M = rb.shape_
        
        # Single end-to-end network instead of multiple blocks
        # This learns the full transformation at once
        self.encoder_decoder = ReducedBasisResNetV2(
            rb, k_in=k_in, k_out=k_out,
            hidden=hidden, depth=depth + layers,  # Combine depth
            dropout=dropout,
            use_input_basis_for_encoder=True
        )
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.encoder_decoder(x)


class IdentityInitRBNet(nn.Module):
    """
    Alternative approach: Start from identity mapping and learn corrections.
    This should have much lower initial loss.
    """
    def __init__(self, rb, k_in: int, k_out: int, hidden: int = 256):
        super().__init__()
        
        self.N, self.M = rb.shape_
        self.D = self.N * self.M
        
        # Store PCA components
        self.register_buffer("in_components_t", 
                           torch.from_numpy(rb.pca_in.components_[:k_in].T.astype(np.float32)))
        self.register_buffer("in_mean", 
                           torch.from_numpy(rb.pca_in.mean_.astype(np.float32)))
        self.register_buffer("out_components", 
                           torch.from_numpy(rb.pca_out.components_[:k_out].astype(np.float32)))
        self.register_buffer("out_mean", 
                           torch.from_numpy(rb.pca_out.mean_.astype(np.float32)))
        
        # Learn only a correction to the identity
        self.correction_net = nn.Sequential(
            nn.Linear(k_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, k_out)
        )
        
        # Initialize correction network to output zeros
        for m in self.correction_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        
        # Initialize last layer to exactly zero
        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)
        
        # Learnable scale for how much correction to apply
        self.correction_scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0, :, :]
        
        B = x.shape[0]
        x_flat = x.reshape(B, self.D)
        
        # Encode with input PCA
        x_centered = x_flat - self.in_mean
        z_in = x_centered @ self.in_components_t
        
        # Start with identity (just pass through input)
        # For identity, we decode input directly as if it were output
        identity_output = x_flat  # Pass through original
        
        # Learn a correction in coefficient space
        correction = self.correction_scale * self.correction_net(z_in)
        
        # Decode correction and add to identity
        correction_decoded = correction @ self.out_components + self.out_mean
        
        # Combine identity with learned correction
        y_flat = 0.9 * identity_output + 0.1 * correction_decoded
        
        return y_flat.reshape(B, 1, self.N, self.M)


def diagnose_model(model, rb, X_sample, Y_sample):
    """
    Diagnose what's happening in the model.
    """
    if isinstance(X_sample, np.ndarray):
        X_sample = torch.from_numpy(X_sample).float()
    if isinstance(Y_sample, np.ndarray):
        Y_sample = torch.from_numpy(Y_sample).float()
    
    if X_sample.ndim == 3:
        X_sample = X_sample.unsqueeze(1)
    if Y_sample.ndim == 3:
        Y_sample = Y_sample.unsqueeze(1)
    
    device = next(model.parameters()).device
    X_sample = X_sample[:4].to(device)  # Just use 4 samples
    Y_sample = Y_sample[:4].to(device)
    
    model.eval()
    with torch.no_grad():
        # Get prediction
        pred = model(X_sample)
        
        # Compute statistics
        print("\nModel Diagnosis:")
        print(f"  Input mean: {X_sample.mean():.4f}, std: {X_sample.std():.4f}")
        print(f"  Target mean: {Y_sample.mean():.4f}, std: {Y_sample.std():.4f}")
        print(f"  Prediction mean: {pred.mean():.4f}, std: {pred.std():.4f}")
        print(f"  MSE Loss: {F.mse_loss(pred, Y_sample):.4f}")
        print(f"  Relative Error: {(pred - Y_sample).abs().mean() / Y_sample.abs().mean():.4f}")
        
        # Check if model is outputting reasonable values
        if pred.abs().max() > 100:
            print("  WARNING: Model outputs are very large!")
        if pred.std() < 0.01:
            print("  WARNING: Model outputs have very low variance!")
        if torch.isnan(pred).any():
            print("  WARNING: Model outputs contain NaN!")



    
    


# ============= Training Utilities =============

def simple_train(model, X_train, Y_train, X_val, Y_val,
                 epochs=50, batch_size=16, lr=1e-3, weight_decay=1e-4,
                 scheduler_step=50, scheduler_gamma=0.8):
    """
    Training loop for the model.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Convert numpy to torch if needed
    if isinstance(X_train, np.ndarray): 
        X_train = torch.from_numpy(X_train).float()
    if isinstance(Y_train, np.ndarray): 
        Y_train = torch.from_numpy(Y_train).float()
    if isinstance(X_val, np.ndarray): 
        X_val = torch.from_numpy(X_val).float()
    if isinstance(Y_val, np.ndarray): 
        Y_val = torch.from_numpy(Y_val).float()

    # Ensure channel dimension
    if X_train.ndim == 3: 
        X_train = X_train.unsqueeze(1)
    if Y_train.ndim == 3: 
        Y_train = Y_train.unsqueeze(1)
    if X_val.ndim == 3: 
        X_val = X_val.unsqueeze(1)
    if Y_val.ndim == 3: 
        Y_val = Y_val.unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_fn = nn.MSELoss()
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch})")
    
    print(f"\nBest validation loss: {best_val_loss:.4e} at epoch {best_epoch}")
    return model

def silent_train(model, X_train, Y_train, X_val, Y_val,
                 epochs=50, batch_size=16, lr=1e-3, weight_decay=1e-4,
                 scheduler_step=50, scheduler_gamma=0.8):
    """Training loop without print statements."""
    
    # Convert numpy to torch if needed
    if isinstance(X_train, np.ndarray): 
        X_train = torch.from_numpy(X_train).float()
    if isinstance(Y_train, np.ndarray): 
        Y_train = torch.from_numpy(Y_train).float()
    if isinstance(X_val, np.ndarray): 
        X_val = torch.from_numpy(X_val).float()
    if isinstance(Y_val, np.ndarray): 
        Y_val = torch.from_numpy(Y_val).float()

    # Ensure channel dimension
    if X_train.ndim == 3: 
        X_train = X_train.unsqueeze(1)
    if Y_train.ndim == 3: 
        Y_train = Y_train.unsqueeze(1)
    if X_val.ndim == 3: 
        X_val = X_val.unsqueeze(1)
    if Y_val.ndim == 3: 
        Y_val = Y_val.unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=scheduler_step, gamma=scheduler_gamma)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        scheduler.step()
    
    return best_val_loss

# ============= Diagnostic Functions =============

def test_pca_reconstruction(rb: PCAReducer, X, Y, k_in=50, k_out=50):
    """Test PCA reconstruction quality."""
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    
    # Test input PCA
    X_flat = X.reshape(len(X), -1)
    X_centered = X_flat - rb.pca_in.mean_
    X_coef = X_centered @ rb.pca_in.components_[:k_in].T
    X_recon = X_coef @ rb.pca_in.components_[:k_in] + rb.pca_in.mean_
    X_recon = X_recon.reshape(X.shape)
    
    # Test output PCA
    Y_flat = Y.reshape(len(Y), -1)
    Y_centered = Y_flat - rb.pca_out.mean_
    Y_coef = Y_centered @ rb.pca_out.components_[:k_out].T
    Y_recon = Y_coef @ rb.pca_out.components_[:k_out] + rb.pca_out.mean_
    Y_recon = Y_recon.reshape(Y.shape)
    
    print("PCA Reconstruction Test:")
    print(f"  Input PCA with k={k_in}:")
    print(f"    X reconstruction error: {np.mean((X - X_recon)**2):.4e}")
    print(f"    Explained variance: {rb.pca_in.explained_variance_ratio_[:k_in].sum():.2%}")
    print(f"  Output PCA with k={k_out}:")
    print(f"    Y reconstruction error: {np.mean((Y - Y_recon)**2):.4e}")
    print(f"    Explained variance: {rb.pca_out.explained_variance_ratio_[:k_out].sum():.2%}")
    print(f"  Baseline X->Y error: {np.mean((X - Y)**2):.4e}")


