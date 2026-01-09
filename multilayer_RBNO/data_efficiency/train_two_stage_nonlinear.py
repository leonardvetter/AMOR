#!/usr/bin/env python3
"""
Two-stage training on Nonlinear Diffusion dataset (nonlinear_diffusion_dataset_big.npz).

Stage 1: Train in PCA space with loss on PCA coefficients.
Stage 2: Fine-tune on real-space data with frozen blocks (or unfrozen with small LR).

Handles arbitrary input/output channels.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from sklearn.decomposition import PCA

from mrbno_two_stage import MRBNOStage1, MRBNOStage2


class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors, computing PCA per channel."""
    
    def __init__(self, svd_solver="auto", whiten=False, random_state=None):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.pca_in_channels = None
        self.pca_out_channels = None
        self.n_channels = None
        self.out_channels = None
        self.shape_ = None
        
    def fit(self, samples, solution):
        if samples.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_in, N, N), got {samples.shape}")
        num_samples, c_in, n, m = samples.shape
        self.n_channels = c_in
        self.shape_ = (n, m)
        
        if solution.ndim == 3:
            solution = solution[:, None, :, :]
        if solution.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_out, N, N) or (num_samples, N, N), got {solution.shape}")
        _, c_out, n_out, m_out = solution.shape
        if (n_out, m_out) != (n, m):
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        
        self.pca_in_channels = []
        for c in range(c_in):
            x_channel = samples[:, c, :, :].reshape(num_samples, -1).astype(np.float32, copy=False)
            pca_ch = PCA(n_components=None, svd_solver=self.svd_solver,
                        whiten=self.whiten, random_state=self.random_state).fit(x_channel)
            self.pca_in_channels.append(pca_ch)
        
        self.pca_out_channels = []
        for c in range(c_out):
            y_channel = solution[:, c, :, :].reshape(num_samples, -1).astype(np.float32, copy=False)
            pca_ch = PCA(n_components=None, svd_solver=self.svd_solver,
                        whiten=self.whiten, random_state=self.random_state).fit(y_channel)
            self.pca_out_channels.append(pca_ch)
        
        return self
    
    @staticmethod
    def _k_from_cutoff(evr, cutoff):
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)
    
    def k_from_cutoff(self, j_in=0.99, j_out=0.99):
        if self.pca_in_channels is None or self.pca_out_channels is None:
            raise ValueError("PCAReducer must be fitted before requesting cutoffs")
        evr_in = np.mean([pca.explained_variance_ratio_ for pca in self.pca_in_channels], axis=0)
        evr_out = np.mean([pca.explained_variance_ratio_ for pca in self.pca_out_channels], axis=0)
        k_in = self._k_from_cutoff(evr_in, j_in)
        k_out = self._k_from_cutoff(evr_out, j_out)
        return k_in, k_out
    
    @property
    def input_components(self):
        if self.pca_in_channels is None:
            raise ValueError("PCAReducer has not been fitted")
        return np.stack([pca.components_ for pca in self.pca_in_channels], axis=0)
    
    @property
    def output_components(self):
        if self.pca_out_channels is None:
            raise ValueError("PCAReducer has not been fitted")
        return np.stack([pca.components_ for pca in self.pca_out_channels], axis=0)
    
    @property
    def input_mean(self):
        if self.pca_in_channels is None:
            raise ValueError("PCAReducer has not been fitted")
        return np.stack([pca.mean_ for pca in self.pca_in_channels], axis=0)
    
    @property
    def output_mean(self):
        if self.pca_out_channels is None:
            raise ValueError("PCAReducer has not been fitted")
        return np.stack([pca.mean_ for pca in self.pca_out_channels], axis=0)


def load_nonlinear_dataset(data_path):
    """Load and preprocess nonlinear diffusion dataset."""
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path, mmap_mode='r')
    
    f = np.array(data['f'], dtype=np.float32)
    m = np.array(data['m'], dtype=np.float32)
    u = np.array(data['u'], dtype=np.float32)
    
    print(f"Dataset shapes: f={f.shape}, m={m.shape}, u={u.shape}")
    
    num_samples = len(f)
    H, W = f.shape[1], f.shape[2]
    
    X = np.stack([f, m], axis=1)
    Y = u[:, np.newaxis, :, :]
    
    print(f"Final dataset shapes: X={X.shape}, Y={Y.shape}")
    print(f"Input channels: {X.shape[1]}, Output channels: {Y.shape[1]}")
    print(f"Spatial dimensions: H={H}, W={W}")
    
    train_frac = 0.7
    val_frac = 0.2
    n_train = int(train_frac * num_samples)
    n_val = int(val_frac * num_samples)
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    
    print(f"\nTrain/Val/Test split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    x_mean = X_train.mean()
    x_std = X_train.std()
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    X_train = (X_train - x_mean) / (x_std + 1e-8)
    Y_train = (Y_train - y_mean) / (y_std + 1e-8)
    X_val = (X_val - x_mean) / (x_std + 1e-8)
    Y_val = (Y_val - y_mean) / (y_std + 1e-8)
    X_test = (X_test - x_mean) / (x_std + 1e-8)
    Y_test = (Y_test - y_mean) / (y_std + 1e-8)
    
    print(f"\nNormalization statistics:")
    print(f"  X: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"  Y: mean={y_mean:.4f}, std={y_std:.4f}")
    
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test), (x_mean, x_std, y_mean, y_std)


def compute_pca_coefficients(X, Y, rb, n_modes_in, n_modes_out):
    """Convert spatial data to PCA coefficients per channel."""
    num_samples, n_channels, H, W = X.shape
    _, out_channels, _, _ = Y.shape
    
    X_pca_list = []
    Y_pca_list = []
    
    batch_size = 1000
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        
        X_pca_channels = []
        for c in range(n_channels):
            X_channel = X_batch[:, c, :, :].reshape(len(X_batch), -1)
            X_channel_centered = X_channel - rb.pca_in_channels[c].mean_
            X_pca_ch = X_channel_centered @ rb.pca_in_channels[c].components_[:n_modes_in].T
            X_pca_channels.append(X_pca_ch)
        X_pca_batch = np.stack(X_pca_channels, axis=1)
        
        Y_pca_channels = []
        for c in range(out_channels):
            Y_channel = Y_batch[:, c, :, :].reshape(len(Y_batch), -1)
            Y_channel_centered = Y_channel - rb.pca_out_channels[c].mean_
            Y_pca_ch = Y_channel_centered @ rb.pca_out_channels[c].components_[:n_modes_out].T
            Y_pca_channels.append(Y_pca_ch)
        Y_pca_batch = np.stack(Y_pca_channels, axis=1)
        
        X_pca_list.append(X_pca_batch)
        Y_pca_list.append(Y_pca_batch)
    
    X_pca = np.concatenate(X_pca_list, axis=0)
    Y_pca = np.concatenate(Y_pca_list, axis=0)
    
    return X_pca, Y_pca


def train_stage1(model, train_loader, val_loader, epochs, lr, device):
    """Train Stage 1 model in PCA space."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 1 Epoch {epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return best_val_loss


def train_stage2(model, train_loader, val_loader, epochs, lr, device):
    """Train Stage 2 model in real space with frozen blocks."""
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 2 Epoch {epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return best_val_loss


def train_stage2_unfrozen(model, train_loader, val_loader, epochs, lr, lr_blocks, device):
    """Train Stage 2 model in real space with all parameters trainable.
    
    Blocks (PCA space parameters) use smaller learning rate lr_blocks.
    Real-space layers (fc0, fc2) use learning rate lr.
    """
    model.to(device)
    
    block_params = []
    real_space_params = []
    for name, param in model.named_parameters():
        if 'blocks' in name:
            block_params.append(param)
        else:
            real_space_params.append(param)
    
    param_groups = [
        {'params': block_params, 'lr': lr_blocks},
        {'params': real_space_params, 'lr': lr}
    ]
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_vals = scheduler.get_last_lr()
            lr_block = lr_vals[0]
            lr_real = lr_vals[1] if len(lr_vals) > 1 else lr_vals[0]
            print(f"[Stage 2 Unfrozen Epoch {epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} lr_real={lr_real:.2e} lr_blocks={lr_block:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-stage training on Nonlinear Diffusion')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/nonlinear_diffusion_dataset_big.npz',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--stage1_epochs', type=int, default=600, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=100, help='Stage 2 epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--width', type=int, default=32, help='Model width')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory (default: $SCRATCH/checkpoints)')
    args = parser.parse_args()
    
    if args.checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.environ.get('SCRATCH', '/tmp'), 'checkpoints')
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    args.data_path = os.path.expandvars(args.data_path)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_nonlinear_dataset(args.data_path)
    H, W = X_train.shape[2], X_train.shape[3]
    in_channels = X_train.shape[1]
    out_channels = Y_train.shape[1]
    
    print("\n" + "="*70)
    print("COMPUTING PCA")
    print("="*70)
    rb = PCAReducerMultiChannel().fit(X_train, Y_train)
    k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    print(f"PCA modes for 99% variance: k_in={k_in}, k_out={k_out}")
    
    print("\n" + "="*70)
    print("STAGE 1: Training in PCA space")
    print("="*70)
    
    X_train_pca, Y_train_pca = compute_pca_coefficients(X_train, Y_train, rb, k_in, k_out)
    X_val_pca, Y_val_pca = compute_pca_coefficients(X_val, Y_val, rb, k_in, k_out)
    
    train_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_train_pca).float(),
        torch.from_numpy(Y_train_pca).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_val_pca).float(),
        torch.from_numpy(Y_val_pca).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    stage1_model = MRBNOStage1(
        in_channels=in_channels, out_channels=out_channels, width=args.width,
        n_modes_in=k_in, n_modes_out=k_out, n_layers=args.n_layers, H=H, W=W,
        norm="group", dropout_p=0.0, residual=True,
        pca_components_in=rb.input_components, pca_mean_in=rb.input_mean,
        pca_components_out=rb.output_components, pca_mean_out=rb.output_mean
    )
    
    train_stage1(stage1_model, train_loader_pca, val_loader_pca, args.stage1_epochs, args.lr, device)
    
    stage1_path = os.path.join(checkpoint_dir, f'stage1_nonlinear_{args.stage1_epochs}epochs.pt')
    torch.save(stage1_model.state_dict(), stage1_path)
    print(f"\nStage 1 checkpoint saved to {stage1_path}")
    
    print("\n" + "="*70)
    print("STAGE 1 TRAINING COMPLETE")
    print("="*70)

