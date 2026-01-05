#!/usr/bin/env python3
"""
Two-stage training on Kolmogorov Flow dataset (2D_NS_Re500.npy).

Stage 1: Train in PCA space with loss on PCA coefficients.
Stage 2: Fine-tune on real-space data with frozen blocks.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import time
from sklearn.decomposition import PCA

from mrbno_two_stage import MRBNOStage1, MRBNOStage2


class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors."""
    
    def __init__(self, svd_solver="auto", whiten=False, random_state=None):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.pca_in = None
        self.pca_out = None
        self.n_channels = None
        self.out_channels = None
        self.shape_ = None
        
    def fit(self, samples, solution):
        if samples.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_in, N, N), got {samples.shape}")
        num_samples, c_in, n, m = samples.shape
        self.n_channels = c_in
        self.shape_ = (n, m)
        x_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
        
        if solution.ndim == 3:
            solution = solution[:, None, :, :]
        if solution.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_out, N, N) or (num_samples, N, N), got {solution.shape}")
        _, c_out, n_out, m_out = solution.shape
        if (n_out, m_out) != (n, m):
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        y_out = solution.reshape(num_samples, -1).astype(np.float32, copy=False)
        
        self.pca_in = PCA(n_components=None, svd_solver=self.svd_solver, 
                          whiten=self.whiten, random_state=self.random_state).fit(x_in)
        self.pca_out = PCA(n_components=None, svd_solver=self.svd_solver, 
                           whiten=self.whiten, random_state=self.random_state).fit(y_out)
        return self
    
    @staticmethod
    def _k_from_cutoff(evr, cutoff):
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)
    
    def k_from_cutoff(self, j_in=0.99, j_out=0.99):
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before requesting cutoffs")
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out
    
    @property
    def input_components(self):
        if self.pca_in is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_in.components_
    
    @property
    def output_components(self):
        if self.pca_out is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_out.components_
    
    @property
    def input_mean(self):
        return self.pca_in.mean_
    
    @property
    def input_mean(self):
        return self.pca_in.mean_
    
    @property
    def output_mean(self):
        return self.pca_out.mean_


def load_kolmogorov_dataset(data_path):
    """Load and preprocess Kolmogorov flow dataset."""
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path, mmap_mode='r')
    
    print(f"Dataset shape: {data.shape}")
    print(f"Interpretation: (num_trajectories={data.shape[0]}, num_timesteps={data.shape[1]}, height={data.shape[2]}, width={data.shape[3]})")
    
    num_trajectories = data.shape[0]
    num_timesteps = data.shape[1] - 1
    total_pairs = num_trajectories * num_timesteps
    
    print(f"\nCreating input-output pairs...")
    print(f"  Trajectories: {num_trajectories}")
    print(f"  Pairs per trajectory: {num_timesteps}")
    print(f"  Total pairs: {total_pairs:,}")
    
    chunk_size = 50
    X_chunks = []
    Y_chunks = []
    
    for chunk_start in range(0, num_trajectories, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_trajectories)
        X_chunk_list = []
        Y_chunk_list = []
        
        for traj_idx in range(chunk_start, chunk_end):
            trajectory = np.array(data[traj_idx], dtype=np.float32)
            for t in range(num_timesteps):
                X_chunk_list.append(trajectory[t])
                Y_chunk_list.append(trajectory[t+1])
        
        X_chunks.append(np.array(X_chunk_list, dtype=np.float32))
        Y_chunks.append(np.array(Y_chunk_list, dtype=np.float32))
        
        if (chunk_start // chunk_size + 1) % 4 == 0:
            print(f"  Processed {chunk_end}/{num_trajectories} trajectories...")
    
    X = np.concatenate(X_chunks, axis=0)
    Y = np.concatenate(Y_chunks, axis=0)
    
    print(f"\nFinal dataset shapes: X={X.shape}, Y={Y.shape}")
    
    X = X[:, np.newaxis, :, :]
    Y = Y[:, np.newaxis, :, :]
    
    print(f"After adding channel dimension: X={X.shape}, Y={Y.shape}")
    
    train_frac = 0.7
    val_frac = 0.2
    n_train = int(train_frac * len(X))
    n_val = int(val_frac * len(X))
    
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
    """Convert spatial data to PCA coefficients."""
    X_pca_list = []
    Y_pca_list = []
    
    batch_size = 1000
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        
        X_flat = X_batch.reshape(len(X_batch), -1)
        Y_flat = Y_batch.reshape(len(Y_batch), -1)
        
        X_centered = X_flat - rb.input_mean
        Y_centered = Y_flat - rb.output_mean
        
        X_pca_batch = X_centered @ rb.input_components[:n_modes_in].T
        Y_pca_batch = Y_centered @ rb.output_components[:n_modes_out].T
        
        X_pca_list.append(X_pca_batch)
        Y_pca_list.append(Y_pca_batch)
    
    X_pca = np.concatenate(X_pca_list, axis=0)
    Y_pca = np.concatenate(Y_pca_list, axis=0)
    
    X_pca = X_pca.reshape(len(X_pca), 1, n_modes_in)
    Y_pca = Y_pca.reshape(len(Y_pca), 1, n_modes_out)
    
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


def main():
    parser = argparse.ArgumentParser(description='Two-stage training on Kolmogorov Flow')
    parser.add_argument('--data_path', type=str, default='$SCRATCH/data/2D_NS_Re500.npy',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--stage1_epochs', type=int, default=300, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=100, help='Stage 2 epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--width', type=int, default=32, help='Model width')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory (default: $SCRATCH/checkpoints)')
    args = parser.parse_args()
    
    # Set checkpoint directory to SCRATCH if not specified
    if args.checkpoint_dir is None:
        scratch_dir = os.environ.get('SCRATCH', '/tmp')
        args.checkpoint_dir = os.path.join(scratch_dir, 'checkpoints')
    
    args.data_path = os.path.expandvars(args.data_path)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_kolmogorov_dataset(args.data_path)
    H, W = X_train.shape[2], X_train.shape[3]
    
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
    
    print(f"PCA coefficient shapes: X_train_pca={X_train_pca.shape}, Y_train_pca={Y_train_pca.shape}")
    
    train_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_train_pca).float(),
        torch.from_numpy(Y_train_pca).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader_pca = DataLoader(TensorDataset(
        torch.from_numpy(X_val_pca).float(),
        torch.from_numpy(Y_val_pca).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    stage1_model = MRBNOStage1(
        in_channels=1, out_channels=1, width=args.width,
        n_modes_in=k_in, n_modes_out=k_out, n_layers=args.n_layers, H=H, W=W,
        norm="group", dropout_p=0.0, residual=True,
        pca_components_in=rb.input_components, pca_mean_in=rb.input_mean,
        pca_components_out=rb.output_components, pca_mean_out=rb.output_mean
    )
    
    train_stage1(stage1_model, train_loader_pca, val_loader_pca, args.stage1_epochs, args.lr, device)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    stage1_path = os.path.join(args.checkpoint_dir, 'stage1_best.pt')
    torch.save(stage1_model.state_dict(), stage1_path)
    print(f"Stage 1 checkpoint saved to {stage1_path}")
    
    print("\n" + "="*70)
    print("STAGE 2: Fine-tuning in real space")
    print("="*70)
    
    stage2_model = MRBNOStage2(stage1_model, in_channels=1, out_channels=1, width=args.width, add_coords=False)
    
    n_stage2_samples = min(1000, len(X_train))
    X_train_stage2 = X_train[:n_stage2_samples]
    Y_train_stage2 = Y_train[:n_stage2_samples]
    X_val_stage2 = X_val[:min(200, len(X_val))]
    Y_val_stage2 = Y_val[:min(200, len(X_val))]
    
    train_loader_stage2 = DataLoader(TensorDataset(
        torch.from_numpy(X_train_stage2).float(),
        torch.from_numpy(Y_train_stage2).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader_stage2 = DataLoader(TensorDataset(
        torch.from_numpy(X_val_stage2).float(),
        torch.from_numpy(Y_val_stage2).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    train_stage2(stage2_model, train_loader_stage2, val_loader_stage2, args.stage2_epochs, args.lr, device)
    
    stage2_path = os.path.join(args.checkpoint_dir, 'stage2_best.pt')
    torch.save(stage2_model.state_dict(), stage2_path)
    print(f"Stage 2 checkpoint saved to {stage2_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

