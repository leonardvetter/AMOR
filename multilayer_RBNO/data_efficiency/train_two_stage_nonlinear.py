#!/usr/bin/env python3
"""
Two-stage training on Nonlinear Diffusion dataset (nonlinear_diffusion_dataset_big.npz).

Stage 1: Train in PCA space with loss on PCA coefficients.
Stage 2: Fine-tune on real-space data with frozen blocks (or unfrozen with small LR).

Uses new MRBNO.py model architecture.
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

from MRBNO import MRBNO


def relative_l1_loss(pred, target, eps=1e-8):
    """Compute relative L1 loss: ||pred - target||_1 / ||target||_1"""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    diff = torch.abs(pred_flat - target_flat)
    norm_target = torch.abs(target_flat)
    return (diff.sum(dim=1) / (norm_target.sum(dim=1) + eps)).mean()


def relative_l2_error(pred, target, eps=1e-8):
    """Compute relative L2 error: ||pred - target||_2 / ||target||_2"""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    diff_norm = torch.norm(pred_flat - target_flat, dim=1)
    target_norm = torch.norm(target_flat, dim=1)
    return (diff_norm / (target_norm + eps)).mean().item()


class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors. Compatible with MRBNO.py."""
    
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
            raise ValueError(f"Expected (num_samples, C_in, N, M), got {samples.shape}")
        num_samples, c_in, n, m = samples.shape
        self.n_channels = c_in
        self.shape_ = (n, m)
        
        if solution.ndim == 3:
            solution = solution[:, None, :, :]
        if solution.ndim != 4:
            raise ValueError(f"Expected (num_samples, C_out, N, M) or (num_samples, N, M), got {solution.shape}")
        _, c_out, n_out, m_out = solution.shape
        if (n_out, m_out) != (n, m):
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        
        x_in = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
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


class MRBNOStage1(MRBNO):
    """Stage 1: Train using MRBNO architecture. Loss computed in PCA space."""
    
    def __init__(self, rb, width, k_in, k_out, layers, dropout=0.0, use_batch_norm=True,
                 use_skip_connections=True, use_coordinates=False, add_constant_mode=False,
                 use_factorized_weights=False):
        super().__init__(
            width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
            dropout=dropout, init_scale=0.1, use_skip_connections=use_skip_connections,
            use_batch_norm=use_batch_norm, use_coordinates=use_coordinates,
            add_constant_mode=add_constant_mode, use_factorized_weights=use_factorized_weights
        )
        self.rb = rb
        self.k_in = k_in
        self.k_out = k_out


class MRBNOStage2(MRBNO):
    """Stage 2: Fine-tune in real space with pre-trained Stage 1 weights."""
    
    def __init__(self, stage1_model, freeze_blocks=True):
        super().__init__(
            width=stage1_model.width,
            rb=stage1_model.rb,
            k_in=stage1_model.k_in,
            k_out=stage1_model.k_out,
            layers=stage1_model.num_layers,
            dropout=0.0,
            init_scale=0.1,
            use_skip_connections=stage1_model.use_skip_connections,
            use_batch_norm=stage1_model.use_batch_norm,
            use_coordinates=stage1_model.use_coordinates,
            add_constant_mode=False,
            use_factorized_weights=False
        )
        
        self.load_state_dict(stage1_model.state_dict(), strict=False)
        
        if freeze_blocks:
            for param in self.block0.parameters():
                param.requires_grad = False
            for param in self.block1.parameters():
                param.requires_grad = False
            for param in self.blocks.parameters():
                param.requires_grad = False
            for param in self.gate0.parameters():
                param.requires_grad = False
            for param in self.gate1.parameters():
                param.requires_grad = False
            for param in self.gates.parameters():
                param.requires_grad = False


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


def spatial_to_pca_coefficients(X, rb, k_in):
    """Convert spatial data to PCA coefficients."""
    num_samples = X.shape[0]
    X_flat = X.reshape(num_samples, -1)
    X_centered = X_flat - rb.pca_in.mean_
    X_pca = X_centered @ rb.pca_in.components_[:k_in].T
    return X_pca

def spatial_to_pca_coefficients_output(Y, rb, k_out):
    """Convert spatial output data to PCA coefficients."""
    num_samples = Y.shape[0]
    Y_flat = Y.reshape(num_samples, -1)
    Y_centered = Y_flat - rb.pca_out.mean_
    Y_pca = Y_centered @ rb.pca_out.components_[:k_out].T
    return Y_pca


def train_stage1(model, train_loader, val_loader, rb, k_out, epochs, lr, device):
    """Train Stage 1 model. Loss computed in PCA space."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    pca_out_mean = torch.from_numpy(rb.pca_out.mean_).float().to(device)
    pca_out_components = torch.from_numpy(rb.pca_out.components_[:k_out]).float().to(device)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred_spatial = model(xb)
            
            pred_flat = pred_spatial.view(pred_spatial.size(0), -1)
            yb_flat = yb.view(yb.size(0), -1)
            
            pred_centered = pred_flat - pca_out_mean
            yb_centered = yb_flat - pca_out_mean
            
            pred_pca = pred_centered @ pca_out_components.T
            yb_pca = yb_centered @ pca_out_components.T
            
            loss = relative_l1_loss(pred_pca, yb_pca)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred_pca, yb_pca) * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred_spatial = model(xb)
                
                pred_flat = pred_spatial.view(pred_spatial.size(0), -1)
                yb_flat = yb.view(yb.size(0), -1)
                
                pred_centered = pred_flat - pca_out_mean
                yb_centered = yb_flat - pca_out_mean
                
                pred_pca = pred_centered @ pca_out_components.T
                yb_pca = yb_centered @ pca_out_components.T
                
                loss = relative_l1_loss(pred_pca, yb_pca)
                val_loss += loss.item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred_pca, yb_pca) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 1 Epoch {epoch+1:03d}/{epochs}] train_rel_l1={train_loss:.4e} train_rel_l2={train_rel_l2:.4e} val_rel_l1={val_loss:.4e} val_rel_l2={val_rel_l2:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return best_val_loss


def train_stage2(model, train_loader, val_loader, epochs, lr, device):
    """Train Stage 2 model in real space with frozen blocks."""
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = relative_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = relative_l1_loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 2 Epoch {epoch+1:03d}/{epochs}] train_rel_l1={train_loss:.4e} train_rel_l2={train_rel_l2:.4e} val_rel_l1={val_loss:.4e} val_rel_l2={val_rel_l2:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return best_val_loss


def train_stage2_unfrozen(model, train_loader, val_loader, epochs, lr, lr_blocks, device):
    """Train Stage 2 model in real space with all parameters trainable.
    
    Blocks (PCA space parameters) use smaller learning rate lr_blocks.
    Real-space layers (lift, proj, skip) use learning rate lr.
    """
    model.to(device)
    
    block_params = []
    real_space_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ['block0', 'block1', 'blocks', 'gate0', 'gate1', 'gates']):
            block_params.append(param)
        else:
            real_space_params.append(param)
    
    param_groups = [
        {'params': block_params, 'lr': lr_blocks},
        {'params': real_space_params, 'lr': lr}
    ]
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = relative_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = relative_l1_loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_vals = scheduler.get_last_lr()
            lr_block = lr_vals[0]
            lr_real = lr_vals[1] if len(lr_vals) > 1 else lr_vals[0]
            print(f"[Stage 2 Unfrozen Epoch {epoch+1:03d}/{epochs}] train_rel_l1={train_loss:.4e} train_rel_l2={train_rel_l2:.4e} val_rel_l1={val_loss:.4e} val_rel_l2={val_rel_l2:.4e} lr_real={lr_real:.2e} lr_blocks={lr_block:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
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
    parser.add_argument('--layers', type=int, default=4, help='Number of layers')
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
    print("STAGE 1: Training in real space, loss in PCA space")
    print("="*70)
    
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float()
    ), batch_size=args.batch_size, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val).float()
    ), batch_size=args.batch_size, shuffle=False)
    
    stage1_model = MRBNOStage1(
        rb=rb, width=args.width, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=0.0, use_batch_norm=True, use_skip_connections=True,
        use_coordinates=False, add_constant_mode=False, use_factorized_weights=False
    )
    
    train_stage1(stage1_model, train_loader, val_loader, rb, k_out, args.stage1_epochs, args.lr, device)
    
    stage1_path = os.path.join(checkpoint_dir, f'stage1_nonlinear_{args.stage1_epochs}epochs.pt')
    torch.save({
        'model_state_dict': stage1_model.state_dict(),
        'rb': rb,
        'k_in': k_in,
        'k_out': k_out,
        'width': args.width,
        'layers': args.layers
    }, stage1_path)
    print(f"\nStage 1 checkpoint saved to {stage1_path}")
    
    print("\n" + "="*70)
    print("STAGE 1 TRAINING COMPLETE")
    print("="*70)
