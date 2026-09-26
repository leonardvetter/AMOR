#!/usr/bin/env python3
"""
Two-stage training on Nonlinear Diffusion dataset.

Stage 1: Train in PCA coefficient space only (removes parts operating on original data).
Stage 2: Complete AMOR.py model with three versions:
  - Version 1: Only new weights trained (frozen blocks)
  - Version 2: All weights trained, higher LR for new weights
  - Version 3: All weights trained equally

Uses AMOR.py from ../models/ with hyperparameters from train_FNO_AMOR_PCANet.ipynb.
"""

import sys
import os
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
from typing import Optional
from sklearn.decomposition import PCA

from AMOR.models.amor import AMOR, ReducedBasisBlock, ScalarGate


def set_global_seed(seed: int) -> int:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _atomic_torch_save(obj, path: str) -> None:
    """Write torch checkpoint via temp file + replace to avoid truncating dest on partial write."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pt", dir=d)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _epoch_checkpoint_paths(checkpoint_dir: str, stage2_name: Optional[str], epoch: int):
    """Return (epoch_named_path, resume_path) for periodic checkpoints."""
    if stage2_name is None:
        return (
            os.path.join(checkpoint_dir, f"stage1_epoch_{epoch}.pt"),
            os.path.join(checkpoint_dir, "stage1_resume.pt"),
        )
    return (
        os.path.join(checkpoint_dir, f"stage2_{stage2_name}_epoch_{epoch}.pt"),
        os.path.join(checkpoint_dir, f"stage2_{stage2_name}_resume.pt"),
    )


def _save_periodic_checkpoint(checkpoint_data: dict, checkpoint_dir: str, stage2_name: Optional[str], epoch: int) -> None:
    """
    Always updates resume atomically. Optional separate epoch_N.pt copy (doubles disk writes).
    ABLATION_SKIP_EPOCH_CKPT=1 skips epoch-named checkpoint files.
    """
    epoch_path, resume_path = _epoch_checkpoint_paths(checkpoint_dir, stage2_name, epoch)
    skip_epoch = os.environ.get("ABLATION_SKIP_EPOCH_CKPT", "").strip().lower() in (
        "1", "true", "yes",
    )
    _atomic_torch_save(checkpoint_data, resume_path)
    if not skip_epoch:
        _atomic_torch_save(checkpoint_data, epoch_path)


def l1_loss(pred, target):
    """Absolute L1 loss (mean over batch). Used as training/validation metric for early stopping."""
    return torch.mean(torch.abs(pred - target))


def mse_loss(pred, target):
    """Mean squared error (mean over batch). Optional training objective."""
    return torch.mean((pred - target) ** 2)


def _primary_train_loss(pred, target, train_loss: str):
    if train_loss == "l1":
        return l1_loss(pred, target)
    if train_loss == "mse":
        return mse_loss(pred, target)
    raise ValueError(f"Unknown train_loss: {train_loss!r} (use 'l1' or 'mse')")


def relative_l1_loss(pred, target, eps=1e-8):
    """Per-sample ||pred-target||_1 / ||target||_1, then mean over the batch (not a ratio of two means)."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    diff = torch.abs(pred_flat - target_flat)
    norm_target = torch.abs(target_flat)
    return (diff.sum(dim=1) / (norm_target.sum(dim=1) + eps)).mean()


def relative_l2_error(pred, target, eps=1e-8):
    """Per-sample L2 ratio ||pred-target||_2 / ||target||_2, then mean over the batch (.item() scalar)."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    diff_norm = torch.norm(pred_flat - target_flat, dim=1)
    target_norm = torch.norm(target_flat, dim=1)
    return (diff_norm / (target_norm + eps)).mean().item()


class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors. Compatible with AMOR.py."""
    
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
        num_samples = samples.shape[0]
        if solution.ndim == 3:
            self.out_channels = 1
            solution = solution[:, np.newaxis, :, :]
        else:
            self.out_channels = solution.shape[1]
        
        self.n_channels = samples.shape[1]
        self.shape_ = (samples.shape[2], samples.shape[3])
        
        x_in_flat = samples.reshape(num_samples, -1).astype(np.float32, copy=False)
        y_out_flat = solution.reshape(num_samples, -1).astype(np.float32, copy=False)

        self.pca_in = PCA(n_components=None, svd_solver=self.svd_solver,
                          whiten=self.whiten, random_state=self.random_state).fit(x_in_flat)
        self.pca_out = PCA(n_components=None, svd_solver=self.svd_solver,
                           whiten=self.whiten, random_state=self.random_state).fit(y_out_flat)
        return self

    def k_from_cutoff(self, j_in=0.99, j_out=0.99):
        cumsum_in = np.cumsum(self.pca_in.explained_variance_ratio_)
        cumsum_out = np.cumsum(self.pca_out.explained_variance_ratio_)
        k_in = np.searchsorted(cumsum_in, j_in) + 1
        k_out = np.searchsorted(cumsum_out, j_out) + 1
        return k_in, k_out


class AMORStage1(nn.Module):
    """
    Stage 1: Train in PCA coefficient space only.
    Removes lift, proj, global_skip that operate on original spatial data.
    Only the ReducedBasisBlocks operate (they work in PCA space internally).
    Stage 1: decode PCA to spatial for blocks, then re-encode to PCA.
    """
    
    def __init__(self, width, rb, k_in, k_out, layers, dropout=0.0, init_scale=0.1,
                 use_skip_connections=True, use_batch_norm=False, add_constant_mode=False,
                 use_factorized_weights=False):
        super().__init__()
        
        if rb.pca_in is None or rb.pca_out is None:
            raise ValueError("PCAReducer must be fitted before creating model")
        
        self.width = width
        self.num_layers = layers
        self.use_skip_connections = use_skip_connections
        self.k_in = k_in
        self.k_out = k_out
        self.rb = rb
        
        D_in = rb.n_channels * rb.shape_[0] * rb.shape_[1]
        D_out = rb.out_channels * rb.shape_[0] * rb.shape_[1]
        
        self.input_dim = D_in
        self.output_dim = D_out
        
        self.block0 = ReducedBasisBlock(
            width=width, rb=rb, k_in=k_in, k_out=k_in,
            dropout=dropout, use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=True, use_batch_norm=use_batch_norm,
            add_constant_mode=add_constant_mode, use_factorized_weights=use_factorized_weights
        )
        self.gate0 = ScalarGate(init=init_scale)
        
        self.block1 = ReducedBasisBlock(
            width=width, rb=rb, k_in=k_in, k_out=k_out,
            dropout=dropout, use_input_basis_for_encoder=True,
            use_input_basis_for_decoder=False, use_batch_norm=use_batch_norm,
            add_constant_mode=add_constant_mode, use_factorized_weights=use_factorized_weights
        )
        self.gate1 = ScalarGate(init=init_scale)
        
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(max(0, layers - 1)):
            self.blocks.append(
                ReducedBasisBlock(
                    width=width, rb=rb, k_in=k_in, k_out=k_out,
                    dropout=dropout, use_input_basis_for_encoder=False,
                    use_input_basis_for_decoder=False, use_batch_norm=use_batch_norm,
                    add_constant_mode=add_constant_mode, use_factorized_weights=use_factorized_weights
                )
            )
            self.gates.append(ScalarGate(init=init_scale))
        
        self.lift = nn.Conv1d(1, width, kernel_size=1)
        self.proj1 = nn.Conv1d(width, width, kernel_size=1)
        self.proj2 = nn.Conv1d(width, 1, kernel_size=1)
        self.act = nn.GELU()
        
        pca_in_mean = torch.from_numpy(rb.pca_in.mean_).float()
        pca_in_components = torch.from_numpy(rb.pca_in.components_[:k_in]).float()
        pca_out_mean = torch.from_numpy(rb.pca_out.mean_).float()
        pca_out_components = torch.from_numpy(rb.pca_out.components_[:k_out]).float()
        
        self.register_buffer('pca_in_mean', pca_in_mean)
        self.register_buffer('pca_in_components', pca_in_components)
        self.register_buffer('pca_out_mean', pca_out_mean)
        self.register_buffer('pca_out_components', pca_out_components)
    
    def _pca_to_spatial(self, x_pca):
        """Convert PCA coefficients to spatial representation for blocks."""
        B = x_pca.shape[0]
        x_spatial_flat = x_pca @ self.pca_in_components + self.pca_in_mean
        x_spatial = x_spatial_flat.view(B, self.rb.n_channels, self.rb.shape_[0], self.rb.shape_[1])
        return x_spatial
    
    def _spatial_to_pca(self, x_spatial):
        """Convert spatial representation back to PCA coefficients."""
        B = x_spatial.shape[0]
        x_spatial_flat = x_spatial.view(B, -1)
        x_centered = x_spatial_flat - self.pca_out_mean
        x_pca = x_centered @ self.pca_out_components.T
        return x_pca
    
    def forward(self, x_pca):
        """
        Forward pass in PCA coefficient space.
        x_pca: (B, k_in) - input PCA coefficients
        returns: (B, k_out) - output PCA coefficients
        """
        B = x_pca.shape[0]
        x_spatial = self._pca_to_spatial(x_pca)
        x_flat = x_spatial.reshape(B, self.input_dim)
        x_seq = x_flat.unsqueeze(1)
        h = self.lift(x_seq)
        
        h0 = self.act(self.gate0(self.block0(h)))
        if self.use_skip_connections:
            h = h0 + h
        else:
            h = h0
        
        h = self.act(self.gate1(self.block1(h)))
        
        for block, gate in zip(self.blocks, self.gates):
            residual = self.act(gate(block(h)))
            if self.use_skip_connections:
                h = h + residual
            else:
                h = residual
        
        h = self.act(self.proj1(h))
        h = self.proj2(h)
        y_flat = h.reshape(B, self.output_dim)
        y_spatial = y_flat.view(B, self.rb.out_channels, self.rb.shape_[0], self.rb.shape_[1])
        y_pca = self._spatial_to_pca(y_spatial)
        
        return y_pca




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

    seed_raw = os.environ.get("DATA_EFF_GLOBAL_SEED", os.environ.get("DATA_EFF_SPLIT_SEED", "0")).strip()
    try:
        split_seed = int(seed_raw)
    except ValueError:
        split_seed = 0
    set_global_seed(split_seed)
    print(f"Global seed set to {split_seed}")
    if split_seed >= 0:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(num_samples)
        X = X[perm]
        Y = Y[perm]
        print(f"Applied deterministic dataset shuffle with DATA_EFF_SPLIT_SEED={split_seed}")
    else:
        print("DATA_EFF_SPLIT_SEED < 0: using original on-disk sample order (not recommended)")
    
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
    """Convert spatial input data to PCA coefficients."""
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


def train_stage1(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, start_epoch=0, train_loss: str = "l1"):
    """Train Stage 1 model in PCA coefficient space."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    
    resume_checkpoint = os.path.join(checkpoint_dir, 'stage1_resume.pt')
    if os.path.exists(resume_checkpoint) and start_epoch == 0:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        except Exception as ex:
            print(
                f"Warning: could not load stage1 resume ({type(ex).__name__}: {ex}); starting fresh.",
                flush=True,
            )
            try:
                os.remove(resume_checkpoint)
            except OSError:
                pass
            checkpoint = None
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', start_epoch)
            print(
                f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4e} @ epoch {best_epoch+1}"
            )
        else:
            best_val_loss = float('inf')
            best_epoch = start_epoch
    else:
        best_val_loss = float('inf')
        best_epoch = start_epoch
    
    patience_counter = 0
    early_stopping_patience = 50
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_l1 = 0.0
        train_rel_l1 = 0.0
        train_rel_l2 = 0.0
        for xb_pca, yb_pca in train_loader:
            xb_pca, yb_pca = xb_pca.to(device), yb_pca.to(device)
            optimizer.zero_grad()
            pred_pca = model(xb_pca)
            loss = _primary_train_loss(pred_pca, yb_pca, train_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_l1 += loss.item() * xb_pca.size(0)
            train_rel_l1 += relative_l1_loss(pred_pca, yb_pca).item() * xb_pca.size(0)
            train_rel_l2 += relative_l2_error(pred_pca, yb_pca) * xb_pca.size(0)
        train_l1 /= len(train_loader.dataset)
        train_rel_l1 /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_l1 = 0.0
        val_rel_l1 = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb_pca, yb_pca in val_loader:
                xb_pca, yb_pca = xb_pca.to(device), yb_pca.to(device)
                pred_pca = model(xb_pca)
                val_l1 += _primary_train_loss(pred_pca, yb_pca, train_loss).item() * xb_pca.size(0)
                val_rel_l1 += relative_l1_loss(pred_pca, yb_pca).item() * xb_pca.size(0)
                val_rel_l2 += relative_l2_error(pred_pca, yb_pca) * xb_pca.size(0)
        val_l1 /= len(val_loader.dataset)
        val_rel_l1 /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        val_monitor = val_rel_l1
        
        if val_monitor < best_val_loss:
            best_val_loss = val_monitor
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, 'stage1_best.pt')
            _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Stage 1 Epoch {epoch+1:03d}/{epochs}] train_{train_loss}={train_l1:.4e} "
                f"train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} "
                f"val_{train_loss}={val_l1:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} "
                f"lr={scheduler.get_last_lr()[0]:.2e} (best val_rel_l1: {best_val_loss:.4e} @ epoch {best_epoch+1})"
            )
        
        if (epoch + 1) % 10 == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'val_rel_l1': val_rel_l1,
                'val_rel_l2': val_rel_l2,
            }
            _save_periodic_checkpoint(checkpoint_data, checkpoint_dir, None, epoch + 1)
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss


def train_stage2(model, train_loader, val_loader, epochs, lr, lr_blocks_factor, device, checkpoint_dir, version_name, start_epoch=0, train_loss: str = "l1"):
    """Train Stage 2 model in real space."""
    model.to(device)
    tl = train_loss

    if lr_blocks_factor < 1.0:
        block_params = []
        new_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name.startswith('block') or name.startswith('gate'):
                    block_params.append(param)
                else:
                    new_params.append(param)
        param_groups = [
            {'params': block_params, 'lr': lr * lr_blocks_factor},
            {'params': new_params, 'lr': lr}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4,
                                       betas=(0.9, 0.999), eps=1e-8)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4,
                                       betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    
    resume_checkpoint = os.path.join(checkpoint_dir, f'stage2_{version_name}_resume.pt')
    if os.path.exists(resume_checkpoint) and start_epoch == 0:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        except Exception as ex:
            print(
                f"Warning: could not load stage2 resume ({type(ex).__name__}: {ex}); starting fresh.",
                flush=True,
            )
            try:
                os.remove(resume_checkpoint)
            except OSError:
                pass
            checkpoint = None
        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', start_epoch)
            print(
                f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4e} @ epoch {best_epoch+1}"
            )
        else:
            best_val_loss = float('inf')
            best_epoch = start_epoch
    else:
        best_val_loss = float('inf')
        best_epoch = start_epoch
    
    patience_counter = 0
    early_stopping_patience = 50
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_l1 = 0.0
        train_rel_l1 = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = _primary_train_loss(pred, yb, train_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_l1 += loss.item() * xb.size(0)
            train_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_l1 /= len(train_loader.dataset)
        train_rel_l1 /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_l1 = 0.0
        val_rel_l1 = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_l1 += _primary_train_loss(pred, yb, train_loss).item() * xb.size(0)
                val_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_l1 /= len(val_loader.dataset)
        val_rel_l1 /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        val_monitor = val_rel_l1
        
        if val_monitor < best_val_loss:
            best_val_loss = val_monitor
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt')
            _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Stage 2 {version_name} Epoch {epoch+1:03d}/{epochs}] train_{train_loss}={train_l1:.4e} "
                f"train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} "
                f"val_{train_loss}={val_l1:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} "
                f"lr={scheduler.get_last_lr()[0]:.2e} (best val_rel_l1: {best_val_loss:.4e} @ epoch {best_epoch+1})"
            )
        
        if (epoch + 1) % 10 == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_rel_l1': val_rel_l1,
                'val_rel_l2': val_rel_l2,
                'best_epoch': best_epoch
            }
            _save_periodic_checkpoint(checkpoint_data, checkpoint_dir, version_name, epoch + 1)
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss


def train_stage2_proj_high_lr(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, version_name, start_epoch=0, proj_lr_factor=5.0):
    """Train Stage 2 with frozen blocks. Higher LR for projection layers only (proj1, proj2)."""
    model.to(device)
    proj_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'proj' in name:
                proj_params.append(param)
            else:
                other_params.append(param)
    param_groups = [
        {'params': proj_params, 'lr': lr * proj_lr_factor},
        {'params': other_params, 'lr': lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    return _train_stage2_loop(model, train_loader, val_loader, epochs, optimizer, scheduler,
                              device, checkpoint_dir, version_name, start_epoch)


def train_stage2_lift_proj_high_lr(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, version_name, start_epoch=0, lift_proj_lr_factor=5.0):
    """Train Stage 2 with frozen blocks. Higher LR for lift/proj layers."""
    model.to(device)
    lift_proj_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lift' in name or 'proj' in name:
                lift_proj_params.append(param)
            else:
                other_params.append(param)
    param_groups = [
        {'params': lift_proj_params, 'lr': lr * lift_proj_lr_factor},
        {'params': other_params, 'lr': lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    return _train_stage2_loop(model, train_loader, val_loader, epochs, optimizer, scheduler,
                              device, checkpoint_dir, version_name, start_epoch)


def train_stage2_lift_proj_first(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, version_name, start_epoch=0):
    """Train Stage 2 with frozen blocks. Phase 1: lift/proj/global_skip only. Phase 2: add skip_conv."""
    model.to(device)
    lift_proj_global_names = ['lift', 'proj1', 'proj2', 'global_skip_proj', 'global_skip_gate']
    for name, param in model.named_parameters():
        if param.requires_grad and 'skip_conv' in name:
            param.requires_grad = False
    phase1_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(phase1_params, lr=lr, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    phase1_epochs = epochs // 2
    return _train_stage2_phased(model, train_loader, val_loader, epochs, optimizer, scheduler,
                                device, checkpoint_dir, version_name, start_epoch,
                                unfreeze_fn=lambda m: _unfreeze_skip_conv(m),
                                unfreeze_epoch=phase1_epochs)


def train_stage2_phased_lift_proj(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, version_name, start_epoch=0):
    """Train Stage 2 with frozen blocks. Phased: proj -> lift -> skip_conv -> global_skip."""
    model.to(device)
    phase_len = epochs // 4
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
    proj_params = []
    for name, param in model.named_parameters():
        if 'proj1' in name or 'proj2' in name:
            param.requires_grad = True
            proj_params.append(param)
    optimizer = torch.optim.AdamW(proj_params, lr=lr, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    phases = [
        (phase_len, lambda m: _unfreeze_lift(m)),
        (2 * phase_len, lambda m: _unfreeze_skip_conv(m)),
        (3 * phase_len, lambda m: _unfreeze_global_skip(m)),
    ]
    return _train_stage2_multi_phased(model, train_loader, val_loader, epochs, lr, optimizer, scheduler,
                                      device, checkpoint_dir, version_name, start_epoch, phases)


def _unfreeze_lift(model):
    for name, param in model.named_parameters():
        if 'lift' in name and param.requires_grad is False:
            param.requires_grad = True


def _unfreeze_skip_conv(model):
    for name, param in model.named_parameters():
        if 'skip_conv' in name and param.requires_grad is False:
            param.requires_grad = True


def _unfreeze_global_skip(model):
    for name, param in model.named_parameters():
        if 'global_skip' in name and param.requires_grad is False:
            param.requires_grad = True


def _train_stage2_loop(model, train_loader, val_loader, epochs, optimizer, scheduler,
                       device, checkpoint_dir, version_name, start_epoch):
    """Generic Stage 2 training loop."""
    resume_checkpoint = os.path.join(checkpoint_dir, f'stage2_{version_name}_resume.pt')
    if os.path.exists(resume_checkpoint) and start_epoch == 0:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', start_epoch)
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4e} @ epoch {best_epoch+1}")
    else:
        best_val_loss = float('inf')
        best_epoch = start_epoch
    patience_counter = 0
    early_stopping_patience = 50
    for epoch in range(start_epoch, epochs):
        model.train()
        train_l1 = 0.0
        train_rel_l1 = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_l1 += loss.item() * xb.size(0)
            train_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_l1 /= len(train_loader.dataset)
        train_rel_l1 /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        model.eval()
        val_l1 = 0.0
        val_rel_l1 = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_l1 += l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_l1 /= len(val_loader.dataset)
        val_rel_l1 /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        scheduler.step()
        if val_rel_l1 < best_val_loss:
            best_val_loss = val_rel_l1
            best_epoch = epoch
            patience_counter = 0
            _atomic_torch_save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt'),
            )
        else:
            patience_counter += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 2 {version_name} Epoch {epoch+1:03d}/{epochs}] train_l1={train_l1:.4e} train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} val_l1={val_l1:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best val_rel_l1: {best_val_loss:.4e} @ epoch {best_epoch+1})")
        if (epoch + 1) % 10 == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'val_rel_l1': val_rel_l1,
                'val_rel_l2': val_rel_l2,
            }
            _save_periodic_checkpoint(checkpoint_data, checkpoint_dir, version_name, epoch + 1)
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    return best_val_loss


def _train_stage2_phased(model, train_loader, val_loader, epochs, optimizer, scheduler,
                         device, checkpoint_dir, version_name, start_epoch, unfreeze_fn, unfreeze_epoch):
    """Stage 2 training with one unfreeze step."""
    best_val_loss, best_epoch, start_epoch = _load_resume_or_init(model, optimizer, scheduler,
                                                                  checkpoint_dir, version_name, start_epoch, device)
    patience_counter = 0
    early_stopping_patience = 50
    for epoch in range(start_epoch, epochs):
        if epoch == unfreeze_epoch:
            print(f"\nUnfreezing skip_conv at epoch {epoch+1}")
            unfreeze_fn(model)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=scheduler.get_last_lr()[0], weight_decay=1e-4,
                                           betas=(0.9, 0.999), eps=1e-8)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch, eta_min=0.0)
        best_val_loss, best_epoch, patience_counter = _run_one_epoch(model, train_loader, val_loader,
            optimizer, scheduler, device, checkpoint_dir, version_name, epoch, epochs,
            best_val_loss, best_epoch, patience_counter, early_stopping_patience)
        if patience_counter >= early_stopping_patience:
            break
    return best_val_loss


def _train_stage2_multi_phased(model, train_loader, val_loader, epochs, lr, optimizer, scheduler,
                               device, checkpoint_dir, version_name, start_epoch, phases):
    """Stage 2 training with multiple unfreeze steps."""
    best_val_loss, best_epoch, start_epoch = _load_resume_or_init(model, optimizer, scheduler,
                                                                  checkpoint_dir, version_name, start_epoch, device)
    patience_counter = 0
    early_stopping_patience = 50
    for epoch in range(start_epoch, epochs):
        for thresh, unfreeze_fn in phases:
            if epoch == thresh:
                unfreeze_fn(model)
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4,
                                               betas=(0.9, 0.999), eps=1e-8)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch, eta_min=0.0)
                print(f"\nUnfreezing next phase at epoch {epoch+1}")
                break
        best_val_loss, best_epoch, patience_counter = _run_one_epoch(model, train_loader, val_loader,
            optimizer, scheduler, device, checkpoint_dir, version_name, epoch, epochs,
            best_val_loss, best_epoch, patience_counter, early_stopping_patience)
        if patience_counter >= early_stopping_patience:
            break
    return best_val_loss


def _load_resume_or_init(model, optimizer, scheduler, checkpoint_dir, version_name, start_epoch, device):
    resume_path = os.path.join(checkpoint_dir, f'stage2_{version_name}_resume.pt')
    if os.path.exists(resume_path) and start_epoch == 0:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and optimizer.param_groups:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, KeyError):
                pass
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except (ValueError, KeyError):
                pass
        start_epoch = ckpt.get('epoch', 0)
        return ckpt.get('best_val_loss', float('inf')), ckpt.get('best_epoch', start_epoch), start_epoch
    return float('inf'), start_epoch, start_epoch


def _run_one_epoch(model, train_loader, val_loader, optimizer, scheduler, device, checkpoint_dir, version_name, epoch, epochs, best_val_loss, best_epoch, patience_counter, early_stopping_patience):
    model.train()
    train_l1 = 0.0
    train_rel_l1 = 0.0
    train_rel_l2 = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = l1_loss(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        train_l1 += loss.item() * xb.size(0)
        train_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
        train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
    train_l1 /= len(train_loader.dataset)
    train_rel_l1 /= len(train_loader.dataset)
    train_rel_l2 /= len(train_loader.dataset)
    model.eval()
    val_l1 = 0.0
    val_rel_l1 = 0.0
    val_rel_l2 = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_l1 += l1_loss(pred, yb).item() * xb.size(0)
            val_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
    val_l1 /= len(val_loader.dataset)
    val_rel_l1 /= len(val_loader.dataset)
    val_rel_l2 /= len(val_loader.dataset)
    scheduler.step()
    if val_rel_l1 < best_val_loss:
        best_val_loss, best_epoch, patience_counter = val_rel_l1, epoch, 0
        _atomic_torch_save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt'),
        )
    else:
        patience_counter += 1
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"[Stage 2 {version_name} Epoch {epoch+1:03d}/{epochs}] train_l1={train_l1:.4e} train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} val_l1={val_l1:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best val_rel_l1: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    if (epoch + 1) % 10 == 0:
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'val_rel_l1': val_rel_l1,
            'val_rel_l2': val_rel_l2,
        }
        _save_periodic_checkpoint(checkpoint_data, checkpoint_dir, version_name, epoch + 1)
    return best_val_loss, best_epoch, patience_counter


def train_stage2_progressive_unfreeze(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, version_name, start_epoch=0):
    """Train Stage 2 with progressive unfreezing: unfreeze blocks gradually."""
    model.to(device)
    
    block_names = []
    for name, param in model.named_parameters():
        if (name.startswith('block') or name.startswith('gate')) and param.requires_grad:
            block_names.append(name)
            param.requires_grad = False
    
    new_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(new_params, lr=lr, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    
    resume_checkpoint = os.path.join(checkpoint_dir, f'stage2_{version_name}_resume.pt')
    if os.path.exists(resume_checkpoint) and start_epoch == 0:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', start_epoch)
        unfreeze_epoch = checkpoint.get('unfreeze_epoch', epochs // 4)
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4e} @ epoch {best_epoch+1}")
    else:
        best_val_loss = float('inf')
        best_epoch = start_epoch
        unfreeze_epoch = epochs // 4
    
    patience_counter = 0
    early_stopping_patience = 50
    
    tl = "l1"
    for epoch in range(start_epoch, epochs):
        if epoch == unfreeze_epoch:
            print(f"\nUnfreezing blocks at epoch {epoch+1}")
            for name, param in model.named_parameters():
                if name in block_names:
                    param.requires_grad = True
            
            block_params = [p for n, p in model.named_parameters() if n in block_names]
            new_params = [p for n, p in model.named_parameters() if n not in block_names and p.requires_grad]
            param_groups = [
                {'params': block_params, 'lr': lr * 0.1},
                {'params': new_params, 'lr': lr}
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4,
                                         betas=(0.9, 0.999), eps=1e-8)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch, eta_min=0.0)
        
        model.train()
        train_l1 = 0.0
        train_rel_l1 = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = _primary_train_loss(pred, yb, tl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_l1 += loss.item() * xb.size(0)
            train_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_l1 /= len(train_loader.dataset)
        train_rel_l1 /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_l1 = 0.0
        val_rel_l1 = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_l1 += _primary_train_loss(pred, yb, tl).item() * xb.size(0)
                val_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_l1 /= len(val_loader.dataset)
        val_rel_l1 /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_rel_l1 < best_val_loss:
            best_val_loss = val_rel_l1
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt')
            _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Stage 2 {version_name} Epoch {epoch+1:03d}/{epochs}] train_{tl}={train_l1:.4e} "
                f"train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} "
                f"val_{tl}={val_l1:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} "
                f"lr={scheduler.get_last_lr()[0]:.2e} (best val_rel_l1: {best_val_loss:.4e} @ epoch {best_epoch+1})"
            )
        
        if (epoch + 1) % 10 == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'unfreeze_epoch': unfreeze_epoch,
                'val_rel_l1': val_rel_l1,
                'val_rel_l2': val_rel_l2,
            }
            _save_periodic_checkpoint(checkpoint_data, checkpoint_dir, version_name, epoch + 1)
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss


def train_stage2_lr_warmup(model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, version_name, start_epoch=0):
    """Train Stage 2 with LR warmup for new layers."""
    model.to(device)
    
    block_params = []
    new_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith('block') or name.startswith('gate'):
                block_params.append(param)
            else:
                new_params.append(param)
    
    param_groups = [
        {'params': block_params, 'lr': lr},
        {'params': new_params, 'lr': lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    
    warmup_epochs = epochs // 10
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) if epoch < warmup_epochs else 1.0
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=0.0)
    
    resume_checkpoint = os.path.join(checkpoint_dir, f'stage2_{version_name}_resume.pt')
    if os.path.exists(resume_checkpoint) and start_epoch == 0:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_warmup_state_dict' in checkpoint:
            scheduler_warmup.load_state_dict(checkpoint['scheduler_warmup_state_dict'])
        if 'scheduler_cosine_state_dict' in checkpoint:
            scheduler_cosine.load_state_dict(checkpoint['scheduler_cosine_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', start_epoch)
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4e} @ epoch {best_epoch+1}")
    else:
        best_val_loss = float('inf')
        best_epoch = start_epoch
    
    patience_counter = 0
    early_stopping_patience = 50
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_l1 = 0.0
        train_rel_l1 = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_l1 += loss.item() * xb.size(0)
            train_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_l1 /= len(train_loader.dataset)
        train_rel_l1 /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_l1 = 0.0
        val_rel_l1 = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_l1 += l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_l1 /= len(val_loader.dataset)
        val_rel_l1 /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()
        
        if val_rel_l1 < best_val_loss:
            best_val_loss = val_rel_l1
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt')
            _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = scheduler_warmup.get_last_lr()[0] if epoch < warmup_epochs else scheduler_cosine.get_last_lr()[0]
            print(f"[Stage 2 {version_name} Epoch {epoch+1:03d}/{epochs}] train_l1={train_l1:.4e} train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} val_l1={val_l1:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} lr={current_lr:.2e} (best val_rel_l1: {best_val_loss:.4e} @ epoch {best_epoch+1})")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_warmup_state_dict': scheduler_warmup.state_dict(),
                'scheduler_cosine_state_dict': scheduler_cosine.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'val_rel_l1': val_rel_l1,
                'val_rel_l2': val_rel_l2,
            }
            _save_periodic_checkpoint(checkpoint_data, checkpoint_dir, version_name, epoch + 1)
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss


def train_stage2_knowledge_distillation(model, stage1_model, train_loader, val_loader, epochs, lr, device, checkpoint_dir, version_name, rb, k_in, k_out, start_epoch=0, alpha=0.3):
    """Train Stage 2 with knowledge distillation from Stage 1."""
    model.to(device)
    stage1_model.to(device)
    stage1_model.eval()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    
    resume_checkpoint = os.path.join(checkpoint_dir, f'stage2_{version_name}_resume.pt')
    if os.path.exists(resume_checkpoint) and start_epoch == 0:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', start_epoch)
        print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4e} @ epoch {best_epoch+1}")
    else:
        best_val_loss = float('inf')
        best_epoch = start_epoch
    
    patience_counter = 0
    early_stopping_patience = 50
    
    def spatial_to_pca_coefficients_batch(x_spatial, rb, k):
        B = x_spatial.shape[0]
        x_flat = x_spatial.view(B, -1).cpu().numpy()
        x_centered = x_flat - rb.pca_in.mean_
        x_pca = x_centered @ rb.pca_in.components_[:k].T
        return torch.from_numpy(x_pca).float().to(x_spatial.device)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_l1 = 0.0
        train_rel_l1 = 0.0
        train_rel_l2 = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            
            loss_task = l1_loss(pred, yb)
            
            with torch.no_grad():
                xb_pca = spatial_to_pca_coefficients_batch(xb, rb, k_in)
                pred_stage1_pca = stage1_model(xb_pca)
                pred_stage1_spatial = pred_stage1_pca @ torch.from_numpy(rb.pca_out.components_[:k_out]).float().to(device) + torch.from_numpy(rb.pca_out.mean_).float().to(device)
                pred_stage1_spatial = pred_stage1_spatial.view(pred.shape)
            
            loss_kd = l1_loss(pred, pred_stage1_spatial)
            loss = alpha * loss_kd + (1 - alpha) * loss_task
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            train_l1 += loss_task.item() * xb.size(0)
            train_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_l1 /= len(train_loader.dataset)
        train_rel_l1 /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_l1 = 0.0
        val_rel_l1 = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_l1 += l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_l1 /= len(val_loader.dataset)
        val_rel_l1 /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        
        if val_rel_l1 < best_val_loss:
            best_val_loss = val_rel_l1
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt')
            _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 2 {version_name} Epoch {epoch+1:03d}/{epochs}] train_l1={train_l1:.4e} train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} val_l1={val_l1:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            }
            _save_periodic_checkpoint(checkpoint_data, checkpoint_dir, version_name, epoch + 1)
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss


def initialize_lift_proj_from_pca(model, rb, k_in, k_out, width, device):
    """Initialize lift and proj layers to approximate PCA encode/decode.
    
    Note: The lift layer is Conv1d(1, width, kernel_size=1), so weight shape is (width, 1, 1).
    The proj2 layer is Conv1d(width, 1, kernel_size=1), so weight shape is (1, width, 1).
    Pointwise convolutions (kernel 1); PCA weights do not map directly to conv kernels.
    Xavier initialization is the default.
    This function is kept for future improvements but currently does minimal initialization.
    """
    with torch.no_grad():
        if hasattr(model, 'lift') and model.lift is not None:
            nn.init.xavier_uniform_(model.lift.weight, gain=0.1)
        
        if hasattr(model, 'proj2') and model.proj2 is not None:
            nn.init.xavier_uniform_(model.proj2.weight, gain=0.1)


def denormalize_y(tensor, y_mean: float, y_std: float):
    """Invert global Y normalization from load_nonlinear_dataset (scalar mean/std)."""
    return tensor * (y_std + 1e-8) + y_mean


def evaluate_model(
    model,
    test_loader,
    device,
    is_stage1=False,
    rb=None,
    k_out=None,
    denorm_y: bool = True,
    y_mean: float | None = None,
    y_std: float | None = None,
):
    """Dataset-averaged relative L1/L2.

    By default undoes Y z-score (denorm_y=True) so reported test metrics match
    paper / original-unit errors. Pass denorm_y=False for normalized-space metrics
    (callers should label those as *_normalized).
    """
    if denorm_y and (y_mean is None or y_std is None):
        raise ValueError("denorm_y=True requires y_mean and y_std from load_nonlinear_dataset")
    model.eval()
    test_rel_l1 = 0.0
    test_rel_l2 = 0.0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            if is_stage1:
                pred = model(xb)
            else:
                pred = model(xb)
            if denorm_y:
                pred = denormalize_y(pred, y_mean, y_std)
                yb = denormalize_y(yb, y_mean, y_std)
            test_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            test_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)

    test_rel_l1 /= len(test_loader.dataset)
    test_rel_l2 /= len(test_loader.dataset)

    return test_rel_l1, test_rel_l2


def evaluate_model_normalized(
    model,
    test_loader,
    device,
    is_stage1=False,
    rb=None,
    k_out=None,
):
    """Relative L1/L2 in z-scored Y space (label results as *_normalized)."""
    return evaluate_model(
        model,
        test_loader,
        device,
        is_stage1=is_stage1,
        rb=rb,
        k_out=k_out,
        denorm_y=False,
        y_mean=None,
        y_std=None,
    )


def load_stage2_v1_from_stage1_checkpoint(stage1_path: str, stage2_best_path: str, device: torch.device):
    """Rebuild frozen-blocks Stage-2 v1 AMOR and load stage2_v1_best weights (spatial test eval)."""
    from AMOR.models.amor import AMOR

    checkpoint = torch.load(stage1_path, map_location=device, weights_only=False)
    rb_data = checkpoint["rb_data"]
    rb = PCAReducerMultiChannel()
    rb.pca_in = rb_data["pca_in"]
    rb.pca_out = rb_data["pca_out"]
    rb.n_channels = rb_data["n_channels"]
    rb.out_channels = rb_data["out_channels"]
    rb.shape_ = rb_data["shape_"]
    k_in = checkpoint["k_in"]
    k_out = checkpoint["k_out"]
    width = checkpoint["width"]
    layers = checkpoint["layers"]

    stage1_model = AMORStage1(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, add_constant_mode=False, use_factorized_weights=False,
    )
    stage1_model.load_state_dict(checkpoint["model_state_dict"])

    model = AMOR(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False, add_constant_mode=False,
        use_factorized_weights=False,
    )
    stage1_state = stage1_model.state_dict()
    stage2_state = model.state_dict()
    for name, param in stage1_state.items():
        if name in stage2_state and (name.startswith("block") or name.startswith("gate")):
            stage2_state[name] = param.clone()
    model.load_state_dict(stage2_state, strict=False)
    for param in model.block0.parameters():
        param.requires_grad = False
    for param in model.block1.parameters():
        param.requires_grad = False
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = False
    for param in model.gate0.parameters():
        param.requires_grad = False
    for param in model.gate1.parameters():
        param.requires_grad = False
    for gate in model.gates:
        for param in gate.parameters():
            param.requires_grad = False

    model.load_state_dict(torch.load(stage2_best_path, map_location=device, weights_only=False))
    model.to(device)
    return model, (k_in, k_out)


def eval_stage2_v1_test(
    stage1_path: str,
    stage2_best_path: str,
    data_path: str,
    device: torch.device,
    batch_size: int = 32,
    denorm_y: bool = True,
):
    """Test-split relative L1/L2 for Stage-2 v1 best checkpoint (default: denormalized Y)."""
    (_, _, _, _, X_test, Y_test), norm_stats = load_nonlinear_dataset(data_path)
    x_mean, x_std, y_mean, y_std = norm_stats
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    model, _ = load_stage2_v1_from_stage1_checkpoint(stage1_path, stage2_best_path, device)
    return evaluate_model(
        model,
        test_loader,
        device,
        is_stage1=False,
        denorm_y=denorm_y,
        y_mean=y_mean,
        y_std=y_std,
    )


def eval_baseline_test(
    baseline_best_path: str,
    baseline_final_path: str,
    data_path: str,
    device: torch.device,
    batch_size: int = 32,
    denorm_y: bool = True,
):
    """Test-split relative L1/L2 for one-stage baseline best checkpoint (default: denormalized Y)."""
    from AMOR.models.amor import AMOR

    (_, _, _, _, X_test, Y_test), norm_stats = load_nonlinear_dataset(data_path)
    y_mean, y_std = norm_stats[2], norm_stats[3]
    meta = torch.load(baseline_final_path, map_location=device, weights_only=False)
    rb = meta["rb"]
    model = AMOR(
        width=meta["width"], rb=rb, k_in=meta["k_in"], k_out=meta["k_out"], layers=meta["layers"],
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False, add_constant_mode=False,
        use_factorized_weights=False,
    )
    model.load_state_dict(torch.load(baseline_best_path, map_location=device, weights_only=False))
    model.to(device)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    return evaluate_model(
        model,
        test_loader,
        device,
        is_stage1=False,
        denorm_y=denorm_y,
        y_mean=y_mean,
        y_std=y_std,
    )
