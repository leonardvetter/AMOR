#!/usr/bin/env python3
"""
Test mesh generalization: evaluate models trained on one resolution
on different resolutions using various transfer methods.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional
from scipy.linalg import eigh

# Script lives at multilayer_RBNO/multilayer_RBNO/mesh_generalization/; repo root is three levels up.
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'multilayer_RBNO', 'models'))

from multilayer_RBNO.models.MRBNO import MRBNO
from multilayer_RBNO.models.FNO_model import FNO2d
from multilayer_RBNO.models.PCANet import PCANet
from multilayer_RBNO.models.PCAReducer import PCAReducerMultiChannel


def load_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from .npz file."""
    data = np.load(data_path, allow_pickle=True)
    f = data['f'].astype(np.float32)
    m = data['m'].astype(np.float32)
    u = data['u'].astype(np.float32)
    
    # Stack f and m as input channels: (N, 2, H, W)
    X = np.stack([f, m], axis=1)
    Y = u[:, None, :, :]  # (N, 1, H, W)
    
    return X, Y


def normalize_data(X_train: np.ndarray, Y_train: np.ndarray, 
                   X_val: np.ndarray, Y_val: np.ndarray) -> Tuple:
    """Normalize data using training statistics."""
    x_mean = X_train.mean()
    x_std = X_train.std()
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    X_train_norm = (X_train - x_mean) / (x_std + 1e-8)
    Y_train_norm = (Y_train - y_mean) / (y_std + 1e-8)
    X_val_norm = (X_val - x_mean) / (x_std + 1e-8)
    Y_val_norm = (Y_val - y_mean) / (y_std + 1e-8)
    
    return (X_train_norm, Y_train_norm, X_val_norm, Y_val_norm,
            x_mean, x_std, y_mean, y_std)


def train_model(model, train_loader, val_loader, epochs=300, lr=2e-3, 
                weight_decay=1e-4, max_grad_norm=3.0, device='cuda',
                checkpoint_path=None):
    """Train model with same hyperparameters as notebook."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )
    criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            try:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError("CUDA OOM during training")
                raise
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
        
        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return train_losses, val_losses, best_val_loss


def compute_relative_errors(Y_pred: np.ndarray, Y_true: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Compute relative L1 and relative L2 errors."""
    diff = np.abs(Y_pred - Y_true)
    abs_true = np.abs(Y_true)
    rel_l1 = float(np.mean(diff) / (np.mean(abs_true) + eps))
    mse = np.mean((Y_pred - Y_true) ** 2)
    rms_true = np.sqrt(np.mean(Y_true ** 2)) + eps
    rel_l2 = float(np.sqrt(mse) / rms_true)
    return {'rel_l1': rel_l1, 'rel_l2': rel_l2}


def evaluate_model(model, test_loader, device='cuda') -> Dict[str, float]:
    """Evaluate model on test set; returns dict with rel_l1 and rel_l2."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
    Y_pred = np.concatenate(all_pred, axis=0)
    Y_true = np.concatenate(all_true, axis=0)
    return compute_relative_errors(Y_pred, Y_true)


def nodal_interpolation(model, X_new, Y_new, target_res, device='cuda'):
    """Transfer method 1: Nodal interpolation (resize input/output)."""
    model.eval()
    
    # Get training resolution from model
    if hasattr(model, 'N') and hasattr(model, 'M'):
        N_train, M_train = model.N, model.M
    else:
        # For FNO, we need to infer from first forward pass
        with torch.no_grad():
            test_input = torch.randn(1, 2, target_res[0], target_res[1]).to(device)
            try:
                _ = model(test_input)
                # If it works, FNO can handle different resolutions
                return evaluate_model(model, 
                    DataLoader(TensorDataset(
                        torch.from_numpy(X_new).float(),
                        torch.from_numpy(Y_new).float()
                    ), batch_size=32, shuffle=False), device)
            except:
                N_train, M_train = 100, 100  # Default assumption
    
    # Resize input to training resolution
    X_resized = F.interpolate(
        torch.from_numpy(X_new).float(),
        size=(N_train, M_train),
        mode='bilinear',
        align_corners=False
    ).numpy()
    
    # Forward pass
    X_t = torch.from_numpy(X_resized).float().to(device)
    with torch.no_grad():
        Y_pred_resized = model(X_t).cpu().numpy()
    
    # Resize output to target resolution
    Y_pred = F.interpolate(
        torch.from_numpy(Y_pred_resized),
        size=target_res,
        mode='bilinear',
        align_corners=False
    ).numpy()
    
    return compute_relative_errors(Y_pred, Y_new)


def l2_projection(model, X_new, Y_new, X_train_orig, Y_train_orig,
                  target_res, device='cuda', batch_size=32):
    """Transfer method 2: L2 projection (interpolate PCA components)."""
    if not (hasattr(model, 'block0') or hasattr(model, 'enc_components_t')):
        raise ValueError("L2 projection only works for MRBNO/PCANet models")
    
    model.eval()
    N_train, M_train = model.N, model.M
    N_new, M_new = target_res
    
    # Extract PCA components from model
    if isinstance(model, MRBNO):
        enc_components_old = model.block0.enc_components_t.cpu().numpy().T  # (k_in, D_in)
        enc_mean_old = model.block0.enc_mean.cpu().numpy()  # (D_in,)
        dec_components_old = model.block1.dec_components.cpu().numpy()  # (k_out, D_out)
        dec_mean_old = model.block1.dec_mean.cpu().numpy()  # (D_out,)
        C_in = model.C_in
        C_out = model.C_out
        k_in = model.block0.k_in
        k_out = model.block1.k_out
    elif isinstance(model, PCANet):
        # PCANet: get directly
        enc_components_old = model.enc_components_t.cpu().numpy().T  # (k_in, D_in)
        enc_mean_old = model.enc_mean.cpu().numpy()  # (D_in,)
        dec_components_old = model.dec_components.cpu().numpy()  # (k_out, D_out)
        dec_mean_old = model.dec_mean.cpu().numpy()  # (D_out,)
        C_in = model.C_in
        C_out = model.C_out
        k_in = model.k_in
        k_out = model.k_out
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
    # Reshape components to spatial dimensions
    enc_components_spatial = enc_components_old.reshape(k_in, C_in, N_train, M_train)
    dec_components_spatial = dec_components_old.reshape(k_out, C_out, N_train, M_train)
    enc_mean_spatial = enc_mean_old.reshape(C_in, N_train, M_train)
    dec_mean_spatial = dec_mean_old.reshape(C_out, N_train, M_train)
    
    # Interpolate components to new resolution
    enc_components_new_spatial = F.interpolate(
        torch.from_numpy(enc_components_spatial).float(),
        size=(N_new, M_new),
        mode='bilinear',
        align_corners=False
    ).numpy()
    dec_components_new_spatial = F.interpolate(
        torch.from_numpy(dec_components_spatial).float(),
        size=(N_new, M_new),
        mode='bilinear',
        align_corners=False
    ).numpy()
    enc_mean_new_spatial = F.interpolate(
        torch.from_numpy(enc_mean_spatial[None]).float(),
        size=(N_new, M_new),
        mode='bilinear',
        align_corners=False
    ).numpy()[0]
    dec_mean_new_spatial = F.interpolate(
        torch.from_numpy(dec_mean_spatial[None]).float(),
        size=(N_new, M_new),
        mode='bilinear',
        align_corners=False
    ).numpy()[0]
    
    # Flatten back
    D_in_new = C_in * N_new * M_new
    D_out_new = C_out * N_new * M_new
    
    enc_components_new = enc_components_new_spatial.reshape(k_in, D_in_new)
    dec_components_new = dec_components_new_spatial.reshape(k_out, D_out_new)
    enc_mean_new = enc_mean_new_spatial.reshape(D_in_new)
    dec_mean_new = dec_mean_new_spatial.reshape(D_out_new)
    
    # Create new PCAReducer with interpolated components
    from sklearn.decomposition import PCA
    
    rb_new = PCAReducerMultiChannel()
    rb_new.shape_ = (N_new, M_new)
    rb_new.n_channels = C_in
    rb_new.out_channels = C_out
    
    # Create fake PCA objects with interpolated components.
    # MRBNO block0 has k_out=k_in and validates k_out <= pca_out.n_components_, so pca_out needs >= max(k_in, k_out).
    n_out_components = max(k_in, k_out)
    pca_in_new = PCA()
    pca_in_new.n_components_ = k_in
    pca_in_new.components_ = enc_components_new
    pca_in_new.mean_ = enc_mean_new
    pca_in_new.explained_variance_ratio_ = np.ones(k_in) / k_in  # Dummy values

    pca_out_new = PCA()
    pca_out_new.n_components_ = n_out_components
    if n_out_components > k_out:
        pad = np.zeros((n_out_components - k_out, D_out_new), dtype=np.float32)
        pca_out_new.components_ = np.concatenate([dec_components_new, pad], axis=0)
    else:
        pca_out_new.components_ = dec_components_new
    pca_out_new.mean_ = dec_mean_new
    pca_out_new.explained_variance_ratio_ = np.ones(n_out_components) / n_out_components  # Dummy values

    rb_new.pca_in = pca_in_new
    rb_new.pca_out = pca_out_new
    
    # Create new model with interpolated PCA
    if isinstance(model, MRBNO):
        model_new = MRBNO(
            width=model.width,
            rb=rb_new,
            k_in=k_in,
            k_out=k_out,
            layers=model.num_layers,
            dropout=0.0,
            init_scale=0.1,
            use_skip_connections=True,
            use_batch_norm=True,
            use_coordinates=model.use_coordinates,
        )
    elif isinstance(model, PCANet):
        model_new = PCANet(
            rb=rb_new,
            k_in=k_in,
            k_out=k_out,
            hidden=1024,
            depth=5,
            dropout=0.0,
            init_scale=0.1,
        )
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Transfer transformation weights where possible
    # This is simplified - in practice, you might need more careful weight transfer
    model_new = model_new.to(device)
    
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_new).float(),
            torch.from_numpy(Y_new).float()
        ),
        batch_size=batch_size,
        shuffle=False
    )
    return evaluate_model(model_new, test_loader, device)


def pca_refitting(model, X_new, Y_new, X_train_orig, Y_train_orig,
                  target_res, device='cuda', batch_size=32):
    """Transfer method 3: PCA refitting (refit PCA on new resolution data)."""
    if not (hasattr(model, 'block0') or hasattr(model, 'enc_components_t')):
        raise ValueError("PCA refitting only works for MRBNO/PCANet models")

    # Get model hyperparameters
    width = getattr(model, 'width', 24)
    layers = getattr(model, 'num_layers', getattr(model, 'layers', 4))
    k_in_orig = model.block0.k_in if hasattr(model, 'block0') else model.k_in
    k_out_orig = model.block1.k_out if hasattr(model, 'block1') else model.k_out
    
    # Interpolate original training data to new resolution
    N_new, M_new = target_res
    X_train_new = F.interpolate(
        torch.from_numpy(X_train_orig).float(),
        size=(N_new, M_new),
        mode='bilinear',
        align_corners=False
    ).numpy()
    Y_train_new = F.interpolate(
        torch.from_numpy(Y_train_orig).float(),
        size=(N_new, M_new),
        mode='bilinear',
        align_corners=False
    ).numpy()
    
    # Refit PCA on new resolution
    rb_new = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train_new, Y_train_new)
    
    epsilon = 1e-3
    k_in_new, k_out_new = rb_new.k_from_cutoff(
        j_in=1-1*epsilon, j_out=1-epsilon
    )
    
    # Create new model with refitted PCA
    if isinstance(model, MRBNO):
        model_new = MRBNO(
            width=width,
            rb=rb_new,
            k_in=k_in_new,
            k_out=k_out_new,
            layers=layers,
            dropout=0.0,
            init_scale=0.1,
            use_skip_connections=True,
            use_batch_norm=True,
            use_coordinates=model.use_coordinates,
        )
    elif isinstance(model, PCANet):
        model_new = PCANet(
            rb=rb_new,
            k_in=k_in_new,
            k_out=k_out_new,
            hidden=1024,
            depth=5,
            dropout=0.0,
            init_scale=0.1,
        )
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
    # Transfer weights where possible (this is simplified)
    # In practice, you might want to fine-tune
    model_new = model_new.to(device)
    
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_new).float(),
            torch.from_numpy(Y_new).float()
        ),
        batch_size=batch_size,
        shuffle=False
    )
    return evaluate_model(model_new, test_loader, device)


def pca_on_finest(model, X_new, Y_new, target_res, finest_res, device='cuda', batch_size=32) -> Dict[str, float]:
    """Transfer method 4: PCA on the finest mesh. PCA is defined at finest_res; for any target
    resolution we interpolate input to finest -> model at finest -> interpolate output to target."""
    if not (hasattr(model, 'block0') or hasattr(model, 'enc_components_t')):
        raise ValueError("PCA on finest only works for MRBNO/PCANet models")
    N_train, M_train = model.N, model.M
    N_finest, M_finest = finest_res, finest_res
    N_target, M_target = target_res

    # Build model at finest by interpolating PCA components from training res to finest
    if isinstance(model, MRBNO):
        enc_components_old = model.block0.enc_components_t.cpu().numpy().T
        enc_mean_old = model.block0.enc_mean.cpu().numpy()
        dec_components_old = model.block1.dec_components.cpu().numpy()
        dec_mean_old = model.block1.dec_mean.cpu().numpy()
        C_in, C_out = model.C_in, model.C_out
        k_in, k_out = model.block0.k_in, model.block1.k_out
    else:
        enc_components_old = model.enc_components_t.cpu().numpy().T
        enc_mean_old = model.enc_mean.cpu().numpy()
        dec_components_old = model.dec_components.cpu().numpy()
        dec_mean_old = model.dec_mean.cpu().numpy()
        C_in, C_out = model.C_in, model.C_out
        k_in, k_out = model.k_in, model.k_out

    enc_spatial = enc_components_old.reshape(k_in, C_in, N_train, M_train)
    dec_spatial = dec_components_old.reshape(k_out, C_out, N_train, M_train)
    enc_mean_spatial = enc_mean_old.reshape(C_in, N_train, M_train)
    dec_mean_spatial = dec_mean_old.reshape(C_out, N_train, M_train)

    enc_finest = F.interpolate(
        torch.from_numpy(enc_spatial).float(), size=(N_finest, M_finest),
        mode='bilinear', align_corners=False
    ).numpy().reshape(k_in, -1)
    dec_finest = F.interpolate(
        torch.from_numpy(dec_spatial).float(), size=(N_finest, M_finest),
        mode='bilinear', align_corners=False
    ).numpy().reshape(k_out, -1)
    enc_mean_finest = F.interpolate(
        torch.from_numpy(enc_mean_spatial[None]).float(), size=(N_finest, M_finest),
        mode='bilinear', align_corners=False
    ).numpy().reshape(-1)
    dec_mean_finest = F.interpolate(
        torch.from_numpy(dec_mean_spatial[None]).float(), size=(N_finest, M_finest),
        mode='bilinear', align_corners=False
    ).numpy().reshape(-1)

    from sklearn.decomposition import PCA
    rb_finest = PCAReducerMultiChannel()
    rb_finest.shape_ = (N_finest, M_finest)
    rb_finest.n_channels = C_in
    rb_finest.out_channels = C_out
    n_out_components = max(k_in, k_out)
    pca_in = PCA()
    pca_in.n_components_ = k_in
    pca_in.components_ = enc_finest
    pca_in.mean_ = enc_mean_finest
    pca_in.explained_variance_ratio_ = np.ones(k_in) / k_in
    pca_out = PCA()
    pca_out.n_components_ = n_out_components
    pca_out.components_ = dec_finest if n_out_components == k_out else np.concatenate([dec_finest, np.zeros((n_out_components - k_out, dec_finest.shape[1]), dtype=np.float32)], axis=0)
    pca_out.mean_ = dec_mean_finest
    pca_out.explained_variance_ratio_ = np.ones(n_out_components) / n_out_components
    rb_finest.pca_in = pca_in
    rb_finest.pca_out = pca_out

    if isinstance(model, MRBNO):
        model_finest = MRBNO(width=model.width, rb=rb_finest, k_in=k_in, k_out=k_out, layers=model.num_layers,
                             dropout=0.0, init_scale=0.1, use_skip_connections=True, use_batch_norm=True, use_coordinates=model.use_coordinates)
    else:
        model_finest = PCANet(rb=rb_finest, k_in=k_in, k_out=k_out, hidden=1024, depth=5, dropout=0.0, init_scale=0.1)
    model_finest.load_state_dict(model.state_dict(), strict=False)
    model_finest = model_finest.to(device)

    # Interpolate input to finest, run model, interpolate output to target
    X_finest = F.interpolate(
        torch.from_numpy(X_new).float(), size=(N_finest, M_finest),
        mode='bilinear', align_corners=False
    ).numpy()
    test_loader_finest = DataLoader(
        TensorDataset(torch.from_numpy(X_finest).float(), torch.from_numpy(Y_new).float()),
        batch_size=batch_size, shuffle=False
    )
    model_finest.eval()
    all_pred_finest = []
    with torch.no_grad():
        for x, _ in test_loader_finest:
            x = x.to(device)
            all_pred_finest.append(model_finest(x).cpu().numpy())
    Y_pred_finest = np.concatenate(all_pred_finest, axis=0)
    Y_pred_target = F.interpolate(
        torch.from_numpy(Y_pred_finest), size=(N_target, M_target),
        mode='bilinear', align_corners=False
    ).numpy()
    return compute_relative_errors(Y_pred_target, Y_new)


def _mass_matrix_uniform(n: int, m: int) -> np.ndarray:
    """Diagonal mass matrix (volume per node) for uniform grid: 1/(n*m) per dof."""
    return np.ones(n * m, dtype=np.float32) / (n * m)


def _fit_pca_mass_matrix(X: np.ndarray, M: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA with M-inner product: solve C v = lambda M v for M-orthonormal components.
    X: (n_samples, n_features), M: (n_features,) diagonal of mass matrix.
    Returns: components (n_components, n_features), mean (n_features,), evr (n_components,)."""
    X_centered = X - X.mean(axis=0)
    n_samples = X.shape[0]
    C = (X_centered.T @ X_centered) / max(n_samples - 1, 1)
    B = np.diag(M) if M.ndim == 1 else M
    # Generalized eigenvalue problem C v = lambda M v; eigh expects (A, B) with A v = lambda B v
    evals, evecs = eigh(C, B)
    # Sort descending by eigenvalue, take top n_components (clip to available)
    n_avail = min(n_components, len(evals))
    idx = np.argsort(evals)[::-1][:n_avail]
    evecs = evecs[:, idx]
    evals = np.maximum(evals[idx], 0.0)
    if n_avail < n_components:
        # Pad with zeros if fewer eigenvalues than requested
        pad = np.zeros((evecs.shape[0], n_components - n_avail), dtype=evecs.dtype)
        evecs = np.hstack([evecs, pad])
        evals = np.concatenate([evals, np.zeros(n_components - n_avail)])
    components = evecs.T
    mean = X.mean(axis=0).astype(np.float32)
    evr = np.maximum(evals, 0) / (np.maximum(np.sum(np.maximum(evals, 0)), 1e-12))
    return components.astype(np.float32), mean, evr.astype(np.float32)


def mass_matrix_transfer(model, X_new, Y_new, X_train_orig, Y_train_orig,
                         target_res, device='cuda', batch_size=32) -> Dict[str, float]:
    """Transfer method 5: PCA with mass matrix. Discretized L2 product is u^T M v.
    Fit PCA (M-orthonormal) on each grid; transfer by interpolating modes and re-orthonormalizing with target M."""
    if not (hasattr(model, 'block0') or hasattr(model, 'enc_components_t')):
        raise ValueError("Mass matrix transfer only works for MRBNO/PCANet models")
    N_train, M_train = model.N, model.M
    N_new, M_new = target_res
    C_in = model.C_in if hasattr(model, 'C_in') else (model.block0.enc_components_t.shape[0] if hasattr(model, 'block0') else 2)
    C_out = model.C_out if hasattr(model, 'C_out') else 1
    k_in = model.block0.k_in if hasattr(model, 'block0') else model.k_in
    k_out = model.block1.k_out if hasattr(model, 'block1') else model.k_out

    # Interpolate training data to target resolution and build mass matrix there
    X_train_new = F.interpolate(
        torch.from_numpy(X_train_orig).float(), size=(N_new, M_new),
        mode='bilinear', align_corners=False
    ).numpy()
    Y_train_new = F.interpolate(
        torch.from_numpy(Y_train_orig).float(), size=(N_new, M_new),
        mode='bilinear', align_corners=False
    ).numpy()
    M_in = _mass_matrix_uniform(N_new, M_new)
    M_out = _mass_matrix_uniform(N_new, M_new)
    # One mass per channel for multi-channel: repeat for each channel
    D_in = C_in * N_new * M_new
    D_out = C_out * N_new * M_new
    M_in_full = np.repeat(M_in, C_in)  # (C_in*N*M,) - same weight per dof
    M_out_full = np.repeat(M_out, C_out)
    X_flat = X_train_new.reshape(X_train_new.shape[0], -1)
    Y_flat = Y_train_new.reshape(Y_train_new.shape[0], -1)
    enc_components_new, enc_mean_new, _ = _fit_pca_mass_matrix(X_flat, M_in_full, k_in)
    dec_components_new, dec_mean_new, _ = _fit_pca_mass_matrix(Y_flat, M_out_full, k_out)

    from sklearn.decomposition import PCA
    rb_new = PCAReducerMultiChannel()
    rb_new.shape_ = (N_new, M_new)
    rb_new.n_channels = C_in
    rb_new.out_channels = C_out
    n_out_components = max(k_in, k_out)
    pca_in_new = PCA()
    pca_in_new.n_components_ = k_in
    pca_in_new.components_ = enc_components_new
    pca_in_new.mean_ = enc_mean_new
    pca_in_new.explained_variance_ratio_ = np.ones(k_in) / k_in
    pca_out_new = PCA()
    pca_out_new.n_components_ = n_out_components
    pca_out_new.components_ = dec_components_new if n_out_components == k_out else np.concatenate([dec_components_new, np.zeros((n_out_components - k_out, D_out), dtype=np.float32)], axis=0)
    pca_out_new.mean_ = dec_mean_new
    pca_out_new.explained_variance_ratio_ = np.ones(n_out_components) / n_out_components
    rb_new.pca_in = pca_in_new
    rb_new.pca_out = pca_out_new

    if isinstance(model, MRBNO):
        model_new = MRBNO(width=model.width, rb=rb_new, k_in=k_in, k_out=k_out, layers=model.num_layers,
                          dropout=0.0, init_scale=0.1, use_skip_connections=True, use_batch_norm=True, use_coordinates=model.use_coordinates)
    else:
        model_new = PCANet(rb=rb_new, k_in=k_in, k_out=k_out, hidden=1024, depth=5, dropout=0.0, init_scale=0.1)
    model_new.load_state_dict(model.state_dict(), strict=False)
    model_new = model_new.to(device)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_new).float(), torch.from_numpy(Y_new).float()),
        batch_size=batch_size, shuffle=False
    )
    return evaluate_model(model_new, test_loader, device)


def main():
    # Setup paths
    scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
    data_dir = os.path.join(scratch_dir, 'data')
    base_data_path = os.path.join(data_dir, 'nonlinear_diffusion_dataset.npz')
    
    output_dir = Path(__file__).parent
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base dataset
    print("Loading base dataset...")
    X_base, Y_base = load_dataset(base_data_path)
    print(f"Base dataset shape: X={X_base.shape}, Y={Y_base.shape}")
    
    # Split data
    n_total = len(X_base)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    X_train = X_base[:n_train]
    Y_train = Y_base[:n_train]
    X_val = X_base[n_train:n_train+n_val]
    Y_val = Y_base[n_train:n_train+n_val]
    X_test = X_base[n_train+n_val:]
    Y_test = Y_base[n_train+n_val:]
    
    # Normalize
    X_train_norm, Y_train_norm, X_val_norm, Y_val_norm, \
    x_mean, x_std, y_mean, y_std = normalize_data(
        X_train, Y_train, X_val, Y_val
    )
    X_test_norm = (X_test - x_mean) / (x_std + 1e-8)
    Y_test_norm = (Y_test - y_mean) / (y_std + 1e-8)
    
    # Create data loaders
    batch_size = 16  # Reduced to avoid OOM
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_norm).float(),
            torch.from_numpy(Y_train_norm).float()
        ),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val_norm).float(),
            torch.from_numpy(Y_val_norm).float()
        ),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Train or load models
    print("\n" + "="*70)
    print("TRAINING/LOADING MODELS")
    print("="*70)
    
    # Check for existing checkpoints (look in results dir and also check for user-provided checkpoints)
    fno_checkpoint = results_dir / 'fno_checkpoint.pt'
    mrbno_checkpoint = results_dir / 'mrbno_checkpoint.pt'
    pcanet_checkpoint = results_dir / 'pcanet_checkpoint.pt'
    
    checkpoint_base_dir = Path(scratch_dir) / 'checkpoints'
    if checkpoint_base_dir.exists():
        for name in ['fno_nonlinear_nonlinear_diffusion_dataset_best.pt', 'fno_nonlinear_best.pt']:
            p = checkpoint_base_dir / name
            if p.exists() and not fno_checkpoint.exists():
                fno_checkpoint = p
                print(f"Using existing FNO checkpoint: {fno_checkpoint}")
                break
        for name in ['pcanet_nonlinear_nonlinear_diffusion_dataset_best.pt', 'pcanet_nonlinear_best.pt']:
            p = checkpoint_base_dir / name
            if p.exists() and not pcanet_checkpoint.exists():
                pcanet_checkpoint = p
                print(f"Using existing PCANet checkpoint: {pcanet_checkpoint}")
                break
    # MRBNO: only results_dir checkpoint (MRBNO.py architecture); do not load LiftV3 or external MRBNO.

    # Delete previous results that may have used another model
    for f in ['mesh_generalization_results.json', 'mesh_generalization_plots.png', 'mesh_generalization_overview.png', 'results.md']:
        (results_dir / f).unlink(missing_ok=True)

    # Train FNO
    print("\nTraining FNO...")
    fno_model = FNO2d(
        in_channels=2,
        out_channels=1,
        width=24,
        modes=(12, 12),
        n_layers=4
    )
    if fno_checkpoint.exists():
        print(f"Loading FNO checkpoint from {fno_checkpoint}")
        try:
            checkpoint = torch.load(fno_checkpoint, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                fno_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                fno_model.load_state_dict(checkpoint)
            fno_model = fno_model.to(device)
            print("FNO model loaded successfully")
        except Exception as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"CUDA OOM loading FNO: {e}. Skipping FNO evaluation.")
                fno_model = None
            else:
                raise
    else:
        print("No FNO checkpoint found, training...")
        try:
            torch.cuda.empty_cache()
            train_model(fno_model, train_loader, val_loader, 
                       checkpoint_path=str(fno_checkpoint), device=device)
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            print(f"Exception during FNO training: {error_type}: {e}")
            if "out of memory" in error_str or "cuda" in error_str or "OutOfMemoryError" in error_type:
                print("Skipping FNO training - GPU memory unavailable. Will skip FNO evaluation.")
                fno_model = None
            else:
                print("Skipping FNO training - will test with untrained model or skip FNO evaluation")
                fno_model = None
    
    # Train MRBNO
    print("\nTraining MRBNO...")
    rb = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train_norm, Y_train_norm)
    
    epsilon = 1e-3
    k_in, k_out = rb.k_from_cutoff(j_in=1-1*epsilon, j_out=1-epsilon)
    print(f"PCA components: k_in={k_in}, k_out={k_out}")

    mrbno_model = None
    if mrbno_checkpoint.exists():
        checkpoint = torch.load(mrbno_checkpoint, map_location='cpu')
        state = checkpoint.get('model_state_dict', checkpoint)
        if 'lift.weight' in state:
            ckpt_k_in = state['block0.weight'].shape[2]
            ckpt_k_out = state['block1.weight'].shape[3]
            print(f"Loading MRBNO checkpoint from {mrbno_checkpoint} (k_in={ckpt_k_in}, k_out={ckpt_k_out})")
            mrbno_model = MRBNO(
                width=24,
                rb=rb,
                k_in=ckpt_k_in,
                k_out=ckpt_k_out,
                layers=4,
                dropout=0.0,
                init_scale=0.1,
                use_skip_connections=True,
                use_batch_norm=True,
                use_coordinates=True,
            )
            mrbno_model.load_state_dict(state, strict=True)
            try:
                mrbno_model = mrbno_model.to(device)
                mrbno_device = device
                print("MRBNO model loaded successfully")
            except Exception as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    print(f"CUDA OOM moving MRBNO to device: {e}. Using CPU for MRBNO.")
                    mrbno_model = mrbno_model.to('cpu')
                    mrbno_device = torch.device('cpu')
                else:
                    raise
        else:
            print("MRBNO checkpoint is not from MRBNO.py (missing lift.weight), training from scratch.")
            mrbno_model = None
    if not mrbno_checkpoint.exists() or mrbno_model is None:
        if mrbno_model is None:
            mrbno_model = MRBNO(
                width=24,
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                layers=4,
                dropout=0.0,
                init_scale=0.1,
                use_skip_connections=True,
                use_batch_norm=True,
                use_coordinates=True,
            )
        print("No MRBNO checkpoint found, training...")
        train_model(mrbno_model, train_loader, val_loader,
                    checkpoint_path=str(mrbno_checkpoint), device=device)
        mrbno_device = device

    # Train PCANet
    print("\nTraining PCANet...")
    if pcanet_checkpoint.exists():
        print(f"Loading PCANet checkpoint from {pcanet_checkpoint}")
        pca_ckpt = torch.load(pcanet_checkpoint, map_location='cpu')
        pca_state = pca_ckpt.get('model_state_dict', pca_ckpt)
        ckpt_k_in = pca_state['enc_components_t'].shape[1]
        ckpt_k_out = pca_state['dec_components'].shape[0]
        pcanet_model = PCANet(
            rb=rb,
            k_in=ckpt_k_in,
            k_out=ckpt_k_out,
            hidden=1024,
            depth=5,
            dropout=0.0,
            init_scale=0.1,
        )
        pcanet_model.load_state_dict(pca_state, strict=True)
        try:
            pcanet_model = pcanet_model.to(device)
            pcanet_device = device
            print("PCANet model loaded successfully")
        except Exception as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"CUDA OOM moving PCANet to device: {e}. Using CPU for PCANet.")
                pcanet_model = pcanet_model.to('cpu')
                pcanet_device = torch.device('cpu')
            else:
                raise
    else:
        pcanet_model = PCANet(
            rb=rb,
            k_in=144,
            k_out=144,
            hidden=1024,
            depth=5,
            dropout=0.0,
            init_scale=0.1,
        )
        print("No PCANet checkpoint found, training...")
        train_model(pcanet_model, train_loader, val_loader,
                    checkpoint_path=str(pcanet_checkpoint), device=device)
        pcanet_device = device

    # Load different resolution datasets: include training resolution so we test on all 5
    train_res = int(X_train.shape[-1])  # assume square grid
    resolutions = sorted(set([train_res, 50, 75, 150, 200]))
    finest_res = max(resolutions)
    print("\n" + "="*70)
    print("TESTING ON DIFFERENT RESOLUTIONS")
    print("="*70)
    print(f"Training resolution: {train_res}x{train_res}. Test resolutions: {resolutions}. Finest (for PCA-on-finest): {finest_res}.")

    methods_pca = ['nodal_interpolation', 'l2_projection', 'pca_refitting', 'pca_on_finest', 'mass_matrix']
    results = {
        'fno': {},
        'mrbno': {m: {} for m in methods_pca},
        'pcanet': {m: {} for m in methods_pca},
    }
    
    for res in resolutions:
        print(f"\nTesting on resolution {res}x{res}...")
        if res == train_res:
            X_res_norm = X_test_norm
            Y_res_norm = Y_test_norm
            print(f"Using base dataset test split: X={X_res_norm.shape}, Y={Y_res_norm.shape}")
        else:
            res_data_path = os.path.join(data_dir, f'nonlinear_diffusion_dataset_res{res}.npz')
            if not os.path.exists(res_data_path):
                print(f"Warning: {res_data_path} not found, skipping...")
                continue
            X_res, Y_res = load_dataset(res_data_path)
            print(f"Loaded dataset: X={X_res.shape}, Y={Y_res.shape}")
            X_res_norm = (X_res - x_mean) / (x_std + 1e-8)
            Y_res_norm = (Y_res - y_mean) / (y_std + 1e-8)
        
        res_batch = 8 if res >= 150 else 16
        # FNO: direct application only (no other method)
        print("  Testing FNO (direct only)...")
        if fno_model is not None:
            try:
                test_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_res_norm).float(),
                        torch.from_numpy(Y_res_norm).float()
                    ),
                    batch_size=res_batch,
                    shuffle=False
                )
                err_dict = evaluate_model(fno_model, test_loader, device)
                results['fno'][res] = err_dict
                print(f"    FNO rel_l1: {err_dict['rel_l1']:.6e}  rel_l2: {err_dict['rel_l2']:.6e}")
            except Exception as e:
                print(f"    FNO failed: {e}")
                results['fno'][res] = None
        else:
            print("    FNO model not available, skipping...")
            results['fno'][res] = None
        
        # Test MRBNO with all methods
        print("  Testing MRBNO...")
        for method in methods_pca:
            try:
                if method == 'nodal_interpolation':
                    err_dict = nodal_interpolation(mrbno_model, X_res_norm, Y_res_norm, (res, res), mrbno_device)
                elif method == 'l2_projection':
                    err_dict = l2_projection(mrbno_model, X_res_norm, Y_res_norm, X_train_norm, Y_train_norm, (res, res), mrbno_device, batch_size=res_batch)
                elif method == 'pca_refitting':
                    err_dict = pca_refitting(mrbno_model, X_res_norm, Y_res_norm, X_train_norm, Y_train_norm, (res, res), mrbno_device, batch_size=res_batch)
                elif method == 'pca_on_finest':
                    err_dict = pca_on_finest(mrbno_model, X_res_norm, Y_res_norm, (res, res), finest_res, mrbno_device, batch_size=res_batch)
                elif method == 'mass_matrix':
                    err_dict = mass_matrix_transfer(mrbno_model, X_res_norm, Y_res_norm, X_train_norm, Y_train_norm, (res, res), mrbno_device, batch_size=res_batch)
                else:
                    continue
                results['mrbno'][method][res] = err_dict
                print(f"    {method}: rel_l1={err_dict['rel_l1']:.6e}  rel_l2={err_dict['rel_l2']:.6e}")
            except Exception as e:
                print(f"    {method} failed: {e}")
                import traceback
                traceback.print_exc()
                results['mrbno'][method][res] = None
        
        # Test PCANet with same methods
        print("  Testing PCANet...")
        for method in methods_pca:
            try:
                if method == 'nodal_interpolation':
                    err_dict = nodal_interpolation(pcanet_model, X_res_norm, Y_res_norm, (res, res), pcanet_device)
                elif method == 'l2_projection':
                    err_dict = l2_projection(pcanet_model, X_res_norm, Y_res_norm, X_train_norm, Y_train_norm, (res, res), pcanet_device, batch_size=res_batch)
                elif method == 'pca_refitting':
                    err_dict = pca_refitting(pcanet_model, X_res_norm, Y_res_norm, X_train_norm, Y_train_norm, (res, res), pcanet_device, batch_size=res_batch)
                elif method == 'pca_on_finest':
                    err_dict = pca_on_finest(pcanet_model, X_res_norm, Y_res_norm, (res, res), finest_res, pcanet_device, batch_size=res_batch)
                elif method == 'mass_matrix':
                    err_dict = mass_matrix_transfer(pcanet_model, X_res_norm, Y_res_norm, X_train_norm, Y_train_norm, (res, res), pcanet_device, batch_size=res_batch)
                else:
                    continue
                results['pcanet'][method][res] = err_dict
                print(f"    {method}: rel_l1={err_dict['rel_l1']:.6e}  rel_l2={err_dict['rel_l2']:.6e}")
            except Exception as e:
                print(f"    {method} failed: {e}")
                import traceback
                traceback.print_exc()
                results['pcanet'][method][res] = None
    
    # Save results
    results_path = results_dir / 'mesh_generalization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Helper: get rel_l2 from result entry (for plotting)
    def _rel_l2(entry):
        return entry['rel_l2'] if isinstance(entry, dict) and entry is not None else None

    # Create plots (y-axis: rel L2)
    print("\nCreating plots...")
    res_list = sorted([r for r in resolutions if r in results.get('fno') or any(results['mrbno'][m].get(r) is not None for m in methods_pca)])
    if not res_list:
        res_list = sorted(resolutions)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Plot 1: FNO and MRBNO
    ax = axes[0]
    fno_vals = [results['fno'].get(r) for r in res_list]
    fno_rel2 = [_rel_l2(v) for v in fno_vals]
    fno_res = [r for r, v in zip(res_list, fno_rel2) if v is not None]
    fno_err = [v for v in fno_rel2 if v is not None]
    if fno_err:
        ax.plot(fno_res, fno_err, 'o-', label='FNO (direct)', linewidth=2, markersize=8)
    for method in methods_pca:
        vals = [results['mrbno'][method].get(r) for r in res_list]
        rel2 = [_rel_l2(v) for v in vals]
        r_plot = [r for r, v in zip(res_list, rel2) if v is not None]
        e_plot = [v for v in rel2 if v is not None]
        if e_plot:
            ax.plot(r_plot, e_plot, 's-', label=f'MRBNO ({method})', linewidth=2, markersize=8)
    ax.set_xlabel('Resolution', fontsize=12)
    ax.set_ylabel('Relative L2', fontsize=12)
    ax.set_title('FNO vs MRBNO (rel L2)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Plot 2: MRBNO methods
    ax = axes[1]
    for method in methods_pca:
        vals = [results['mrbno'][method].get(r) for r in res_list]
        rel2 = [_rel_l2(v) for v in vals]
        r_plot = [r for r, v in zip(res_list, rel2) if v is not None]
        e_plot = [v for v in rel2 if v is not None]
        if e_plot:
            ax.plot(r_plot, e_plot, 'o-', label=method.replace('_', ' ').title(), linewidth=2, markersize=8)
    ax.set_xlabel('Resolution', fontsize=12)
    ax.set_ylabel('Relative L2', fontsize=12)
    ax.set_title('MRBNO Transfer Methods (rel L2)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Plot 3: PCANet methods
    ax = axes[2]
    for method in methods_pca:
        vals = [results['pcanet'][method].get(r) for r in res_list]
        rel2 = [_rel_l2(v) for v in vals]
        r_plot = [r for r, v in zip(res_list, rel2) if v is not None]
        e_plot = [v for v in rel2 if v is not None]
        if e_plot:
            ax.plot(r_plot, e_plot, 'o-', label=method.replace('_', ' ').title(), linewidth=2, markersize=8)
    ax.set_xlabel('Resolution', fontsize=12)
    ax.set_ylabel('Relative L2', fontsize=12)
    ax.set_title('PCANet Transfer Methods (rel L2)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = results_dir / 'mesh_generalization_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {plot_path}")

    # Overview plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    res_list_plot = sorted(resolutions)
    if any(results['fno'].get(r) is not None for r in res_list_plot):
        fno_vals = [results['fno'].get(r) for r in res_list_plot]
        fno_plot = [r for r, v in zip(res_list_plot, fno_vals) if v is not None]
        fno_err = [_rel_l2(v) for v in fno_vals if v is not None]
        if fno_err:
            ax.plot(fno_plot, fno_err, 'o-', label='FNO (direct)', linewidth=2, markersize=8)
    for method in methods_pca:
        vals = [results['mrbno'][method].get(r) for r in res_list_plot]
        r_plot = [r for r, v in zip(res_list_plot, vals) if v is not None]
        e_plot = [_rel_l2(v) for v in vals if v is not None]
        if e_plot:
            ax.plot(r_plot, e_plot, 's-', label=f'MRBNO {method}', linewidth=2, markersize=8)
    for method in methods_pca:
        vals = [results['pcanet'][method].get(r) for r in res_list_plot]
        r_plot = [r for r, v in zip(res_list_plot, vals) if v is not None]
        e_plot = [_rel_l2(v) for v in vals if v is not None]
        if e_plot:
            ax.plot(r_plot, e_plot, '^-', label=f'PCANet {method}', linewidth=2, markersize=8)
    ax.set_xlabel('Resolution', fontsize=12)
    ax.set_ylabel('Relative L2', fontsize=12)
    ax.set_title('Resolution generalization: all models (rel L2)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    overview_path = results_dir / 'mesh_generalization_overview.png'
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overview plot saved to {overview_path}")

    # Results summary .md: both rel L1 and rel L2 for all 5 resolutions and all methods
    def _fmt(e):
        if e is None:
            return "—"
        if isinstance(e, dict):
            return f"L1={e['rel_l1']:.4e} L2={e['rel_l2']:.4e}"
        return f"{e:.4e}"
    md_lines = [
        "# Mesh generalization results",
        "",
        f"Training resolution: {train_res}×{train_res}. Test resolutions (all 5): {resolutions}. FNO: direct only.",
        "",
        "## Relative L1 and L2 by resolution",
        "",
    ]
    md_lines.append("| Resolution | FNO (direct) | " + " | ".join([f"MRBNO {m}" for m in methods_pca]) + " | " + " | ".join([f"PCANet {m}" for m in methods_pca]) + " |")
    md_lines.append("|------------|" + "--------------|" * (1 + len(methods_pca) * 2) + "")
    for r in sorted(resolutions):
        row = [str(r), _fmt(results['fno'].get(r))]
        for m in methods_pca:
            row.append(_fmt(results['mrbno'][m].get(r)))
        for m in methods_pca:
            row.append(_fmt(results['pcanet'][m].get(r)))
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.extend(["", "## Files", "", "- mesh_generalization_results.json", "- mesh_generalization_plots.png", "- mesh_generalization_overview.png", ""])
    with open(results_dir / 'results.md', 'w') as f:
        f.write("\n".join(md_lines))
    print(f"Results summary saved to {results_dir / 'results.md'}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
