#!/usr/bin/env python3
"""
Test mesh generalization: evaluate models trained on one resolution
on different resolutions using various transfer methods.
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import copy
from typing import Dict, Tuple, List, Optional
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

# Script lives at AMOR/mesh_generalization/; repo root is three levels up.
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'AMOR', 'models'))

from AMOR.models.amor import AMOR
from AMOR.models.FNO_model import FNO2d
from AMOR.models.PCANet import PCANet
from AMOR.models.PCAReducer import PCAReducerMultiChannel


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
                checkpoint_path=None, early_stopping_patience=50):
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
    def _rel_l1(pred, target):
        batch = pred.shape[0]
        pred = pred.reshape(batch, -1)
        target = target.reshape(batch, -1)
        return ((pred - target).abs().sum(1) / target.abs().sum(1).clamp_min(1e-8)).mean()
    criterion = _rel_l1
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_losses = []
    val_losses = []
    best_model_state = copy.deepcopy(model.state_dict())

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
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} (patience={early_stopping_patience})")
            break

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    model.load_state_dict(best_model_state)
    model.load_state_dict(best_model_state)
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


# ============================================================
# P1 FEM utilities for mesh-to-mesh transfer
# Exact strategies from scratch/fenics_mesh_transfer_demo_no_dolfin.py:
#   A. Algebraic transfer / prolongation matrix
#   B. Target-space L² projection (sparse solve)
#   C. Same L² projection, dense scipy solve
#   D. Oversampled least-squares / pseudoinverse fit
#   D+. Weighted variant of D
# ============================================================

def _make_mesh(n):
    """UnitSquareMesh(n,n): (n+1)^2 nodes, 2*n^2 triangles on [0,1]^2."""
    def nid(i, j): return i * (n + 1) + j
    nodes = np.array([[i / n, j / n] for i in range(n + 1) for j in range(n + 1)], dtype=float)
    tris = []
    for i in range(n):
        for j in range(n):
            tris.append([nid(i, j), nid(i+1, j), nid(i+1, j+1)])
            tris.append([nid(i, j), nid(i+1, j+1), nid(i, j+1)])
    return nodes, np.array(tris, dtype=int), n


def _tri_area(v):
    return 0.5 * ((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
                - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1]))


def _find_tri(n_mesh, x, y):
    xi = x * n_mesh; yi = y * n_mesh
    i = int(min(xi, n_mesh - 1)); j = int(min(yi, n_mesh - 1))
    fx = xi - i; fy = yi - j
    return 2 * (i * n_mesh + j) if (fx >= fy) else 2 * (i * n_mesh + j) + 1


def _bary(v, x, y):
    T = np.array([[v[1, 0] - v[0, 0], v[2, 0] - v[0, 0]],
                  [v[1, 1] - v[0, 1], v[2, 1] - v[0, 1]]])
    lm = np.linalg.solve(T, np.array([x - v[0, 0], y - v[0, 1]]))
    return np.array([1.0 - lm[0] - lm[1], lm[0], lm[1]])


def _clamp(val, lo=1e-10, hi=None):
    if hi is None: hi = 1.0 - lo
    return min(max(val, lo), hi)


def _eval_p1_batch(nodes, tris, n_mesh, coeffs, pts):
    """Evaluate P1 FEM function at many points."""
    out = np.empty(len(pts))
    for q, (x, y) in enumerate(pts):
        x = _clamp(x); y = _clamp(y)
        tid = _find_tri(n_mesh, x, y)
        tri = tris[tid]
        lm = _bary(nodes[tri], x, y)
        out[q] = float(lm @ coeffs[tri])
    return out


def _assemble_mass(nodes, tris):
    """Consistent P1 mass matrix M_ij = ∫ φ_i φ_j dx."""
    N = len(nodes)
    rows, cols, vals = [], [], []
    for tri in tris:
        v = nodes[tri]
        A = abs(_tri_area(v))
        for p in range(3):
            for q in range(3):
                rows.append(tri[p]); cols.append(tri[q])
                vals.append(A / 12.0 * (2.0 if p == q else 1.0))
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _assemble_rhs(nodes_t, tris_t, nodes_s, tris_s, n_s, coeffs_s):
    """RHS for L² projection: b_i = ∫ u_s(x) φ_i^(t)(x) dx (3-pt edge-midpoint quadrature)."""
    b = np.zeros(len(nodes_t))
    for tri in tris_t:
        v = nodes_t[tri]
        A = abs(_tri_area(v))
        m01 = 0.5 * (v[0] + v[1]); m12 = 0.5 * (v[1] + v[2]); m20 = 0.5 * (v[2] + v[0])

        def _us(pt):
            x = _clamp(pt[0]); y = _clamp(pt[1])
            tid = _find_tri(n_s, x, y)
            t = tris_s[tid]; lm = _bary(nodes_s[t], x, y)
            return float(lm @ coeffs_s[t])

        u01 = _us(m01); u12 = _us(m12); u20 = _us(m20)
        b[tri[0]] += (A / 6.0) * (u01 + u20)
        b[tri[1]] += (A / 6.0) * (u01 + u12)
        b[tri[2]] += (A / 6.0) * (u12 + u20)
    return b


def _create_transfer_matrix(nodes_s, tris_s, n_s, nodes_t):
    """Prolongation matrix P (n_t × n_s): P[i,j] = barycentric weight of source node j for target node i."""
    nt, ns = len(nodes_t), len(nodes_s)
    rows, cols, vals = [], [], []
    for i, (x, y) in enumerate(nodes_t):
        x = _clamp(x); y = _clamp(y)
        tid = _find_tri(n_s, x, y)
        tri = tris_s[tid]; lm = _bary(nodes_s[tri], x, y)
        for k in range(3):
            rows.append(i); cols.append(tri[k]); vals.append(float(lm[k]))
    return sp.csr_matrix((vals, (rows, cols)), shape=(nt, ns))


def _grid_to_mesh(n):
    """Treat n×n pixel grid as a P1 mesh: make_mesh(n-1) gives n^2 nodes, one per pixel."""
    return _make_mesh(n - 1)


def _apply_transfer(model, X_test, Y_test, n_train, transfer_fn, device, batch_size):
    """Run model at n_train resolution (bilinear input), apply FEM transfer_fn to each output field."""
    n_test = X_test.shape[-1]
    if n_test != n_train:
        X_in = F.interpolate(torch.from_numpy(X_test).float(),
                             size=(n_train, n_train), mode='bilinear',
                             align_corners=False).numpy()
    else:
        X_in = X_test

    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_in).float()),
                        batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    Y_pred_train = np.concatenate(preds, axis=0)  # (N, C, n_train, n_train)

    if n_test == n_train:
        Y_pred = Y_pred_train
    else:
        N, C = Y_pred_train.shape[:2]
        Y_pred = np.zeros((N, C, n_test, n_test), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                Y_pred[i, c] = transfer_fn(Y_pred_train[i, c], n_train, n_test)

    return compute_relative_errors(Y_pred, Y_test)


def strategy_algebraic(model, X_new, Y_new, n_train, device, batch_size=32):
    """Method A: algebraic transfer via prolongation matrix (exact from scratch file)."""
    def transfer(field, n_src, n_tgt):
        nodes_s, tris_s, ns = _grid_to_mesh(n_src)
        nodes_t, _, _ = _grid_to_mesh(n_tgt)
        P = _create_transfer_matrix(nodes_s, tris_s, ns, nodes_t)
        return (P @ field.flatten()).reshape(n_tgt, n_tgt).astype(np.float32)
    return _apply_transfer(model, X_new, Y_new, n_train, transfer, device, batch_size)


def strategy_l2_sparse(model, X_new, Y_new, n_train, device, batch_size=32):
    """Method B: target L² projection via sparse mass-matrix solve (exact from scratch file)."""
    def transfer(field, n_src, n_tgt):
        nodes_s, tris_s, ns = _grid_to_mesh(n_src)
        nodes_t, tris_t, _ = _grid_to_mesh(n_tgt)
        M_t = _assemble_mass(nodes_t, tris_t)
        b = _assemble_rhs(nodes_t, tris_t, nodes_s, tris_s, ns, field.flatten())
        return spla.spsolve(M_t, b).reshape(n_tgt, n_tgt).astype(np.float32)
    return _apply_transfer(model, X_new, Y_new, n_train, transfer, device, batch_size)


def strategy_l2_dense(model, X_new, Y_new, n_train, device, batch_size=32):
    """Method C: same L² projection, dense scipy solve (exact from scratch file)."""
    def transfer(field, n_src, n_tgt):
        nodes_s, tris_s, ns = _grid_to_mesh(n_src)
        nodes_t, tris_t, _ = _grid_to_mesh(n_tgt)
        M_t = _assemble_mass(nodes_t, tris_t)
        b = _assemble_rhs(nodes_t, tris_t, nodes_s, tris_s, ns, field.flatten())
        return la.solve(M_t.toarray(), b, assume_a="sym").reshape(n_tgt, n_tgt).astype(np.float32)
    return _apply_transfer(model, X_new, Y_new, n_train, transfer, device, batch_size)


def strategy_pseudoinverse(model, X_new, Y_new, n_train, device, batch_size=32, nx_samp=41):
    """Method D: oversampled pseudoinverse fit (exact from scratch file)."""
    def transfer(field, n_src, n_tgt):
        nodes_s, tris_s, ns = _grid_to_mesh(n_src)
        nodes_t, tris_t, nt = _grid_to_mesh(n_tgt)
        xx = np.linspace(0.0, 1.0, nx_samp)
        pts = np.array([[x, y] for y in xx for x in xx])
        y_samp = _eval_p1_batch(nodes_s, tris_s, ns, field.flatten(), pts)
        n_dofs = len(nodes_t)
        rows_S, cols_S, vals_S = [], [], []
        for q, (x, y) in enumerate(pts):
            x = _clamp(x); y = _clamp(y)
            tid = _find_tri(nt, x, y); tri = tris_t[tid]; lm = _bary(nodes_t[tri], x, y)
            for k in range(3):
                rows_S.append(q); cols_S.append(tri[k]); vals_S.append(float(lm[k]))
        S_t = sp.csr_matrix((vals_S, (rows_S, cols_S)), shape=(len(pts), n_dofs)).toarray()
        return (la.pinv(S_t) @ y_samp).reshape(n_tgt, n_tgt).astype(np.float32)
    return _apply_transfer(model, X_new, Y_new, n_train, transfer, device, batch_size)


def strategy_weighted_pseudoinverse(model, X_new, Y_new, n_train, device, batch_size=32, nx_samp=41):
    """Method D+: weighted least-squares variant (exact from scratch file)."""
    def transfer(field, n_src, n_tgt):
        nodes_s, tris_s, ns = _grid_to_mesh(n_src)
        nodes_t, tris_t, nt = _grid_to_mesh(n_tgt)
        xx = np.linspace(0.0, 1.0, nx_samp)
        pts = np.array([[x, y] for y in xx for x in xx])
        y_samp = _eval_p1_batch(nodes_s, tris_s, ns, field.flatten(), pts)
        n_dofs = len(nodes_t)
        rows_S, cols_S, vals_S = [], [], []
        for q, (x, y) in enumerate(pts):
            x = _clamp(x); y = _clamp(y)
            tid = _find_tri(nt, x, y); tri = tris_t[tid]; lm = _bary(nodes_t[tri], x, y)
            for k in range(3):
                rows_S.append(q); cols_S.append(tri[k]); vals_S.append(float(lm[k]))
        S_t = sp.csr_matrix((vals_S, (rows_S, cols_S)), shape=(len(pts), n_dofs)).toarray()
        dx = 1.0 / (nx_samp - 1)
        w = np.full(len(pts), dx * dx); Wsqrt = np.sqrt(w)
        return la.lstsq(S_t * Wsqrt[:, None], y_samp * Wsqrt)[0].reshape(n_tgt, n_tgt).astype(np.float32)
    return _apply_transfer(model, X_new, Y_new, n_train, transfer, device, batch_size)


def nodal_interpolation(model, X_new, Y_new, target_res, device='cuda'):
    """Transfer method 1: Nodal interpolation (resize input/output)."""
    model.eval()
    
    # Get training resolution from model
    if hasattr(model, 'N') and hasattr(model, 'M'):
        N_train, M_train = model.N, model.M
    else:
        # FNO: infer channel count from first forward pass
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


def _amor_decoder_rows_and_means(model: AMOR):
    """Physical decoder rows Φ (k,D) and means — not dual encoder E."""
    k_in = int(getattr(model, "k_in", model.block0.k_in))
    k_out = int(getattr(model, "k_out", model.block1.k_out))
    if getattr(model, "shared_fixed_basis", None) is not None:
        h = model.shared_fixed_basis
        phi_in = h.in_comp.detach().cpu().numpy()[:k_in]
        mean_in = h.in_mean.detach().cpu().numpy()
        phi_out = h.out_comp.detach().cpu().numpy()[:k_out]
        mean_out = h.out_mean.detach().cpu().numpy()
    else:
        # block0 decodes with input Φ; transition (block1) decodes with output Φ
        phi_in = model.block0.dec_components.detach().cpu().numpy()[:k_in]
        mean_in = model.block0.dec_mean.detach().cpu().numpy()
        phi_out = model.block1.dec_components.detach().cpu().numpy()[:k_out]
        mean_out = model.block1.dec_mean.detach().cpu().numpy()
    return phi_in, mean_in, phi_out, mean_out


def _model_grid_hw(model):
    """(N, M) spatial size for AMOR/PCANet."""
    if hasattr(model, "N") and hasattr(model, "M"):
        return int(model.N), int(model.M)
    sh = getattr(model, "spatial_shape", None)
    if sh is not None and len(sh) >= 2:
        return int(sh[0]), int(sh[1])
    raise AttributeError("model has no N/M or spatial_shape")


def l2_projection(model, X_new, Y_new, X_train_orig, Y_train_orig,
                  target_res, device='cuda', batch_size=32, dual_encoder: bool = True,
                  return_model: bool = False):
    """Transfer method 2: interpolate decoder modes B; dual encoder E=pinv(B).

    On equal-weight grids no mass renormalization is required. ``dual_encoder``
    (default True) sets ``encoder_components_ = pinv(B)`` so encode uses E, not Bᵀ.
    """
    if not (hasattr(model, 'block0') or hasattr(model, 'enc_components_t')):
        raise ValueError("L2 projection only works for AMOR/PCANet models")
    
    model.eval()
    N_train, M_train = _model_grid_hw(model)
    N_new, M_new = target_res
    
    # Extract physical decoder Φ (never dual E from enc_components_t)
    if isinstance(model, AMOR):
        enc_components_old, enc_mean_old, dec_components_old, dec_mean_old = _amor_decoder_rows_and_means(model)
        C_in = model.C_in
        C_out = model.C_out
        k_in = int(getattr(model, "k_in", model.block0.k_in))
        k_out = int(getattr(model, "k_out", model.block1.k_out))
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

    # Dual encoder on equal-weight grid: E = pinv(B), B unchanged (no renormalization).
    if dual_encoder:
        E_in = np.linalg.pinv(enc_components_new).T.astype(np.float32)  # (k, D) rows
        E_out = np.linalg.pinv(dec_components_new).T.astype(np.float32)
        # pinv(B) is (D,k); rows = pinv(B).T → (k,D). Check E B^T ≈ I.
        err_in = float(np.linalg.norm(E_in @ enc_components_new.T - np.eye(k_in)))
        err_out = float(np.linalg.norm(E_out @ dec_components_new.T - np.eye(k_out)))
        print(f"grid dual: ||E_in B_in^T - I||_F={err_in:.3e}  ||E_out B_out^T - I||_F={err_out:.3e}", flush=True)
    else:
        E_in = E_out = None
    
    # Create new PCAReducer with interpolated components
    from sklearn.decomposition import PCA
    
    rb_new = PCAReducerMultiChannel()
    rb_new.shape_ = (N_new, M_new)
    rb_new.n_channels = C_in
    rb_new.out_channels = C_out
    
    # Create fake PCA objects with interpolated components.
    # AMOR block0 has k_out=k_in and validates k_out <= pca_out.n_components_, so pca_out needs >= max(k_in, k_out).
    n_out_components = max(k_in, k_out)
    pca_in_new = PCA()
    pca_in_new.n_components_ = k_in
    pca_in_new.components_ = enc_components_new.astype(np.float32)
    pca_in_new.mean_ = enc_mean_new.astype(np.float32)
    pca_in_new.explained_variance_ratio_ = np.ones(k_in) / k_in  # Dummy values
    if E_in is not None:
        pca_in_new.encoder_components_ = E_in

    pca_out_new = PCA()
    pca_out_new.n_components_ = n_out_components
    if n_out_components > k_out:
        pad = np.zeros((n_out_components - k_out, D_out_new), dtype=np.float32)
        pca_out_new.components_ = np.concatenate([dec_components_new.astype(np.float32), pad], axis=0)
        if E_out is not None:
            pad_e = np.zeros((n_out_components - k_out, D_out_new), dtype=np.float32)
            pca_out_new.encoder_components_ = np.concatenate([E_out, pad_e], axis=0)
    else:
        pca_out_new.components_ = dec_components_new.astype(np.float32)
        if E_out is not None:
            pca_out_new.encoder_components_ = E_out
    pca_out_new.mean_ = dec_mean_new.astype(np.float32)
    pca_out_new.explained_variance_ratio_ = np.ones(n_out_components) / n_out_components  # Dummy values

    rb_new.pca_in = pca_in_new
    rb_new.pca_out = pca_out_new
    
    # Create new model with interpolated PCA
    if isinstance(model, AMOR):
        model_new = AMOR(
            width=model.width,
            rb=rb_new,
            k_in=k_in,
            k_out=k_out,
            layers=model.num_layers,
            dropout=0.0,
            init_scale=0.1,
            use_skip_connections=True,
            use_batch_norm=False,
            share_bases=True,
            use_coordinates=False,
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

    # Transfer learnable θ only (bases already set from interpolated Φ / dual E).
    old_sd = model.state_dict()
    new_sd = model_new.state_dict()
    filtered = {
        k: v
        for k, v in old_sd.items()
        if k in new_sd
        and new_sd[k].shape == v.shape
        and "components" not in k
        and not k.endswith("_mean")
        and "shared_fixed_basis" not in k
    }
    model_new.load_state_dict(filtered, strict=False)
    model_new = model_new.to(device)
    if return_model:
        return model_new

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
        raise ValueError("PCA refitting only works for AMOR/PCANet models")

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
    if isinstance(model, AMOR):
        model_new = AMOR(
            width=width,
            rb=rb_new,
            k_in=k_in_new,
            k_out=k_out_new,
            layers=layers,
            dropout=0.0,
            init_scale=0.1,
            use_skip_connections=True,
            use_batch_norm=False,
            use_coordinates=False,
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
    
    # Transfer only compatible parameters.
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
    resolution: interpolate input to finest, run model at finest, interpolate output to target."""
    if not (hasattr(model, 'block0') or hasattr(model, 'enc_components_t')):
        raise ValueError("PCA on finest only works for AMOR/PCANet models")
    N_train, M_train = model.N, model.M
    N_finest, M_finest = finest_res, finest_res
    N_target, M_target = target_res

    # Build model at finest by interpolating PCA components from training res to finest
    if isinstance(model, AMOR):
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

    if isinstance(model, AMOR):
        model_finest = AMOR(width=model.width, rb=rb_finest, k_in=k_in, k_out=k_out, layers=model.num_layers,
                             dropout=0.0, init_scale=0.1, use_skip_connections=True, use_batch_norm=False, use_coordinates=False)
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
    """Fit PCA with M-inner product: solve C v = λ M v for M-orthonormal components.

    X: (n_samples, n_features). M may be a 1-d nodal weight vector (uniform-grid
    mass, stored as diag) or a full dense SPD matrix — not restricted to diagonal
    operators beyond that convenience form.
    Returns: components (n_components, n_features), mean, evr.
    """
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
        raise ValueError("Mass matrix transfer only works for AMOR/PCANet models")
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

    if isinstance(model, AMOR):
        model_new = AMOR(width=model.width, rb=rb_new, k_in=k_in, k_out=k_out, layers=model.num_layers,
                          dropout=0.0, init_scale=0.1, use_skip_connections=True, use_batch_norm=False, use_coordinates=False)
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
    data_dir = expand('${AMOR_SCRATCH_DATA}')  # rename path on a different cluster
    base_data_path = os.path.join(data_dir, 'nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3.npz')
    
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
    amor_checkpoint = results_dir / 'amor_checkpoint.pt'
    pcanet_checkpoint = results_dir / 'pcanet_checkpoint.pt'
    
    checkpoint_base_dir = Path(data_dir) / 'checkpoints'
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
    # AMOR: only results_dir checkpoint (AMOR.py architecture); do not load LiftV3 or external AMOR.

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
    
    # Train AMOR
    print("\nTraining AMOR...")
    rb = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train_norm, Y_train_norm)
    
    epsilon = 1e-3
    k_in, k_out = rb.k_from_cutoff(j_in=1-1*epsilon, j_out=1-epsilon)
    print(f"PCA components: k_in={k_in}, k_out={k_out}")

    amor_model = None
    if amor_checkpoint.exists():
        checkpoint = torch.load(amor_checkpoint, map_location='cpu')
        state = checkpoint.get('model_state_dict', checkpoint)
        if 'lift.weight' in state:
            ckpt_k_in = state['block0.weight'].shape[2]
            ckpt_k_out = state['block1.weight'].shape[3]
            print(f"Loading AMOR checkpoint from {amor_checkpoint} (k_in={ckpt_k_in}, k_out={ckpt_k_out})")
            amor_model = AMOR(
                width=24,
                rb=rb,
                k_in=ckpt_k_in,
                k_out=ckpt_k_out,
                layers=4,
                dropout=0.0,
                init_scale=0.1,
                use_skip_connections=True,
                use_batch_norm=False,
                use_coordinates=False,
            )
            amor_model.load_state_dict(state, strict=True)
            try:
                amor_model = amor_model.to(device)
                amor_device = device
                print("AMOR model loaded successfully")
            except Exception as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    print(f"CUDA OOM moving AMOR to device: {e}. Using CPU for AMOR.")
                    amor_model = amor_model.to('cpu')
                    amor_device = torch.device('cpu')
                else:
                    raise
        else:
            print("AMOR checkpoint is not from AMOR.py (missing lift.weight), training from scratch.")
            amor_model = None
    if not amor_checkpoint.exists() or amor_model is None:
        if amor_model is None:
            amor_model = AMOR(
                width=24,
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                layers=4,
                dropout=0.0,
                init_scale=0.1,
                use_skip_connections=True,
                use_batch_norm=False,
                use_coordinates=False,
            )
        print("No AMOR checkpoint found, training...")
        train_model(amor_model, train_loader, val_loader,
                    checkpoint_path=str(amor_checkpoint), device=device)
        amor_device = device

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

    # Load datasets at training resolution plus four other grid sizes (five total)
    train_res = int(X_train.shape[-1])  # assume square grid
    resolutions = sorted(set([train_res, 50, 75, 150, 200]))
    finest_res = max(resolutions)
    print("\n" + "="*70)
    print("TESTING ON DIFFERENT RESOLUTIONS")
    print("="*70)
    print(f"Training resolution: {train_res}x{train_res}. Test resolutions: {resolutions}.")

    # Transfer strategies — match exactly the 4+1 methods from scratch/fenics_mesh_transfer_demo_no_dolfin.py
    methods_fem = ['algebraic', 'l2_sparse', 'l2_dense', 'pseudoinverse', 'weighted_pseudoinverse']
    strategy_fns = {
        'algebraic':            strategy_algebraic,
        'l2_sparse':            strategy_l2_sparse,
        'l2_dense':             strategy_l2_dense,
        'pseudoinverse':        strategy_pseudoinverse,
        'weighted_pseudoinverse': strategy_weighted_pseudoinverse,
    }
    results = {
        'fno':    {m: {} for m in methods_fem},
        'amor':  {m: {} for m in methods_fem},
        'pcanet': {m: {} for m in methods_fem},
    }
    
    for res in resolutions:
        print(f"\nTesting on resolution {res}x{res}...")
        if res == train_res:
            X_res_norm = X_test_norm
            Y_res_norm = Y_test_norm
            print(f"Using base dataset test split: X={X_res_norm.shape}, Y={Y_res_norm.shape}")
        else:
            res_data_path = os.path.join(data_dir, f'nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_res{res}.npz')
            if not os.path.exists(res_data_path):
                print(f"Warning: {res_data_path} not found, skipping...")
                continue
            X_res, Y_res = load_dataset(res_data_path)
            print(f"Loaded dataset: X={X_res.shape}, Y={Y_res.shape}")
            X_res_norm = (X_res - x_mean) / (x_std + 1e-8)
            Y_res_norm = (Y_res - y_mean) / (y_std + 1e-8)
        
        res_batch = 8 if res >= 150 else 16

        # Apply all 5 FEM strategies to every model
        for model_name, mdl, dev in [('fno', fno_model, device),
                                      ('amor', amor_model, amor_device),
                                      ('pcanet', pcanet_model, pcanet_device)]:
            if mdl is None:
                print(f"  {model_name.upper()} not available, skipping...")
                for m in methods_fem:
                    results[model_name][m][res] = None
                continue
            print(f"  Testing {model_name.upper()} with all FEM strategies...")
            for method in methods_fem:
                try:
                    err_dict = strategy_fns[method](mdl, X_res_norm, Y_res_norm, train_res, dev, batch_size=res_batch)
                    results[model_name][method][res] = err_dict
                    print(f"    {method}: rel_l1={err_dict['rel_l1']:.6e}  rel_l2={err_dict['rel_l2']:.6e}")
                except Exception as e:
                    print(f"    {method} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results[model_name][method][res] = None
    
    # Save results
    results_path = results_dir / 'mesh_generalization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Helper: get rel_l2 from result entry (for plotting)
    def _rel_l2(entry):
        return entry['rel_l2'] if isinstance(entry, dict) and entry is not None else None

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
        f"Training resolution: {train_res}×{train_res}. Test resolutions: {resolutions}. Transfer strategies: A=algebraic, B=l2_sparse, C=l2_dense, D=pseudoinverse, D+=weighted_pseudoinverse.",
        "",
        "## Relative L1 and L2 by resolution",
        "",
    ]
    md_lines.append("| Resolution | " + " | ".join([f"FNO {m}" for m in methods_fem]) + " | " + " | ".join([f"AMOR {m}" for m in methods_fem]) + " | " + " | ".join([f"PCANet {m}" for m in methods_fem]) + " |")
    md_lines.append("|------------|" + "--------------|" * (len(methods_fem) * 3) + "")
    for r in sorted(resolutions):
        row = [str(r)]
        for m in methods_fem:
            row.append(_fmt(results['fno'][m].get(r)))
        for m in methods_fem:
            row.append(_fmt(results['amor'][m].get(r)))
        for m in methods_fem:
            row.append(_fmt(results['pcanet'][m].get(r)))
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.extend(["", "## Files", "", "- mesh_generalization_results.json", ""])
    with open(results_dir / 'results.md', 'w') as f:
        f.write("\n".join(md_lines))
    print(f"Results summary saved to {results_dir / 'results.md'}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
