"""Training loop for the Kolmogorov flow trainers.

The target is a forward Euler step, y_{k+1} = y_k + dt * f_theta(y_k), so the
loop differs from the field-to-field loop in loop2d.py. Supports resuming from a
previous run via ``initial_state_dict`` and ``epoch_offset``.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(model, train_loader, val_loader, epochs=300, lr=2e-3, weight_decay=1e-4,
                max_grad_norm=3.0, device='cuda', early_stopping_patience=50, dt=1.0,
                initial_state_dict=None, resume_best_val_rel_l1=None, resume_best_epoch=0,
                epoch_offset=0, max_epochs_display=None):
    """Train with L1; select best checkpoint by validation relative L1."""
    start_time = time.time()

    model.to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    from AMOR.training.loop2d import l1_loss
    criterion = l1_loss

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_val_rel_l1 = float('inf') if resume_best_val_rel_l1 is None else resume_best_val_rel_l1
    best_epoch = resume_best_epoch
    patience_counter = 0

    has_val = len(val_loader.dataset) > 0
    best_model_state = copy.deepcopy(model.state_dict())
    display_max = max_epochs_display if max_epochs_display is not None else epoch_offset + epochs

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = xb + dt * model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation phase
        if has_val:
            model.eval()
            val_loss = 0.0
            val_rel_l1_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = xb + dt * model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item() * xb.size(0)
                    b = pred.size(0)
                    pred_flat = pred.view(b, -1)
                    y_flat = yb.view(b, -1)
                    diff_l1 = torch.abs(pred_flat - y_flat).sum(dim=1)
                    y_l1 = torch.abs(y_flat).sum(dim=1).clamp(min=1e-8)
                    val_rel_l1_sum += (diff_l1 / y_l1).sum().item()
                    val_count += b
            val_loss /= len(val_loader.dataset)
            val_rel_l1 = val_rel_l1_sum / max(1, val_count)
        else:
            val_loss = float('inf')
            val_rel_l1 = train_loss
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping logic
        if has_val:
            if val_rel_l1 < best_val_rel_l1:
                best_val_loss = val_loss
                best_val_rel_l1 = val_rel_l1
                best_epoch = epoch_offset + epoch + 1
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
        else:
            if train_loss < best_val_rel_l1:
                best_val_loss = train_loss
                best_val_rel_l1 = train_loss
                best_epoch = epoch_offset + epoch + 1
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
        
        scheduler.step()
        
        abs_ep = epoch_offset + epoch + 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if has_val:
                print(
                    f"[{abs_ep:03d}/{display_max}] train_l1={train_loss:.4e} "
                    f"val_l1={val_loss:.4e} val_rel_l1={val_rel_l1:.4e} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"(best val_rel_l1: {best_val_rel_l1:.4e} @ epoch {best_epoch})"
                )
            else:
                print(
                    f"[{abs_ep:03d}/{display_max}] train={train_loss:.4e} (no val) "
                    f"lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch})"
                )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {abs_ep} (patience={early_stopping_patience})")
            break

    model.load_state_dict(best_model_state)
    training_time = time.time() - start_time
    epochs_ran = epoch_offset + epoch + 1

    return train_losses, val_losses, best_val_rel_l1, training_time, best_epoch, epochs_ran


def compute_rel_metrics(model, loader, device, xy_mean, xy_std, dt=1.0):
    """Relative L1/L2 on denormalized fields (original units)."""
    mean_t = torch.tensor(float(xy_mean), device=device, dtype=torch.float32)
    std_t = torch.tensor(float(xy_std), device=device, dtype=torch.float32)
    model.eval()
    rel_l1_sum, rel_l2_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = xb + dt * model(xb)
            pred_phys = pred * std_t + mean_t
            y_phys = yb * std_t + mean_t
            b = pred_phys.size(0)
            pred_flat = pred_phys.view(b, -1)
            y_flat = y_phys.view(b, -1)
            diff_l1 = torch.abs(pred_flat - y_flat).sum(dim=1)
            y_l1 = torch.abs(y_flat).sum(dim=1).clamp(min=1e-8)
            rel_l1_sum += (diff_l1 / y_l1).sum().item()
            diff_l2 = torch.norm(pred_flat - y_flat, p=2, dim=1)
            y_l2 = torch.norm(y_flat, p=2, dim=1).clamp(min=1e-8)
            rel_l2_sum += (diff_l2 / y_l2).sum().item()
            n += b
    return (rel_l1_sum / n if n else float('nan'), rel_l2_sum / n if n else float('nan'))


def compute_rel_metrics_normalized(model, loader, device, dt=1.0):
    """Compute mean relative L1/L2 errors on a split."""
    model.eval()
    rel_l1_sum, rel_l2_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = xb + dt * model(xb)
            b = pred.size(0)
            pred_flat = pred.view(b, -1)
            y_flat = yb.view(b, -1)
            diff_l1 = torch.abs(pred_flat - y_flat).sum(dim=1)
            y_l1 = torch.abs(y_flat).sum(dim=1).clamp(min=1e-8)
            rel_l1_sum += (diff_l1 / y_l1).sum().item()
            diff_l2 = torch.norm(pred_flat - y_flat, p=2, dim=1)
            y_l2 = torch.norm(y_flat, p=2, dim=1).clamp(min=1e-8)
            rel_l2_sum += (diff_l2 / y_l2).sum().item()
            n += b
    return (rel_l1_sum / n if n else float('nan'), rel_l2_sum / n if n else float('nan'))
