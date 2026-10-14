"""Training loop shared by the 2D field-to-field trainers (Helmholtz, nonlinear diffusion).

AdamW + CosineAnnealingLR, L1 objective, gradient clipping, early stopping on
validation loss. Best weights are kept in memory and restored at the end.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def l1_loss(pred, target):
    """L1 loss used for training."""
    return torch.mean(torch.abs(pred - target))


def train_model(model, train_loader, val_loader, device, hyperparams, loss_fn=l1_loss):
    """
    Train model with AdamW + CosineAnnealingLR + early stopping.
    Saves best model weights during training and restores them at the end.
    ``loss_fn(pred, target)`` defaults to the absolute L1 loss. With ``val_loader=None`` there is no
    validation, no early stopping and no restore: the model after the last epoch is kept.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams['epochs'], eta_min=0.0)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(hyperparams['epochs']):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['max_grad_norm'])
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        if val_loader is None:
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{hyperparams['epochs']}: train={train_loss:.6f}")
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss - hyperparams['early_stopping_min_delta']:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hyperparams['early_stopping_patience']:
            print(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{hyperparams['epochs']}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f}")

    if val_loader is None:
        return float('nan'), hyperparams['epochs'] - 1

    # Restore best model weights
    model.load_state_dict(best_model_state)
    return best_val_loss, best_epoch
