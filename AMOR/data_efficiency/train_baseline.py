#!/usr/bin/env python3
"""
Baseline training: Train AMOR.py directly on full 20k samples (one-stage).
PCA is computed on ALL train/val data to match Stage 1 for fair weight comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from train_two_stage import (
    load_nonlinear_dataset,
    PCAReducerMultiChannel,
    l1_loss,
    relative_l1_loss,
    relative_l2_error,
    evaluate_model,
    _atomic_torch_save,
)
from AMOR.models.amor import AMOR


def train_baseline(model, train_loader, val_loader, test_loader, epochs, lr, device, checkpoint_dir, start_epoch=0, y_mean=None, y_std=None):
    """Train baseline model."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                                   betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    
    resume_checkpoint = os.path.join(checkpoint_dir, 'baseline_resume.pt')
    if os.path.exists(resume_checkpoint) and start_epoch == 0:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        except (OSError, EOFError, RuntimeError) as e:
            print(f"Warning: could not load resume ({e}); removing corrupt file and starting fresh.")
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
            print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4e} @ epoch {best_epoch+1}")
        else:
            best_val_loss = float('inf')
            best_epoch = start_epoch
    else:
        best_val_loss = float('inf')
        best_epoch = start_epoch

    skip_epoch_ckpt = os.environ.get("ABLATION_SKIP_EPOCH_CKPT", "").strip().lower() in ("1", "true", "yes")
    
    patience_counter = 0
    early_stopping_patience = 50
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
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
            train_loss += loss.item() * xb.size(0)
            train_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
            train_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_rel_l1 /= len(train_loader.dataset)
        train_rel_l2 /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_rel_l1 = 0.0
        val_rel_l2 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = l1_loss(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_rel_l1 += relative_l1_loss(pred, yb).item() * xb.size(0)
                val_rel_l2 += relative_l2_error(pred, yb) * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_rel_l1 /= len(val_loader.dataset)
        val_rel_l2 /= len(val_loader.dataset)
        
        scheduler.step()
        
        val_monitor = val_rel_l1
        if val_monitor < best_val_loss:
            best_val_loss = val_monitor
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(checkpoint_dir, 'baseline_best.pt')
            _atomic_torch_save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Baseline Epoch {epoch+1:03d}/{epochs}] train_l1={train_loss:.4e} train_rel_l1={train_rel_l1:.4e} train_rel_l2={train_rel_l2:.4e} val_l1={val_loss:.4e} val_rel_l1={val_rel_l1:.4e} val_rel_l2={val_rel_l2:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best val_rel_l1: {best_val_loss:.4e} @ epoch {best_epoch+1})")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'baseline_epoch_{epoch+1}.pt')
            resume_path = os.path.join(checkpoint_dir, 'baseline_resume.pt')
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            }
            _atomic_torch_save(checkpoint_data, resume_path)
            if not skip_epoch_ckpt:
                _atomic_torch_save(checkpoint_data, checkpoint_path)
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    best_path = os.path.join(checkpoint_dir, 'baseline_best.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))

    print("\n" + "=" * 70)
    print("EVALUATING baseline_best.pt (train / val / test)")
    print("Test metrics: after undoing Y z-score. Train/val also reported denormalized.")
    print("=" * 70)
    train_rel_l1, train_rel_l2 = evaluate_model(
        model, train_loader, device, is_stage1=False, y_mean=y_mean, y_std=y_std
    )
    val_rel_l1, val_rel_l2 = evaluate_model(
        model, val_loader, device, is_stage1=False, y_mean=y_mean, y_std=y_std
    )
    test_rel_l1, test_rel_l2 = evaluate_model(
        model, test_loader, device, is_stage1=False, y_mean=y_mean, y_std=y_std
    )
    print(f"Train relative L1: {train_rel_l1:.4e}  Train relative L2: {train_rel_l2:.4e}")
    print(f"Val relative L1:   {val_rel_l1:.4e}  Val relative L2:   {val_rel_l2:.4e}")
    print(f"Test relative L1:  {test_rel_l1:.4e}  Test relative L2:  {test_rel_l2:.4e}")

    return best_val_loss, test_rel_l1, test_rel_l2


def main():
    parser = argparse.ArgumentParser(description='Baseline training on full dataset')
    parser.add_argument('--data_path', type=str, default=expand('${AMOR_SCRATCH_DATA}/nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_big.npz'))  # rename path on a different cluster
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--k_in', type=int, default=144)
    parser.add_argument('--k_out', type=int, default=144)
    parser.add_argument('--initial_checkpoint', type=str, default=None,
                       help='Path to initial checkpoint to load weights from (before training starts)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    base_dir = os.environ.get(
        "ABLATION_BASE_DIR", expand("${AMOR_DATA_ROOT}/data_efficiency")
    )
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_nonlinear_dataset(args.data_path)
    
    print("\n" + "="*70)
    print("BASELINE: Training on full dataset (all 20k samples)")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    print("\n" + "="*70)
    print("COMPUTING PCA ON ALL TRAIN/VAL DATA")
    print("="*70)
    print("Note: PCA computed on all train/val data to match Stage 1")
    
    X_all_train_val = np.concatenate([X_train, X_val], axis=0)
    Y_all_train_val = np.concatenate([Y_train, Y_val], axis=0)
    rb = PCAReducerMultiChannel().fit(X_all_train_val, Y_all_train_val)
    if args.k_in is None or args.k_out is None:
        k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    else:
        k_in, k_out = args.k_in, args.k_out
    print(f"Using k_in={k_in}, k_out={k_out}")
    
    print("\n" + "="*70)
    print("BASELINE: Training AMOR on full dataset")
    print("="*70)
    
    model = AMOR(
        width=args.width, rb=rb, k_in=k_in, k_out=k_out, layers=args.layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False, add_constant_mode=False,
        use_factorized_weights=False
    )
    
    if args.initial_checkpoint and os.path.exists(args.initial_checkpoint):
        print(f"\nLoading initial weights from: {args.initial_checkpoint}")
        init_ckpt = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt['model_state_dict'], strict=False)
        print("Initial weights loaded (PCA space weights from Stage 1)")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(),
                     torch.from_numpy(Y_train).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(),
                     torch.from_numpy(Y_val).float()),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                     torch.from_numpy(Y_test).float()),
        batch_size=args.batch_size, shuffle=False
    )
    
    best_val_loss, test_rel_l1, test_rel_l2 = train_baseline(
        model, train_loader, val_loader, test_loader, args.epochs, args.lr, device, checkpoint_dir,
        start_epoch=0, y_mean=float(norm_stats[2]), y_std=float(norm_stats[3]),
    )
    
    checkpoint_path = os.path.join(checkpoint_dir, 'baseline_final.pt')
    _atomic_torch_save({
        'model_state_dict': model.state_dict(),
        'rb': rb,
        'k_in': k_in,
        'k_out': k_out,
        'width': args.width,
        'layers': args.layers
    }, checkpoint_path)
    
    print(f"\nBaseline checkpoint saved to {checkpoint_path}")
    
    resume_path = os.path.join(checkpoint_dir, 'baseline_resume.pt')
    if os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == '__main__':
    main()
