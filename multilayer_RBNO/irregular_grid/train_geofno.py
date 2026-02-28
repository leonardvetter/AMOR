"""
Training script for MRBNO on Geo-FNO datasets (airfoil or elasticity).

Uses MRBNO.py model with direct handling of irregular grids (3D input).
Hyperparameters match train_FNO_MRBNO_PCANet.ipynb notebook.

Usage:
    python train_geofno.py --dataset_path <path_to_dataset> --dataset_type <airfoil|elasticity>
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from timeit import default_timer

# Import from local directory first (for MRBNO and PCAReducer with 3D support)
local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, local_dir)
sys.path.insert(1, os.path.join(os.path.dirname(local_dir), 'models'))

from geofno_loader import prepare_geofno_data
from losses import LpLoss
from MRBNO import MRBNO
from PCAReducer import PCAReducerMultiChannel


def relative_l1_loss(pred, target):
    """Compute relative L1 loss: ||pred - target||_1 / ||target||_1"""
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    error = torch.abs(pred - target)
    return torch.mean(error / (torch.abs(target) + 1e-10))


def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=3.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        batch_data, batch_target, batch_material = batch
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        
        optimizer.zero_grad()
        output = model(batch_data)
        loss = criterion(output, batch_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data, batch_target, batch_material = batch
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            output = model(batch_data)
            loss = criterion(output, batch_target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train MRBNO on Geo-FNO datasets')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to Geo-FNO dataset directory')
    parser.add_argument('--dataset_type', type=str, default='airfoil',
                       choices=['airfoil', 'elasticity', 'pipe', 'plasticity', 'car-cfd', 'channel-shocks'],
                       help='Type of dataset')
    parser.add_argument('--n_modes', type=int, default=144,
                       help='Number of PCA modes (default: 144, matches notebook)')
    parser.add_argument('--width', type=int, default=24,
                       help='Model width (default: 24, matches notebook)')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of layers (default: 4, matches notebook)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32, matches notebook)')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of epochs (default: 300, matches notebook)')
    parser.add_argument('--lr', type=float, default=2e-3,
                       help='Learning rate (default: 2e-3, matches notebook)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4, matches notebook)')
    parser.add_argument('--max_grad_norm', type=float, default=3.0,
                       help='Gradient clipping max norm (default: 3.0, matches notebook)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50 epochs, matches notebook)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                       help='Minimum change to qualify as improvement for early stopping (default: 0.0, matches notebook)')
    parser.add_argument('--use_coordinates', action='store_true', default=False,
                       help='Use coordinate embedding (default: False, matches notebook)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MRBNO Training on Geo-FNO Dataset")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Type: {args.dataset_type}")
    print(f"Device: {args.device}")
    print()
    
    device = torch.device(args.device)
    
    print("Loading dataset...")
    datasets, mesh_coords, material_props = prepare_geofno_data(
        args.dataset_path,
        dataset_type=args.dataset_type,
        train_split=0.7,
        val_split=0.2,
        normalize=False,
        use_geofno_split=True
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets.get('val', None)
    test_dataset = datasets['test']
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    if val_dataset is not None:
        print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Get data shapes
    train_data_3d = train_dataset.data.numpy()  # (num_samples, C, N)
    train_targets_3d = train_dataset.targets.numpy()  # (num_samples, C_out, N)
    
    print(f"\nData shapes:")
    print(f"  Input: {train_data_3d.shape}")
    print(f"  Output: {train_targets_3d.shape}")
    
    print("\nFitting PCAReducerMultiChannel...")
    # PCAReducer now accepts 3D data directly
    rb = PCAReducerMultiChannel(
        svd_solver="auto",
        whiten=False,
        random_state=42,
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False
    ).fit(train_data_3d, train_targets_3d)
    
    # Determine k_in and k_out from explained variance ratio
    # Use cutoff to keep modes explaining 99.9% variance, but cap at n_modes
    k_in, k_out = rb.k_from_cutoff(j_in=0.999, j_out=0.999)
    k_in = min(k_in, args.n_modes)
    k_out = min(k_out, args.n_modes)
    
    print(f"PCA modes: k_in={k_in}, k_out={k_out} (requested {args.n_modes})")
    
    print("\nCreating model...")
    model = MRBNO(
        width=args.width,
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=args.n_layers,
        dropout=0.0,
        init_scale=0.1,
        use_skip_connections=True,
        use_batch_norm=True,
        use_coordinates=args.use_coordinates,
        add_constant_mode=False,
        use_factorized_weights=False
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Use relative L1 loss matching notebook
    criterion = relative_l1_loss
    print("Using Relative L1 Loss")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing LR schedule matching notebook
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    print(f"Using CosineAnnealingLR (T_max={args.epochs}, eta_min=0.0)")
    
    print("\nStarting training...")
    print("=" * 60)
    print(f"Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LR schedule: CosineAnnealingLR")
    print(f"  Loss: Relative L1")
    print(f"  Gradient clipping: max_norm={args.max_grad_norm}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Early stopping min_delta: {args.min_delta}")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        t1 = default_timer()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.max_grad_norm)
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
        else:
            val_loss = float('inf')
        scheduler.step()
        t2 = default_timer()
        
        # Print every 10 epochs or first epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_loader is not None:
                print(f"[{epoch+1:03d}/{args.epochs}] train={train_loss:.4e} val={val_loss:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
            else:
                print(f"[{epoch+1:03d}/{args.epochs}] train={train_loss:.4e} (no val) lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
        
        # Early stopping and save best model (only if validation set exists)
        if val_loader is not None:
            improvement = best_val_loss - val_loss
            if improvement > args.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                checkpoint_dir = '/mnt/c/Users/dvod/MRBNO/data/checkpoints/geofno'
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_best_{args.dataset_type}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'args': vars(args),
                    'rb_shape': rb.shape_,
                    'rb_n_channels': rb.n_channels,
                    'rb_out_channels': rb.out_channels,
                    'k_in': k_in,
                    'k_out': k_out,
                }, checkpoint_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4e} at epoch {best_epoch+1}")
                break
        else:
            best_val_loss = train_loss
    
    print("=" * 60)
    print("Training complete!")
    if val_loader is not None:
        print(f"Best validation loss: {best_val_loss:.4e}")
    
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
