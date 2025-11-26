#!/usr/bin/env python3
"""
Standalone training script for MRBNO Lift v3 on Kolmogorov Flow dataset (Re=40).

This script trains MRBNO on the Kolmogorov flow dataset (Re=40) and saves the checkpoint.
The notebook can then load this checkpoint for evaluation and analysis.

Usage:
    python train_mrbno_kolmogorov_Re40.py [--batch_size BATCH_SIZE] [--epochs EPOCHS]
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import argparse
import time

# Add models directory to path
sys.path.append('models')

from MRBNO_lift_v3 import MRBNOLiftV3
from MRBNO_simple import PCAReducerMultiChannel


def load_kolmogorov_dataset(data_path='/pscratch/sd/d/dvod/2D_NS_Re40.npy'):
    """Load and preprocess Kolmogorov flow dataset."""
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path, mmap_mode='r')
    
    print(f"Dataset shape: {data.shape}")
    print(f"Interpretation: (num_trajectories={data.shape[0]}, num_timesteps={data.shape[1]}, height={data.shape[2]}, width={data.shape[3]})")
    
    # Create input-output pairs: u(t) -> u(t+1)
    num_trajectories = data.shape[0]
    num_timesteps = data.shape[1] - 1  # 500 pairs per trajectory
    total_pairs = num_trajectories * num_timesteps
    
    print(f"\nCreating input-output pairs...")
    print(f"  Trajectories: {num_trajectories}")
    print(f"  Pairs per trajectory: {num_timesteps}")
    print(f"  Total pairs: {total_pairs:,}")
    
    # Process in chunks to be memory-efficient, convert to float32 immediately
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
    
    # Add channel dimension for input (1 channel: velocity field)
    X = X[:, np.newaxis, :, :]  # (N, 1, 64, 64)
    Y = Y[:, np.newaxis, :, :]  # (N, 1, 64, 64)
    
    print(f"After adding channel dimension: X={X.shape}, Y={Y.shape}")
    
    # Train/val split (70/20/10 like notebook)
    actual_samples = len(X)
    train_frac = 0.7
    val_frac = 0.2
    n_train = int(train_frac * actual_samples)
    n_val = int(val_frac * actual_samples)
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    
    print(f"\nTrain/Val split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Normalize using train statistics only
    x_mean = X_train.mean()
    x_std = X_train.std()
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    X_train = (X_train - x_mean) / (x_std + 1e-8)
    Y_train = (Y_train - y_mean) / (y_std + 1e-8)
    X_val = (X_val - x_mean) / (x_std + 1e-8)
    Y_val = (Y_val - y_mean) / (y_std + 1e-8)
    
    print(f"\nNormalization statistics:")
    print(f"  X: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"  Y: mean={y_mean:.4f}, std={y_std:.4f}")
    
    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).float()
    
    return X_train_t, Y_train_t, X_val_t, Y_val_t, X_train, Y_train


def train_model(model, train_loader, val_loader, epochs=300, lr=2e-3, weight_decay=1e-4, 
                max_grad_norm=3.0, device='cuda', early_stopping_patience=50):
    """Train model with cosine annealing LR schedule and early stopping."""
    start_time = time.time()
    
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    has_val = len(val_loader.dataset) > 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
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
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
        else:
            val_loss = float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping logic
        improved = False
        if has_val:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_epoch = epoch
                patience_counter = 0
                improved = True
            else:
                patience_counter += 1
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if has_val:
                print(f"[{epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
            else:
                print(f"[{epoch+1:03d}/{epochs}] train={train_loss:.4e} (no val) lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1} (patience={early_stopping_patience})")
            break
    
    training_time = time.time() - start_time
    
    return train_losses, val_losses, best_val_loss, training_time, best_epoch + 1


def main():
    parser = argparse.ArgumentParser(description='Train MRBNO on Kolmogorov Flow dataset')
    parser.add_argument('--data_path', type=str, default='/pscratch/sd/d/dvod/2D_NS_Re40.npy',
                        help='Path to Kolmogorov flow dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16 to avoid OOM)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=3.0,
                        help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load dataset
    X_train_t, Y_train_t, X_val_t, Y_val_t, X_train, Y_train = load_kolmogorov_dataset(args.data_path)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=args.batch_size, shuffle=False)
    
    print(f"\nData loaders created: batch_size={args.batch_size}")
    
    # Compute PCA (same as notebook)
    print("\n" + "="*70)
    print("COMPUTING PCA")
    print("="*70)
    rb = PCAReducerMultiChannel(
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False,
    ).fit(X_train, Y_train)
    
    epsilon = 1e-3
    k_in, k_out = rb.k_from_cutoff(j_in=1-1*epsilon, j_out=1-epsilon)
    
    print(f"PCA components: k_in={k_in}, k_out={k_out}")
    print(f"Boundary condition awareness: bc_weight={rb.bc_weight}, threshold={rb.boundary_threshold}")
    
    # Create MRBNO model (same as notebook)
    print("\n" + "="*70)
    print("CREATING MRBNO LIFT V3 MODEL")
    print("="*70)
    model = MRBNOLiftV3(
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
    
    params = sum(p.numel() for p in model.parameters())
    print(f"MRBNO Lift v3 Parameters: {params:,}")
    
    # Check GPU memory
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU Memory before training:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {total - reserved:.2f} GB")
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING MRBNO LIFT V3")
    print("="*70)
    train_losses, val_losses, best_val_loss, training_time, best_epoch = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=device,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'mrbno_lift_v3_kolmogorov_Re40_best.pt')
    torch.save(model.state_dict(), checkpoint_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4e} (epoch {best_epoch})")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == '__main__':
    main()

