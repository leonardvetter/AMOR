"""
Training script for MRBNO on Geo-FNO datasets (airfoil or elasticity).

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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from geofno_loader import prepare_geofno_data
from losses import LpLoss
from MRBNO import MRBNO
from PCAReducer import PCAReducerMultiChannel


def train_epoch(model, dataloader, optimizer, criterion, device, dataset_type):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        batch_data, batch_target, batch_material = batch
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        
        # Reshape to 4D for new MRBNO model
        batch_data_4d = reshape_to_4d(batch_data, dataset_type)
        batch_target_4d = reshape_to_4d(batch_target, dataset_type)
        
        optimizer.zero_grad()
        output_4d = model(batch_data_4d)
        
        # Reshape back to 3D for loss computation
        output = reshape_from_4d(output_4d, dataset_type, batch_target.shape)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device, dataset_type):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data, batch_target, batch_material = batch
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            # Reshape to 4D for new MRBNO model
            batch_data_4d = reshape_to_4d(batch_data, dataset_type)
            batch_target_4d = reshape_to_4d(batch_target, dataset_type)
            
            output_4d = model(batch_data_4d)
            
            # Reshape back to 3D for loss computation
            output = reshape_from_4d(output_4d, dataset_type, batch_target.shape)
            loss = criterion(output, batch_target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def reshape_to_4d(data, dataset_type):
    """
    Reshape 3D data (B, C, N) to 4D format (B, C, N, M) for new MRBNO model.
    
    Args:
        data: Input data of shape (B, C, N) or numpy array of shape (num_samples, C, N)
        dataset_type: 'airfoil' or 'elasticity'
    
    Returns:
        Reshaped data of shape (B, C, N, M)
    """
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
        is_tensor = True
    else:
        data_np = data
        is_tensor = False
    
    if dataset_type == 'airfoil':
        # Airfoil: (B, 6, 11271) -> (B, 6, 221, 51)
        B, C, N = data_np.shape
        height, width = 221, 51
        if N != height * width:
            raise ValueError(f"Expected {height * width} nodes for airfoil, got {N}")
        data_4d = data_np.reshape(B, C, height, width)
    elif dataset_type == 'elasticity':
        # Elasticity: (B, 2, 972) -> (B, 2, 36, 27) since 972 = 36 * 27
        B, C, N = data_np.shape
        # Find factors of N
        if N == 972:
            height, width = 36, 27
        else:
            # Try to find reasonable factors
            import math
            sqrt_n = int(math.sqrt(N))
            for h in range(sqrt_n, 0, -1):
                if N % h == 0:
                    height, width = h, N // h
                    break
            else:
                height, width = 1, N
        data_4d = data_np.reshape(B, C, height, width)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if is_tensor:
        return torch.from_numpy(data_4d).to(data.device if isinstance(data, torch.Tensor) else 'cpu')
    else:
        return data_4d


def reshape_from_4d(data_4d, dataset_type, original_shape):
    """
    Reshape 4D data (B, C, N, M) back to 3D format (B, C, N).
    
    Args:
        data_4d: Input data of shape (B, C, N, M)
        dataset_type: 'airfoil' or 'elasticity'
        original_shape: Original shape (B, C, N)
    
    Returns:
        Reshaped data of shape (B, C, N)
    """
    if isinstance(data_4d, torch.Tensor):
        data_np = data_4d.detach().cpu().numpy()
        is_tensor = True
        device = data_4d.device
    else:
        data_np = data_4d
        is_tensor = False
    
    B, C, N, M = data_np.shape
    N_flat = N * M
    data_3d = data_np.reshape(B, C, N_flat)
    
    # Ensure it matches original shape
    if data_3d.shape != original_shape:
        # Truncate or pad if needed
        B_orig, C_orig, N_orig = original_shape
        if N_flat > N_orig:
            data_3d = data_3d[:, :, :N_orig]
        elif N_flat < N_orig:
            padding = np.zeros((B, C, N_orig - N_flat), dtype=data_3d.dtype)
            data_3d = np.concatenate([data_3d, padding], axis=2)
    
    if is_tensor:
        return torch.from_numpy(data_3d).to(device)
    else:
        return data_3d




def main():
    parser = argparse.ArgumentParser(description='Train MRBNO on Geo-FNO datasets')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to Geo-FNO dataset file (.npz, .h5, or .hdf5)')
    parser.add_argument('--dataset_type', type=str, default='airfoil',
                       choices=['airfoil', 'elasticity'],
                       help='Type of dataset')
    parser.add_argument('--n_modes', type=int, default=144,
                       help='Number of PCA modes (default: 144, equivalent to 12x12 in Geo-FNO)')
    parser.add_argument('--width', type=int, default=32,
                       help='Model width (default: 32, matches Geo-FNO)')
    parser.add_argument('--n_layers', type=int, default=5,
                       help='Number of layers (default: 5, matches Geo-FNO)')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch size (default: 20, matches Geo-FNO)')
    parser.add_argument('--epochs', type=int, default=501,
                       help='Number of epochs (default: 501, matches Geo-FNO)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001, matches Geo-FNO)')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                       choices=['cosine', 'step'],
                       help='Learning rate schedule: cosine (CosineAnnealingLR) or step (StepLR)')
    parser.add_argument('--T_max', type=int, default=None,
                       help='T_max for CosineAnnealingLR (default: epochs)')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Minimum learning rate for CosineAnnealingLR (default: 1e-6)')
    parser.add_argument('--step_size', type=int, default=50,
                       help='LR scheduler step size for StepLR (default: 50)')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='LR scheduler gamma for StepLR (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4, matches Geo-FNO)')
    parser.add_argument('--add_coords', action='store_true', default=True,
                       help='Add coordinate channels as extra input channels (recommended for irregular grids)')
    parser.add_argument('--use_grid_skip', action='store_true', default=False,
                       help='Use grid-based skip connections (like Geo-FNO b layers)')
    parser.add_argument('--use_input_skip', action='store_true', default=False,
                       help='Use direct skip connection from input to output')
    parser.add_argument('--use_weighted_loss', action='store_true', default=False,
                       help='Use weighted loss for high-stress regions (beta=0.5)')
    parser.add_argument('--use_multiplicative_conditioning', action='store_true', default=False,
                       help='Use multiplicative material conditioning instead of additive')
    parser.add_argument('--use_material_adaptive_pca', action='store_true', default=False,
                       help='Use material-adaptive PCA basis transformation')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50 epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-6,
                       help='Minimum change to qualify as improvement for early stopping (default: 1e-6)')
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
    
    material_prop_dim = None
    if material_props is not None:
        material_prop_dim = material_props.shape[1]
        print(f"Material properties loaded: {material_prop_dim} properties per sample")
    
    print("\nPreparing data for new MRBNO model (4D format)...")
    # Reshape training data to 4D format
    train_data_3d = train_dataset.data.numpy()  # (num_samples, C, N)
    train_targets_3d = train_dataset.targets.numpy()  # (num_samples, C_out, N)
    
    train_data_4d = reshape_to_4d(train_data_3d, args.dataset_type)  # (num_samples, C, H, W)
    train_targets_4d = reshape_to_4d(train_targets_3d, args.dataset_type)  # (num_samples, C_out, H, W)
    
    print(f"Reshaped data: {train_data_4d.shape} -> {train_targets_4d.shape}")
    
    print("\nFitting PCAReducerMultiChannel...")
    rb = PCAReducerMultiChannel(
        svd_solver="auto",
        whiten=False,
        random_state=42,
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False
    ).fit(train_data_4d, train_targets_4d)
    
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
        use_coordinates=args.add_coords,
        add_constant_mode=False,
        use_factorized_weights=False
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Use relative L1 loss for training/validation
    if args.use_weighted_loss:
        criterion = LpLoss(d=2, p=1, size_average=False, reduction=True)
        criterion.forward = lambda x, y: criterion.weighted_rel_l1(x, y, beta=0.5)
        print("Using Weighted Relative L1 Loss (beta=0.5)")
    else:
        criterion = LpLoss(d=2, p=1, size_average=False, reduction=True)
        criterion.forward = lambda x, y: criterion.rel_l1(x, y)
        print("Using Relative L1 Loss")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine annealing LR schedule
    if args.lr_schedule == 'cosine':
        T_max = args.T_max if args.T_max is not None else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=args.eta_min
        )
        print(f"Using CosineAnnealingLR (T_max={T_max}, eta_min={args.eta_min})")
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
        print(f"Using StepLR (step_size={args.step_size}, gamma={args.gamma})")
    
    print("\nStarting training...")
    print("=" * 60)
    print(f"Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LR schedule: {args.lr_schedule}")
    print(f"  Loss: Relative L1")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Early stopping min_delta: {args.min_delta}")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        t1 = default_timer()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.dataset_type)
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device, args.dataset_type)
        else:
            val_loss = float('inf')
        scheduler.step()
        t2 = default_timer()
        
        # Print every 10 epochs or first epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_loader is not None:
                print(f"Epoch {epoch+1}/{args.epochs}: Time = {t2-t1:.2f}s, Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
            else:
                print(f"Epoch {epoch+1}/{args.epochs}: Time = {t2-t1:.2f}s, Train Loss = {train_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping and save best model (only if validation set exists)
        if val_loader is not None:
            improvement = best_val_loss - val_loss
            if improvement > args.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
                checkpoint_dir = os.path.join(scratch_dir, 'checkpoints', 'geofno')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_suffix = ''
                if args.use_weighted_loss:
                    checkpoint_suffix = '_weighted_loss'
                elif args.use_multiplicative_conditioning:
                    checkpoint_suffix = '_multiplicative_cond'
                elif args.use_material_adaptive_pca:
                    checkpoint_suffix = '_adaptive_pca'
                checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best_{}{}.pt'.format(args.dataset_type, checkpoint_suffix))
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
                print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                break
        else:
            best_val_loss = train_loss
    
    print("=" * 60)
    print("Training complete!")
    if val_loader is not None:
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device, args.dataset_type)
    print(f"Test loss (full mesh): {test_loss:.6f}")


if __name__ == "__main__":
    main()

