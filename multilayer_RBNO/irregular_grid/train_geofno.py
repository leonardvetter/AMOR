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

from geofno_loader import prepare_geofno_data
from data_utils import compute_pca_for_model
from MRBNO_lift_irregular import MRBNOLiftIrregular
from losses import LpLoss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    has_material = hasattr(model, 'use_material_conditioning') and model.use_material_conditioning
    
    for batch in dataloader:
        batch_data, batch_target, batch_material = batch
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        
        batch_material = batch_material.to(device) if has_material and batch_material.numel() > 0 else None
        
        optimizer.zero_grad()
        output = model(batch_data, material_props=batch_material)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    has_material = hasattr(model, 'use_material_conditioning') and model.use_material_conditioning
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data, batch_target, batch_material = batch
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            batch_material = batch_material.to(device) if has_material and batch_material.numel() > 0 else None
            
            output = model(batch_data, material_props=batch_material)
            loss = criterion(output, batch_target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_airfoil_downsampled(model, dataloader, criterion, device):
    """
    Validate airfoil model on downsampled mesh (56x51 instead of 221x51).
    
    Downsampling: take every 4th row in X direction (r1=4), keep all in Y direction (r2=1).
    Original: 221 x 51 = 11271 nodes
    Downsampled: 56 x 51 = 2856 nodes
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    has_material = hasattr(model, 'use_material_conditioning') and model.use_material_conditioning
    
    original_height, original_width = 221, 51
    downsampled_height, downsampled_width = 56, 51
    r1, r2 = 4, 1
    
    downsampled_indices = []
    for i in range(0, original_height, r1):
        row_start = i * original_width
        row_end = row_start + original_width
        downsampled_indices.extend(range(row_start, row_end))
    
    downsampled_indices = torch.tensor(downsampled_indices, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data, batch_target, batch_material = batch
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            batch_material = batch_material.to(device) if has_material and batch_material.numel() > 0 else None
            
            output = model(batch_data, material_props=batch_material)
            
            batch_size = output.shape[0]
            output_downsampled = output[:, :, downsampled_indices]
            target_downsampled = batch_target[:, :, downsampled_indices]
            
            loss = criterion(output_downsampled, target_downsampled)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


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
    
    print("\nComputing PCA basis...")
    train_data_np = train_dataset.data.numpy()
    pca_components, pca_mean = compute_pca_for_model(
        train_data_np,
        n_modes=args.n_modes
    )
    
    print(f"PCA computed: {pca_components.shape[0]} modes, {pca_components.shape[1]} nodes")
    
    print("\nCreating model...")
    model = MRBNOLiftIrregular(
        in_channels=train_dataset.data.shape[1],
        out_channels=train_dataset.targets.shape[1],
        width=args.width,
        n_modes=min(args.n_modes, pca_components.shape[0]),
        n_layers=args.n_layers,
        add_coords=args.add_coords,
        use_grid_skip=args.use_grid_skip,
        use_input_skip=args.use_input_skip,
        norm="group",
        dropout_p=0.0,
        residual=True,
        pca_components=pca_components,
        pca_mean=pca_mean,
        mesh_coords=mesh_coords,
        material_prop_dim=material_prop_dim,
        num_nodes=mesh_coords.shape[0],
        use_multiplicative_conditioning=args.use_multiplicative_conditioning,
        material_adaptive_pca=args.use_material_adaptive_pca
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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
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
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test loss (full mesh): {test_loss:.6f}")
    
    if args.dataset_type == 'airfoil':
        print("\nComputing test loss on downsampled mesh (56x51)...")
        test_loss_downsampled = validate_airfoil_downsampled(model, test_loader, criterion, device)
        print(f"Test loss (downsampled 56x51): {test_loss_downsampled:.6f}")


if __name__ == "__main__":
    main()

