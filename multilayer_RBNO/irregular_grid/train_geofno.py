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

from geofno_loader import prepare_geofno_data
from data_utils import compute_pca_for_model
from MRBNO_lift_irregular import MRBNOLiftIrregular


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data, batch_target in dataloader:
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        
        optimizer.zero_grad()
        output = model(batch_data)
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
    
    with torch.no_grad():
        for batch_data, batch_target in dataloader:
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
                       help='Path to Geo-FNO dataset file (.npz, .h5, or .hdf5)')
    parser.add_argument('--dataset_type', type=str, default='airfoil',
                       choices=['airfoil', 'elasticity'],
                       help='Type of dataset')
    parser.add_argument('--n_modes', type=int, default=64,
                       help='Number of PCA modes')
    parser.add_argument('--width', type=int, default=32,
                       help='Model width')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--add_coords', action='store_true',
                       help='Add coordinate channels')
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
    datasets, mesh_coords = prepare_geofno_data(
        args.dataset_path,
        dataset_type=args.dataset_type,
        train_split=0.7,
        val_split=0.2,
        normalize=True
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    test_dataset = datasets['test']
    
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
        norm="group",
        dropout_p=0.0,
        residual=True,
        pca_components=pca_components,
        pca_mean=pca_mean,
        mesh_coords=mesh_coords
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
            checkpoint_dir = os.path.join(scratch_dir, 'checkpoints', 'geofno')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_best_{args.dataset_type}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()

