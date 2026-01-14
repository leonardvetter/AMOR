"""
Compute error metrics (rel L1, rel L2, Geo-FNO metric) for all dataset splits.

Geo-FNO metric is relative L2 loss (LpLoss with p=2, size_average=False).
"""

import torch
import numpy as np
import argparse
import os

from geofno_loader import prepare_geofno_data
from data_utils import compute_pca_for_model
from MRBNO_lift_irregular import MRBNOLiftIrregular
from losses import LpLoss
from torch.utils.data import DataLoader


def compute_metrics(model, dataloader, device):
    """Compute rel L1, rel L2, and Geo-FNO (rel L2) metrics."""
    model.eval()
    
    rel_l1_criterion = LpLoss(d=2, p=1, size_average=False, reduction=True)
    rel_l1_criterion.forward = lambda x, y: rel_l1_criterion.rel_l1(x, y)
    
    rel_l2_criterion = LpLoss(d=2, p=2, size_average=False, reduction=True)
    rel_l2_criterion.forward = lambda x, y: rel_l2_criterion.rel(x, y)
    
    geofno_criterion = LpLoss(d=2, p=2, size_average=False, reduction=True)
    geofno_criterion.forward = lambda x, y: geofno_criterion.rel(x, y)
    
    has_material = hasattr(model, 'use_material_conditioning') and model.use_material_conditioning
    
    rel_l1_total = 0.0
    rel_l2_total = 0.0
    geofno_total = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data, batch_target, batch_material = batch
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            batch_material = batch_material.to(device) if has_material and batch_material.numel() > 0 else None
            
            output = model(batch_data, material_props=batch_material)
            
            rel_l1_loss = rel_l1_criterion(output, batch_target)
            rel_l2_loss = rel_l2_criterion(output, batch_target)
            geofno_loss = geofno_criterion(output, batch_target)
            
            rel_l1_total += rel_l1_loss.item()
            rel_l2_total += rel_l2_loss.item()
            geofno_total += geofno_loss.item()
            num_batches += 1
    
    return {
        'rel_l1': rel_l1_total / num_batches if num_batches > 0 else 0.0,
        'rel_l2': rel_l2_total / num_batches if num_batches > 0 else 0.0,
        'geofno': geofno_total / num_batches if num_batches > 0 else 0.0
    }


def load_model(checkpoint_path, dataset_type, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    datasets, mesh_coords, material_props = prepare_geofno_data(
        args['dataset_path'],
        dataset_type=dataset_type,
        normalize=False,
        use_geofno_split=True
    )
    
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    val_dataset = datasets.get('val', None)
    
    material_prop_dim = None
    if material_props is not None:
        material_prop_dim = material_props.shape[1]
    
    train_data_np = train_dataset.data.numpy()
    pca_components, pca_mean = compute_pca_for_model(
        train_data_np,
        n_modes=args['n_modes']
    )
    
    model = MRBNOLiftIrregular(
        in_channels=train_dataset.data.shape[1],
        out_channels=train_dataset.targets.shape[1],
        width=args['width'],
        n_modes=min(args['n_modes'], pca_components.shape[0]),
        n_layers=args['n_layers'],
        add_coords=args.get('add_coords', True),
        use_grid_skip=args.get('use_grid_skip', False),
        use_input_skip=args.get('use_input_skip', False),
        norm="group",
        dropout_p=0.0,
        residual=True,
        pca_components=pca_components,
        pca_mean=pca_mean,
        mesh_coords=mesh_coords,
        material_prop_dim=material_prop_dim,
        num_nodes=mesh_coords.shape[0],
        use_multiplicative_conditioning=args.get('use_multiplicative_conditioning', False),
        material_adaptive_pca=args.get('use_material_adaptive_pca', False)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='Compute error metrics for all splits')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['elasticity', 'airfoil'],
                       help='Dataset type')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("=" * 80)
    print(f"Computing Metrics for {args.dataset_type.upper()} Dataset")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print()
    
    model, train_dataset, val_dataset, test_dataset = load_model(
        args.checkpoint_path, args.dataset_type, device
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Computing metrics on TRAIN split...")
    train_metrics = compute_metrics(model, train_loader, device)
    
    print("Computing metrics on TEST split...")
    test_metrics = compute_metrics(model, test_loader, device)
    
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print("Computing metrics on VALIDATION split...")
        val_metrics = compute_metrics(model, val_loader, device)
    else:
        val_metrics = None
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Split':<15} {'Rel L1':<15} {'Rel L2':<15} {'Geo-FNO (Rel L2)':<15}")
    print("-" * 80)
    print(f"{'Train':<15} {train_metrics['rel_l1']:<15.6f} {train_metrics['rel_l2']:<15.6f} {train_metrics['geofno']:<15.6f}")
    if val_metrics is not None:
        print(f"{'Validation':<15} {val_metrics['rel_l1']:<15.6f} {val_metrics['rel_l2']:<15.6f} {val_metrics['geofno']:<15.6f}")
    print(f"{'Test':<15} {test_metrics['rel_l1']:<15.6f} {test_metrics['rel_l2']:<15.6f} {test_metrics['geofno']:<15.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

