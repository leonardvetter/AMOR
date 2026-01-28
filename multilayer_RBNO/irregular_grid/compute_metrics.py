"""
Compute error metrics (rel L1, rel L2, Geo-FNO metric) for all dataset splits.

Geo-FNO metric is relative L2 loss (LpLoss with p=2, size_average=False).
"""

import torch
import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from geofno_loader import prepare_geofno_data
from losses import LpLoss
from MRBNO import MRBNO
from PCAReducer import PCAReducerMultiChannel
from torch.utils.data import DataLoader


def reshape_to_4d(data, dataset_type):
    """Reshape 3D data (B, C, N) to 4D format (B, C, N, M) for new MRBNO model."""
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
        is_tensor = True
        device = data.device
    else:
        data_np = data
        is_tensor = False
    
    if dataset_type == 'airfoil':
        B, C, N = data_np.shape
        height, width = 221, 51
        data_4d = data_np.reshape(B, C, height, width)
    elif dataset_type == 'elasticity':
        B, C, N = data_np.shape
        if N == 972:
            height, width = 36, 27
        else:
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
        return torch.from_numpy(data_4d).to(device)
    else:
        return data_4d


def reshape_from_4d(data_4d, dataset_type, original_shape):
    """Reshape 4D data (B, C, N, M) back to 3D format (B, C, N)."""
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
    
    if data_3d.shape != original_shape:
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


def compute_metrics(model, dataloader, device, dataset_type):
    """Compute rel L1, rel L2, and Geo-FNO (rel L2) metrics."""
    model.eval()
    
    rel_l1_criterion = LpLoss(d=2, p=1, size_average=False, reduction=True)
    rel_l1_criterion.forward = lambda x, y: rel_l1_criterion.rel_l1(x, y)
    
    rel_l2_criterion = LpLoss(d=2, p=2, size_average=False, reduction=True)
    rel_l2_criterion.forward = lambda x, y: rel_l2_criterion.rel(x, y)
    
    geofno_criterion = LpLoss(d=2, p=2, size_average=False, reduction=True)
    geofno_criterion.forward = lambda x, y: geofno_criterion.rel(x, y)
    
    rel_l1_total = 0.0
    rel_l2_total = 0.0
    geofno_total = 0.0
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
    
    # Reshape training data to 4D format
    train_data_3d = train_dataset.data.numpy()
    train_targets_3d = train_dataset.targets.numpy()
    train_data_4d = reshape_to_4d(train_data_3d, dataset_type)
    train_targets_4d = reshape_to_4d(train_targets_3d, dataset_type)
    
    # Recreate PCAReducerMultiChannel
    rb = PCAReducerMultiChannel(
        svd_solver="auto",
        whiten=False,
        random_state=42,
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False
    ).fit(train_data_4d, train_targets_4d)
    
    # Get k_in and k_out from checkpoint or recompute
    k_in = checkpoint.get('k_in', args.get('n_modes', 144))
    k_out = checkpoint.get('k_out', args.get('n_modes', 144))
    
    # Create model
    model = MRBNO(
        width=args['width'],
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=args['n_layers'],
        dropout=0.0,
        init_scale=0.1,
        use_skip_connections=True,
        use_batch_norm=True,
        use_coordinates=args.get('add_coords', True),
        add_constant_mode=False,
        use_factorized_weights=False
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
    train_metrics = compute_metrics(model, train_loader, device, args.dataset_type)
    
    print("Computing metrics on TEST split...")
    test_metrics = compute_metrics(model, test_loader, device, args.dataset_type)
    
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print("Computing metrics on VALIDATION split...")
        val_metrics = compute_metrics(model, val_loader, device, args.dataset_type)
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

