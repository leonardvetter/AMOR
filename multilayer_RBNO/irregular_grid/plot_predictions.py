"""
Plot MRBNO predictions for Geo-FNO datasets.

Plots one sample (default index 42) showing:
- Input
- Output (prediction)
- Ground truth
- Error between prediction and ground truth
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Import from local directory first (for MRBNO and PCAReducer with 3D support)
local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, local_dir)
# Don't import from models directory - use local PCAReducer that handles 3D

from geofno_loader import prepare_geofno_data
from MRBNO import MRBNO
# Import local PCAReducer (handles 3D data)
import importlib.util
pca_path = os.path.join(local_dir, 'PCAReducer.py')
spec = importlib.util.spec_from_file_location("PCAReducer", pca_path)
PCAReducer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PCAReducer_module)
PCAReducerMultiChannel = PCAReducer_module.PCAReducerMultiChannel




def load_model(checkpoint_path, dataset_type, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint validation loss: {checkpoint['val_loss']:.6f}")
    
    datasets, mesh_coords, material_props = prepare_geofno_data(
        args['dataset_path'],
        dataset_type=dataset_type,
        normalize=False,
        use_geofno_split=True
    )
    
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    
    # PCAReducer now accepts 3D data directly
    train_data_3d = train_dataset.data.numpy()
    train_targets_3d = train_dataset.targets.numpy()
    
    # Recreate PCAReducerMultiChannel
    rb = PCAReducerMultiChannel(
        svd_solver="auto",
        whiten=False,
        random_state=42,
        bc_weight=1.0,
        boundary_threshold=0.1,
        use_boundary_weighting=False
    ).fit(train_data_3d, train_targets_3d)
    
    # Get k_in and k_out from checkpoint
    k_in = checkpoint.get('k_in', args.get('n_modes', 144))
    k_out = checkpoint.get('k_out', args.get('n_modes', 144))
    
    # Create model - use exact args from checkpoint
    use_coords = args.get('add_coords', args.get('use_coordinates', False))
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
        use_coordinates=use_coords,
        add_constant_mode=False,
        use_factorized_weights=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, test_dataset, mesh_coords, material_props


def _plot_four_panel_scatter(xy, pred_1d, true_1d, abs_error_1d, rel_error_1d, title_prefix, sample_idx, save_path=None,
                             mesh_title='Mesh / point cloud', pred_title='MRBNO prediction', gt_title='Ground truth', error_title='Absolute error'):
    """Generic 4-panel plot: mesh, ground truth, prediction, error. xy shape (N, 2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmin = min(pred_1d.min(), true_1d.min())
    vmax = max(pred_1d.max(), true_1d.max())
    cmap = 'RdBu_r'

    ax = axes[0, 0]
    ax.scatter(xy[:, 0], xy[:, 1], c='steelblue', s=8, alpha=0.7, edgecolors='none')
    ax.set_title(f'{mesh_title}\nSample {sample_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=pred_1d, cmap=cmap, s=15, vmin=vmin, vmax=vmax)
    ax.set_title(f'{pred_title}\nSample {sample_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=true_1d, cmap=cmap, s=15, vmin=vmin, vmax=vmax)
    ax.set_title(f'{gt_title}\nSample {sample_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=abs_error_1d, cmap='hot', s=15)
    ax.set_title(f'{error_title}\nSample {sample_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='|pred − truth|')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title_prefix, fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def plot_elasticity_prediction(model, test_dataset, mesh_coords, material_props, sample_idx, device, save_path=None):
    """Plot elasticity: mesh, ground truth (stress σ), prediction, error."""
    sample_data = test_dataset.data[sample_idx:sample_idx+1].to(device)
    sample_target = test_dataset.targets[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        prediction = model(sample_data)
    pred_sigma = prediction[0, 0].cpu().numpy()
    true_sigma = sample_target[0, 0].cpu().numpy()
    abs_error = np.abs(pred_sigma - true_sigma)
    rel_error = abs_error / (np.abs(true_sigma) + 1e-8)
    xy = np.asarray(mesh_coords)
    if xy.shape[1] > 2:
        xy = xy[:, :2]
    _plot_four_panel_scatter(
        xy, pred_sigma, true_sigma, abs_error, rel_error,
        'Elasticity: stress field (σ)', sample_idx, save_path,
        mesh_title='Mesh (X, Y)', pred_title='Prediction (σ)', gt_title='Ground truth (σ)', error_title='Error |pred − truth|'
    )
    print(f"Elasticity sample {sample_idx}: pred range [{pred_sigma.min():.4f}, {pred_sigma.max():.4f}], mean abs err {abs_error.mean():.4f}, mean rel err {rel_error.mean():.4f}")


def plot_airfoil_prediction(model, test_dataset, mesh_coords, sample_idx, device, save_path=None, title_prefix='Airfoil: Q field'):
    """Plot airfoil (or pipe/channel-shocks): mesh, ground truth Q, prediction, error."""
    sample_data = test_dataset.data[sample_idx:sample_idx+1].to(device)
    sample_target = test_dataset.targets[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        prediction = model(sample_data)
    pred_q = prediction[0, 0].cpu().numpy()
    true_q = sample_target[0, 0].cpu().numpy()
    abs_error = np.abs(pred_q - true_q)
    rel_error = abs_error / (np.abs(true_q) + 1e-8)
    xy = np.asarray(mesh_coords)
    if xy.shape[1] > 2:
        xy = xy[:, :2]
    _plot_four_panel_scatter(
        xy, pred_q, true_q, abs_error, rel_error,
        title_prefix, sample_idx, save_path,
        mesh_title='Mesh (X, Y)', pred_title='Prediction (Q)', gt_title='Ground truth (Q)', error_title='Error |pred − truth|'
    )
    print(f"Sample {sample_idx}: pred range [{pred_q.min():.4f}, {pred_q.max():.4f}], mean abs err {abs_error.mean():.4f}")


def plot_plasticity_prediction(model, test_dataset, mesh_coords, sample_idx, device, save_path=None):
    """Plasticity: output (2480, num_nodes). Plot first output channel; mesh is placeholder so use node-index layout."""
    sample_data = test_dataset.data[sample_idx:sample_idx+1].to(device)
    sample_target = test_dataset.targets[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        prediction = model(sample_data)
    # Use first channel (one time/spatial slice) for visualization
    pred_1d = prediction[0, 0].cpu().numpy()   # (num_nodes,)
    true_1d = sample_target[0, 0].cpu().numpy()
    abs_error = np.abs(pred_1d - true_1d)
    rel_error = abs_error / (np.abs(true_1d) + 1e-8)
    num_nodes = pred_1d.size
    # No real mesh: use 2D layout by node index (e.g. 10x11 for 101 nodes)
    n2 = int(np.ceil(np.sqrt(num_nodes)))
    n1 = (num_nodes + n2 - 1) // n2
    node_x = np.arange(num_nodes) % n1
    node_y = np.arange(num_nodes) // n1
    xy = np.stack([node_x, node_y], axis=1).astype(np.float64)
    _plot_four_panel_scatter(
        xy, pred_1d, true_1d, abs_error, rel_error,
        'Plasticity: first output channel (node layout)', sample_idx, save_path,
        mesh_title='Node layout (index)', pred_title='Prediction', gt_title='Ground truth', error_title='Error |pred − truth|'
    )
    print(f"Plasticity sample {sample_idx}: pred range [{pred_1d.min():.4f}, {pred_1d.max():.4f}], mean abs err {abs_error.mean():.4f}")


def plot_car_cfd_prediction(model, test_dataset, mesh_coords, sample_idx, device, save_path=None):
    """Car-CFD: 3D mesh, pressure output. Use 2D projection (X, Y) for visualization."""
    sample_data = test_dataset.data[sample_idx:sample_idx+1].to(device)
    sample_target = test_dataset.targets[sample_idx:sample_idx+1].to(device)
    with torch.no_grad():
        prediction = model(sample_data)
    pred_1d = prediction[0, 0].cpu().numpy()
    true_1d = sample_target[0, 0].cpu().numpy()
    abs_error = np.abs(pred_1d - true_1d)
    rel_error = abs_error / (np.abs(true_1d) + 1e-8)
    xy = np.asarray(mesh_coords)
    if xy.shape[1] >= 2:
        xy = xy[:, :2]
    else:
        xy = np.stack([np.arange(len(pred_1d)), np.zeros(len(pred_1d))], axis=1)
    _plot_four_panel_scatter(
        xy, pred_1d, true_1d, abs_error, rel_error,
        'Car-CFD: pressure (X–Y projection)', sample_idx, save_path,
        mesh_title='Point cloud (X, Y)', pred_title='Prediction (pressure)', gt_title='Ground truth (pressure)', error_title='Error |pred − truth|'
    )
    print(f"Car-CFD sample {sample_idx}: pred range [{pred_1d.min():.4f}, {pred_1d.max():.4f}], mean abs err {abs_error.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot MRBNO predictions')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['elasticity', 'airfoil', 'pipe', 'plasticity', 'car-cfd', 'channel-shocks'],
                       help='Dataset type')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--sample_idx', type=int, default=42,
                       help='Sample index to plot (default: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: $SCRATCH/data/geofno_predictions)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu to avoid OOM)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    if args.output_dir is None:
        scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
        args.output_dir = os.path.join(scratch_dir, 'data', 'geofno_predictions')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MRBNO Prediction Visualization")
    print("=" * 60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print()
    
    model, test_dataset, mesh_coords, material_props = load_model(
        args.checkpoint_path, args.dataset_type, device
    )
    
    if args.sample_idx >= len(test_dataset):
        print(f"Warning: Sample index {args.sample_idx} >= dataset size {len(test_dataset)}. Using index 0.")
        args.sample_idx = 0
    
    if args.dataset_type == 'elasticity':
        save_path = os.path.join(args.output_dir, f'elasticity_prediction_sample_{args.sample_idx}.png')
        plot_elasticity_prediction(model, test_dataset, mesh_coords, material_props, args.sample_idx, device, save_path)
    elif args.dataset_type == 'airfoil':
        save_path = os.path.join(args.output_dir, f'airfoil_prediction_sample_{args.sample_idx}.png')
        plot_airfoil_prediction(model, test_dataset, mesh_coords, args.sample_idx, device, save_path, title_prefix='Airfoil: Q field')
    elif args.dataset_type == 'pipe':
        save_path = os.path.join(args.output_dir, f'pipe_prediction_sample_{args.sample_idx}.png')
        plot_airfoil_prediction(model, test_dataset, mesh_coords, args.sample_idx, device, save_path, title_prefix='Pipe: Q field')
    elif args.dataset_type == 'channel-shocks':
        save_path = os.path.join(args.output_dir, f'channel_shocks_prediction_sample_{args.sample_idx}.png')
        plot_airfoil_prediction(model, test_dataset, mesh_coords, args.sample_idx, device, save_path, title_prefix='Channel-shocks: Q field')
    elif args.dataset_type == 'plasticity':
        save_path = os.path.join(args.output_dir, f'plasticity_prediction_sample_{args.sample_idx}.png')
        plot_plasticity_prediction(model, test_dataset, mesh_coords, args.sample_idx, device, save_path)
    elif args.dataset_type == 'car-cfd':
        save_path = os.path.join(args.output_dir, f'car_cfd_prediction_sample_{args.sample_idx}.png')
        plot_car_cfd_prediction(model, test_dataset, mesh_coords, args.sample_idx, device, save_path)
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()

