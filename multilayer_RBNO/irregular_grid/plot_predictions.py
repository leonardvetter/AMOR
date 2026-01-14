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

from geofno_loader import prepare_geofno_data
from data_utils import compute_pca_for_model
from MRBNO_lift_irregular import MRBNOLiftIrregular


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
        num_nodes=mesh_coords.shape[0]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, test_dataset, mesh_coords, material_props


def plot_elasticity_prediction(model, test_dataset, mesh_coords, material_props, sample_idx, device, save_path=None):
    """Plot elasticity prediction for one sample."""
    sample_data = test_dataset.data[sample_idx:sample_idx+1].to(device)
    sample_target = test_dataset.targets[sample_idx:sample_idx+1].to(device)
    
    if material_props is not None:
        sample_material = test_dataset.material_props[sample_idx:sample_idx+1].to(device)
    else:
        sample_material = None
    
    with torch.no_grad():
        prediction = model(sample_data, material_props=sample_material)
    
    input_xy = sample_data[0].cpu().numpy().T
    pred_sigma = prediction[0, 0].cpu().numpy()
    true_sigma = sample_target[0, 0].cpu().numpy()
    error = pred_sigma - true_sigma
    abs_error = np.abs(error)
    rel_error = abs_error / (np.abs(true_sigma) + 1e-8)
    
    xy = mesh_coords
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Input: mesh coordinates (X, Y)
    ax = axes[0, 0]
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c='blue', s=10, alpha=0.6, edgecolors='k', linewidths=0.1)
    ax.set_title(f'Input: Mesh Coordinates (X, Y)\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Use same colormap and scale for prediction and ground truth
    vmin = min(pred_sigma.min(), true_sigma.min())
    vmax = max(pred_sigma.max(), true_sigma.max())
    cmap = 'RdBu_r'
    
    # Prediction
    ax = axes[0, 1]
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=pred_sigma, cmap=cmap, s=50, vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.2)
    ax.set_title(f'Prediction: Stress Field (σ)\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Predicted stress')
    ax.grid(True, alpha=0.3)
    
    # Ground truth
    ax = axes[1, 0]
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=true_sigma, cmap=cmap, s=50, vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.2)
    ax.set_title(f'Ground Truth: Stress Field (σ)\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='True stress')
    ax.grid(True, alpha=0.3)
    
    # Relative error
    ax = axes[1, 1]
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=rel_error, cmap='hot', s=50, edgecolors='k', linewidths=0.2)
    ax.set_title(f'Relative Error: |Prediction - Ground Truth| / |Ground Truth|\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Relative error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved elasticity prediction plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    print(f"\nElasticity Sample {sample_idx} Statistics:")
    print(f"  Prediction range: [{pred_sigma.min():.6f}, {pred_sigma.max():.6f}]")
    print(f"  Ground truth range: [{true_sigma.min():.6f}, {true_sigma.max():.6f}]")
    print(f"  Mean absolute error: {abs_error.mean():.6f}")
    print(f"  Max absolute error: {abs_error.max():.6f}")
    print(f"  Mean relative error: {rel_error.mean():.6f}")
    print(f"  Max relative error: {rel_error.max():.6f}")


def plot_airfoil_prediction(model, test_dataset, mesh_coords, sample_idx, device, save_path=None):
    """Plot airfoil prediction for one sample."""
    sample_data = test_dataset.data[sample_idx:sample_idx+1].to(device)
    sample_target = test_dataset.targets[sample_idx:sample_idx+1].to(device)
    
    with torch.no_grad():
        prediction = model(sample_data, material_props=None)
    
    pred_q = prediction[0, 0].cpu().numpy()
    true_q = sample_target[0, 0].cpu().numpy()
    error = pred_q - true_q
    abs_error = np.abs(error)
    rel_error = abs_error / (np.abs(true_q) + 1e-8)
    
    xy = mesh_coords
    
    original_height, original_width = 221, 51
    pred_q_2d = pred_q.reshape(original_height, original_width)
    true_q_2d = true_q.reshape(original_height, original_width)
    rel_error_2d = rel_error.reshape(original_height, original_width)
    xy_2d = xy.reshape(original_height, original_width, 2)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Use same colormap and scale for prediction and ground truth
    vmin = min(pred_q_2d.min(), true_q_2d.min())
    vmax = max(pred_q_2d.max(), true_q_2d.max())
    
    # For better visualization, focus colormap on the range where most data falls
    # Use percentiles to avoid extreme outliers dominating the colormap
    p5_pred, p95_pred = np.percentile(pred_q_2d, [5, 95])
    p5_true, p95_true = np.percentile(true_q_2d, [5, 95])
    vmin_focused = min(p5_pred, p5_true)
    vmax_focused = max(p95_pred, p95_true)
    
    cmap = 'RdBu_r'
    n_levels = 50
    
    # Input: mesh coordinates - show as scatter to see grid points
    ax = axes[0, 0]
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c='blue', s=2, alpha=0.6)
    ax.set_title(f'Input: Mesh Coordinates (X, Y)\nSample {sample_idx} - {len(xy)} grid points')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Prediction - use focused colormap range for better detail
    ax = axes[0, 1]
    im = ax.contourf(xy_2d[:, :, 0], xy_2d[:, :, 1], pred_q_2d, levels=n_levels, cmap=cmap, vmin=vmin_focused, vmax=vmax_focused, extend='both')
    ax.contour(xy_2d[:, :, 0], xy_2d[:, :, 1], pred_q_2d, levels=20, colors='k', alpha=0.2, linewidths=0.5)
    ax.set_title(f'Prediction: Q Field\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Predicted Q field')
    ax.grid(True, alpha=0.3)
    
    # Ground truth - use same focused colormap range
    ax = axes[1, 0]
    im = ax.contourf(xy_2d[:, :, 0], xy_2d[:, :, 1], true_q_2d, levels=n_levels, cmap=cmap, vmin=vmin_focused, vmax=vmax_focused, extend='both')
    ax.contour(xy_2d[:, :, 0], xy_2d[:, :, 1], true_q_2d, levels=20, colors='k', alpha=0.2, linewidths=0.5)
    ax.set_title(f'Ground Truth: Q Field\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='True Q field')
    ax.grid(True, alpha=0.3)
    
    # Relative error
    ax = axes[1, 1]
    im = ax.contourf(xy_2d[:, :, 0], xy_2d[:, :, 1], rel_error_2d, levels=50, cmap='hot')
    ax.set_title(f'Relative Error: |Prediction - Ground Truth| / |Ground Truth|\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Relative error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved airfoil prediction plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    print(f"\nAirfoil Sample {sample_idx} Statistics:")
    print(f"  Prediction range: [{pred_q.min():.6f}, {pred_q.max():.6f}]")
    print(f"  Ground truth range: [{true_q.min():.6f}, {true_q.max():.6f}]")
    print(f"  Mean absolute error: {abs_error.mean():.6f}")
    print(f"  Max absolute error: {abs_error.max():.6f}")
    print(f"  Mean relative error: {rel_error.mean():.6f}")
    print(f"  Max relative error: {rel_error.max():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Plot MRBNO predictions')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['elasticity', 'airfoil'],
                       help='Dataset type')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--sample_idx', type=int, default=42,
                       help='Sample index to plot (default: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: $SCRATCH/data/geofno_predictions)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
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
        plot_elasticity_prediction(model, test_dataset, mesh_coords, material_props, 
                                  args.sample_idx, device, save_path)
    elif args.dataset_type == 'airfoil':
        save_path = os.path.join(args.output_dir, f'airfoil_prediction_sample_{args.sample_idx}.png')
        plot_airfoil_prediction(model, test_dataset, mesh_coords, args.sample_idx, device, save_path)
    
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()

