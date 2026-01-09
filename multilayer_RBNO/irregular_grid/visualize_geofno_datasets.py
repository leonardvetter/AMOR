"""
Visualize Geo-FNO datasets to understand their structure.

Plots one sample from each dataset (airfoil and elasticity) to visualize:
- Input fields
- Output fields
- Mesh structure
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from geofno_loader import load_geofno_dataset


def plot_elasticity_sample(data_dir, sample_idx=0, save_path=None):
    """Plot elasticity dataset sample."""
    data, targets, mesh_coords = load_geofno_dataset(data_dir, dataset_type='elasticity')
    
    input_rr = data[sample_idx, 0, :]  # (num_nodes,)
    output_sigma = targets[sample_idx, 0, :]  # (num_nodes,)
    xy = mesh_coords  # (num_nodes, 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Input: material properties
    ax = axes[0]
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=input_rr, cmap='viridis', s=10, edgecolors='k', linewidths=0.1)
    ax.set_title(f'Input: Material Properties (rr)\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Material property value')
    
    # Output: stress field
    ax = axes[1]
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=output_sigma, cmap='RdBu_r', s=10, edgecolors='k', linewidths=0.1)
    ax.set_title(f'Output: Stress Field (σ)\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Stress value')
    
    # Mesh structure
    ax = axes[2]
    ax.scatter(xy[:, 0], xy[:, 1], c='gray', s=5, alpha=0.5)
    ax.set_title(f'Mesh Structure\n{len(xy)} nodes')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved elasticity visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    print(f"\nElasticity Dataset Sample {sample_idx}:")
    print(f"  Input shape: {input_rr.shape}")
    print(f"  Output shape: {output_sigma.shape}")
    print(f"  Mesh coords shape: {xy.shape}")
    print(f"  Input range: [{input_rr.min():.4f}, {input_rr.max():.4f}]")
    print(f"  Output range: [{output_sigma.min():.4f}, {output_sigma.max():.4f}]")
    print(f"  Mesh X range: [{xy[:, 0].min():.4f}, {xy[:, 0].max():.4f}]")
    print(f"  Mesh Y range: [{xy[:, 1].min():.4f}, {xy[:, 1].max():.4f}]")


def plot_airfoil_sample(data_dir, sample_idx=0, save_path=None):
    """Plot airfoil dataset sample."""
    data, targets, mesh_coords = load_geofno_dataset(data_dir, dataset_type='airfoil')
    
    input_coords = data[sample_idx, :, :].T  # (num_nodes, 2) - X, Y coordinates
    output_q = targets[sample_idx, 0, :]  # (num_nodes,)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Input: mesh coordinates (geometry)
    ax = axes[0]
    scatter = ax.scatter(input_coords[:, 0], input_coords[:, 1], c='blue', s=1, alpha=0.6)
    ax.set_title(f'Input: Mesh Geometry (X, Y coordinates)\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    
    # Output: Q field
    ax = axes[1]
    scatter = ax.scatter(input_coords[:, 0], input_coords[:, 1], c=output_q, cmap='RdBu_r', s=1, edgecolors='none')
    ax.set_title(f'Output: Q Field (pressure/velocity)\nSample {sample_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Q field value')
    
    # Mesh structure
    ax = axes[2]
    ax.scatter(input_coords[:, 0], input_coords[:, 1], c='gray', s=1, alpha=0.5)
    ax.set_title(f'Mesh Structure\n{len(input_coords)} nodes')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved airfoil visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    print(f"\nAirfoil Dataset Sample {sample_idx}:")
    print(f"  Input coords shape: {input_coords.shape}")
    print(f"  Output shape: {output_q.shape}")
    print(f"  Input X range: [{input_coords[:, 0].min():.4f}, {input_coords[:, 0].max():.4f}]")
    print(f"  Input Y range: [{input_coords[:, 1].min():.4f}, {input_coords[:, 1].max():.4f}]")
    print(f"  Output range: [{output_q.min():.4f}, {output_q.max():.4f}]")


def main():
    scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
    geofno_data_dir = os.path.join(scratch_dir, 'data', 'Geo-FNO', 'data')
    
    elasticity_dir = os.path.join(geofno_data_dir, 'diffeo')
    airfoil_dir = os.path.join(geofno_data_dir, 'naca')
    
    output_dir = os.path.join(scratch_dir, 'data', 'geofno_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Geo-FNO Dataset Visualization")
    print("=" * 60)
    
    if os.path.exists(elasticity_dir):
        print("\nPlotting elasticity dataset...")
        elasticity_path = os.path.join(output_dir, 'elasticity_sample_0.png')
        plot_elasticity_sample(elasticity_dir, sample_idx=0, save_path=elasticity_path)
    else:
        print(f"\nElasticity dataset not found at {elasticity_dir}")
    
    if os.path.exists(airfoil_dir):
        print("\nPlotting airfoil dataset...")
        airfoil_path = os.path.join(output_dir, 'airfoil_sample_0.png')
        plot_airfoil_sample(airfoil_dir, sample_idx=0, save_path=airfoil_path)
    else:
        print(f"\nAirfoil dataset not found at {airfoil_dir}")
    
    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

