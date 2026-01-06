"""
Test script for irregular grid MRBNO implementation.

Tests data loading, PCA computation, and model instantiation with synthetic data.
"""

import sys
import os

print("Checking dependencies...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy not available: {e}")
    sys.exit(1)

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"  ✗ PyTorch not available: {e}")
    print("  This test requires PyTorch. Please run in an environment with PyTorch installed.")
    sys.exit(1)

try:
    from codbench_elasticity_loader import prepare_codbench_elasticity_data, CoDBenchElasticityDataset
    print("  ✓ codbench_elasticity_loader imported")
except ImportError as e:
    print(f"  ✗ Failed to import codbench_elasticity_loader: {e}")
    sys.exit(1)

try:
    from pca_utils import compute_pca_pointcloud
    print("  ✓ pca_utils imported")
except ImportError as e:
    print(f"  ✗ Failed to import pca_utils: {e}")
    sys.exit(1)

try:
    from MRBNO_lift_irregular import MRBNOLiftIrregular
    print("  ✓ MRBNO_lift_irregular imported")
except ImportError as e:
    print(f"  ✗ Failed to import MRBNO_lift_irregular: {e}")
    sys.exit(1)

print()


def create_synthetic_elasticity_data(num_samples=100, num_nodes=500, num_channels=1, coord_dim=2):
    """
    Create synthetic elasticity dataset for testing.
    
    Args:
        num_samples: Number of samples
        num_nodes: Number of mesh nodes
        num_channels: Number of input/output channels
        coord_dim: Coordinate dimension (2D or 3D)
    
    Returns:
        data: Input data (num_samples, num_channels, num_nodes)
        targets: Output stress data (num_samples, num_channels, num_nodes)
        mesh_coords: Mesh coordinates (num_nodes, coord_dim)
    """
    np.random.seed(42)
    
    mesh_coords = np.random.rand(num_nodes, coord_dim).astype(np.float32)
    
    data = np.random.randn(num_samples, num_channels, num_nodes).astype(np.float32)
    
    targets = data * 0.5 + np.random.randn(num_samples, num_channels, num_nodes).astype(np.float32) * 0.1
    
    return data, targets, mesh_coords


def test_data_loader():
    """Test data loader functionality."""
    print("=" * 60)
    print("Testing data loader...")
    print("=" * 60)
    
    data, targets, mesh_coords = create_synthetic_elasticity_data(num_samples=100, num_nodes=500)
    
    print(f"Created synthetic data:")
    print(f"  Input shape: {data.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Mesh coords shape: {mesh_coords.shape}")
    
    dataset = CoDBenchElasticityDataset(data, targets, mesh_coords)
    print(f"\nDataset length: {len(dataset)}")
    
    sample_data, sample_target = dataset[0]
    print(f"Sample input shape: {sample_data.shape}")
    print(f"Sample target shape: {sample_target.shape}")
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch_data, batch_target = next(iter(dataloader))
    print(f"Batch input shape: {batch_data.shape}")
    print(f"Batch target shape: {batch_target.shape}")
    
    print("✓ Data loader test passed!")
    return data, targets, mesh_coords


def test_pca_computation(data):
    """Test PCA computation."""
    print("\n" + "=" * 60)
    print("Testing PCA computation...")
    print("=" * 60)
    
    pca_components, pca_mean = compute_pca_pointcloud(data, n_components=50)
    
    print(f"PCA components shape: {pca_components.shape}")
    print(f"PCA mean shape: {pca_mean.shape}")
    print(f"Number of nodes: {data.shape[2]}")
    
    assert pca_components.shape[0] == 50, "Wrong number of PCA components"
    assert pca_components.shape[1] == data.shape[2], "PCA components should match number of nodes"
    assert pca_mean.shape[0] == data.shape[2], "PCA mean should match number of nodes"
    
    print("✓ PCA computation test passed!")
    return pca_components, pca_mean


def test_model_instantiation(pca_components, pca_mean, mesh_coords):
    """Test model instantiation and forward pass."""
    print("\n" + "=" * 60)
    print("Testing model instantiation...")
    print("=" * 60)
    
    num_nodes = mesh_coords.shape[0]
    n_modes = min(50, pca_components.shape[0])
    
    model = MRBNOLiftIrregular(
        in_channels=1,
        out_channels=1,
        width=32,
        n_modes=n_modes,
        n_layers=2,
        add_coords=True,
        norm="group",
        dropout_p=0.0,
        residual=True,
        pca_components=pca_components,
        pca_mean=pca_mean,
        mesh_coords=mesh_coords
    )
    
    print(f"Model created successfully!")
    print(f"  Input channels: 1")
    print(f"  Output channels: 1")
    print(f"  Width: 32")
    print(f"  Number of modes: {n_modes}")
    print(f"  Number of layers: 2")
    print(f"  Add coords: True")
    
    batch_size = 4
    x = torch.randn(batch_size, 1, num_nodes)
    
    print(f"\nTesting forward pass with input shape: {x.shape}")
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    assert y.shape == (batch_size, 1, num_nodes), f"Expected output shape {(batch_size, 1, num_nodes)}, got {y.shape}"
    
    print("✓ Model instantiation and forward pass test passed!")
    return model


def test_full_pipeline():
    """Test full pipeline: data -> PCA -> model."""
    print("\n" + "=" * 60)
    print("Testing full pipeline...")
    print("=" * 60)
    
    data, targets, mesh_coords = create_synthetic_elasticity_data(num_samples=50, num_nodes=300)
    
    train_data = data[:40]
    train_targets = targets[:40]
    
    max_components = min(40, train_data.shape[2])
    n_components = min(32, max_components)
    pca_components, pca_mean = compute_pca_pointcloud(train_data, n_components=n_components)
    
    n_modes = min(n_components, pca_components.shape[0])
    model = MRBNOLiftIrregular(
        in_channels=1,
        out_channels=1,
        width=16,
        n_modes=n_modes,
        n_layers=2,
        add_coords=True,
        pca_components=pca_components,
        pca_mean=pca_mean,
        mesh_coords=mesh_coords
    )
    
    dataset = CoDBenchElasticityDataset(data[40:], targets[40:], mesh_coords)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data, batch_target in dataloader:
            output = model(batch_data)
            loss = torch.nn.functional.mse_loss(output, batch_target)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Average MSE loss on test set: {avg_loss:.6f}")
    print(f"Processed {num_batches} batches")
    
    print("✓ Full pipeline test passed!")
    return True


def main():
    """Run all tests."""
    print("Starting irregular grid MRBNO tests...")
    print()
    
    try:
        data, targets, mesh_coords = test_data_loader()
        pca_components, pca_mean = test_pca_computation(data)
        model = test_model_instantiation(pca_components, pca_mean, mesh_coords)
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

