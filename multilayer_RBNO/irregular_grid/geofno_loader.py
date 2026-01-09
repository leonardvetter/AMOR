"""
Geo-FNO dataset loader for irregular grid datasets (airfoil, elasticity).

Handles loading of Geo-FNO datasets which contain irregular mesh data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict
import os

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_geofno_airfoil(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Geo-FNO airfoil dataset.
    
    Airfoil data is on a regular grid, but we can extract it as point cloud.
    Input: X and Y coordinates (num_samples, height, width, 2)
    Output: Q field (num_samples, height, width)
    
    Args:
        data_dir: Directory containing NACA_Cylinder_*.npy files
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes)
        targets: Output data (num_samples, num_channels, num_nodes)
        mesh_coords: Mesh coordinates (num_nodes, coord_dim) - constant across samples
    """
    x_file = os.path.join(data_dir, 'NACA_Cylinder_X.npy')
    y_file = os.path.join(data_dir, 'NACA_Cylinder_Y.npy')
    q_file = os.path.join(data_dir, 'NACA_Cylinder_Q.npy')
    
    if not os.path.exists(x_file):
        raise FileNotFoundError(f"Airfoil X file not found: {x_file}")
    if not os.path.exists(y_file):
        raise FileNotFoundError(f"Airfoil Y file not found: {y_file}")
    if not os.path.exists(q_file):
        raise FileNotFoundError(f"Airfoil Q file not found: {q_file}")
    
    input_x = np.load(x_file)
    input_y = np.load(y_file)
    output_q = np.load(q_file)[:, 4]
    
    num_samples, height, width = input_x.shape
    
    input_data = np.stack([input_x, input_y], axis=-1)
    input_data = input_data.reshape(num_samples, height * width, 2)
    
    output_data = output_q.reshape(num_samples, height * width)
    
    mesh_coords = input_data[0]
    
    data = input_data.transpose(0, 2, 1)
    targets = output_data[:, np.newaxis, :]
    
    return data, targets, mesh_coords


def load_geofno_elasticity(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load Geo-FNO elasticity dataset.
    
    Elasticity data is on irregular mesh.
    Input: xy (mesh coordinates) - (num_nodes, 2, num_samples) -> (num_samples, num_nodes, 2)
    Output: sigma (stress) - (num_nodes, num_samples) -> (num_samples, num_nodes)
    Material properties: rr (42 values per sample) - used as input features
    
    Args:
        data_dir: Directory containing Random_UnitCell_*.npy files
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes) - mesh coordinates as channels
        targets: Output data (num_samples, num_channels, num_nodes) - stress field
        mesh_coords: Mesh coordinates (num_nodes, coord_dim) - constant across samples
        material_props: Material properties (num_samples, num_props) - optional, None if not found
    """
    xy_files = [f for f in os.listdir(data_dir) if f.startswith('Random_UnitCell_XY') and f.endswith('.npy')]
    sigma_files = [f for f in os.listdir(data_dir) if f.startswith('Random_UnitCell_sigma') and f.endswith('.npy')]
    rr_files = [f for f in os.listdir(data_dir) if f.startswith('Random_UnitCell_rr') and f.endswith('.npy')]
    
    if not xy_files or not sigma_files:
        raise FileNotFoundError(f"Required elasticity files not found in {data_dir}. Need Random_UnitCell_XY*.npy, Random_UnitCell_sigma*.npy")
    
    xy_file = os.path.join(data_dir, xy_files[0])
    sigma_file = os.path.join(data_dir, sigma_files[0])
    
    if len(xy_files) > 1 or len(sigma_files) > 1:
        print(f"Warning: Multiple elasticity files found. Using: {xy_files[0]}, {sigma_files[0]}")
    
    input_xy = np.load(xy_file)
    input_sigma = np.load(sigma_file)
    
    input_sigma = input_sigma.T
    input_xy = input_xy.transpose(2, 0, 1)
    
    num_samples, num_nodes, coord_dim = input_xy.shape
    num_samples_sigma, num_nodes_sigma = input_sigma.shape
    
    if num_samples != num_samples_sigma:
        raise ValueError(f"Number of samples mismatch: xy has {num_samples}, sigma has {num_samples_sigma}")
    
    if num_nodes != num_nodes_sigma:
        raise ValueError(f"Number of nodes mismatch: xy has {num_nodes}, sigma has {num_nodes_sigma}")
    
    mesh_coords = input_xy[0]
    
    data = input_xy.transpose(0, 2, 1)
    targets = input_sigma[:, np.newaxis, :]
    
    material_props = None
    if rr_files:
        rr_file = os.path.join(data_dir, rr_files[0])
        input_rr = np.load(rr_file)
        input_rr = input_rr.T
        if input_rr.shape[0] == num_samples:
            material_props = input_rr
            print(f"Loaded material properties (rr): shape {material_props.shape}")
        else:
            print(f"Warning: Material properties shape {input_rr.shape} doesn't match number of samples {num_samples}")
    
    return data, targets, mesh_coords, material_props


def load_geofno_dataset(data_path: str, dataset_type: str = 'auto') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load Geo-FNO dataset from directory or file.
    
    Args:
        data_path: Path to dataset directory (for airfoil/elasticity) or file
        dataset_type: 'airfoil', 'elasticity', or 'auto' (detect from path)
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes)
        targets: Output data (num_samples, num_channels, num_nodes)
        mesh_coords: Mesh coordinates (num_nodes, coord_dim)
        material_props: Material properties (num_samples, num_props) or None
    """
    if dataset_type == 'auto':
        if 'naca' in data_path.lower() or 'airfoil' in data_path.lower():
            dataset_type = 'airfoil'
        elif 'diffeo' in data_path.lower() or 'elastic' in data_path.lower():
            dataset_type = 'elasticity'
        else:
            raise ValueError(f"Could not auto-detect dataset type from path: {data_path}")
    
    if dataset_type == 'airfoil':
        data, targets, mesh_coords = load_geofno_airfoil(data_path)
        return data, targets, mesh_coords, None
    elif dataset_type == 'elasticity':
        return load_geofno_elasticity(data_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def prepare_geofno_data(data_path: str,
                        dataset_type: str = 'auto',
                        output_dir: Optional[str] = None,
                        train_split: float = 0.7,
                        val_split: float = 0.2,
                        normalize: bool = True,
                        random_seed: int = 42) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare Geo-FNO dataset (airfoil or elasticity) for training.
    
    Args:
        data_path: Path to dataset directory (for airfoil/elasticity)
        dataset_type: 'airfoil', 'elasticity', or 'auto' (detect from path)
        output_dir: Directory to save processed data (default: $SCRATCH/data/geofno)
        train_split: Fraction of data for training
        val_split: Fraction of data for validation (test gets remainder)
        normalize: Whether to normalize data using training statistics
        random_seed: Random seed for train/val/test split
        
    Returns:
        data_dict: Dictionary with keys 'train', 'val', 'test' containing datasets
        mesh_coords: Mesh coordinates (num_nodes, coord_dim)
    """
    from data_utils import IrregularGridDataset
    
    if output_dir is None:
        scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
        output_dir = os.path.join(scratch_dir, 'data', 'geofno')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Geo-FNO dataset from {data_path}...")
    data, targets, mesh_coords, material_props = load_geofno_dataset(data_path, dataset_type)
    
    print(f"Dataset loaded:")
    print(f"  Input shape: {data.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Mesh coords shape: {mesh_coords.shape}")
    print(f"  Number of samples: {data.shape[0]}")
    print(f"  Number of nodes: {mesh_coords.shape[0]}")
    
    num_samples = data.shape[0]
    indices = np.random.RandomState(random_seed).permutation(num_samples)
    
    n_train = int(train_split * num_samples)
    n_val = int(val_split * num_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    data_train = data[train_indices]
    targets_train = targets[train_indices]
    data_val = data[val_indices]
    targets_val = targets[val_indices]
    data_test = data[test_indices]
    targets_test = targets[test_indices]
    
    if normalize:
        data_mean = data_train.mean(axis=(0, 2), keepdims=True)
        data_std = data_train.std(axis=(0, 2), keepdims=True) + 1e-8
        targets_mean = targets_train.mean(axis=(0, 2), keepdims=True)
        targets_std = targets_train.std(axis=(0, 2), keepdims=True) + 1e-8
        
        data_train = (data_train - data_mean) / data_std
        data_val = (data_val - data_mean) / data_std
        data_test = (data_test - data_mean) / data_std
        targets_train = (targets_train - targets_mean) / targets_std
        targets_val = (targets_val - targets_mean) / targets_std
        targets_test = (targets_test - targets_mean) / targets_std
        
        print(f"Data normalized using training statistics")
    
    material_props_train = material_props[train_indices] if material_props is not None else None
    material_props_val = material_props[val_indices] if material_props is not None else None
    material_props_test = material_props[test_indices] if material_props is not None else None
    
    train_dataset = IrregularGridDataset(data_train, targets_train, mesh_coords, material_props_train)
    val_dataset = IrregularGridDataset(data_val, targets_val, mesh_coords, material_props_val)
    test_dataset = IrregularGridDataset(data_test, targets_test, mesh_coords, material_props_test)
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }, mesh_coords, material_props
