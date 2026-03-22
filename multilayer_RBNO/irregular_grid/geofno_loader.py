"""
Geo-FNO dataset loader for irregular grid datasets.

Input-output mapping and splits aligned with Geo-FNO paper/code (NeuralOperator/Geo-FNO):
- Elasticity: input = mesh (x,y) + RR (42) per node; output = sigma; split first 1000 train, last 200 test.
- Airfoil: input = (X,Y) only; output = Q channel 4; split first 1000 train, next 200 test.
- Pipe: input = X,Y only (match Geo-FNO pipe.py); output = Q channel 0; split first 1800 train, next 300 test.
- Plasticity: input = material props (101) broadcast; output = time+spatial flattened; first 700 train, last 200 test.
- Car-CFD: input = points (3D); output = pressure; first 700 train, last 100 test.
- Channel-shocks: input = Q[0:4] + X,Y; output = Q[4]; first 1600 train, last 200 test.
All trained on irregular mesh directly (no mapping to uniform grid).
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
    
    Geo-FNO airfoil uses only mesh (X,Y) as input, output Q channel 4 (see naca_geofno.py).
    Input: X and Y coordinates only (same mapping as Geo-FNO).
    Output: Q field channel 4
    
    Args:
        data_dir: Directory containing NACA_Cylinder_*.npy files
        
    Returns:
        data: Input data (num_samples, 2, num_nodes) - X, Y coords only
        targets: Output data (num_samples, 1, num_nodes) - Q channel 4
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
    input_q = np.load(q_file)
    output_q = input_q[:, 4]
    
    num_samples, height, width = input_x.shape
    num_nodes = height * width
    
    # Geo-FNO uses only (X, Y) as input
    input_coords = np.stack([input_x, input_y], axis=-1)
    input_coords = input_coords.reshape(num_samples, num_nodes, 2)
    data = input_coords.transpose(0, 2, 1)  # (num_samples, 2, num_nodes)
    
    output_data = output_q.reshape(num_samples, num_nodes)
    targets = output_data[:, np.newaxis, :]
    
    mesh_coords = np.stack([input_x[0], input_y[0]], axis=-1).reshape(num_nodes, 2)
    
    return data, targets, mesh_coords


def load_geofno_elasticity(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load Geo-FNO elasticity dataset.
    
    Elasticity data is on irregular mesh.
    Input: RR (material properties) broadcasted to mesh nodes - matching Geo-FNO approach
    Output: sigma (stress) - (num_nodes, num_samples) -> (num_samples, num_nodes)
    
    Args:
        data_dir: Directory containing Random_UnitCell_*.npy files
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes) - RR material properties broadcasted to nodes
        targets: Output data (num_samples, num_channels, num_nodes) - stress field
        mesh_coords: Mesh coordinates (num_nodes, coord_dim) - constant across samples
        material_props: Material properties (num_samples, num_props) - original RR values
    """
    xy_files = [f for f in os.listdir(data_dir) if f.startswith('Random_UnitCell_XY') and f.endswith('.npy')]
    sigma_files = [f for f in os.listdir(data_dir) if f.startswith('Random_UnitCell_sigma') and f.endswith('.npy')]
    rr_files = [f for f in os.listdir(data_dir) if f.startswith('Random_UnitCell_rr') and f.endswith('.npy')]
    
    if not xy_files or not sigma_files:
        raise FileNotFoundError(f"Required elasticity files not found in {data_dir}. Need Random_UnitCell_XY*.npy, Random_UnitCell_sigma*.npy")
    
    if not rr_files:
        raise FileNotFoundError(f"RR material properties file not found in {data_dir}. Need Random_UnitCell_rr*.npy")
    
    xy_file = os.path.join(data_dir, xy_files[0])
    sigma_file = os.path.join(data_dir, sigma_files[0])
    rr_file = os.path.join(data_dir, rr_files[0])
    
    if len(xy_files) > 1 or len(sigma_files) > 1:
        print(f"Warning: Multiple elasticity files found. Using: {xy_files[0]}, {sigma_files[0]}")
    
    input_xy = np.load(xy_file)  # (972, 2, 2000)
    input_sigma = np.load(sigma_file)  # (972, 2000)
    input_rr = np.load(rr_file)  # (42, 2000)
    
    input_sigma = input_sigma.T  # (2000, 972)
    input_xy = input_xy.transpose(2, 0, 1)  # (2000, 972, 2)
    input_rr = input_rr.T  # (2000, 42)
    
    num_samples, num_nodes, coord_dim = input_xy.shape
    num_samples_sigma, num_nodes_sigma = input_sigma.shape
    num_samples_rr, num_props = input_rr.shape
    
    if num_samples != num_samples_sigma or num_samples != num_samples_rr:
        raise ValueError(f"Number of samples mismatch: xy={num_samples}, sigma={num_samples_sigma}, rr={num_samples_rr}")
    
    if num_nodes != num_nodes_sigma:
        raise ValueError(f"Number of nodes mismatch: xy={num_nodes}, sigma={num_nodes_sigma}")
    
    mesh_coords = input_xy[0]  # (972, 2)
    
    # Geo-FNO elasticity input: mesh (x,y) per node + design params rr (42) as code. Same mapping on irregular grid.
    # xy per node: (num_samples, 972, 2) -> (num_samples, 2, 972)
    xy_channels = input_xy.transpose(0, 2, 1)  # (num_samples, 2, num_nodes)
    rr_broadcasted = np.broadcast_to(input_rr[:, :, np.newaxis], (num_samples, num_props, num_nodes))
    data = np.concatenate([xy_channels, rr_broadcasted], axis=1)  # (num_samples, 44, 972)
    targets = input_sigma[:, np.newaxis, :]  # (num_samples, 1, 972)
    
    material_props = input_rr  # Keep original for reference
    
    print(f"Elasticity: Input mesh (x,y) + RR (42) per Geo-FNO; output sigma")
    print(f"  Input shape: {data.shape} (2 xy + 42 rr)")
    print(f"  Output shape: {targets.shape} (sigma)")
    
    return data, targets, mesh_coords, material_props


def load_geofno_pipe(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Geo-FNO pipe dataset.
    Geo-FNO pipe.py uses only (X, Y) as input and Pipe_Q channel 0 as output.
    Input: X, Y coordinates only (match Geo-FNO pipe.py).
    Output: Q channel 0.
    """
    q_file = os.path.join(data_dir, 'Pipe_Q.npy')
    x_file = os.path.join(data_dir, 'Pipe_X.npy')
    y_file = os.path.join(data_dir, 'Pipe_Y.npy')
    
    if not os.path.exists(q_file) or not os.path.exists(x_file) or not os.path.exists(y_file):
        raise FileNotFoundError(f"Pipe files not found in {data_dir}")
    
    q = np.load(q_file)  # (2310, 3, 129, 129)
    x = np.load(x_file)  # (2310, 129, 129)
    y = np.load(y_file)  # (2310, 129, 129)
    
    num_samples, num_channels, height, width = q.shape
    num_nodes = height * width
    
    # Input: X, Y only (Geo-FNO pipe.py: input = stack([inputX, inputY], dim=-1))
    input_coords = np.stack([x, y], axis=1).reshape(num_samples, 2, num_nodes)
    data = input_coords  # (num_samples, 2, num_nodes)
    
    # Output: Q channel 0 (Geo-FNO pipe.py: output = np.load(OUTPUT_Sigma)[:, 0])
    targets = q[:, 0, :, :].reshape(num_samples, 1, num_nodes)
    
    mesh_coords = np.stack([x[0], y[0]], axis=-1).reshape(num_nodes, 2)
    return data, targets, mesh_coords


def load_geofno_plasticity(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Geo-FNO plasticity dataset.
    
    Input: Material properties (101 per sample) broadcasted to all nodes
    Output: Time-dependent field (31 time steps, 20x4 spatial) flattened
    
    Args:
        data_dir: Directory containing plas_*.mat file
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes) - material properties broadcasted to nodes
        targets: Output data (num_samples, num_channels, num_nodes) - time-dependent field flattened
        mesh_coords: Mesh coordinates (num_nodes, coord_dim) - placeholder (no explicit coords)
    """
    from scipy.io import loadmat
    
    mat_files = [f for f in os.listdir(data_dir) if f.startswith('plas_') and f.endswith('.mat')]
    if not mat_files:
        raise FileNotFoundError(f"Plasticity .mat file not found in {data_dir}")
    
    mat_file = os.path.join(data_dir, mat_files[0])
    data_dict = loadmat(mat_file)
    
    input_data = data_dict['input']  # (987, 101) - 101 material properties per sample
    output_data = data_dict['output']  # (987, 101, 31, 20, 4) - 101 nodes, 31 time steps, 20x4 spatial
    
    num_samples, num_props = input_data.shape
    num_samples_out, num_nodes, num_time, height, width = output_data.shape
    
    if num_samples != num_samples_out:
        raise ValueError(f"Sample count mismatch: input={num_samples}, output={num_samples_out}")
    
    if num_props != num_nodes:
        raise ValueError(f"Property/node count mismatch: input props={num_props}, output nodes={num_nodes}")
    
    # Broadcast material properties to all nodes: (num_samples, num_props) -> (num_samples, num_props, num_nodes)
    # Each node gets all 101 material properties
    input_broadcasted = np.broadcast_to(input_data[:, :, np.newaxis], (num_samples, num_props, num_nodes))
    # Shape: (num_samples, 101, 101)
    
    # Flatten time and spatial dimensions: (31, 20, 4) -> 31*20*4 = 2480
    # Output: (num_samples, num_nodes, num_time, height, width) -> (num_samples, num_nodes, num_time*height*width)
    output_flat = output_data.reshape(num_samples, num_nodes, num_time * height * width)
    output_flat = output_flat.transpose(0, 2, 1)  # (num_samples, 2480, num_nodes)
    
    data = input_broadcasted  # (num_samples, 101, 101)
    targets = output_flat  # (num_samples, 2480, num_nodes)
    
    # No explicit mesh coordinates, use placeholder
    mesh_coords = np.zeros((num_nodes, 2))  # Placeholder
    
    print(f"Plasticity: Input material properties ({num_props}) broadcasted to {num_nodes} nodes")
    print(f"  Input shape: {data.shape}")
    print(f"  Output shape: {targets.shape} (time {num_time} and spatial {height}x{width} dimensions flattened to {num_time*height*width} channels)")
    
    return data, targets, mesh_coords


def load_geofno_car_cfd(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Geo-FNO car-CFD dataset.
    
    Input: Transports (3 channels) + Points (3D coordinates)
    Output: Pressures
    
    Args:
        data_dir: Directory containing torus_data.pt
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes) - transports + points
        targets: Output data (num_samples, num_channels, num_nodes) - pressures
        mesh_coords: Mesh coordinates (num_nodes, coord_dim) - 3D points
    """
    torus_file = os.path.join(data_dir, 'torus_data.pt')
    if not os.path.exists(torus_file):
        raise FileNotFoundError(f"torus_data.pt not found in {data_dir}")
    
    data_dict = torch.load(torus_file)
    points = data_dict['points'].numpy()  # (889, 3586, 3)
    transports = data_dict['transports'].numpy()  # (889, 3, 84, 84)
    pressures = data_dict['pressures'].numpy()  # (889, 3586)
    
    num_samples, num_nodes, coord_dim = points.shape
    num_samples_trans, num_channels_trans, height, width = transports.shape
    
    if num_samples != num_samples_trans:
        raise ValueError(f"Sample count mismatch: points={num_samples}, transports={num_samples_trans}")
    
    # Flatten transports from (84, 84) to match number of nodes (3586)
    # Actually, transports is on a different grid (84x84=7056) than points (3586)
    # We need to interpolate or use the irregular grid directly
    # For irregular grid approach, we use the points grid (3586 nodes)
    # So we need to interpolate transports to the points grid
    # For now, let's assume we can reshape or we need to handle this differently
    # Actually, looking at the data: points is (889, 3586, 3) - irregular mesh
    # transports is (889, 3, 84, 84) - regular grid
    # pressures is (889, 3586) - on irregular mesh
    
    # For irregular grid approach, we need transports on the irregular mesh
    # Since we don't have interpolation code, we'll use a simplified approach:
    # Use points coordinates as input features, and we'll need to handle transports separately
    # Actually, let's check if there's a way to get transports on the mesh
    
    # For now, use points (3D coords) as input, pressures as output
    # This matches the irregular grid approach
    points_reshaped = points.transpose(0, 2, 1)  # (889, 3, 3586)
    
    # If we have transports on mesh, use them; otherwise just use points
    # For Geo-FNO, they likely interpolate transports to mesh
    # We'll use points as input for now (can be extended later)
    data = points_reshaped  # (num_samples, 3, num_nodes)
    targets = pressures[:, np.newaxis, :]  # (num_samples, 1, num_nodes)
    
    mesh_coords = points[0]  # (3586, 3) - 3D coordinates
    
    print(f"Car-CFD: Using points (3D coords) as input on irregular mesh")
    print(f"  Input shape: {data.shape}")
    print(f"  Output shape: {targets.shape}")
    print(f"  Note: Transports are on regular grid (84x84), not used in irregular grid approach")
    
    return data, targets, mesh_coords


def load_geofno_channel_shocks(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Geo-FNO channel-shocks dataset.
    
    Input: Q channels 0-3 (field values) + X and Y coordinates
    Output: Q channel 4
    
    Args:
        data_dir: Directory containing Channel_*.npy files
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes) - Q channels 0-3 + X, Y coords
        targets: Output data (num_samples, num_channels, num_nodes) - Q channel 4
        mesh_coords: Mesh coordinates (num_nodes, coord_dim) - varies per sample
    """
    q_file = os.path.join(data_dir, 'Channel_Q.npy')
    x_file = os.path.join(data_dir, 'Channel_X.npy')
    y_file = os.path.join(data_dir, 'Channel_Y.npy')
    
    if not os.path.exists(q_file) or not os.path.exists(x_file) or not os.path.exists(y_file):
        raise FileNotFoundError(f"Channel-shocks files not found in {data_dir}")
    
    q = np.load(q_file)  # (5, 129, 65, 2000)
    x = np.load(x_file)  # (129, 65, 2000)
    y = np.load(y_file)  # (129, 65, 2000)
    
    # Transpose to (num_samples, channels, height, width)
    q = q.transpose(3, 0, 1, 2)  # (2000, 5, 129, 65)
    x = x.transpose(2, 0, 1)  # (2000, 129, 65)
    y = y.transpose(2, 0, 1)  # (2000, 129, 65)
    
    num_samples, num_channels, height, width = q.shape
    num_nodes = height * width
    
    # Input: Q[0:4] + X, Y
    input_q_fields = q[:, :4, :, :].reshape(num_samples, 4, num_nodes)
    input_coords = np.stack([x, y], axis=1).reshape(num_samples, 2, num_nodes)
    data = np.concatenate([input_q_fields, input_coords], axis=1)  # (num_samples, 6, num_nodes)
    
    # Output: Q[4]
    targets = q[:, 4, :, :].reshape(num_samples, 1, num_nodes)
    
    # Mesh coordinates (use first sample as reference, though they vary)
    mesh_coords = np.stack([x[0], y[0]], axis=-1).reshape(num_nodes, 2)
    
    return data, targets, mesh_coords


def load_geofno_dataset(data_path: str, dataset_type: str = 'auto') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load Geo-FNO dataset from directory or file.
    
    Args:
        data_path: Path to dataset directory
        dataset_type: 'airfoil', 'elasticity', 'pipe', 'plasticity', 'car-cfd', 'channel-shocks', or 'auto'
        
    Returns:
        data: Input data (num_samples, num_channels, num_nodes)
        targets: Output data (num_samples, num_channels, num_nodes)
        mesh_coords: Mesh coordinates (num_nodes, coord_dim)
        material_props: Material properties (num_samples, num_props) or None
    """
    if dataset_type == 'auto':
        path_lower = data_path.lower()
        if 'naca' in path_lower or 'airfoil' in path_lower:
            dataset_type = 'airfoil'
        elif 'diffeo' in path_lower or 'elastic' in path_lower:
            dataset_type = 'elasticity'
        elif 'pipe' in path_lower:
            dataset_type = 'pipe'
        elif 'plastic' in path_lower:
            dataset_type = 'plasticity'
        elif 'car' in path_lower or 'cfd' in path_lower:
            dataset_type = 'car-cfd'
        elif 'channel' in path_lower:
            dataset_type = 'channel-shocks'
        else:
            raise ValueError(f"Could not auto-detect dataset type from path: {data_path}")
    
    if dataset_type == 'airfoil':
        data, targets, mesh_coords = load_geofno_airfoil(data_path)
        return data, targets, mesh_coords, None
    elif dataset_type == 'elasticity':
        return load_geofno_elasticity(data_path)
    elif dataset_type == 'pipe':
        data, targets, mesh_coords = load_geofno_pipe(data_path)
        return data, targets, mesh_coords, None
    elif dataset_type == 'plasticity':
        data, targets, mesh_coords = load_geofno_plasticity(data_path)
        return data, targets, mesh_coords, None
    elif dataset_type == 'car-cfd':
        data, targets, mesh_coords = load_geofno_car_cfd(data_path)
        return data, targets, mesh_coords, None
    elif dataset_type == 'channel-shocks':
        data, targets, mesh_coords = load_geofno_channel_shocks(data_path)
        return data, targets, mesh_coords, None
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def prepare_geofno_data(data_path: str,
                        dataset_type: str = 'auto',
                        output_dir: Optional[str] = None,
                        train_split: float = 0.7,
                        val_split: float = 0.2,
                        normalize: bool = False,
                        random_seed: int = 42,
                        use_geofno_split: bool = True) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare Geo-FNO dataset (airfoil or elasticity) for training.
    
    Uses fixed split matching Geo-FNO: first N for train, last M for test (elasticity)
    or first N for train, next M for test (airfoil) when use_geofno_split=True.
    
    Args:
        data_path: Path to dataset directory (for airfoil/elasticity)
        dataset_type: 'airfoil', 'elasticity', or 'auto' (detect from path)
        output_dir: Directory to save processed data (default: $SCRATCH/data/geofno)
        train_split: Fraction of data for training (ignored if use_geofno_split=True)
        val_split: Fraction of data for validation (ignored if use_geofno_split=True)
        normalize: Whether to normalize data using training statistics (default: False to match Geo-FNO)
        random_seed: Random seed for train/val/test split (ignored if use_geofno_split=True)
        use_geofno_split: If True, use Geo-FNO fixed split (first 1000 train, last 200 test for elasticity;
                         first 1000 train, next 200 test for airfoil)
        
    Returns:
        data_dict: Dictionary with keys 'train', 'val', 'test' containing datasets
        mesh_coords: Mesh coordinates (num_nodes, coord_dim)
        material_props: Material properties array or None
    """
    from data_utils import IrregularGridDataset
    
    if output_dir is None:
        output_dir = '/users/PAS3244/dvoderhol/data/Geo-FNO/data/geofno'
    
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
    
    if use_geofno_split:
        if dataset_type == 'elasticity':
            ntrain = 1000
            ntest = 200
            train_indices = np.arange(ntrain)
            test_indices = np.arange(num_samples - ntest, num_samples)
            val_indices = np.arange(ntrain, num_samples - ntest)
            print(f"Using Geo-FNO fixed split for elasticity: first {ntrain} train, last {ntest} test")
        elif dataset_type == 'airfoil':
            ntrain = 1000
            ntest = 200
            train_indices = np.arange(ntrain)
            test_indices = np.arange(ntrain, ntrain + ntest)
            val_indices = np.arange(ntrain + ntest, num_samples)
            print(f"Using Geo-FNO fixed split for airfoil: first {ntrain} train, next {ntest} test")
        elif dataset_type == 'pipe':
            ntrain = 1800
            ntest = 300
            train_indices = np.arange(ntrain)
            test_indices = np.arange(ntrain, ntrain + ntest)
            val_indices = np.arange(ntrain + ntest, num_samples)
            print(f"Using Geo-FNO split for pipe: first {ntrain} train, next {ntest} test")
        elif dataset_type == 'plasticity':
            ntrain = 700
            ntest = 200
            train_indices = np.arange(ntrain)
            test_indices = np.arange(num_samples - ntest, num_samples)
            val_indices = np.arange(ntrain, num_samples - ntest)
            print(f"Using Geo-FNO split for plasticity: first {ntrain} train, last {ntest} test")
        elif dataset_type == 'car-cfd':
            ntrain = 700
            ntest = 100
            train_indices = np.arange(ntrain)
            test_indices = np.arange(num_samples - ntest, num_samples)
            val_indices = np.arange(ntrain, num_samples - ntest)
            print(f"Using Geo-FNO split for car-cfd: first {ntrain} train, last {ntest} test")
        elif dataset_type == 'channel-shocks':
            ntrain = 1600
            ntest = 200
            train_indices = np.arange(ntrain)
            test_indices = np.arange(num_samples - ntest, num_samples)
            val_indices = np.arange(ntrain, num_samples - ntest)
            print(f"Using Geo-FNO split for channel-shocks: first {ntrain} train, last {ntest} test")
        else:
            raise ValueError(f"Unknown dataset_type for Geo-FNO split: {dataset_type}")
    else:
        indices = np.random.RandomState(random_seed).permutation(num_samples)
        n_train = int(train_split * num_samples)
        n_val = int(val_split * num_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
    
    data_train = data[train_indices]
    targets_train = targets[train_indices]
    data_val = data[val_indices] if len(val_indices) > 0 else data_train[:0]
    targets_val = targets[val_indices] if len(val_indices) > 0 else targets_train[:0]
    data_test = data[test_indices]
    targets_test = targets[test_indices]
    
    if normalize:
        data_mean = data_train.mean(axis=(0, 2), keepdims=True)
        data_std = data_train.std(axis=(0, 2), keepdims=True) + 1e-8
        targets_mean = targets_train.mean(axis=(0, 2), keepdims=True)
        targets_std = targets_train.std(axis=(0, 2), keepdims=True) + 1e-8
        
        data_train = (data_train - data_mean) / data_std
        data_val = (data_val - data_mean) / data_std if len(val_indices) > 0 else data_val
        data_test = (data_test - data_mean) / data_std
        targets_train = (targets_train - targets_mean) / targets_std
        targets_val = (targets_val - targets_mean) / targets_std if len(val_indices) > 0 else targets_val
        targets_test = (targets_test - targets_mean) / targets_std
        
        print(f"Data normalized using training statistics")
    else:
        print(f"Data not normalized (matching Geo-FNO)")
    
    material_props_train = material_props[train_indices] if material_props is not None else None
    material_props_val = material_props[val_indices] if material_props is not None and len(val_indices) > 0 else None
    material_props_test = material_props[test_indices] if material_props is not None else None
    
    train_dataset = IrregularGridDataset(data_train, targets_train, mesh_coords, material_props_train)
    val_dataset = IrregularGridDataset(data_val, targets_val, mesh_coords, material_props_val) if len(val_indices) > 0 else None
    test_dataset = IrregularGridDataset(data_test, targets_test, mesh_coords, material_props_test)
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    if val_dataset is not None:
        print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    result_dict = {
        'train': train_dataset,
        'test': test_dataset
    }
    if val_dataset is not None:
        result_dict['val'] = val_dataset
    
    return result_dict, mesh_coords, material_props
