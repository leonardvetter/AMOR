"""
Data loading utilities for irregular grid datasets (e.g., Geo-FNO airfoil, elasticity).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import os

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from pca_utils import compute_pca_pointcloud


class IrregularGridDataset(Dataset):
    """
    Dataset for irregular grid data with constant mesh.
    
    Args:
        data: Input data of shape (num_samples, num_channels, num_nodes)
        targets: Target data of shape (num_samples, num_channels, num_nodes)
        mesh_coords: Mesh coordinates of shape (num_nodes, coord_dim)
    """
    def __init__(self, data: np.ndarray, targets: np.ndarray, mesh_coords: np.ndarray):
        if data.ndim == 2:
            data = data[:, np.newaxis, :]
        if targets.ndim == 2:
            targets = targets[:, np.newaxis, :]
        
        self.data = torch.from_numpy(data.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.mesh_coords = torch.from_numpy(mesh_coords.astype(np.float32))
        
        assert data.shape[0] == targets.shape[0], "Data and targets must have same number of samples"
        assert data.shape[2] == targets.shape[2] == mesh_coords.shape[0], "Number of nodes must match"
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def compute_pca_for_model(data: np.ndarray, n_modes: int = 256, 
                          explained_variance_ratio: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA basis for MRBNO model from training data.
    
    Args:
        data: Training data of shape (num_samples, num_channels, num_nodes)
        n_modes: Number of PCA modes to keep
        explained_variance_ratio: Alternative to n_modes - keeps enough modes to explain this variance
    
    Returns:
        pca_components: PCA components (n_modes, num_nodes)
        pca_mean: Mean vector (num_nodes,)
    """
    return compute_pca_pointcloud(data, n_components=n_modes, 
                                   explained_variance_ratio=explained_variance_ratio)
