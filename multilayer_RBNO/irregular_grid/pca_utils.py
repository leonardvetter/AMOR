"""
Utilities for computing PCA basis on point cloud data for irregular grids.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple


def compute_pca_pointcloud(data: np.ndarray, n_components: int = None, 
                           explained_variance_ratio: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA basis for point cloud data.
    
    Args:
        data: Input data of shape (num_samples, num_channels, num_nodes) or (num_samples, num_nodes)
              If 2D, treats as (num_samples, num_nodes) with single channel
        n_components: Number of PCA components to keep. If None, uses explained_variance_ratio.
        explained_variance_ratio: Variance ratio to retain if n_components is None.
    
    Returns:
        pca_components: PCA components matrix (n_components, num_nodes)
        pca_mean: Mean vector (num_nodes,)
    """
    if data.ndim == 2:
        data_flat = data.astype(np.float32)
    elif data.ndim == 3:
        num_samples, num_channels, num_nodes = data.shape
        data_flat = data.reshape(num_samples * num_channels, num_nodes).astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
    
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(data_flat)
    
    if n_components is None:
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(cumsum_var, explained_variance_ratio) + 1
        n_components = min(n_components, len(pca.explained_variance_ratio_))
        pca_components = pca.components_[:n_components, :]
    else:
        pca_components = pca.components_[:n_components, :]
    
    return pca_components, pca.mean_
