"""Kolmogorov flow dataset: trajectory snapshots as consecutive (y_k, y_{k+1}) pairs.

The raw array is (num_trajectories, num_timesteps, H, W). Pairs are laid out in
trajectory-major order and then split sequentially, the same 70/20/remainder
rule as nonlinear diffusion and Helmholtz. The weight seed does not enter.
Both fields are z-scored with one scalar mean and std from the training split.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Reynolds numbers published in the paper and the array each was trained on.
DEFAULT_DATA_PATHS = {
    40: expand("${AMOR_DATA_ROOT}/2D_NS_Re40.npy"),
    500: expand("${AMOR_DATA_ROOT}/2D_NS_Re500.npy"),
    5000: expand("${AMOR_DATA_ROOT}/2D_NS_Re5000.npy"),
}


def load_kolmogorov_dataset(data_path, return_norm_stats=False, seed: int = 42):
    """Load pairs in trajectory-major order and split them sequentially.

    ``seed`` permutes pairs before the 70/20/remainder split. Normalization is
    fit on that seed's training pairs.
    """
    print(f"Loading dataset from {data_path}...")
    data = np.load(data_path, mmap_mode='r')
    
    print(f"Dataset shape: {data.shape}")
    print(f"Interpretation: (num_trajectories={data.shape[0]}, num_timesteps={data.shape[1]}, height={data.shape[2]}, width={data.shape[3]})")
    
    # Create input-output pairs: u(t) -> u(t+1)
    num_trajectories = data.shape[0]
    num_timesteps = data.shape[1] - 1  # 500 pairs per trajectory
    total_pairs = num_trajectories * num_timesteps
    
    print(f"\nCreating input-output pairs...")
    print(f"  Trajectories: {num_trajectories}")
    print(f"  Pairs per trajectory: {num_timesteps}")
    print(f"  Total pairs: {total_pairs:,}")
    
    # Process in chunks to be memory-efficient, convert to float32 immediately
    chunk_size = 50
    X_chunks = []
    Y_chunks = []
    
    for chunk_start in range(0, num_trajectories, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_trajectories)
        X_chunk_list = []
        Y_chunk_list = []
        
        for traj_idx in range(chunk_start, chunk_end):
            trajectory = np.array(data[traj_idx], dtype=np.float32)
            for t in range(num_timesteps):
                X_chunk_list.append(trajectory[t])
                Y_chunk_list.append(trajectory[t+1])
        
        X_chunks.append(np.array(X_chunk_list, dtype=np.float32))
        Y_chunks.append(np.array(Y_chunk_list, dtype=np.float32))
        
        if (chunk_start // chunk_size + 1) % 4 == 0:
            print(f"  Processed {chunk_end}/{num_trajectories} trajectories...")
    
    X = np.concatenate(X_chunks, axis=0)
    Y = np.concatenate(Y_chunks, axis=0)
    
    print(f"\nFinal dataset shapes: X={X.shape}, Y={Y.shape}")
    
    # Add channel dimension for input (1 channel: velocity field)
    X = X[:, np.newaxis, :, :]
    Y = Y[:, np.newaxis, :, :]
    
    print(f"After adding channel dimension: X={X.shape}, Y={Y.shape}")
    
    from AMOR.data.splits import seeded_split

    n = len(X)
    train_idx, val_idx, test_idx = seeded_split(n, seed)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"\nTrain/val/test split (seed {seed}, 70%/20%/remainder):")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Unified normalization using combined X/Y train statistics so that
    # the Euler residual (yb - xb) is meaningful in normalized space.
    xy_mean = np.concatenate([X_train.flatten(), Y_train.flatten()]).mean()
    xy_std = np.concatenate([X_train.flatten(), Y_train.flatten()]).std()

    X_train = (X_train - xy_mean) / (xy_std + 1e-8)
    Y_train = (Y_train - xy_mean) / (xy_std + 1e-8)
    X_val = (X_val - xy_mean) / (xy_std + 1e-8)
    Y_val = (Y_val - xy_mean) / (xy_std + 1e-8)
    X_test = (X_test - xy_mean) / (xy_std + 1e-8)
    Y_test = (Y_test - xy_mean) / (xy_std + 1e-8)

    print(f"\nNormalization statistics (unified X/Y):")
    print(f"  mean={xy_mean:.4f}, std={xy_std:.4f}")
    
    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).float()
    
    out = (X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, X_train, Y_train)
    if return_norm_stats:
        return out + (float(xy_mean), float(xy_std))
    return out
