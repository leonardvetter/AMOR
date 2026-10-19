"""3D elastostatics NPZ utilities (mmap-friendly, random IID split)."""

from __future__ import annotations

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset


@dataclass
class PDE3DSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def open_elastostatics_mmap(npz_path: Path | str):
    """Elastostatics NPZ: X=E (n,N,N,N), Y=(ux,uy,uz) (n,3,N,N,N).

    Returns channel-first mmap views: X (n,1,N,N,N), Y (n,3,N,N,N).
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    z = np.load(npz_path, mmap_mode="r")
    x = z["X"]
    y = z["Y"]
    if x.ndim != 4:
        raise ValueError(f"Expected X (n,N,N,N), got {x.shape}")
    if y.ndim != 5 or y.shape[1] != 3:
        raise ValueError(f"Expected Y (n,3,N,N,N), got {y.shape}")
    if x.shape[1:] != y.shape[2:]:
        raise ValueError(f"Spatial mismatch X{x.shape} vs Y{y.shape}")
    x_ch = x[:, np.newaxis, ...]
    try:
        meta = np.load(npz_path, allow_pickle=True)["meta"].item()
    except Exception:
        meta = {}
    return x_ch, y, meta


def random_split(
    n_samples: int,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    seed: int = 42,
) -> PDE3DSplit:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return PDE3DSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def _online_mean_std(arr_mmap: np.ndarray, indices: Sequence[int], channel_axis: int = 1):
    sample0 = np.asarray(arr_mmap[indices[0]])
    c = sample0.shape[channel_axis]
    mean = np.zeros(c, dtype=np.float64)
    m2 = np.zeros(c, dtype=np.float64)
    count = 0
    for idx in indices:
        sample = np.asarray(arr_mmap[idx], dtype=np.float64)
        flat = sample.reshape(c, -1)
        batch_count = flat.shape[1]
        batch_mean = flat.mean(axis=1)
        batch_var = flat.var(axis=1)
        if count == 0:
            mean = batch_mean
            m2 = batch_var * batch_count
            count = batch_count
            continue
        total = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * (batch_count / total)
        m2 = m2 + batch_var * batch_count + (delta ** 2) * (count * batch_count / total)
        count = total
    std = np.sqrt(m2 / max(count, 1))
    shape = [1] * sample0.ndim
    shape[channel_axis] = c
    return mean.reshape(shape).astype(np.float32), std.reshape(shape).astype(np.float32)


def compute_norm_stats(x_mmap, y_mmap, train_idx):
    x_mean, x_std = _online_mean_std(x_mmap, train_idx)
    y_mean, y_std = _online_mean_std(y_mmap, train_idx)
    return x_mean, x_std, y_mean, y_std


class PDE3DDataset(Dataset):
    def __init__(self, x_mmap, y_mmap, indices, x_mean, x_std, y_mean, y_std):
        self.x_mmap = x_mmap
        self.y_mmap = y_mmap
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        import torch

        idx = int(self.indices[i])
        x = (np.asarray(self.x_mmap[idx], dtype=np.float32) - self.x_mean) / (self.x_std + 1e-8)
        y = (np.asarray(self.y_mmap[idx], dtype=np.float32) - self.y_mean) / (self.y_std + 1e-8)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_dataloaders(x_mmap, y_mmap, split, norm, batch_size, num_workers=0, generator=None):
    import torch

    x_mean, x_std, y_mean, y_std = norm
    loaders = {}
    for name, idx in (
        ("train", split.train_idx),
        ("val", split.val_idx),
        ("test", split.test_idx),
    ):
        ds = PDE3DDataset(x_mmap, y_mmap, idx, x_mean, x_std, y_mean, y_std)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if name == "train" else None,
        )
    return loaders


def save_split_artifact(path: Path | str, split: PDE3DSplit, norm, hyperparams: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": split.train_idx.tolist(),
        "val_idx": split.val_idx.tolist(),
        "test_idx": split.test_idx.tolist(),
        "dataset_meta": meta,
        "hyperparameters": hyperparams,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed(
        path.with_suffix(".norm.npz"),
        x_mean=norm[0],
        x_std=norm[1],
        y_mean=norm[2],
        y_std=norm[3],
    )
