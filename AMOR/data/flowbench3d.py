"""FlowBench 3D LDC data utilities (mmap-friendly, geometry-aware splits)."""

from __future__ import annotations

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset


SCRATCH_ROOT = Path(expand("${AMOR_SCRATCH}"))
DEFAULT_DATA_DIR = SCRATCH_ROOT / "data/FlowBench/LDC_NS_3D"
DEFAULT_RESULTS_DIR = SCRATCH_ROOT / "flowbench_ldc3d_model_comparison"


@dataclass
class FlowBenchSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_geoms: np.ndarray
    val_geoms: np.ndarray
    test_geoms: np.ndarray


@dataclass
class NormStats:
    x_mean: np.ndarray  # (1, C, 1, 1, 1)
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def open_flowbench_mmap(data_dir: Path | str = DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    x_path = data_dir / "LDC_3d_X.npz"
    y_path = data_dir / "LDC_3d_Y.npz"
    for p in (x_path, y_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
    x_npz = np.load(x_path, mmap_mode="r")
    y_npz = np.load(y_path, mmap_mode="r")
    return x_npz["data"], y_npz["data"]


def geometry_ids(x_mmap: np.ndarray) -> np.ndarray:
    """Per-sample geometry tag from channel 1 (constant field)."""
    return np.asarray(x_mmap[:, 1, 0, 0, 0], dtype=np.float64)


def geometry_split(
    geom_per_sample: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    seed: int = 42,
) -> FlowBenchSplit:
    unique_geoms = np.unique(geom_per_sample)
    rng = np.random.default_rng(seed)
    unique_geoms = unique_geoms.copy()
    rng.shuffle(unique_geoms)

    n = len(unique_geoms)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_geoms = unique_geoms[:n_train]
    val_geoms = unique_geoms[n_train:n_train + n_val]
    test_geoms = unique_geoms[n_train + n_val:]

    train_set = set(train_geoms.tolist())
    val_set = set(val_geoms.tolist())
    test_set = set(test_geoms.tolist())

    train_idx, val_idx, test_idx = [], [], []
    for i, g in enumerate(geom_per_sample):
        if g in train_set:
            train_idx.append(i)
        elif g in val_set:
            val_idx.append(i)
        elif g in test_set:
            test_idx.append(i)
        else:
            raise RuntimeError(f"Sample {i} geometry {g} not assigned to any split")

    return FlowBenchSplit(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        train_geoms=train_geoms,
        val_geoms=val_geoms,
        test_geoms=test_geoms,
    )


def _online_mean_std(
    arr_mmap: np.ndarray,
    indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std; mmap samples are (C, D, H, W)."""
    sample0 = np.asarray(arr_mmap[indices[0]])
    if sample0.ndim != 4:
        raise ValueError(f"Expected sample shape (C, D, H, W), got {sample0.shape}")
    c = sample0.shape[0]
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
  # (1, C, 1, 1, 1) for broadcast with (B, C, D, H, W)
    mean = mean.reshape(1, c, 1, 1, 1).astype(np.float32)
    std = std.reshape(1, c, 1, 1, 1).astype(np.float32)
    return mean, std


def compute_norm_stats(
    x_mmap: np.ndarray,
    y_mmap: np.ndarray,
    train_idx: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean, x_std = _online_mean_std(x_mmap, train_idx)
    y_mean, y_std = _online_mean_std(y_mmap, train_idx)
    return x_mean, x_std, y_mean, y_std


def load_samples_for_pca(
    arr_mmap: np.ndarray,
    indices: Sequence[int],
    max_samples: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if max_samples is not None and len(idx) > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    return np.stack([np.asarray(arr_mmap[i]) for i in idx], axis=0)


class FlowBenchLDC3DDataset(Dataset):
  """Single-sample mmap dataset with normalization."""

  def __init__(
      self,
      x_mmap: np.ndarray,
      y_mmap: np.ndarray,
      indices: Sequence[int],
      x_mean: np.ndarray,
      x_std: np.ndarray,
      y_mean: np.ndarray,
      y_std: np.ndarray,
  ):
      self.x_mmap = x_mmap
      self.y_mmap = y_mmap
      self.indices = np.asarray(indices, dtype=np.int64)
      self.x_mean = x_mean
      self.x_std = x_std
      self.y_mean = y_mean
      self.y_std = y_std

  def __len__(self) -> int:
      return len(self.indices)

  def __getitem__(self, i: int):
      import torch

      idx = int(self.indices[i])
      x = np.asarray(self.x_mmap[idx], dtype=np.float32)
      y = np.asarray(self.y_mmap[idx], dtype=np.float32)
      while x.ndim > 4 and x.shape[0] == 1:
          x = x[0]
      while y.ndim > 4 and y.shape[0] == 1:
          y = y[0]
      if x.ndim != 4 or y.ndim != 4:
          raise ValueError(f"Bad sample shapes at idx={idx}: x={x.shape} y={y.shape}")
      # Norm stats are (1, C, 1, 1, 1); add batch dim so broadcast matches (B, C, D, H, W).
      x = (x[np.newaxis, ...] - self.x_mean) / (self.x_std + 1e-8)
      y = (y[np.newaxis, ...] - self.y_mean) / (self.y_std + 1e-8)
      x = np.squeeze(x, axis=0)
      y = np.squeeze(y, axis=0)
      return torch.from_numpy(x), torch.from_numpy(y)


def make_dataloaders(
    x_mmap,
    y_mmap,
    split: FlowBenchSplit,
    norm: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    num_workers: int = 0,
    generator=None,
) -> Dict[str, DataLoader]:
    import torch

    x_mean, x_std, y_mean, y_std = norm
    loaders = {}
    for name, idx in (
        ("train", split.train_idx),
        ("val", split.val_idx),
        ("test", split.test_idx),
    ):
        ds = FlowBenchLDC3DDataset(x_mmap, y_mmap, idx, x_mean, x_std, y_mean, y_std)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if name == "train" else None,
        )
    return loaders


def save_split_artifact(path: Path | str, split: FlowBenchSplit, norm, hyperparams: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": split.train_idx.tolist(),
        "val_idx": split.val_idx.tolist(),
        "test_idx": split.test_idx.tolist(),
        "train_geoms": split.train_geoms.tolist(),
        "val_geoms": split.val_geoms.tolist(),
        "test_geoms": split.test_geoms.tolist(),
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
