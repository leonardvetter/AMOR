"""FlowBench 3D LDC metrics and inference timing (test metrics after undoing z-score)."""

from __future__ import annotations

import time

import numpy as np
import torch


def denorm(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return y_norm * y_std + y_mean


@torch.no_grad()
def eval_split(model, loader, device, y_mean, y_std, relative_field_errors):
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)
    rel_l1s, rel_l2s = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        pred_p = denorm(pred, y_mean, y_std)
        true_p = denorm(yb, y_mean, y_std)
        l1, l2 = relative_field_errors(pred_p, true_p)
        rel_l1s.append(l1)
        rel_l2s.append(l2)
    return float(np.mean(rel_l1s)), float(np.mean(rel_l2s))


@torch.no_grad()
def benchmark_inference(model, loader, device, warmup: int = 3, max_samples: int = 50):
    times = []
    n = 0
    for xb, _ in loader:
        xb = xb.to(device)
        if n < warmup:
            model(xb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            n += 1
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(xb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if len(times) >= max_samples:
            break
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n_timed": int(len(arr)),
        "mean_s": float(arr.mean()) if len(arr) else float("nan"),
        "std_s": float(arr.std()) if len(arr) else float("nan"),
        "median_s": float(np.median(arr)) if len(arr) else float("nan"),
    }


def mean_epoch_time(history: list[dict]) -> float | None:
    times = [h["epoch_time_s"] for h in history if "epoch_time_s" in h]
    return float(np.mean(times)) if times else None


def model_memory_bytes(n_params: int, bytes_per_param: int = 4) -> int:
    """Trainable weight storage (fp32 default); excludes PCA buffers and optimizer state."""
    return int(n_params) * int(bytes_per_param)


def model_memory_mb(n_params: int, bytes_per_param: int = 4) -> float:
    return model_memory_bytes(n_params, bytes_per_param) / (1024**2)
