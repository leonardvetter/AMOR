#!/usr/bin/env python3
"""Shared helpers for 3D FNO/AMOR trainers (elastostatics)."""


from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import json
import os
import sys
import time

import numpy as np
import torch

PKG_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

from AMOR.models.FNO3d_neuralop import FNO3d
from AMOR.models.amor import AMOR
from AMOR.models.PCAReducerVolume import PCAReducerVolume

from AMOR.training.torch_utils import (
    clone_state_dict,
    get_model_state_dict,
    relative_field_errors as relative_field_errors_on_tensors,
    strip_compile_prefix,
    unwrap_compiled,
)

HYPERPARAMETERS = {
    "layers": 4,
    "width": 24,
    "pca_j_in": 0.999,
    "pca_j_out": 0.999,
    "pca_eps": None,
    "fno_modes": (12, 12, 12),
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "max_grad_norm": 3.0,
    "epochs": 300,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 0.0,
    "batch_size": 1,
    "seed": 42,
    "train_frac": 0.7,
    "val_frac": 0.2,
    "test_frac": 0.1,
    "num_workers": 0,
    "pca_batch_size": 5,
    "use_factorized_weights": False,
    "use_factorized_weights_auto_threshold": 10000,
}

RESULTS_ROOT = expand("${AMOR_DATA_ROOT}/3d_elastostatics/model_comparison")


def pca_variance_targets(hyperparams: dict) -> tuple[float, float, float, float]:
    """eps -> variance target j = 1-eps for fit and mode cutoff (repo eps1e-1 convention)."""
    eps = hyperparams.get("pca_eps")
    if eps is not None:
        j = 1.0 - float(eps)
        return j, j, j, j
    j_in = float(hyperparams["pca_j_in"])
    j_out = float(hyperparams["pca_j_out"])
    return j_in, j_out, j_in, j_out


def j_variance_tag(j: float) -> str:
    pct = int(round(float(j) * 1000))
    if pct % 10 == 0 and pct < 1000:
        return f"j{pct // 10}"
    return f"j{pct}"



def relative_field_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[float, float]:
    """Relative L1/L2 after undoing z-score (reported test metric)."""
    y_mean_t = torch.as_tensor(y_mean, device=pred.device, dtype=pred.dtype)
    y_std_t = torch.as_tensor(y_std, device=pred.device, dtype=pred.dtype)
    pred_dn = pred * y_std_t + y_mean_t
    true_dn = target * y_std_t + y_mean_t
    return relative_field_errors_on_tensors(pred_dn, true_dn)


def relative_field_errors_normalized(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    """Relative L1/L2 in z-scored (normalized) space."""
    return relative_field_errors_on_tensors(pred, target)


def load_epoch_log(epoch_log_path: str) -> tuple[list[dict], list[float], dict]:
    """Parse epoch log: per-epoch records, wall times, optional training_end summary."""
    history: list[dict] = []
    times: list[float] = []
    training_end: dict = {}
    if not os.path.exists(epoch_log_path):
        return history, times, training_end
    with open(epoch_log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "training_end":
                training_end = rec
            elif "epoch_time_s" in rec:
                history.append(rec)
                times.append(float(rec["epoch_time_s"]))
    return history, times, training_end


def summarize_epoch_times(epoch_log_path: str) -> dict:
    history, times, training_end = load_epoch_log(epoch_log_path)
    if not times:
        return {}
    out = {
        "epoch_times_s": times,
        "epoch_time_s_mean": float(np.mean(times)),
        "epoch_time_s_median": float(np.median(times)),
        "epoch_time_s_min": float(np.min(times)),
        "epoch_time_s_max": float(np.max(times)),
        "n_epochs_logged": len(times),
    }
    if training_end:
        if "total_train_s" in training_end:
            out["train_loop_s"] = float(training_end["total_train_s"])
        best_val = training_end.get("best_val_rel_l1", training_end.get("best_val_loss"))
        if best_val is not None:
            out["best_val_l1_from_log"] = float(best_val)
        if "best_epoch" in training_end:
            out["best_epoch_from_log"] = int(training_end["best_epoch"])
    return out


def benchmark_inference(model, loader, device, y_mean, y_std, n_warmup: int = 5) -> dict:
    model.eval()
    seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            for _ in range(n_warmup):
                model(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            break
        total_s = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            t0 = time.perf_counter()
            pred = model(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_s += time.perf_counter() - t0
            relative_field_errors(pred, yb.to(device), y_mean, y_std)
            n += xb.size(0)
    s_per = total_s / max(n, 1)
    return {
        "inference_s_per_sample": s_per,
        "inference_ms_per_sample": s_per * 1000.0,
        "inference_n_samples": n,
    }


def load_state_dict_compat(model, ckpt_path: str, device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    state = strip_compile_prefix(state)
    unwrap_compiled(model).load_state_dict(state, strict=False)


def save_state_dict_compat(model, ckpt_path: str) -> None:
    state = strip_compile_prefix(clone_state_dict(get_model_state_dict(model)))
    torch.save(state, ckpt_path)



def build_model(model_name: str, x_mmap, y_mmap, rb: PCAReducerVolume | None, hyperparams: dict):
    _, c_in, d, h, w = x_mmap.shape
    _, c_out, _, _, _ = y_mmap.shape

    if model_name == "FNO":
        return FNO3d(
            in_channels=c_in,
            out_channels=c_out,
            width=hyperparams["width"],
            modes=hyperparams["fno_modes"],
            n_layers=hyperparams["layers"],
        )

    if model_name == "AMOR":
        if rb is None:
            raise ValueError("PCA reducer required for AMOR")
        if hyperparams.get("k_in") is not None:
            k_in = min(int(hyperparams["k_in"]), rb.pca_in.n_components_)
            k_out = min(int(hyperparams["k_out"]), rb.pca_out.n_components_)
            tag = f"fixed k_in={k_in} k_out={k_out}"
        else:
            j_in, j_out, _, _ = pca_variance_targets(hyperparams)
            k_in, k_out = rb.k_from_cutoff(j_in=j_in, j_out=j_out)
            k_in = min(k_in, rb.pca_in.n_components_)
            k_out = min(k_out, rb.pca_out.n_components_)
            eps = hyperparams.get("pca_eps")
            tag = f"eps={eps:g}" if eps is not None else f"j_in={j_in}, j_out={j_out}"
        print(f"  AMOR PCA modes: k_in={k_in} k_out={k_out} ({tag})")
        fw = hyperparams.get("use_factorized_weights", False)
        fw_thr = hyperparams.get("use_factorized_weights_auto_threshold", 10000)
        core = AMOR(
            width=hyperparams["width"],
            rb=rb,
            k_in=k_in,
            k_out=k_out,
            layers=hyperparams["layers"],
            dropout=0.0,
            init_scale=0.1,
            use_skip_connections=True,
            use_batch_norm=False,
            use_coordinates=False,
            add_constant_mode=False,
            use_factorized_weights=fw,
            use_factorized_weights_auto_threshold=fw_thr,
        )
        return core, k_in, k_out

    raise ValueError(model_name)


