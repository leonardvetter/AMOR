#!/usr/bin/env python3
"""Independent AMOR timing sweeps (modes + width) for elastostatics N64 n1000.

Complements matched sweeps in timing_scaling.json:
  - modes: fixed width=24, k_in = k_out = k for k in grid
  - width: fixed k_in = k_out = 16, vary width

Writes timing_independent.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_ROOT, "models"))

from bench_elastostatics_timing_scaling import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_OUTDIR,
    LAYERS,
    build_amor,
    channel_norm_batch,
    make_synthetic_rb,
    measure,
)
from AMOR.datasets.elastostatics3d import open_elastostatics_mmap, random_split  # noqa: E402

# Dense equal-rank grid: k_in = k_out = k
AMOR_KS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
WIDTHS = [12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
AMOR_WIDTH_K = 16  # k_in = k_out
MODES_WIDTH = 24
SEED = 42


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_mmap, y_mmap, _ = open_elastostatics_mmap(args.data_path)
    split = random_split(x_mmap.shape[0], seed=SEED)
    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    N = int(x_mmap.shape[-1])
    c_in, c_out = int(x_mmap.shape[1]), int(y_mmap.shape[1])
    n_train = len(train_idx)
    xb, yb = channel_norm_batch(x_mmap, y_mmap, int(train_idx[0]), train_idx)
    print(
        f"independent timing X{tuple(xb.shape)} Y{tuple(yb.shape)} "
        f"n_train={n_train} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    runs = []

    print("\n=== Independent AMOR modes: k_in=k_out @ w=24 ===", flush=True)
    for k in AMOR_KS:
        tag = f"AMOR_k{k}_k{k}_w{MODES_WIDTH}_indep"
        rb = make_synthetic_rb(N, c_in, c_out, k, k)
        rec = measure(tag, build_amor(rb, MODES_WIDTH, k, k), xb, yb, n_train)
        runs.append(
            {
                **rec,
                "model": "AMOR",
                "sweep": "independent_modes",
                "width": MODES_WIDTH,
                "modes": None,
                "k_in": k,
                "k_out": k,
                "matched_to_m": None,
                "layers": LAYERS,
            }
        )

    print(f"\n=== Independent AMOR width @ k_in=k_out={AMOR_WIDTH_K} ===", flush=True)
    k = AMOR_WIDTH_K
    for w in WIDTHS:
        tag = f"AMOR_k{k}_k{k}_w{w}_indep"
        if w == MODES_WIDTH and k in AMOR_KS:
            src = next(r for r in runs if r["k_in"] == k and r["k_out"] == k and r["width"] == w)
            runs.append({**src, "sweep": "independent_width", "tag": tag + "_widsweep"})
            continue
        rb = make_synthetic_rb(N, c_in, c_out, k, k)
        rec = measure(tag, build_amor(rb, w, k, k), xb, yb, n_train)
        runs.append(
            {
                **rec,
                "model": "AMOR",
                "sweep": "independent_width",
                "width": w,
                "modes": None,
                "k_in": k,
                "k_out": k,
                "matched_to_m": None,
                "layers": LAYERS,
            }
        )

    payload = {
        "task": "elastostatics independent AMOR timing (k_in=k_out)",
        "gpu": torch.cuda.get_device_name(0),
        "amor_ks": AMOR_KS,
        "amor_width_k": AMOR_WIDTH_K,
        "widths": WIDTHS,
        "modes_width": MODES_WIDTH,
        "n_train": n_train,
        "runs": runs,
    }
    out = out_dir / "timing_independent.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}", flush=True)




if __name__ == "__main__":
    main()
