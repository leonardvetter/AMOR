#!/usr/bin/env python3
"""Launch best-known AMOR mass-PCA config on easy 10k rib (same hparams as easy_10000_best)."""

from __future__ import annotations

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import argparse
import os
import subprocess
import sys

_SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Train mass-PCA AMOR on easy rib 10k")
    p.add_argument(
        "--data_path",
        default=expand("${AMOR_DATA_ROOT}/aircraft_rib/datasets/aircraft_rib_cantilever_tip.npz"),  # rename path on a different cluster
    )
    p.add_argument(
        "--checkpoint_dir",
        default=expand("${AMOR_DATA_ROOT}/aircraft_rib/amor_rib_h15_gp10000_easy_best_mass"),  # rename path on a different cluster
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    time_buf = os.environ.get("AMOR_SLURM_BUFFER_SEC") or os.environ.get("RIB_TIME_BUFFER_SEC") or "1200"

    cmd = [
        sys.executable,
        str(_SCRIPTS / "train_amor_aircraft_rib_linear_mass.py"),
        "--data_path",
        args.data_path,
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--use_solve_ok_only",
        "--use_coordinates",
        "--use_film_conditioning",
        "--film_hidden",
        "64",
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        "300",
        "--lr",
        "2e-3",
        "--weight_decay",
        "1e-4",
        "--max_grad_norm",
        "3.0",
        "--early_stopping_patience",
        "50",
        "--width",
        "16",
        "--layers",
        "5",
        "--k_in",
        "100",
        "--k_out",
        "30",
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--time_buffer_sec",
        str(time_buf),
    ]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
