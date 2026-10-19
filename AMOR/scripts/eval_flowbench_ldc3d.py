#!/usr/bin/env python3
"""Denormalized rel metrics + inference timing for FlowBench 3D LDC checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)

from AMOR.models.PCAReducerVolume import PCAReducerVolume

from AMOR.data.flowbench3d import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    geometry_ids,
    geometry_split,
    make_dataloaders,
    open_flowbench_mmap,
)
from AMOR.training.metrics3d import (
    benchmark_inference,
    eval_split,
    mean_epoch_time,
)
from train_flowbench_ldc3d_single import HYPERPARAMETERS, build_model, relative_field_errors


def load_hyperparams(results_path: str) -> dict:
    hyperparams = dict(HYPERPARAMETERS)
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        hyperparams.update(saved.get("hyperparameters", {}))
        for key in ("k_in", "k_out"):
            if key in saved:
                hyperparams[key] = saved[key]
    return hyperparams


def discover_completed(results_dir: str) -> list[tuple[str, str | None]]:
    jobs: list[tuple[str, str | None]] = []
    for ckpt in sorted(glob.glob(os.path.join(results_dir, "ldc3d_*_model.pt"))):
        stem = os.path.basename(ckpt).replace("_model.pt", "")
        if stem == "ldc3d_FNO":
            jobs.append(("FNO", None))
        elif stem == "ldc3d_AMOR":
            jobs.append(("AMOR", None))
        elif stem.startswith("ldc3d_AMOR_"):
            jobs.append(("AMOR", stem.replace("ldc3d_AMOR_", "")))
    return jobs


def load_amor_rb(
    results_dir: str,
    stem: str,
    split,
    x_mmap,
    y_mmap,
    hyperparams: dict,
) -> PCAReducerVolume:
    pca_path = os.path.join(results_dir, f"{stem}_pca.npz")
    if os.path.exists(pca_path):
        return PCAReducerVolume.load(pca_path)

    k_in_fixed = hyperparams.get("k_in")
    k_out_fixed = hyperparams.get("k_out")
    min_pca_out = k_out_fixed
    if k_in_fixed is not None and k_out_fixed is not None:
        min_pca_out = max(int(k_in_fixed), int(k_out_fixed))

    print(f"No {pca_path}; refitting PCA (slow)...")
    return PCAReducerVolume(random_state=hyperparams["seed"]).fit_from_mmap(
        x_mmap,
        y_mmap,
        split.train_idx,
        variance_target=hyperparams["pca_j_in"],
        variance_target_out=hyperparams["pca_j_out"],
        min_components_in=k_in_fixed,
        min_components_out=min_pca_out,
        mmap_dir=os.environ.get("TMPDIR", "/tmp"),
    )


def load_state_dict_compat(model, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)


def eval_one(model_name: str, run_tag: str | None, results_dir: str, data_dir: str, device):
    stem = f"ldc3d_{model_name}" + (f"_{run_tag}" if run_tag else "")
    ckpt = os.path.join(results_dir, f"{stem}_model.pt")
    results_path = os.path.join(results_dir, f"{stem}_results.json")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    hyperparams = load_hyperparams(results_path)
    norm_npz = np.load(os.path.join(results_dir, "ldc3d_split.norm.npz"))
    y_mean = torch.from_numpy(norm_npz["y_mean"].astype(np.float32))
    y_std = torch.from_numpy(norm_npz["y_std"].astype(np.float32))
    norm = (norm_npz["x_mean"], norm_npz["x_std"], norm_npz["y_mean"], norm_npz["y_std"])

    x_mmap, y_mmap = open_flowbench_mmap(data_dir)
    split = geometry_split(
        geometry_ids(x_mmap),
        train_frac=hyperparams["train_frac"],
        val_frac=hyperparams["val_frac"],
        seed=hyperparams["seed"],
    )
    loaders = make_dataloaders(x_mmap, y_mmap, split, norm, batch_size=1, num_workers=0)

    rb = None
    if model_name == "AMOR":
        rb = load_amor_rb(results_dir, stem, split, x_mmap, y_mmap, hyperparams)
    out = build_model(model_name, x_mmap, rb, hyperparams)
    model = out[0] if model_name == "AMOR" else out
    load_state_dict_compat(model, ckpt, device)
    model.to(device).eval()

    train_l1, train_l2 = eval_split(
        model, loaders["train"], device, y_mean, y_std, relative_field_errors
    )
    test_l1, test_l2 = eval_split(
        model, loaders["test"], device, y_mean, y_std, relative_field_errors
    )
    infer = benchmark_inference(model, loaders["test"], device)
    epoch_mean = mean_epoch_time([])
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        epoch_mean = mean_epoch_time(saved.get("epoch_history", []))

    return {
        "model": model_name,
        "run_tag": run_tag,
        "checkpoint": ckpt,
        "k_in": hyperparams.get("k_in"),
        "k_out": hyperparams.get("k_out"),
        "train_rel_l1": train_l1,
        "train_rel_l2": train_l2,
        "test_rel_l1": test_l1,
        "test_rel_l2": test_l2,
        "mean_epoch_time_s": epoch_mean,
        "inference_per_sample_s": infer,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--model", default=None, choices=["FNO", "AMOR"])
    p.add_argument("--run-tag", default=None)
    p.add_argument("--all-completed", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all_completed:
        jobs = discover_completed(args.results_dir)
        if not jobs:
            raise SystemExit(f"No checkpoints in {args.results_dir}")
    else:
        if args.model is None:
            raise SystemExit("Pass --model or --all-completed")
        jobs = [(args.model, args.run_tag)]

    reports = []
    for model_name, run_tag in jobs:
        print(f"=== {model_name} {run_tag or ''} ===")
        report = eval_one(model_name, run_tag, args.results_dir, args.data_dir, device)
        reports.append(report)
        stem = f"ldc3d_{model_name}" + (f"_{run_tag}" if run_tag else "")
        one_path = os.path.join(args.results_dir, f"{stem}_metrics.json")
        with open(one_path, "w") as f:
            json.dump(report, f, indent=2)

    out_path = os.path.join(args.results_dir, "ldc3d_metrics.json")
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(json.dumps(reports, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
