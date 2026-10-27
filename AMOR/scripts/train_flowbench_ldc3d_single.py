#!/usr/bin/env python3
"""
Train FNO or AMOR on FlowBench 3D LDC — field-to-field steady NS surrogate.

Mapping (same 128^3 grid):
    X (2 ch): Re/BC field + geometry ID field  ->  Y (4 ch): u, v, w, p

Training matches repo standard (Helmholtz / nonlinear diffusion):
    AdamW lr=2e-3, weight_decay=1e-4, CosineAnnealingLR T_max=epochs,
    L1 train loss, grad clip 3.0, early stopping patience 20.

Split: by geometry ID (no leakage across Re cases of same geometry).

Usage:
    python train_flowbench_ldc3d_single.py --model FNO
    python train_flowbench_ldc3d_single.py --model AMOR
    python train_flowbench_ldc3d_single.py --model FNO --dry-run
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(PKG_ROOT))

from AMOR.models.amor import AMOR
from AMOR.models.PCAReducerVolume import (
    PCAReducerVolume,
    fit_volume_laplacian_basis,
)

from AMOR.training.metrics3d import (
    benchmark_inference,
    eval_split,
    mean_epoch_time,
    model_memory_mb,
)
from AMOR.training.torch_utils import (
    clone_state_dict,
    get_model_state_dict,
    relative_field_errors,
    unwrap_compiled,
)
from AMOR.training.loop3d import (
    atomic_torch_save,
    filter_checkpoint_state,
    l1_loss,
    log_event,
    relative_l1_loss,
    set_seed,
    train_model,
)
from AMOR.data.flowbench3d import (
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    compute_norm_stats,
    geometry_ids,
    geometry_split,
    make_dataloaders,
    open_flowbench_mmap,
    save_split_artifact,
)

HYPERPARAMETERS = {
    "layers": 4,
    "width": 24,
    "pca_j_in": 0.99,
    "pca_j_out": 0.99,
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
    "use_factorized_weights": "auto",
    "use_factorized_weights_auto_threshold": 1000,
    "share_bases": True,
    "basis_type": "pca",
    "laplacian_periodic": False,
    "trainable_basis": False,
    "trainable_basis_param": "qr",
}

DATA_DIR = os.environ.get("FLOWBENCH_DATA_DIR", str(DEFAULT_DATA_DIR))
RESULTS_DIR = os.environ.get("FLOWBENCH_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))

def build_model(model_name: str, x_mmap, rb: PCAReducerVolume | None, hyperparams: dict):
    _, c_in, d, h, w = x_mmap.shape
    c_out = 4

    if model_name == "FNO":
        from AMOR.models.FNO3d_neuralop import FNO3d  # noqa: WPS433 — optional dep (neuralop)

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
        if hyperparams.get("k_in") is not None and hyperparams.get("k_out") is not None:
            k_in = int(hyperparams["k_in"])
            k_out = int(hyperparams["k_out"])
            mode = "fixed"
        else:
            k_in, k_out = rb.k_from_cutoff(
                j_in=hyperparams["pca_j_in"],
                j_out=hyperparams["pca_j_out"],
            )
            mode = "cutoff"
        k_in = min(k_in, rb.pca_in.n_components_)
        k_out = min(k_out, rb.pca_out.n_components_)
        print(
            f"  AMOR PCA modes: k_in={k_in}, k_out={k_out} ({mode}"
            + (
                f", j_in={hyperparams['pca_j_in']}, j_out={hyperparams['pca_j_out']})"
                if mode == "cutoff"
                else ")"
            )
        )
        fw = hyperparams.get("use_factorized_weights", "auto")
        fw_thr = hyperparams.get("use_factorized_weights_auto_threshold", 1000)
        trainable = bool(hyperparams.get("trainable_basis", False))
        share = bool(hyperparams.get("share_bases", True)) and not trainable
        print(
            f"  basis_type={hyperparams.get('basis_type', 'pca')} "
            f"trainable_basis={trainable} share_bases={share} "
            f"trainable_param={hyperparams.get('trainable_basis_param', 'qr')}"
        )
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
            share_bases=share,
            trainable_basis=trainable,
            trainable_basis_param=hyperparams.get("trainable_basis_param", "qr"),
        )
        return core, k_in, k_out

    raise ValueError(model_name)


def parse_args():
    p = argparse.ArgumentParser(description="Train FNO or AMOR on FlowBench 3D LDC")
    p.add_argument("--model", required=True, choices=["FNO", "AMOR"])
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--dry-run", action="store_true", help="Print config + one forward pass, no training")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="early-stopping patience (default from HYPERPARAMETERS)",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--k-in", type=int, default=None, help="Fixed AMOR input PCA modes")
    p.add_argument("--k-out", type=int, default=None, help="Fixed AMOR output PCA modes")
    p.add_argument("--run-tag", default=None, help="Suffix for result/log filenames (e.g. k150_k20)")
    p.add_argument("--seed", type=int, default=None, help="Run seed for initialization, batch order, geometry split, and basis RNG.")
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for the geometry split and basis RNG (default: --seed).",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help="Warm-start: load trainable weights from this checkpoint (fresh optimizer/scheduler).",
    )
    p.add_argument(
        "--factorized-weights",
        choices=["true", "false", "auto"],
        default=None,
        help="AMOR: dense (false), factorized (true), or auto by k_in*k_out threshold",
    )
    p.add_argument(
        "--factorized-threshold",
        type=int,
        default=None,
        help="auto factorized when k_in*k_out exceeds this (default 1000)",
    )
    p.add_argument(
        "--basis-type",
        choices=["pca", "laplacian", "learned_qr"],
        default="pca",
        help="AMOR basis: pca | fixed 3D laplacian | learned Cayley on frozen laplacian span",
    )
    p.add_argument(
        "--laplacian-periodic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="periodic grid Laplacian (default Dirichlet, better for LDC walls)",
    )
    p.add_argument(
        "--amp",
        choices=["bf16"],
        default=None,
        help="Training autocast. bf16 does not use a GradScaler. Eval stays fp32.",
    )
    p.add_argument(
        "--no-block-checkpoint",
        action="store_true",
        help="Keep block activations (faster). Default recomputes them to save GPU memory.",
    )
    p.add_argument(
        "--probe-batches",
        type=int,
        default=0,
        help="Run this many training batches, print peak GPU memory, and exit.",
    )
    return p.parse_args()


def print_training_summary(args, split, x_mmap, y_mmap, hyperparams):
    print("=" * 72)
    print("FlowBench 3D LDC — training setup")
    print("=" * 72)
    print(f"Model:           {args.model}")
    print(f"Task:            field-to-field X(2ch,128³) -> Y(4ch,128³)")
    print(f"Data:            {args.data_dir}")
    print(f"Results:         {args.results_dir}")
    print(f"X shape:         {tuple(x_mmap.shape)}")
    print(f"Y shape:         {tuple(y_mmap.shape)}")
    print(f"Split (by geom): train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}")
    print(f"  geoms:         train={len(split.train_geoms)} val={len(split.val_geoms)} test={len(split.test_geoms)}")
    print(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    print("=" * 72)


def main():
    import os as _os
    _env_bin = _os.path.join(sys.prefix, "bin")
    if _env_bin not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = _env_bin + ":" + _os.environ.get("PATH", "")

    args = parse_args()
    hyperparams = dict(HYPERPARAMETERS)
    if args.epochs is not None:
        hyperparams["epochs"] = args.epochs
    if args.patience is not None:
        hyperparams["early_stopping_patience"] = args.patience
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.seed is not None:
        hyperparams["seed"] = int(args.seed)
    hyperparams["split_seed"] = int(
        args.split_seed if args.split_seed is not None else hyperparams["seed"]
    )
    if args.k_in is not None:
        hyperparams["k_in"] = args.k_in
    if args.k_out is not None:
        hyperparams["k_out"] = args.k_out
    if args.factorized_weights is not None:
        hyperparams["use_factorized_weights"] = (
            args.factorized_weights
            if args.factorized_weights == "auto"
            else args.factorized_weights == "true"
        )
    if args.factorized_threshold is not None:
        hyperparams["use_factorized_weights_auto_threshold"] = args.factorized_threshold
    hyperparams["basis_type"] = args.basis_type
    hyperparams["laplacian_periodic"] = bool(args.laplacian_periodic)
    if args.amp is not None:
        hyperparams["amp"] = args.amp
    if args.no_block_checkpoint:
        os.environ["AMOR_BLOCK_CHECKPOINT"] = "0"
    if args.probe_batches:
        hyperparams["probe_batches"] = int(args.probe_batches)
        hyperparams["epochs"] = 1
    if args.basis_type == "learned_qr":
        # Full QR_(k,D) OOM at 128³; learn Cayley rotation of frozen Laplacian span.
        hyperparams["trainable_basis"] = True
        hyperparams["trainable_basis_param"] = "cayley_frozen"
        hyperparams["share_bases"] = False
    elif args.basis_type == "laplacian":
        hyperparams["trainable_basis"] = False
        hyperparams["share_bases"] = True
    else:
        hyperparams["trainable_basis"] = False
        hyperparams["share_bases"] = True

    run_tag = args.run_tag
    if run_tag is None and args.k_in is not None and args.k_out is not None:
        run_tag = f"k{args.k_in}_k{args.k_out}"
        if args.basis_type != "pca":
            run_tag = f"{run_tag}_{args.basis_type}"
    file_stem = f"ldc3d_{args.model}" + (f"_{run_tag}" if run_tag else "")

    os.makedirs(args.results_dir, exist_ok=True)
    result_path = os.path.join(args.results_dir, f"{file_stem}_results.json")
    if os.path.exists(result_path) and not args.dry_run and not os.environ.get("AMOR_PROTOCOL_OUT"):
        print(f"Result exists: {result_path}\nDelete to re-run.")
        sys.exit(0)

    from AMOR.data.splits import apply_weight_seed, shuffle_generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    wall_start = time.time()
    run_log_path = os.path.join(args.results_dir, f"{file_stem}_run_log.jsonl")
    log_event(run_log_path, "job_start", model=args.model, device=str(device))

    print("Opening mmap arrays (may take ~1-2 min)...")
    t_mmap = time.time()
    x_mmap, y_mmap = open_flowbench_mmap(args.data_dir)
    log_event(run_log_path, "mmap_open", seconds=round(time.time() - t_mmap, 1))
    geom = geometry_ids(x_mmap)
    split = geometry_split(
        geom,
        train_frac=hyperparams["train_frac"],
        val_frac=hyperparams["val_frac"],
        seed=int(hyperparams["split_seed"]),
    )
    print_training_summary(args, split, x_mmap, y_mmap, hyperparams)

    print("Computing normalization stats on train split...")
    t_norm = time.time()
    norm = compute_norm_stats(x_mmap, y_mmap, split.train_idx)
    log_event(run_log_path, "norm_stats", seconds=round(time.time() - t_norm, 1))
    save_split_artifact(
        os.path.join(args.results_dir, "ldc3d_split.json"),
        split,
        norm,
        hyperparams,
    )

    loaders = make_dataloaders(
        x_mmap, y_mmap, split, norm, hyperparams["batch_size"], hyperparams["num_workers"],
        generator=shuffle_generator(int(hyperparams["seed"])),
    )

    rb = None
    k_in = k_out = None
    if args.model == "AMOR":
        n_train = len(split.train_idx)
        k_in_fixed = hyperparams.get("k_in")
        k_out_fixed = hyperparams.get("k_out")
        if k_in_fixed is None or k_out_fixed is None:
            if args.basis_type != "pca":
                raise SystemExit("--basis-type laplacian/learned_qr requires --k-in and --k-out")
        pca_tag = (
            f"k{k_in_fixed}_k{k_out_fixed}"
            if k_in_fixed is not None and k_out_fixed is not None
            else run_tag
        )
        t_pca = time.time()
        if args.basis_type in ("laplacian", "learned_qr"):
            lap_tag = f"{pca_tag}_laplacian"
            if hyperparams["laplacian_periodic"]:
                lap_tag += "_periodic"
            basis_path = os.path.join(args.results_dir, f"ldc3d_AMOR_{lap_tag}_basis.npz")
            if os.path.exists(basis_path):
                print(f"Loading cached Laplacian basis: {basis_path}")
                rb = PCAReducerVolume.load(basis_path)
                rb.basis_type = "laplacian"
                log_event(run_log_path, "laplacian_load", path=basis_path, seconds=round(time.time() - t_pca, 1))
            else:
                print(
                    f"Building 3D Laplacian basis k_in={k_in_fixed} k_out={k_out_fixed} "
                    f"periodic={hyperparams['laplacian_periodic']} on {n_train} train means..."
                )
                rb = fit_volume_laplacian_basis(
                    x_mmap,
                    y_mmap,
                    split.train_idx,
                    int(k_in_fixed),
                    int(k_out_fixed),
                    laplacian_periodic=hyperparams["laplacian_periodic"],
                    random_state=int(hyperparams["split_seed"]),
                )
                rb.save(basis_path)
                log_event(
                    run_log_path,
                    "laplacian_fit",
                    path=basis_path,
                    seconds=round(time.time() - t_pca, 1),
                    k_in=int(k_in_fixed),
                    k_out=int(k_out_fixed),
                    periodic=hyperparams["laplacian_periodic"],
                )
            # Already sized to k_in/k_out; still truncate defensively.
            rb.truncate(int(k_in_fixed), int(k_out_fixed))
        else:
            pca_path = os.path.join(args.results_dir, f"ldc3d_AMOR_{pca_tag}_pca.npz")
            if os.path.exists(pca_path):
                print(f"Loading cached PCA: {pca_path}")
                rb = PCAReducerVolume.load(pca_path)
                log_event(run_log_path, "pca_load", path=pca_path, seconds=round(time.time() - t_pca, 1))
            else:
                print(f"Fitting PCA on full train set ({n_train} samples)...")
                min_pca_out = k_out_fixed
                if k_in_fixed is not None and k_out_fixed is not None:
                    min_pca_out = max(int(k_in_fixed), int(k_out_fixed))
                mmap_dir = os.environ.get("TMPDIR", "/tmp")
                rb = PCAReducerVolume(random_state=int(hyperparams["split_seed"])).fit_from_mmap(
                    x_mmap,
                    y_mmap,
                    split.train_idx,
                    variance_target=hyperparams["pca_j_in"],
                    variance_target_out=hyperparams["pca_j_out"],
                    min_components_in=k_in_fixed,
                    min_components_out=min_pca_out,
                    mmap_dir=mmap_dir,
                )
                pca_k_in, pca_k_out = rb.k_from_cutoff(
                    j_in=hyperparams["pca_j_in"],
                    j_out=hyperparams["pca_j_out"],
                )
                log_event(
                    run_log_path,
                    "pca_fit",
                    seconds=round(time.time() - t_pca, 1),
                    n_train=n_train,
                    pca_k_in=int(pca_k_in),
                    pca_k_out=int(pca_k_out),
                    pca_n_in=int(rb.pca_in.n_components_),
                    pca_n_out=int(rb.pca_out.n_components_),
                )
            if k_in_fixed is not None and k_out_fixed is not None:
                keep_k_in, keep_k_out = int(k_in_fixed), int(k_out_fixed)
            else:
                keep_k_in, keep_k_out = rb.k_from_cutoff(
                    j_in=hyperparams["pca_j_in"], j_out=hyperparams["pca_j_out"]
                )
            rb.truncate(keep_k_in, keep_k_out)
            rb.save(pca_path)
            log_event(
                run_log_path,
                "pca_saved",
                path=pca_path,
                kept_k_in=int(rb.pca_in.n_components_),
                kept_k_out=int(rb.pca_out.n_components_),
            )
        gc.collect()
    apply_weight_seed(hyperparams["seed"])
    build_out = build_model(args.model, x_mmap, rb, hyperparams)
    if args.model == "AMOR":
        model, k_in, k_out = build_out
        _, _, d, h, w = x_mmap.shape
        vol = (int(d), int(h), int(w))
        if rb.shape_ != vol:
            raise ValueError(f"PCA shape_ {rb.shape_} != volume {vol} (expected native 3D)")
        if model.spatial_ndim != 3 or model.spatial_shape != vol:
            raise ValueError(
                f"AMOR not 3D-native: spatial_ndim={model.spatial_ndim} "
                f"spatial_shape={model.spatial_shape} expected {vol}"
            )
        print(f"  AMOR 3D-native: shape_={rb.shape_} spatial_ndim={model.spatial_ndim}")
    else:
        model = build_out
    n_params = sum(p.numel() for p in model.parameters())
    mem_mb = model_memory_mb(n_params)
    print(f"Parameters: {n_params:,}  model_mem={mem_mb:.2f} MiB")
    if os.environ.get("AMOR_PROTOCOL_OUT"):
        from AMOR.training.loop3d import relative_l1_loss
        from AMOR.training.torch_utils import run_external_protocol
        model = model.to(device)
        run_external_protocol(
            model, loaders["train"], device,
            lambda m, x, y: relative_l1_loss(m(x), y),
            {"entry": os.environ.get("AMOR_PROTOCOL_ENTRY", f"flowbench_{args.model}"),
             "dataset": "flowbench", "model": args.model, "params": n_params,
             "precision": "fp32",
             "block_checkpoint": os.environ.get("AMOR_BLOCK_CHECKPOINT", "1"),
             "note": "fp32, no bf16; block checkpoint follows AMOR_BLOCK_CHECKPOINT"},
        )
        return

    xb, yb = next(iter(loaders["train"]))
    if xb.ndim != 5 or yb.ndim != 5:
        raise ValueError(
            f"Expected batched tensors (B,C,D,H,W); got x={tuple(xb.shape)} y={tuple(yb.shape)}"
        )
    xb, yb = xb.to(device), yb.to(device)
    model = model.to(device)
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume-from not found: {args.resume_from}")
        print(f"Warm-start: loading trainable weights from {args.resume_from}")
        state = torch.load(args.resume_from, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # 'missing' = PCA basis buffers (rebuilt from the fit, not stored in trainable-only ckpt).
        # 'unexpected' must be empty; anything here means a real weight-shape/name mismatch.
        if unexpected:
            raise RuntimeError(f"resume checkpoint has unexpected keys (shape/config mismatch): {unexpected[:8]}")
        print(f"  loaded {len(state)} trainable tensors; missing(buffers)={len(missing)} unexpected={len(unexpected)}")
        log_event(run_log_path, "resume_load", path=args.resume_from,
                  loaded=len(state), missing=len(missing), unexpected=len(unexpected))
    with torch.no_grad():
        pred = model(xb)
    print(f"Dry forward: x {tuple(xb.shape)} -> y {tuple(pred.shape)} (target {tuple(yb.shape)})")
    log_event(
        run_log_path,
        "dry_forward",
        setup_s=round(time.time() - wall_start, 1),
        n_params=n_params,
        model_mem_mb=round(mem_mb, 3),
        **(
            {"amor_3d_shape": list(rb.shape_), "spatial_ndim": model.spatial_ndim}
            if args.model == "AMOR"
            else {}
        ),
    )
    if args.dry_run:
        print("Dry-run complete. Review config above before submitting Slurm jobs.")
        return

    if args.model == "FNO":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile skipped: {exc}")
    else:
        print("torch.compile skipped for AMOR (GPU memory)")

    t0 = time.time()
    epoch_log_path = os.path.join(args.results_dir, f"{file_stem}_epoch_log.jsonl")
    ckpt_path = os.path.join(args.results_dir, f"{file_stem}_model.pt")
    last_ckpt_path = os.path.join(args.results_dir, f"{file_stem}_last.pt")
    print(f"Epoch log: {epoch_log_path}")
    log_event(run_log_path, "training_start", epoch_log=epoch_log_path)
    best_val_rel, best_epoch, history, train_loop_s = train_model(
        model, loaders["train"], loaders["val"], device, hyperparams,
        epoch_log_path=epoch_log_path,
        checkpoint_path=ckpt_path,
        last_checkpoint_path=last_ckpt_path,
        checkpoint_trainable_only=(args.model == "AMOR"),
    )
    if args.probe_batches:
        print("PROBE done", flush=True)
        return
    elapsed = time.time() - t0
    print(
        f"Training done: loop={train_loop_s:.1f}s wall={elapsed:.1f}s, "
        f"best epoch {best_epoch+1}, best val rel L1 {best_val_rel:.6f}"
    )

    final_raw = get_model_state_dict(model)
    final_state = filter_checkpoint_state(final_raw) if args.model == "AMOR" else clone_state_dict(final_raw)
    atomic_torch_save(final_state, ckpt_path)

    metrics = {}
    model.eval()
    for split_name, loader in loaders.items():
        if split_name == "test":
            continue
        rel_l1, rel_l2 = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                l1, l2 = relative_field_errors(pred, yb)
                rel_l1.append(l1)
                rel_l2.append(l2)
        metrics[f"{split_name}_rel_l1_normalized"] = float(np.mean(rel_l1))
        metrics[f"{split_name}_rel_l2_normalized"] = float(np.mean(rel_l2))
        print(
            f"{split_name} (normalized): rel L1={metrics[f'{split_name}_rel_l1_normalized']:.4f}  "
            f"rel L2={metrics[f'{split_name}_rel_l2_normalized']:.4f}"
        )

    y_mean_t = torch.from_numpy(norm[2].astype(np.float32))
    y_std_t = torch.from_numpy(norm[3].astype(np.float32))
    test_phys_l1, test_phys_l2 = eval_split(
        model, loaders["test"], device, y_mean_t, y_std_t, relative_field_errors
    )
    metrics["test_rel_l1"] = test_phys_l1
    metrics["test_rel_l2"] = test_phys_l2
    print(f"test: rel L1={test_phys_l1:.4f} rel L2={test_phys_l2:.4f}")

    train_phys_l1, train_phys_l2 = eval_split(
        model, loaders["train"], device, y_mean_t, y_std_t, relative_field_errors
    )
    infer = benchmark_inference(model, loaders["test"], device)
    epoch_mean_s = mean_epoch_time(history)
    print(
        f"test: rel L1={test_phys_l1:.4f} rel L2={test_phys_l2:.4f} | "
        f"inference={infer['mean_s']:.4f}s/sample | mean epoch={epoch_mean_s:.1f}s"
    )

    metrics_report = {
        "train_rel_l1": train_phys_l1,
        "train_rel_l2": train_phys_l2,
        "test_rel_l1": test_phys_l1,
        "test_rel_l2": test_phys_l2,
        "mean_epoch_time_s": epoch_mean_s,
        "inference_per_sample_s": infer,
    }
    metrics_path = os.path.join(args.results_dir, f"{file_stem}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "run_tag": run_tag,
                "checkpoint": ckpt_path,
                "k_in": int(k_in) if k_in is not None else None,
                "k_out": int(k_out) if k_out is not None else None,
                **metrics_report,
            },
            f,
            indent=2,
        )

    result = {
        "model": args.model,
        "dataset": "FlowBench_LDC_NS_3D",
        "mapping": "field-to-field X(2,128,128,128)->Y(4,128,128,128)",
        "hyperparameters": hyperparams,
        "num_params": n_params,
        "model_mem_mb": round(mem_mb, 3),
        "best_val_rel_l1": float(best_val_rel),
        "best_epoch": int(best_epoch + 1),
        "training_time_s": elapsed,
        "train_loop_s": train_loop_s,
        "setup_time_s": t0 - wall_start,
        "total_wall_s": time.time() - wall_start,
        "epoch_log": epoch_log_path,
        "run_log": run_log_path,
        "epoch_history": history,
        **metrics,
        **metrics_report,
        "metrics_json": metrics_path,
        "checkpoint": ckpt_path,
    }
    if k_in is not None:
        result["k_in"] = int(k_in)
        result["k_out"] = int(k_out)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log_event(run_log_path, "job_done", total_wall_s=round(time.time() - wall_start, 1), result=result_path)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
