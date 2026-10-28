#!/usr/bin/env python3
"""Train FNO or AMOR on 3D elastostatics: E(x) -> (ux,uy,uz).

Recommended fair compare (from timing probe):
  N=64, layers=4, FNO modes=12^3, AMOR pca_j=0.999, width 24.

Optimizer/schedule: AdamW 2e-3, wd 1e-4, L1, cosine annealing (see train_3d_common).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_ROOT, "models"))

from AMOR.models.PCAReducerVolume import PCAReducerVolume  # noqa: E402
from AMOR.data.elastostatics3d import (  # noqa: E402
    PDE3DSplit,
    compute_norm_stats,
    make_dataloaders,
    open_elastostatics_mmap,
    random_split,
    save_split_artifact,
)
from AMOR.training.loop3d import log_event, set_seed, train_model  # noqa: E402
from AMOR.training.torch_utils import (  # noqa: E402
    clone_state_dict,
    get_model_state_dict,
    unwrap_compiled,
)
from AMOR.training.trainer3d import (  # noqa: E402
    HYPERPARAMETERS,
    benchmark_inference,
    build_model,
    j_variance_tag,
    load_epoch_log,
    load_state_dict_compat,
    pca_variance_targets,
    relative_field_errors,
    save_state_dict_compat,
    strip_compile_prefix,
    summarize_epoch_times,
)

DATA_DIR = expand("${AMOR_SCRATCH_DATA}/3d_elastostatics")
RESULTS_ROOT = expand("${AMOR_SCRATCH_DATA}/3d_elastostatics/model_comparison")


def data_path_for_N(N: int) -> str:
    return os.path.join(DATA_DIR, f"elastostatics3d_N{N}_n100.npz")


def run_prefix(N: int, model: str, hyperparams: dict) -> str:
    base = f"elast_N{N}_{model}"
    if model == "FNO":
        modes = hyperparams.get("fno_modes", (12, 12, 12))
        m = modes[0] if isinstance(modes, (tuple, list)) else modes
        base += f"_m{int(m)}"
        base += f"_w{int(hyperparams.get('width', 24))}"
        base += f"_L{int(hyperparams.get('layers', 4))}"
        base += "_neuralop"
        return base
    w = int(hyperparams.get("width", 24))
    n_layers = int(hyperparams.get("layers", 4))
    base += f"_w{w}_L{n_layers}"
    if hyperparams.get("k_in") is not None and hyperparams.get("k_out") is not None:
        base += f"_k{hyperparams['k_in']}_k{hyperparams['k_out']}"
    elif hyperparams.get("pca_j_in") is not None:
        j_in = float(hyperparams["pca_j_in"])
        j_out = float(hyperparams.get("pca_j_out", j_in))
        if abs(j_in - j_out) < 1e-9:
            base += f"_{j_variance_tag(j_in)}"
        else:
            base += f"_{j_variance_tag(j_in)}_{j_variance_tag(j_out)}"
    return base


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["FNO", "AMOR"])
    p.add_argument("--N", type=int, default=64, help="grid size (32/64/128)")
    p.add_argument("--data-path", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the sample split, normalization set, and PCA RNG (default: 42).",
    )
    p.add_argument(
        "--factorized-weights",
        choices=("true", "false"),
        default="false",
        help="AMOR coefficient map. false is a dense map.",
    )
    p.add_argument("--train-frac", type=float, default=None)
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--pca-j", type=float, default=0.999,
                   help="AMOR variance fraction (default 0.999)")
    p.add_argument("--fno-modes", type=int, default=12,
                   help="FNO modes per axis (default 12)")
    p.add_argument("--k-in", type=int, default=None)
    p.add_argument("--k-out", type=int, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def main():
    env_bin = os.path.join(sys.prefix, "bin")
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + ":" + os.environ.get("PATH", "")

    args = parse_args()
    N = int(args.N)
    args.data_path = args.data_path or data_path_for_N(N)
    args.results_dir = args.results_dir or os.path.join(RESULTS_ROOT, f"N{N}")

    hyperparams = dict(HYPERPARAMETERS)
    hyperparams["fno_modes"] = (12, 12, 12)
    hyperparams["pca_j_in"] = 0.999
    hyperparams["pca_j_out"] = 0.999
    hyperparams.pop("pca_eps", None)

    if args.epochs is not None:
        hyperparams["epochs"] = args.epochs
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.lr is not None:
        hyperparams["lr"] = float(args.lr)
    if args.width is not None:
        hyperparams["width"] = int(args.width)
    if args.layers is not None:
        hyperparams["layers"] = int(args.layers)
    if args.weight_decay is not None:
        hyperparams["weight_decay"] = float(args.weight_decay)
    if args.patience is not None:
        hyperparams["early_stopping_patience"] = int(args.patience)
    if args.fno_modes is not None:
        m = int(args.fno_modes)
        hyperparams["fno_modes"] = (m, m, m)
    if args.seed is not None:
        hyperparams["seed"] = int(args.seed)
    hyperparams["split_seed"] = int(
        args.split_seed if args.split_seed is not None else hyperparams["seed"]
    )
    hyperparams["use_factorized_weights"] = args.factorized_weights == "true"
    if args.train_frac is not None:
        hyperparams["train_frac"] = float(args.train_frac)
    hyperparams["val_frac"] = 0.2 if args.val_frac is None else float(args.val_frac)
    hyperparams["test_frac"] = 1.0 - float(hyperparams["train_frac"]) - float(hyperparams["val_frac"])
    if args.pca_j is not None:
        hyperparams["pca_j_in"] = float(args.pca_j)
        hyperparams["pca_j_out"] = float(args.pca_j)
    if args.k_in is not None:
        hyperparams["k_in"] = args.k_in
        hyperparams["k_out"] = args.k_out

    os.makedirs(args.results_dir, exist_ok=True)
    prefix = run_prefix(N, args.model, hyperparams)
    result_path = os.path.join(args.results_dir, f"{prefix}_results.json")
    if os.path.exists(result_path) and not args.dry_run and not args.eval_only and not os.environ.get("AMOR_PROTOCOL_OUT"):
        print(f"Result exists: {result_path}\nDelete to re-run.")
        sys.exit(0)

    from AMOR.data.splits import apply_weight_seed, shuffle_generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    wall_start = time.time()
    run_log_path = os.path.join(args.results_dir, f"{prefix}_run_log.jsonl")
    log_event(run_log_path, "job_start", pde="elastostatics", model=args.model, N=N)

    print("Opening mmap arrays...")
    x_mmap, y_mmap, ds_meta = open_elastostatics_mmap(args.data_path)
    split = random_split(
        x_mmap.shape[0],
        train_frac=hyperparams["train_frac"],
        val_frac=hyperparams["val_frac"],
        seed=int(hyperparams["split_seed"]),
    )
    if args.limit_train is not None or args.limit_eval is not None:
        lt = args.limit_train if args.limit_train is not None else len(split.train_idx)
        le = args.limit_eval
        split = PDE3DSplit(
            train_idx=split.train_idx[:lt],
            val_idx=split.val_idx[:le] if le is not None else split.val_idx,
            test_idx=split.test_idx[:le] if le is not None else split.test_idx,
        )

    print("=" * 72)
    print(f"3D elastostatics — E(x) -> (ux,uy,uz) on {N}^3")
    print(f"Model={args.model}  data={args.data_path}")
    print(f"X{tuple(x_mmap.shape)} Y{tuple(y_mmap.shape)}")
    print(f"split train/val/test={len(split.train_idx)}/{len(split.val_idx)}/{len(split.test_idx)}")
    print(json.dumps(hyperparams, indent=2))
    print("=" * 72)

    norm = compute_norm_stats(x_mmap, y_mmap, split.train_idx)
    save_split_artifact(
        os.path.join(args.results_dir, f"elast_N{N}_split.json"),
        split, norm, hyperparams, ds_meta,
    )
    loaders = make_dataloaders(
        x_mmap, y_mmap, split, norm, hyperparams["batch_size"], hyperparams["num_workers"],
        generator=shuffle_generator(int(hyperparams["seed"])),
    )

    rb = None
    k_in = k_out = None
    if args.model == "AMOR":
        j_in, j_out, var_in, var_out = pca_variance_targets(hyperparams)
        print(f"Fitting PCA variance_target in/out={var_in:.4g}/{var_out:.4g} ...")
        t_pca = time.time()
        fit_kw = dict(variance_target=var_in, variance_target_out=var_out)
        if hyperparams.get("k_in") is not None:
            fit_kw["n_components_in"] = int(hyperparams["k_in"])
            fit_kw["n_components_out"] = int(hyperparams["k_out"])
        rb = PCAReducerVolume(random_state=int(hyperparams["split_seed"])).fit_from_mmap(
            x_mmap, y_mmap, split.train_idx, **fit_kw,
            mmap_dir=os.environ.get("TMPDIR", "/tmp"),
        )
        if hyperparams.get("k_in") is not None:
            k_in = min(int(hyperparams["k_in"]), rb.pca_in.n_components_)
            k_out = min(int(hyperparams["k_out"]), rb.pca_out.n_components_)
        else:
            k_in, k_out = rb.k_from_cutoff(j_in=j_in, j_out=j_out)
        rb.truncate(int(k_in), int(k_out))
        print(f"AMOR modes k_in={k_in} k_out={k_out}  pca_s={time.time()-t_pca:.1f}")
        gc.collect()

    apply_weight_seed(hyperparams["seed"])
    build_out = build_model(args.model, x_mmap, y_mmap, rb, hyperparams)
    if args.model == "AMOR":
        model, k_in, k_out = build_out
    else:
        model = build_out
    n_params = sum(p.numel() for p in model.parameters())
    n_blocks = model.n_nonlocal_blocks() if hasattr(model, "n_nonlocal_blocks") else None
    print(f"Parameters: {n_params:,}  blocks={n_blocks}")
    if os.environ.get("AMOR_PROTOCOL_OUT"):
        from AMOR.training.loop3d import relative_l1_loss
        from AMOR.training.torch_utils import run_external_protocol
        model = model.to(device)
        run_external_protocol(
            model, loaders["train"], device,
            lambda m, x, y: relative_l1_loss(m(x), y),
            {"entry": os.environ.get("AMOR_PROTOCOL_ENTRY", f"elastostatics_{args.model}"),
             "dataset": "elastostatics", "model": args.model, "params": n_params,
             "precision": "fp32",
             "block_checkpoint": os.environ.get("AMOR_BLOCK_CHECKPOINT", "1")},
        )
        return

    xb, yb = next(iter(loaders["train"]))
    xb, yb = xb.to(device), yb.to(device)
    model = model.to(device)
    with torch.no_grad():
        pred = model(xb)
    print(f"Dry forward: x{tuple(xb.shape)} -> y{tuple(pred.shape)} (tgt{tuple(yb.shape)})")
    if args.dry_run:
        print("Dry-run complete.")
        return

    ckpt_path = os.path.join(args.results_dir, f"{prefix}_model.pt")
    epoch_log_path = os.path.join(args.results_dir, f"{prefix}_epoch_log.jsonl")

    if args.eval_only:
        load_state_dict_compat(model, ckpt_path, device)
        best_val, best_epoch, train_loop_s, elapsed = 0.0, 0, 0.0, 0.0
        history, _, training_end = load_epoch_log(epoch_log_path)
        if training_end:
            best_val = float(training_end.get("best_val_rel_l1", training_end.get("best_val_loss", 0.0)))
            best_epoch = int(training_end.get("best_epoch", 1)) - 1
            train_loop_s = float(training_end.get("total_train_s", 0.0))
            elapsed = train_loop_s
    else:
        if not args.no_compile:
            try:
                model = torch.compile(model)
                print("torch.compile enabled")
            except Exception as exc:
                print(f"torch.compile skipped: {exc}")
        t0 = time.time()
        log_event(run_log_path, "training_start", epoch_log=epoch_log_path)
        best_val, best_epoch, history, train_loop_s = train_model(
            model, loaders["train"], loaders["val"], device, hyperparams,
            epoch_log_path=epoch_log_path,
            checkpoint_path=ckpt_path,
        )
        elapsed = time.time() - t0
        print(
            f"Training done: loop={train_loop_s:.1f}s wall={elapsed:.1f}s, "
            f"best epoch {best_epoch+1}, best val L1 {best_val:.6f}"
        )
        save_state_dict_compat(model, ckpt_path)

    y_mean, y_std = norm[2], norm[3]
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
                l1, l2 = relative_field_errors(pred, yb, y_mean, y_std)
                rel_l1.append(l1)
                rel_l2.append(l2)
        metrics[f"{split_name}_rel_l1"] = float(np.mean(rel_l1))
        metrics[f"{split_name}_rel_l2"] = float(np.mean(rel_l2))
        print(f"{split_name}: rel L1={metrics[f'{split_name}_rel_l1']:.4f}  "
              f"rel L2={metrics[f'{split_name}_rel_l2']:.4f}")

    test_rel_l1, test_rel_l2 = [], []
    with torch.no_grad():
        for xb, yb in loaders["test"]:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            l1, l2 = relative_field_errors(pred, yb, y_mean, y_std)
            test_rel_l1.append(l1)
            test_rel_l2.append(l2)
    metrics["test_rel_l1"] = float(np.mean(test_rel_l1))
    metrics["test_rel_l2"] = float(np.mean(test_rel_l2))
    print(f"test: rel L1={metrics['test_rel_l1']:.4f}  "
          f"rel L2={metrics['test_rel_l2']:.4f}")

    infer_stats = benchmark_inference(model, loaders["test"], device, y_mean, y_std)
    metrics.update(infer_stats)
    print(f"inference: {infer_stats['inference_ms_per_sample']:.2f} ms/sample")

    epoch_stats = summarize_epoch_times(epoch_log_path)
    if epoch_stats:
        metrics.update(epoch_stats)
        print(f"epoch time mean={epoch_stats['epoch_time_s_mean']:.2f}s "
              f"median={epoch_stats['epoch_time_s_median']:.2f}s "
              f"n={epoch_stats['n_epochs_logged']}")

    result = {
        "model": args.model,
        "pde": "elastostatics",
        "N": N,
        "dataset": args.data_path,
        "mapping": f"E({N}^3)->(ux,uy,uz)({N}^3)",
        "error_space": "physical_denormalized",
        "hyperparameters": hyperparams,
        "num_params": n_params,
        "n_blocks": n_blocks,
        "best_val_l1": float(best_val),
        "best_epoch": int(best_epoch + 1),
        "training_time_s": elapsed,
        "train_loop_s": train_loop_s,
        "total_wall_s": time.time() - wall_start,
        "epoch_log": epoch_log_path,
        "run_log": run_log_path,
        "epoch_history": history,
        **metrics,
        "checkpoint": ckpt_path,
    }
    if k_in is not None:
        result["k_in"] = int(k_in)
        result["k_out"] = int(k_out)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    log_event(run_log_path, "job_done", result=result_path)
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
