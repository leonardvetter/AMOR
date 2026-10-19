#!/usr/bin/env python3
"""AMOR on the Geo-FNO airfoil / elasticity benchmarks (Table tab:irregular), one (dataset, seed) per call.

Protocol:
  published split 1000 train / 200 test; Z-score (inputs per node and channel, outputs per channel) and reduced bases
  fit on the train split only;
  mass-weighted PCA with the lumped node mass of the mesh, (k_in, k_out) from 99.9% explained variance;
  AMOR with factorized weights, width = floor(k_out / 2), 6 blocks (airfoil) / 4 blocks (elasticity);
  AdamW (lr 2e-3, wd 1e-4), cosine annealing, HP["epochs"] epochs (500 in this file), grad-clip 3.0, relative L2 training loss on the
  physical (denormalised) fields, as in Geo-FNO. No validation set / early stopping: the final model is evaluated.
Errors: mean per-sample relative L1 / L2 of the physical fields on the 200 test samples, in percent.
Seeds vary the model initialisation and the batch order (the split is the published one).

    python AMOR/scripts/train_geofno.py --dataset airfoil --seed 0 --data-dir /path/to/parent
    python AMOR/scripts/train_geofno.py --smoke
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import torch
from torch.utils.data import DataLoader, TensorDataset

from AMOR.data import geofno as gdata
from AMOR.models.amor import AMOR
from AMOR.models.PCAReducer import PCAReducerMultiChannelMass
from AMOR.training.loop2d import train_model

# --data-dir overrides this. AMOR_DATA_DIR is the env override. The fallback is
# <repo>/AMOR/data, which is not AMOR_DATA_ROOT.
DEFAULT_DATA = Path(os.environ.get("AMOR_DATA_DIR", REPO_ROOT / "AMOR" / "data"))
DEFAULT_OUT = REPO_ROOT / "experiments_out" / "geofno"

CFG = {  # per-dataset model size: layers from the SI table, width = floor(k_out / 2), k from 99.9% variance
    "airfoil": {"layers": 6},
    "elasticity": {"layers": 4},
}
HP = {
    "lr": 2e-3, "weight_decay": 1e-4, "max_grad_norm": 3.0, "epochs": 500, "batch_size": 20,
    "early_stopping_patience": 10**9, "early_stopping_min_delta": 0.0,  # unused: no validation set
    "variance_cutoff": 0.999, "max_pca_modes": 500, "init_scale": 0.1, "input_norm": "per-node", "input_std_floor": 0.01,
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    try:
        return {"hash": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:  # noqa: BLE001
        return {"hash": None, "dirty": None}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_rel_l2_loss(y_mean: float, y_std: float):
    """Per-sample relative L2 of the physical fields (Geo-FNO training loss), from normalised pred / target."""

    def loss(pred, y):
        num = ((pred - y) * y_std).flatten(1).norm(dim=1)
        den = (y * y_std + y_mean).flatten(1).norm(dim=1).clamp_min(1e-12)
        return (num / den).mean()

    return loss


@torch.no_grad()
def predict(model, X: np.ndarray, bs: int = 50) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + bs])).numpy() for i in range(0, len(X), bs)], axis=0)


def rel_errors(pred: np.ndarray, true: np.ndarray) -> dict:
    """Per-sample relative L1 / L2 errors in percent (arrays over the test samples) and their means."""
    e = (pred - true).astype(np.float64).reshape(len(pred), -1)
    t = true.astype(np.float64).reshape(len(true), -1)
    l1 = 100 * np.abs(e).sum(1) / np.abs(t).sum(1)
    l2 = 100 * np.linalg.norm(e, axis=1) / np.linalg.norm(t, axis=1)
    return {"rel_l1": float(l1.mean()), "rel_l2": float(l2.mean()), "per_sample_rel_l1": l1.tolist(),
            "per_sample_rel_l2": l2.tolist()}


def run(dataset: str, seed: int, hp: dict, data_dir: Path, out_dir: Path, device, force: bool, n_train: int | None) -> dict:
    res_path = out_dir / "results" / f"{dataset}_seed{seed}.json"
    if res_path.exists() and not force:
        print(f"{res_path} exists, skipping (use --force)")
        return json.loads(res_path.read_text())
    t_start = time.time()
    print(f"\n=== {dataset} seed {seed} device={device} ===", flush=True)
    X, Y, tr_idx, te_idx = gdata.load(dataset, data_dir)
    if n_train is not None:
        tr_idx = tr_idx[:n_train]
    Xtr, Ytr, Xte, Yte = X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx]
    del X, Y
    print(f"train {Xtr.shape} test {Xte.shape}", flush=True)

    # Z-score from the training split only. Inputs (the node coordinates) per node and channel over the training
    # samples, as Geo-FNO's UnitGaussianNormalizer: the coordinates differ between samples by ~1e-3 of their spatial
    # extent, so a channel-wise Z-score leaves the sample-dependent signal tiny and training stalls on the mean field.
    # The std is floored at a fraction of the channel's median node std (nodes that never move). Outputs: channel-wise.
    ax = (0, 2, 3)
    xm, xs0 = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    xs = np.maximum(xs0, hp["input_std_floor"] * np.median(xs0, axis=(2, 3), keepdims=True))
    ym, ys = Ytr.mean(ax, keepdims=True), Ytr.std(ax, keepdims=True) + 1e-8
    Xtr_n, Ytr_n = ((Xtr - xm) / xs).astype(np.float32), ((Ytr - ym) / ys).astype(np.float32)
    Xte_n = ((Xte - xm) / xs).astype(np.float32)

    # Mass-weighted PCA (lumped node mass of the mean training mesh), modes from 99.9% explained variance.
    mass = gdata.node_mass(dataset, Xtr)
    print(f"node mass: sum {mass.sum():.4f}, min {mass.min():.3e}, max {mass.max():.3e}", flush=True)
    set_all_seeds(seed)
    K = min(hp["max_pca_modes"], len(Xtr_n) - 1)
    rb = PCAReducerMultiChannelMass(random_state=seed).fit(
        Xtr_n, Ytr_n, k_in=K, k_out=K, mass_matrix=sp.diags(mass, format="csr"))
    k_in, k_out = rb.k_from_cutoff(j_in=hp["variance_cutoff"], j_out=hp["variance_cutoff"])
    if max(k_in, k_out) >= K:
        raise RuntimeError(f"variance cutoff not reached with {K} PCA modes (k_in={k_in}, k_out={k_out})")
    width, layers = max(1, k_out // 2), CFG[dataset]["layers"]
    print(f"k_in={k_in} k_out={k_out} width={width} layers={layers}", flush=True)

    set_all_seeds(seed)
    model = AMOR(width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers, dropout=0.0,
                 init_scale=hp["init_scale"], use_skip_connections=True, use_batch_norm=False,
                 use_coordinates=False, use_factorized_weights=True)
    n_params = sum(p.numel() for p in model.parameters())
    g = torch.Generator().manual_seed(seed)  # batch order
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(Ytr_n)),
                        batch_size=hp["batch_size"], shuffle=True, generator=g, num_workers=0)
    print(f"[AMOR] {n_params:,} parameters, training {hp['epochs']} epochs", flush=True)
    t0 = time.time()
    train_model(model, loader, None, device, hp, loss_fn=make_rel_l2_loss(float(ym.reshape(-1)[0]), float(ys.reshape(-1)[0])))
    train_time = time.time() - t0

    model = model.cpu()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{dataset}_seed{seed}.pt")
    pred = predict(model, Xte_n) * ys + ym  # physical units
    err = rel_errors(pred, Yte)
    print(f"[AMOR] test rel L1 {err['rel_l1']:.4f}%  rel L2 {err['rel_l2']:.4f}%  (train {train_time:.0f}s)", flush=True)

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"{dataset}_seed{seed}.npz", pred=pred.astype(np.float32), test_idx=te_idx)

    out = {
        "dataset": dataset, "seed": seed, **err,
        "model": {"k_in": int(k_in), "k_out": int(k_out), "width": int(width), "layers": int(layers),
                  "num_params": int(n_params), "factorized": True, "mass": "lumped node mass (mean training mesh)"},
        "training": {"train_time_s": train_time, "device": str(device), "n_train": int(len(Xtr)), "n_test": int(len(Xte))},
        "config": {k: v for k, v in hp.items()},
        "env": {"git": git_info(), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "python": platform.python_version(),
                "date": datetime.datetime.now().isoformat(timespec="seconds"), "wall_time_s": time.time() - t_start},
    }
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {res_path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["airfoil", "elasticity"], nargs="+", default=["airfoil", "elasticity"])
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--all-seeds", action="store_true", help="seeds 0-4")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None, help="debug: use only the first n training samples")
    ap.add_argument("--smoke", action="store_true", help="3 epochs, 200 training samples, writes to <out>_smoke")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="directory that contains Airfoil_dataset/ and Elasticity_dataset/ "
        "(default: $AMOR_DATA_DIR, else <repo>/AMOR/data)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    hp, out_dir, n_train = dict(HP), a.out_dir, a.n_train
    if a.smoke:
        hp["epochs"], n_train = 3, 200
        out_dir = out_dir.with_name(out_dir.name + "_smoke")
    if a.epochs is not None:
        hp["epochs"] = a.epochs
    seeds = [0, 1, 2, 3, 4] if a.all_seeds else a.seed
    device = pick_device(a.device)
    for ds in a.dataset:
        for s in seeds:
            run(ds, s, hp, a.data_dir, out_dir, device, a.force, n_train)


if __name__ == "__main__":
    main()
