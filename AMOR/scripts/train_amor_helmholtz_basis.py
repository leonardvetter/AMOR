#!/usr/bin/env python3
"""
AMOR basis ablation on Helmholtz datasets (default: sigma=0.2, rho=0.4).

Parametric steady map  m(x) -> (u_real, u_imag);  pred = model(x)  (no increment).
One basis per invocation for slurm-array sweeps. GPU-only: the basis is fit with
only the k_in / k_out modes the model consumes, stored as on-device buffers, so
there is no host<->device basis transfer during training.

Frozen capacity for sigma=0.2 rho=0.4 (from train_helmholtz.py):
  k_in=144, k_out=144, width=20, layers=3, lr=2e-3, wd=1e-4, batch=8,
  epochs=300, patience=50, grad-clip=3.0, AdamW + CosineAnnealingLR, L1 loss.
Only the basis is changed across the sweep.

Metrics recorded: physical (de-normalized magnitude) test rel L1 / L2,
training time per epoch (s), inference time per sample (ms).
"""

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path as _PathForAmor

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(SCRIPT_DIR)

from AMOR.paths import expand  # noqa: E402

from AMOR.models.amor import AMOR  # noqa: E402

from AMOR.models.PCAReducer import (  # noqa: E402
    PCAReducerMultiChannel,
    _pca_like_from_rows,
    fit_alternative_basis,
    fit_random_init_basis,
)
from AMOR.models.prior_cov import LowRankPriorCov  # noqa: E402

LEARNED_BASIS_TYPES = ("learned_qr", "learned_cayley")
FIXED_BASIS_TYPES = ("pca", "chebyshev", "legendre", "jacobi", "laplacian")
ALL_BASIS_TYPES = FIXED_BASIS_TYPES + LEARNED_BASIS_TYPES + ("dis",)


def _npz_str(d, key: str, default: str = "") -> str:
    if key not in d.files:
        return default
    v = d[key]
    if isinstance(v, np.ndarray):
        v = v.item() if v.ndim == 0 else v
    if isinstance(v, bytes):
        return v.decode()
    return str(v)

# Per-variant AMOR capacity (matches train_helmholtz.py).
VARIANT_CFG = {
    "sigma_0.2_rho_0.4": dict(k_in=144, k_out=144, width=20, layers=3),
    "sigma_0.1_rho_0.6": dict(k_in=144, k_out=144, width=20, layers=3),
    "sigma_0.4_rho_0.2": dict(k_in=144, k_out=144, width=24, layers=4),
}
VARIANT_FILE = {
    "sigma_0.2_rho_0.4": "helmholtz_sigma_0.2_rho_0.4.npz",
    "sigma_0.1_rho_0.6": "helmholtz_sigma_0.1_rho_0.6.npz",
    "sigma_0.4_rho_0.2": "helmholtz_sigma_0.4_rho_0.2.npz",
}


def _is_learned(basis_type: str) -> bool:
    return basis_type in LEARNED_BASIS_TYPES


def _learnable_param(basis_type: str) -> str:
    return basis_type.replace("learned_", "")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_helmholtz(path: str, num_samples: int):
    data = np.load(path, allow_pickle=True)
    m = data["m"][:num_samples]
    u_real = data["u_real"][:num_samples]
    u_imag = data["u_imag"][:num_samples]
    X = m[:, np.newaxis, :, :].astype(np.float32)           # (N, 1, H, W)
    Y = np.stack([u_real, u_imag], axis=1).astype(np.float32)  # (N, 2, H, W)
    return X, Y


def _rb_from_dis_rows(dis_rows_path, X_train, Y_train, k_in, k_out):
    """Build a PCAReducer from precomputed dual-encoder DIS rows."""
    from AMOR.models.dis_dual import assert_dual, encoder_from_cinv, phi_from_psi

    d = np.load(dis_rows_path)
    rows_in = d["rows_in"].astype(np.float64)
    rows_out = d["rows_out"].astype(np.float64)
    n = int(d["nx"]); c_in = int(d["c_in"]); c_out = int(d["c_out"])
    num_samples = X_train.shape[0]
    x_mean = X_train.reshape(num_samples, -1).mean(0)
    y_mean = Y_train.reshape(num_samples, -1).mean(0)
    rb = PCAReducerMultiChannel()
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.shape_ = (n, n)
    rb.output_shape_ = (c_out, n, n)
    ortho = _npz_str(d, "ortho_metric")
    input_metric = _npz_str(d, "input_metric")
    use_cinv = ortho == "C_inv" or input_metric == "prior_cov" or "cov_V" in d.files
    k_in_eff = min(int(k_in), rows_in.shape[0])
    k_out_eff = min(int(k_out), rows_out.shape[0])
    if "rows_enc_in" in d.files:
        E = np.asarray(d["rows_enc_in"][:k_in_eff], dtype=np.float64)
        Phi = rows_in[:k_in_eff]
        assert_dual(E, Phi, name="load helm DIS_in")
        rb.pca_in = _pca_like_from_rows(x_mean, Phi, num_samples)
        rb.pca_in.encoder_components_ = E.astype(np.float32)
        print(f"DIS dual-encoder in({Phi.shape[0]}) out({k_out_eff})", flush=True)
    elif use_cinv and "cov_V" in d.files:
        cov = LowRankPriorCov.from_save_dict(d)
        k_in_eff = min(k_in_eff, int(cov.rank))
        phi = phi_from_psi(rows_in[:k_in_eff], cov)
        E = encoder_from_cinv(phi, cov)
        assert_dual(E, phi, name="migrate helm DIS_in")
        rb.pca_in = _pca_like_from_rows(x_mean, phi, num_samples)
        rb.pca_in.encoder_components_ = E.astype(np.float32)
        print(f"DIS migrated ψ→Φ/E in({phi.shape[0]}) cov_rank={cov.rank}", flush=True)
    else:
        rb.pca_in = _pca_like_from_rows(x_mean, rows_in[:k_in_eff], num_samples)
    rb.pca_out = _pca_like_from_rows(y_mean, rows_out[:k_out_eff], num_samples)
    if "rows_enc_out" in d.files:
        rb.pca_out.encoder_components_ = np.asarray(d["rows_enc_out"][:k_out_eff], dtype=np.float32)
    else:
        rb.pca_out.encoder_components_ = np.asarray(rb.pca_out.components_, dtype=np.float32)
    rb.basis_type = "dis"
    rb.output_on_increment = False
    return rb


def fit_basis(basis_type, X_train, Y_train, k_in, k_out, random_state,
              laplacian_periodic, jacobi_alpha, jacobi_beta, dis_rows_path=None):
    if basis_type == "dis":
        if not dis_rows_path or not os.path.isfile(dis_rows_path):
            raise SystemExit(f"dis basis needs --dis_rows_path (got {dis_rows_path}); run fit_helmholtz_dis_basis.py first")
        return _rb_from_dis_rows(dis_rows_path, X_train, Y_train, k_in, k_out)
    if _is_learned(basis_type):
        return fit_random_init_basis(
            X_train, Y_train, random_state=random_state,
            n_components_in=k_in, n_components_out=k_out,
        )
    rb = fit_alternative_basis(
        X_train, Y_train, basis_type,
        random_state=random_state,
        laplacian_periodic=laplacian_periodic,
        jacobi_alpha=jacobi_alpha, jacobi_beta=jacobi_beta,
        n_components_in=k_in, n_components_out=k_out,
    )
    rb.output_on_increment = False
    return rb


def train(model, train_loader, val_loader, device, args):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    l1 = lambda p, t: torch.mean(torch.abs(p - t))

    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    epochs_ran = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epochs_ran = epoch + 1
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = l1(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += l1(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= (args.epochs + 1 if _is_learned(args.basis_type) else args.early_stopping_patience):
            print(f"  Early stop @ epoch {epoch+1} (best {best_epoch+1})")
            break
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{args.epochs} val={val_loss:.6f}", flush=True)
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    return best_val, best_epoch + 1, epochs_ran, train_time


@torch.no_grad()
def eval_rel(model, loader, device, y_mean, y_std):
    """De-normalized |u| magnitude rel L1 / L2 (%), the physical Helmholtz metric."""
    ym = torch.from_numpy(y_mean).float().to(device)
    ys = torch.from_numpy(y_std).float().to(device)
    model.eval()
    l1s, l2s = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pd = model(xb) * ys + ym
        gt = yb * ys + ym
        pm = torch.sqrt(pd[:, 0] ** 2 + pd[:, 1] ** 2 + 1e-10)
        tm = torch.sqrt(gt[:, 0] ** 2 + gt[:, 1] ** 2 + 1e-10)
        diff = (pm - tm).reshape(pm.size(0), -1)
        tmf = tm.reshape(tm.size(0), -1)
        l1s.append((diff.abs().sum(1) / (tmf.abs().sum(1) + 1e-10)).cpu())
        l2s.append((diff.norm(dim=1) / (tmf.norm(dim=1) + 1e-10)).cpu())
    l1 = torch.cat(l1s).mean().item() * 100
    l2 = torch.cat(l2s).mean().item() * 100
    return l1, l2


@torch.no_grad()
def measure_inference_ms(model, loader, device, n_warmup=2, n_iter=5):
    model.eval()
    xb = next(iter(loader))[0].to(device)
    bs = xb.size(0)
    for _ in range(n_warmup):
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (n_iter * bs) * 1000.0


def main():
    p = argparse.ArgumentParser(description="AMOR Helmholtz basis ablation")
    p.add_argument("--variant", choices=list(VARIANT_CFG), default="sigma_0.2_rho_0.4")
    p.add_argument("--data_dir", default=expand("${AMOR_SCRATCH_DATA}"))  # rename path on a different cluster
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--basis_type", choices=ALL_BASIS_TYPES, default="pca")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--laplacian_periodic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--jacobi_alpha", type=float, default=1.0)
    p.add_argument("--jacobi_beta", type=float, default=1.0)
    p.add_argument("--dis_rows_path", default=None,
                   help="faithful-DIS rows .npz from fit_helmholtz_dis_basis.py (required for --basis_type dis)")
    p.add_argument("--tag", default=None,
                   help="Filename tag (default: basis_type). Writes amor_helmholtz_{variant}_{tag}_*.json/pt")
    p.add_argument("--layers", type=int, default=None,
                   help="Override total nonlocal blocks (incl. transition). Default: variant cfg.")
    p.add_argument(
        "--use_factorized_weights",
        choices=("auto", "true", "false"),
        default="auto",
        help="Factorized T (default auto).",
    )
    p.add_argument("--skip_if_exists", action="store_true")
    args = p.parse_args()

    cfg = VARIANT_CFG[args.variant]
    k_in, k_out, width, layers = cfg["k_in"], cfg["k_out"], cfg["width"], cfg["layers"]
    if args.layers is not None:
        layers = int(args.layers)
    fact = {"auto": "auto", "true": True, "false": False}[args.use_factorized_weights]

    if args.checkpoint_dir is None:
        args.checkpoint_dir = expand(f"${{AMOR_SCRATCH_DATA}}/helmholtz_{args.variant}_basis")  # rename path on a different cluster
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = args.tag or args.basis_type
    metrics_path = os.path.join(args.checkpoint_dir, f"amor_helmholtz_{args.variant}_{tag}_metrics.json")
    if args.skip_if_exists and os.path.isfile(metrics_path):
        print(f"Skip: {metrics_path} exists")
        return

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} | variant={args.variant} | basis={args.basis_type}")
    set_seed(args.seed)

    data_path = os.path.join(args.data_dir, VARIANT_FILE[args.variant])
    X, Y = load_helmholtz(data_path, args.num_samples)
    n = len(X)
    n_tr, n_val = int(0.7 * n), int(0.2 * n)
    Xtr, Xval, Xte = X[:n_tr], X[n_tr:n_tr + n_val], X[n_tr + n_val:]
    Ytr, Yval, Yte = Y[:n_tr], Y[n_tr:n_tr + n_val], Y[n_tr + n_val:]

    Xm = Xtr.mean(axis=(0, 2, 3), keepdims=True); Xs = Xtr.std(axis=(0, 2, 3), keepdims=True)
    Ym = Ytr.mean(axis=(0, 2, 3), keepdims=True); Ys = Ytr.std(axis=(0, 2, 3), keepdims=True)
    Xtr = (Xtr - Xm) / (Xs + 1e-8); Xval = (Xval - Xm) / (Xs + 1e-8); Xte = (Xte - Xm) / (Xs + 1e-8)
    Ytr = (Ytr - Ym) / (Ys + 1e-8); Yval = (Yval - Ym) / (Ys + 1e-8); Yte = (Yte - Ym) / (Ys + 1e-8)
    y_mean_c = Ym.reshape(1, 2, 1, 1).astype(np.float32)
    y_std_c = Ys.reshape(1, 2, 1, 1).astype(np.float32)

    def loader(Xa, Ya, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(Xa), torch.from_numpy(Ya)),
                          batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
    train_loader = loader(Xtr, Ytr, True)
    val_loader = loader(Xval, Yval, False)
    test_loader = loader(Xte, Yte, False)

    print(f"Fitting basis '{args.basis_type}' (k_in={k_in}, k_out={k_out})...")
    rb = fit_basis(args.basis_type, Xtr, Ytr, k_in, k_out, args.random_state,
                   args.laplacian_periodic, args.jacobi_alpha, args.jacobi_beta,
                   dis_rows_path=args.dis_rows_path)
    k_in = min(k_in, rb.pca_in.n_components_)
    k_out = min(k_out, rb.pca_out.n_components_)
    if hasattr(rb, "truncate"):
        rb.truncate(k_in, k_out)

    learned = _is_learned(args.basis_type)
    model = AMOR(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False, add_constant_mode=False,
        use_factorized_weights=fact,
        trainable_basis=learned,
        trainable_basis_param=_learnable_param(args.basis_type) if learned else "qr",
        share_bases=not learned,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_blocks = model.n_nonlocal_blocks()
    print(f"Params: {n_params:,}  blocks={n_blocks}")

    best_val, best_epoch, epochs_ran, train_time = train(model, train_loader, val_loader, device, args)
    sec_per_epoch = train_time / max(1, epochs_ran)

    test_l1, test_l2 = eval_rel(model, test_loader, device, y_mean_c, y_std_c)
    val_l1, val_l2 = eval_rel(model, val_loader, device, y_mean_c, y_std_c)
    infer_ms = measure_inference_ms(model, test_loader, device)

    print(f"TEST rel L1={test_l1:.4f}%  rel L2={test_l2:.4f}%")
    print(f"train {sec_per_epoch:.2f} s/epoch ({epochs_ran} epochs) | inference {infer_ms:.4f} ms/sample")

    summary = {
        "dataset": f"helmholtz_{args.variant}",
        "basis_type": args.basis_type,
        "tag": tag,
        "dis_rows_path": args.dis_rows_path if args.basis_type == "dis" else None,
        "dis_version": "dual_encoder" if args.basis_type == "dis" else None,
        "output_on_increment": False,
        "trainable_basis": learned,
        "k_in": k_in, "k_out": k_out, "width": width, "layers": layers,
        "n_blocks": n_blocks,
        "use_factorized_weights": str(fact),
        "batch_size": args.batch_size,
        "epochs_ran": epochs_ran, "best_epoch": best_epoch,
        "test_rel_l1": test_l1, "test_rel_l2": test_l2,
        "val_rel_l1": val_l1, "val_rel_l2": val_l2,
        "train_sec_per_epoch": sec_per_epoch,
        "inference_ms_per_sample": infer_ms,
        "training_time_sec": train_time,
        "num_params": n_params,
        "laplacian_periodic": args.laplacian_periodic,
        "jacobi_alpha": args.jacobi_alpha, "jacobi_beta": args.jacobi_beta,
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    ckpt = os.path.join(args.checkpoint_dir, f"amor_helmholtz_{args.variant}_{tag}_best.pt")
    torch.save({"model_state_dict": model.state_dict(), **summary}, ckpt)
    print(f"Saved metrics: {metrics_path}\nSaved ckpt: {ckpt}")


if __name__ == "__main__":
    main()
