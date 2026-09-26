#!/usr/bin/env python3
"""FNO vs AMOR timing scaling on elastostatics N=64³ n=1000.

Fair mode pairing:
  For each FNO modes m (all axes), measure neuralop-style Fourier truncation
  rel-L2 on train E and u. Match smallest PCA ranks k_in / k_out with
  mean rel-L2 <= that Fourier floor (Gram PCA on train split).

Sweeps (shared else: layers=4, AdamW 2e-3, wd 1e-4, L1, batch=1, fp32):
  1) modes: fixed width, FNO m vs AMOR (k_in(m), k_out(m))
  2) width/channels: fixed m (and matched k), vary width

Reports train_step_ms, est_epoch_s = step_ms * n_train / 1000, infer_ms.
AMOR bases for timing are synthetic with matched (k_in,k_out) shapes
(GPU cost = real truncated bases).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG_ROOT, "models"))

from AMOR.models.amor import AMOR  # noqa: E402
from AMOR.models.FNO3d_neuralop import FNO3d  # noqa: E402
from AMOR.models.PCAReducerVolume import PCAReducerVolume  # noqa: E402
from AMOR.datasets.elastostatics3d import open_elastostatics_mmap, random_split  # noqa: E402

from AMOR.paths import expand

DEFAULT_DATA = expand("${AMOR_DATA_ROOT}/3d_elastostatics/elastostatics3d_N64_n1000.npz")
DEFAULT_OUTDIR = expand("${AMOR_DATA_ROOT}/3d_elastostatics/timing_scaling_N64_n1000")

LAYERS = 4
LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 42
FS = 18

# Dense grids for smoother scaling curves (publication figures).
DEFAULT_MODES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
DEFAULT_WIDTHS = [12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
MODES_SWEEP_WIDTH = 24
WIDTH_SWEEP_MODES = (12, 24)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fno_retained_coeffs(m: int) -> int:
    return int(m * m * (m // 2 + 1))


def fourier_truncate_batch(x: torch.Tensor, m: int) -> torch.Tensor:
    """Neuralop SpectralConv keep-mask; x (..., N,N,N)."""
    n_last = m // 2 + 1
    X = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")
    X = torch.fft.fftshift(X, dim=(-3, -2))
    out = torch.zeros_like(X)
    d, h, w = X.shape[-3:]
    kept_d = min(m, d)
    kept_h = min(m, h)
    kept_w = min(n_last, w)
    cd, ch = d // 2, h // 2
    nd, pd = kept_d // 2, kept_d // 2 + kept_d % 2
    nh, ph = kept_h // 2, kept_h // 2 + kept_h % 2
    out[..., cd - nd : cd + pd, ch - nh : ch + ph, :kept_w] = X[
        ..., cd - nd : cd + pd, ch - nh : ch + ph, :kept_w
    ]
    out = torch.fft.ifftshift(out, dim=(-3, -2))
    return torch.fft.irfftn(out, s=x.shape[-3:], dim=(-3, -2, -1), norm="forward")


@torch.no_grad()
def mean_rel_norms_fourier(arr_mmap, indices, modes, batch=8, device="cuda"):
    """arr: (N,C,D,H,W). Mean relative L1 / L2 / Linf over train (joint channels)."""
    out = {
        int(m): {"l1": [], "l2": [], "linf": []}
        for m in modes
    }
    n = len(indices)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        slabs = [np.asarray(arr_mmap[int(j)], dtype=np.float32) for j in indices[i0:i1]]
        x = torch.from_numpy(np.stack(slabs, axis=0)).to(device)
        flat = x.reshape(x.shape[0], -1)
        denom_l1 = flat.abs().sum(dim=1).clamp_min(1e-12)
        denom_l2 = torch.linalg.vector_norm(flat, dim=1).clamp_min(1e-12)
        denom_inf = flat.abs().amax(dim=1).clamp_min(1e-12)
        for m in modes:
            xhat = fourier_truncate_batch(x, int(m))
            err = (x - xhat).reshape(x.shape[0], -1)
            out[int(m)]["l1"].append((err.abs().sum(dim=1) / denom_l1).cpu().numpy())
            out[int(m)]["l2"].append(
                (torch.linalg.vector_norm(err, dim=1) / denom_l2).cpu().numpy()
            )
            out[int(m)]["linf"].append((err.abs().amax(dim=1) / denom_inf).cpu().numpy())
        if (i0 // batch) % 25 == 0:
            print(f"  fourier samples {i1}/{n}", flush=True)
        del x
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            torch.cuda.empty_cache()
    return {
        m: {
            "l1": float(np.mean(np.concatenate(v["l1"]))),
            "l2": float(np.mean(np.concatenate(v["l2"]))),
            "linf": float(np.mean(np.concatenate(v["linf"]))),
        }
        for m, v in out.items()
    }


# backward-compat alias used nowhere after refactor
def mean_rel_l2_fourier(arr_mmap, indices, modes, batch=8, device="cuda"):
    full = mean_rel_norms_fourier(arr_mmap, indices, modes, batch=batch, device=device)
    return {m: full[m]["l2"] for m in modes}


def fill_train_mat(arr_mmap, indices) -> np.ndarray:
    n = len(indices)
    feat0 = np.asarray(arr_mmap[int(indices[0])], dtype=np.float32).reshape(-1)
    mat = np.empty((n, feat0.size), dtype=np.float32)
    mat[0] = feat0
    for row in range(1, n):
        mat[row] = np.asarray(arr_mmap[int(indices[row])], dtype=np.float32).reshape(-1)
        if row % 100 == 0:
            print(f"  load {row}/{n}", flush=True)
    return mat


def gram_pca_rel_curves(
    mat: np.ndarray,
    max_k: int | None = None,
    max_k_l1_linf: int = 256,
    sample_batch: int = 4,
) -> dict:
    """Gram PCA; mean relative L1/L2/Linf vs k (1-indexed curves).

    L2 uses residual-energy identity over all ranks. L1/Linf use explicit
    cumulative recon up to min(r, max_k_l1_linf) (cost O(n k p)).
    """
    n, p = mat.shape
    mean = mat.mean(axis=0).astype(np.float64)
    X = mat.astype(np.float32, copy=True)
    X -= mean.astype(np.float32)

    print(f"  Gram {n}x{n} ...", flush=True)
    gram = (X @ X.T).astype(np.float64)
    print("  eigh ...", flush=True)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-12
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    r = len(eigvals)
    if max_k is not None:
        r = min(r, int(max_k))
        eigvals = eigvals[:r]
        eigvecs = eigvecs[:, :r]
    evr = eigvals / eigvals.sum()

    s = np.sqrt(eigvals)
    scores = eigvecs[:, :r] * s[None, :]

    # L2 via energy (fast, all ranks)
    xc_norm2 = np.einsum("ij,ij->i", X, X)
    mean_norm2 = float(np.dot(mean, mean))
    cross = 2.0 * (X.astype(np.float64) @ mean)
    x_norm_l2 = np.sqrt(np.maximum(xc_norm2 + mean_norm2 + cross, 1e-20))
    scores2 = scores.astype(np.float64) ** 2
    cum_explained = np.cumsum(scores2, axis=1)
    resid_e = np.maximum(xc_norm2[:, None] - cum_explained, 0.0)
    mean_rel_l2 = (np.sqrt(resid_e) / x_norm_l2[:, None]).mean(axis=0)

    r_ll = min(r, int(max_k_l1_linf))
    print(f"  build V ({p} x {r_ll}) for L1/Linf ...", flush=True)
    A = (eigvecs[:, :r_ll] / s[None, :r_ll]).astype(np.float64)
    V = (X.astype(np.float64).T @ A).astype(np.float32)

    print(f"  L1/Linf curves k=1..{r_ll} (batch={sample_batch}) ...", flush=True)
    sum_rel_l1 = np.zeros(r_ll, dtype=np.float64)
    sum_rel_inf = np.zeros(r_ll, dtype=np.float64)
    for i0 in range(0, n, sample_batch):
        i1 = min(i0 + sample_batch, n)
        b = i1 - i0
        orig = mat[i0:i1].astype(np.float64)
        denom_l1 = np.abs(orig).sum(axis=1).clip(min=1e-12)
        denom_inf = np.abs(orig).max(axis=1).clip(min=1e-12)
        recon = np.broadcast_to(mean[None, :], (b, p)).copy()
        sc = scores[i0:i1, :r_ll].astype(np.float64)
        for k in range(r_ll):
            recon += sc[:, k : k + 1] * V[:, k].astype(np.float64)
            err = orig - recon
            sum_rel_l1[k] += (np.abs(err).sum(axis=1) / denom_l1).sum()
            sum_rel_inf[k] += (np.abs(err).max(axis=1) / denom_inf).sum()
        if (i0 // sample_batch) % 20 == 0:
            print(f"    samples {i1}/{n}", flush=True)

    return {
        "evr": evr.astype(np.float64),
        "mean_rel_l1": sum_rel_l1 / n,
        "mean_rel_l2": mean_rel_l2.astype(np.float64),
        "mean_rel_linf": sum_rel_inf / n,
        "n_components": int(r),
        "n_components_l1_linf": int(r_ll),
    }


def gram_pca_rel_l2_curve(mat: np.ndarray, max_k: int | None = None) -> dict:
    return gram_pca_rel_curves(mat, max_k=max_k)


def match_k(mean_rel: np.ndarray, target: float, key: str = "rel") -> dict:
    idx = np.where(mean_rel <= target)[0]
    if idx.size == 0:
        return {
            "k": None,
            f"achieved_{key}": float(mean_rel[-1]),
            "matched": False,
            "note": "PCA never reaches Fourier floor",
        }
    k = int(idx[0] + 1)
    return {
        "k": k,
        f"achieved_{key}": float(mean_rel[k - 1]),
        "matched": True,
    }


def make_synthetic_rb(n: int, c_in: int, c_out: int, k_in: int, k_out: int) -> PCAReducerVolume:
    from sklearn.decomposition import PCA

    spatial = (n, n, n)
    d_in = c_in * n * n * n
    d_out = c_out * n * n * n
    rb = PCAReducerVolume()
    rb.n_channels = c_in
    rb.out_channels = c_out
    rb.volume_shape_ = spatial
    rb.shape_ = spatial
    rb.output_shape_ = (c_out, *spatial)

    def _mk(k, dim):
        pca = PCA(n_components=k)
        pca.mean_ = np.zeros(dim, dtype=np.float32)
        pca.components_ = np.random.randn(k, dim).astype(np.float32) / np.sqrt(dim)
        pca.explained_variance_ratio_ = np.full(k, 1.0 / k, dtype=np.float32)
        pca.n_components_ = k
        pca.n_features_in_ = dim
        return pca

    rb.pca_in = _mk(k_in, d_in)
    rb.pca_out = _mk(k_out, d_out)
    return rb


def build_fno(c_in, c_out, width, modes):
    return FNO3d(
        in_channels=c_in,
        out_channels=c_out,
        width=width,
        modes=modes,
        n_layers=LAYERS,
    )


def build_amor(rb, width, k_in, k_out):
    return AMOR(
        width=width,
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=LAYERS,
        dropout=0.0,
        init_scale=0.1,
        use_skip_connections=True,
        use_batch_norm=False,
        use_coordinates=False,
        add_constant_mode=False,
        use_factorized_weights=False,
        share_bases=True,
        trainable_basis=False,
    )


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_train_step(model, x, y, n_iter=10, n_warmup=4):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean(torch.abs(model(x) - y))
        loss.backward()
        opt.step()
    cuda_sync()
    ms = []
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        loss = torch.mean(torch.abs(model(x) - y))
        loss.backward()
        opt.step()
        e.record()
        cuda_sync()
        ms.append(s.elapsed_time(e))
    return float(np.median(ms)), float(np.mean(ms))


@torch.no_grad()
def time_infer(model, x, n_iter=20, n_warmup=6):
    model.eval()
    for _ in range(n_warmup):
        model(x)
    cuda_sync()
    ms = []
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        model(x)
        e.record()
        cuda_sync()
        ms.append(s.elapsed_time(e))
    return float(np.median(ms)), float(np.mean(ms))


def measure(tag, model, x, y, n_train: int) -> dict:
    model = model.cuda()
    x = x.cuda()
    y = y.cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    row = {"tag": tag, "ok": True, "error": None, "n_params": n_params(model)}
    try:
        tr_med, tr_mean = time_train_step(model, x, y)
        inf_med, inf_mean = time_infer(model, x)
        row.update(
            {
                "train_step_ms": tr_med,
                "train_step_ms_mean": tr_mean,
                "infer_ms": inf_med,
                "infer_ms_mean": inf_mean,
                "est_epoch_s": tr_med * n_train / 1000.0,
                "peak_mem_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
            }
        )
    except RuntimeError as exc:
        row["ok"] = False
        row["error"] = "OOM" if "out of memory" in str(exc).lower() else f"{type(exc).__name__}: {exc}"
        if "out of memory" not in str(exc).lower():
            row["traceback"] = traceback.format_exc()[-1500:]
        torch.cuda.empty_cache()
    print(
        f"{tag:42s} ok={row['ok']}  "
        f"step={row.get('train_step_ms') and round(row['train_step_ms'], 2)}ms  "
        f"epoch≈{row.get('est_epoch_s') and round(row['est_epoch_s'], 2)}s  "
        f"infer={row.get('infer_ms') and round(row['infer_ms'], 2)}ms  "
        f"params={row['n_params']}"
        + (f"  ERR={row['error']}" if row.get("error") else ""),
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return row


def channel_norm_batch(x_mmap, y_mmap, idx0, train_idx):
    """Normalize one sample with train channel stats (streaming)."""
    # cheap: use first 64 train samples for stats if n large
    use = train_idx[: min(64, len(train_idx))]
    xs = np.stack([np.asarray(x_mmap[int(i)], dtype=np.float64) for i in use], axis=0)
    ys = np.stack([np.asarray(y_mmap[int(i)], dtype=np.float64) for i in use], axis=0)
    # (B,C,...) -> mean/std over B and spatial
    x_mean = xs.mean(axis=(0, 2, 3, 4), keepdims=False).astype(np.float32)
    x_std = xs.std(axis=(0, 2, 3, 4), keepdims=False).astype(np.float32) + 1e-8
    y_mean = ys.mean(axis=(0, 2, 3, 4), keepdims=False).astype(np.float32)
    y_std = ys.std(axis=(0, 2, 3, 4), keepdims=False).astype(np.float32) + 1e-8
    x = (np.asarray(x_mmap[idx0], dtype=np.float32) - x_mean[:, None, None, None]) / x_std[
        :, None, None, None
    ]
    y = (np.asarray(y_mmap[idx0], dtype=np.float32) - y_mean[:, None, None, None]) / y_std[
        :, None, None, None
    ]
    return torch.from_numpy(x[None]).contiguous(), torch.from_numpy(y[None]).contiguous()


def run_match(x_mmap, y_mmap, train_idx, modes, device, fourier_batch, out_dir: Path) -> dict:
    print("=== Fourier truncation floors (rel L1/L2/Linf) ===", flush=True)
    f_err = mean_rel_norms_fourier(x_mmap, train_idx, modes, batch=fourier_batch, device=device)
    u_err = mean_rel_norms_fourier(
        y_mmap, train_idx, modes, batch=max(1, fourier_batch // 2), device=device
    )
    for m in modes:
        print(
            f"  m={m:3d}  M={fno_retained_coeffs(m):7d}  "
            f"E L1/L2/Linf={f_err[m]['l1']:.4e}/{f_err[m]['l2']:.4e}/{f_err[m]['linf']:.4e}  "
            f"u L1/L2/Linf={u_err[m]['l1']:.4e}/{u_err[m]['l2']:.4e}/{u_err[m]['linf']:.4e}",
            flush=True,
        )

    print("=== PCA input (E) Gram ===", flush=True)
    x_mat = fill_train_mat(x_mmap, train_idx)
    pca_in = gram_pca_rel_curves(x_mat)
    del x_mat

    print("=== PCA output (u) Gram ===", flush=True)
    y_mat = fill_train_mat(y_mmap, train_idx)
    pca_out = gram_pca_rel_curves(y_mat)
    del y_mat

    def pack_match(pca, four, field: str) -> dict:
        out = {}
        for norm, curve_key in [
            ("l1", "mean_rel_l1"),
            ("l2", "mean_rel_l2"),
            ("linf", "mean_rel_linf"),
        ]:
            mk = match_k(pca[curve_key], four[norm], key=f"rel_{norm}")
            out[norm] = {
                "fourier_rel": four[norm],
                "k": mk["k"],
                "matched": mk["matched"],
                "achieved_rel": mk.get(f"achieved_rel_{norm}", mk.get("achieved_rel")),
            }
            if not mk["matched"]:
                out[norm]["note"] = mk.get("note")
        return out

    matches = []
    for m in modes:
        kin = pack_match(pca_in, f_err[m], "E")
        kout = pack_match(pca_out, u_err[m], "u")
        # L2 keys kept for timing / backward compat
        row = {
            "fno_modes_m": int(m),
            "fno_retained_coeffs_M": fno_retained_coeffs(m),
            "by_norm": {
                "E": kin,
                "u": kout,
            },
            "fourier_rel_l1_E": f_err[m]["l1"],
            "fourier_rel_l2_E": f_err[m]["l2"],
            "fourier_rel_linf_E": f_err[m]["linf"],
            "fourier_rel_l1_u": u_err[m]["l1"],
            "fourier_rel_l2_u": u_err[m]["l2"],
            "fourier_rel_linf_u": u_err[m]["linf"],
            "k_in_l1": kin["l1"]["k"],
            "k_in_l2": kin["l2"]["k"],
            "k_in_linf": kin["linf"]["k"],
            "k_out_l1": kout["l1"]["k"],
            "k_out_l2": kout["l2"]["k"],
            "k_out_linf": kout["linf"]["k"],
            "k_in": kin["l2"]["k"],
            "k_out": kout["l2"]["k"],
            "k_in_matched": kin["l2"]["matched"],
            "k_out_matched": kout["l2"]["matched"],
            "k_in_achieved_rel_l2": kin["l2"]["achieved_rel"],
            "k_out_achieved_rel_l2": kout["l2"]["achieved_rel"],
        }
        matches.append(row)
        print(
            f"  m={m:3d}: "
            f"k_in L1/L2/Linf={kin['l1']['k']}/{kin['l2']['k']}/{kin['linf']['k']}  "
            f"k_out L1/L2/Linf={kout['l1']['k']}/{kout['l2']['k']}/{kout['linf']['k']}",
            flush=True,
        )

    result = {
        "n_train": int(len(train_idx)),
        "seed": SEED,
        "metric": (
            "mean_i ||phi - P phi||_p / ||phi||_p on train for p in {1,2,inf}; "
            "k_in/k_out default keys use p=2"
        ),
        "fourier_mask": "neuralop rfftn+fftshift keep (m,m,m//2+1)",
        "pca": "train Gram PCA, mean-centered, flattened channels",
        "modes": list(modes),
        "matches": matches,
        "pca_in_n_components": pca_in["n_components"],
        "pca_out_n_components": pca_out["n_components"],
        "amor_j99_reference": {"k_in": 18, "k_out": 7, "note": "from N64 n1000 train fit"},
    }
    curves = out_dir / "recon_match_pca_curves.npz"
    np.savez_compressed(
        curves,
        evr_in=pca_in["evr"],
        mean_rel_l1_in=pca_in["mean_rel_l1"],
        mean_rel_l2_in=pca_in["mean_rel_l2"],
        mean_rel_linf_in=pca_in["mean_rel_linf"],
        evr_out=pca_out["evr"],
        mean_rel_l1_out=pca_out["mean_rel_l1"],
        mean_rel_l2_out=pca_out["mean_rel_l2"],
        mean_rel_linf_out=pca_out["mean_rel_linf"],
    )
    result["pca_curves"] = str(curves)
    match_path = out_dir / "recon_match.json"
    match_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {match_path}", flush=True)
    return result


def run_timing(x_mmap, y_mmap, train_idx, match: dict, widths, modes_width, width_modes, out_dir: Path):
    N = int(x_mmap.shape[-1])
    c_in, c_out = int(x_mmap.shape[1]), int(y_mmap.shape[1])
    n_train = len(train_idx)
    xb, yb = channel_norm_batch(x_mmap, y_mmap, int(train_idx[0]), train_idx)
    print(
        f"timing batch X{tuple(xb.shape)} Y{tuple(yb.shape)} n_train={n_train}",
        flush=True,
    )

    by_m = {int(r["fno_modes_m"]): r for r in match["matches"]}
    runs = []

    print(f"\n=== Modes sweep @ width={modes_width} ===", flush=True)
    for m, row in sorted(by_m.items()):
        if m > N:
            continue
        tag = f"FNO_m{m}_w{modes_width}"
        rec = measure(tag, build_fno(c_in, c_out, modes_width, (m, m, m)), xb, yb, n_train)
        runs.append(
            {
                **rec,
                "model": "FNO",
                "sweep": "modes",
                "width": modes_width,
                "modes": m,
                "k_in": None,
                "k_out": None,
                "matched_to_m": m,
            }
        )
        ki, ko = row.get("k_in"), row.get("k_out")
        if ki is None or ko is None:
            print(f"  skip AMOR for m={m}: unmatched k", flush=True)
            continue
        tag = f"AMOR_k{ki}_k{ko}_w{modes_width}_match_m{m}"
        rb = make_synthetic_rb(N, c_in, c_out, int(ki), int(ko))
        rec = measure(tag, build_amor(rb, modes_width, int(ki), int(ko)), xb, yb, n_train)
        runs.append(
            {
                **rec,
                "model": "AMOR",
                "sweep": "modes",
                "width": modes_width,
                "modes": None,
                "k_in": int(ki),
                "k_out": int(ko),
                "matched_to_m": m,
                "fourier_rel_l2_E": row["fourier_rel_l2_E"],
                "fourier_rel_l2_u": row["fourier_rel_l2_u"],
            }
        )

    print("\n=== Width/channels sweep ===", flush=True)
    existing_tags = {r["tag"] for r in runs}
    for m_fix in width_modes:
        row = by_m.get(int(m_fix))
        if row is None:
            continue
        ki, ko = row.get("k_in"), row.get("k_out")
        for w in widths:
            tag = f"FNO_m{m_fix}_w{w}"
            if tag not in existing_tags:
                rec = measure(
                    tag, build_fno(c_in, c_out, w, (m_fix, m_fix, m_fix)), xb, yb, n_train
                )
                runs.append(
                    {
                        **rec,
                        "model": "FNO",
                        "sweep": "width",
                        "width": w,
                        "modes": m_fix,
                        "k_in": None,
                        "k_out": None,
                        "matched_to_m": m_fix,
                    }
                )
                existing_tags.add(tag)
            else:
                # already timed in modes sweep; also usable for width curves
                for r in runs:
                    if r["tag"] == tag:
                        r["sweep"] = "modes+width"
                        break
            if ki is None or ko is None:
                continue
            tag = f"AMOR_k{ki}_k{ko}_w{w}_match_m{m_fix}"
            if tag in existing_tags:
                for r in runs:
                    if r["tag"] == tag:
                        r["sweep"] = "modes+width"
                        break
                continue
            rb = make_synthetic_rb(N, c_in, c_out, int(ki), int(ko))
            rec = measure(tag, build_amor(rb, w, int(ki), int(ko)), xb, yb, n_train)
            runs.append(
                {
                    **rec,
                    "model": "AMOR",
                    "sweep": "width",
                    "width": w,
                    "modes": None,
                    "k_in": int(ki),
                    "k_out": int(ko),
                    "matched_to_m": m_fix,
                }
            )
            existing_tags.add(tag)

    payload = {
        "task": "elastostatics N64 n1000 FNO/AMOR timing scaling",
        "gpu": torch.cuda.get_device_name(0),
        "shared": {
            "layers": LAYERS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "loss": "L1",
            "batch_size": 1,
            "optimizer": "AdamW",
            "n_train": n_train,
            "est_epoch_s": "train_step_ms * n_train / 1000",
            "amor_bases": "synthetic shapes matched (k_in,k_out)",
            "modes_sweep_width": modes_width,
            "width_sweep_modes": list(width_modes),
            "widths": list(widths),
        },
        "recon_match": match,
        "runs": runs,
    }
    out = out_dir / "timing_scaling.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}", flush=True)
    return payload



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUTDIR)
    ap.add_argument("--modes", type=int, nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    ap.add_argument("--modes-sweep-width", type=int, default=MODES_SWEEP_WIDTH)
    ap.add_argument("--width-sweep-modes", type=int, nargs="+", default=list(WIDTH_SWEEP_MODES))
    ap.add_argument("--fourier-batch", type=int, default=4)
    ap.add_argument("--match-only", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    modes_sweep_width = int(args.modes_sweep_width)
    width_sweep_modes = tuple(int(m) for m in args.width_sweep_modes)

    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")

    x_mmap, y_mmap, meta = open_elastostatics_mmap(args.data_path)
    split = random_split(x_mmap.shape[0], seed=seed)
    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    print(
        f"data={args.data_path} X{x_mmap.shape} Y{y_mmap.shape} "
        f"train={len(train_idx)} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    match_path = out_dir / "recon_match.json"
    if args.timing_only and match_path.exists():
        match = json.loads(match_path.read_text())
    else:
        t0 = time.time()
        match = run_match(
            x_mmap, y_mmap, train_idx, args.modes, device, args.fourier_batch, out_dir
        )
        print(f"match elapsed {time.time()-t0:.1f}s", flush=True)

    if args.match_only:
        return

    t1 = time.time()
    payload = run_timing(
        x_mmap,
        y_mmap,
        train_idx,
        match,
        args.widths,
        modes_sweep_width,
        width_sweep_modes,
        out_dir,
    )
    print(f"timing elapsed {time.time()-t1:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
