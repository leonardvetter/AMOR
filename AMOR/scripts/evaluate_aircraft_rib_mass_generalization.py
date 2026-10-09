#!/usr/bin/env python3
"""Cross-mesh tip eval: h15-trained mass-PCA AMOR → finer tip meshes (h11/h7/h5).

Methods
  refit   — M-PCA on target snapshots; load learnable θ only (frozen).
  interp  — P1-prolong coarse ``pca_mass`` modes/means (B = P Φ unchanged),
            then paper dual encoder E = G^{-1}(B M), G = B M Bᵀ (Cholesky only
            for applying G^{-1}). Learnable T is unchanged; μ₂ = P μ₁.

Coordinates and batch-norm are always off in the rebuilt model (match current
trainers). If a legacy checkpoint still has ``coord_embed.*``, pass
``--use_coordinates_override on`` (or ``auto``) so θ can load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AMOR.models.amor import AMOR  # noqa: E402
from AMOR.models.PCAReducer import PCAReducerMultiChannelMass, _MassPCAView  # noqa: E402
from AMOR.models.mass.mesh_mass import lumped_nodal_mass, mass_diag_stacked  # noqa: E402


def _load_dataset(path: str, use_solve_ok_only: bool) -> Dict[str, np.ndarray]:
    d0 = np.load(path, allow_pickle=True)
    if use_solve_ok_only and "solve_ok" in d0.files:
        ok = np.asarray(d0["solve_ok"], dtype=bool).reshape(-1)
        return {
            k: np.asarray(d0[k])[ok] if np.asarray(d0[k]).shape[0] == len(ok) else np.asarray(d0[k])
            for k in d0.files
        }
    return {k: np.asarray(d0[k]) for k in d0.files}


def _ensure_pca_mass(ckpt: dict, state: dict) -> None:
    """Fill pca_mass from shared fixed-basis buffers when the trainer omitted it."""
    if ckpt.get("pca_mass") is not None:
        return

    def _np(name: str) -> np.ndarray:
        t = state[name]
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        return np.asarray(t)

    if "shared_fixed_basis.in_comp" not in state:
        raise ValueError("checkpoint missing pca_mass and shared_fixed_basis buffers")
    ckpt["pca_mass"] = {
        "pca_in_components": _np("shared_fixed_basis.in_comp"),
        "pca_in_mean": _np("shared_fixed_basis.in_mean"),
        "pca_out_components": _np("shared_fixed_basis.out_comp"),
        "pca_out_mean": _np("shared_fixed_basis.out_mean"),
    }
    print("pca_mass recovered from shared_fixed_basis buffers", flush=True)


def _fill_training_norm(ckpt: dict, coarse_path: str) -> None:
    """Recompute coarse train-split z-score stats when a basis-run ckpt omitted them."""
    if ckpt.get("normalization") and ckpt.get("load_norm"):
        return
    d = _load_dataset(coarse_path, bool(ckpt.get("use_solve_ok_only", False)))
    e = np.asarray(d["E_nodal"], dtype=np.float32)
    u = np.asarray(d["u"], dtype=np.float32)
    x = e[:, None, :]
    y = np.transpose(u, (0, 2, 1)).astype(np.float32)
    s = x.shape[0]
    seed = int(ckpt.get("seed", 42))
    train_frac = float(ckpt.get("train_frac", 0.7))
    val_frac = float(ckpt.get("val_frac", 0.2))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(s)
    n_train = max(1, int(train_frac * s))
    n_val = int(val_frac * s)
    if n_train + n_val > s:
        n_val = max(0, s - n_train)
    tr = perm[:n_train]
    x_mu = x[tr].mean(axis=(0, 2), keepdims=True)
    x_std = x[tr].std(axis=(0, 2), keepdims=True) + 1e-8
    y_mu = y[tr].mean(axis=(0, 2), keepdims=True)
    y_std = y[tr].std(axis=(0, 2), keepdims=True) + 1e-8
    if not ckpt.get("normalization"):
        ckpt["normalization"] = {
            "x_mu": x_mu.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "y_mu": y_mu.astype(np.float32),
            "y_std": y_std.astype(np.float32),
        }
        print(
            f"normalization recomputed from coarse train split "
            f"(seed={seed}, train_frac={train_frac}, n_train={n_train})",
            flush=True,
        )
    if not ckpt.get("load_norm") and "load" in d:
        load = np.asarray(d["load"], dtype=np.float32).reshape(-1)
        ckpt["load_norm"] = {
            "mu": float(np.mean(load[tr])),
            "std": float(np.std(load[tr]) + 1e-8),
        }


def _norm_stats_from_ckpt(ckpt: dict) -> Dict[str, np.ndarray]:
    n = ckpt.get("normalization")
    if not n:
        raise ValueError("checkpoint missing normalization")
    out = {}
    for k in ("x_mu", "x_std", "y_mu", "y_std"):
        v = n[k]
        out[k] = np.asarray(v, dtype=np.float32) if not hasattr(v, "numpy") else np.asarray(v.numpy(), dtype=np.float32)
    return out


def _zscore_fields(x: np.ndarray, y: np.ndarray, stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    xn = ((x - stats["x_mu"]) / stats["x_std"]).astype(np.float32)
    yn = ((y - stats["y_mu"]) / stats["y_std"]).astype(np.float32)
    return xn, yn


def _denorm_pred(pred: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (pred * stats["y_std"] + stats["y_mu"]).astype(np.float32)


def _metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    s = pred.shape[0]
    pf = pred.reshape(s, -1)
    tf = true.reshape(s, -1)
    rel_l1 = np.sum(np.abs(pf - tf), axis=1) / np.maximum(np.sum(np.abs(tf), axis=1), 1e-30)
    rel_l2 = np.sqrt(np.sum((pf - tf) ** 2, axis=1)) / np.maximum(np.sqrt(np.sum(tf**2, axis=1)), 1e-30)
    return {
        "mse": float(np.mean((pred - true) ** 2)),
        "rel_l1": float(np.mean(rel_l1)),
        "rel_l2": float(np.mean(rel_l2)),
    }


def _tet_centroids(nodes: np.ndarray, elems: np.ndarray) -> np.ndarray:
    return nodes[elems].mean(axis=1)


def _point_in_tet_bary(
    p: np.ndarray, xa: np.ndarray, xb: np.ndarray, xc: np.ndarray, xd: np.ndarray, tol: float = 1e-9
) -> Tuple[np.ndarray, bool]:
    v0 = xb - xa
    v1 = xc - xa
    v2 = xd - xa
    v = p - xa
    mat = np.stack([v0, v1, v2], axis=1)
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-30:
        return np.zeros(4), False
    inv = np.linalg.inv(mat)
    l1, l2, l3 = inv @ v
    l0 = 1.0 - l1 - l2 - l3
    lam = np.array([l0, l1, l2, l3], dtype=np.float64)
    inside = bool(np.all(lam >= -tol) and np.all(lam <= 1.0 + tol))
    return lam, inside


class CoarseTetLocator:
    def __init__(self, nodes: np.ndarray, elems: np.ndarray, n_probe: int = 48):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elems = np.asarray(elems, dtype=np.int64)
        self.centroids = _tet_centroids(self.nodes, self.elems)
        self._tree = cKDTree(self.centroids)
        self._node_tree = cKDTree(self.nodes)
        self.n_probe = int(n_probe)

    def locate(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        p = np.asarray(p, dtype=np.float64).reshape(3)
        _, idx = self._tree.query(p, k=min(self.n_probe, len(self.elems)))
        idx = np.atleast_1d(idx)
        for e in idx:
            tri = self.elems[int(e)]
            xa, xb, xc, xd = self.nodes[tri[0]], self.nodes[tri[1]], self.nodes[tri[2]], self.nodes[tri[3]]
            lam, ins = _point_in_tet_bary(p, xa, xb, xc, xd)
            if ins:
                return tri, lam, True
        return self.elems[int(idx[0])], np.ones(4) / 4.0, False


def build_prolongation_sparse(
    nodes_c: np.ndarray, elems_c: np.ndarray, nodes_f: np.ndarray, n_probe: int
) -> sp.csr_matrix:
    """P (N_f, N_c): fine nodal value = P1 coarse basis at the fine node."""
    loc = CoarseTetLocator(nodes_c, elems_c, n_probe=n_probe)
    n_f, n_c = nodes_f.shape[0], nodes_c.shape[0]
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i in range(n_f):
        tri, lam, ok = loc.locate(nodes_f[i])
        if ok:
            for j in range(4):
                rows.append(i)
                cols.append(int(tri[j]))
                data.append(float(lam[j]))
        else:
            jn = int(loc._node_tree.query(nodes_f[i], k=1)[1])
            rows.append(i)
            cols.append(jn)
            data.append(1.0)
    return sp.coo_matrix((data, (rows, cols)), shape=(n_f, n_c), dtype=np.float64).tocsr()


def _dual_encoder_rows_paper(B: np.ndarray, M, *, name: str) -> Tuple[np.ndarray, dict]:
    """Paper Sec. 2.5 dual encoder: keep decoder B, set E = G^{-1} (B M).

    B is unchanged (prolonged modes). G = B M Bᵀ; Cholesky only applies G^{-1}.
    Returns encoder rows E (k, D) and residual diagnostics.
    """
    import scipy.linalg as sla
    import scipy.sparse as sp

    B = np.asarray(B, dtype=np.float64)
    if not sp.issparse(M):
        raise TypeError("dual encoder requires sparse consistent mass_matrix")
    M = M.tocsr()
    BM = (M @ B.T).T  # (k, D) = B M
    G = BM @ B.T  # (k, k) = B M Bᵀ
    G = 0.5 * (G + G.T)
    try:
        c, lower = sla.cho_factor(G, lower=True, check_finite=False)
        E = sla.cho_solve((c, lower), BM, check_finite=False)
    except Exception:
        E = np.linalg.solve(G, BM)
    dual_err = float(np.linalg.norm(E @ B.T - np.eye(B.shape[0]), ord="fro"))
    print(f"{name}: ||E B^T - I||_F = {dual_err:.3e}  (B unchanged)", flush=True)
    return E.astype(np.float32), {"dual_err_EB_I": dual_err, "k": int(B.shape[0])}


def _prolong_scalar_modes(P: sp.csr_matrix, mean_c: np.ndarray, comp_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean_f = np.asarray(P @ mean_c.reshape(-1), dtype=np.float64).reshape(-1)
    comp_f = np.asarray(P @ comp_c.T, dtype=np.float64).T
    return mean_f.astype(np.float32), comp_f.astype(np.float32)


def _prolong_vector_modes(
    P: sp.csr_matrix, mean_c: np.ndarray, comp_c: np.ndarray, n_c: int, out_ch: int
) -> Tuple[np.ndarray, np.ndarray]:
    mean_c3 = mean_c.reshape(out_ch, n_c)
    mean_f3 = np.stack([P @ mean_c3[c] for c in range(out_ch)], axis=0)
    mean_f = mean_f3.reshape(-1)
    k = comp_c.shape[0]
    n_f = P.shape[0]
    comp_f = np.zeros((k, out_ch * n_f), dtype=np.float64)
    for j in range(k):
        row = comp_c[j].reshape(out_ch, n_c)
        for c in range(out_ch):
            comp_f[j, c * n_f : (c + 1) * n_f] = P @ row[c]
    return mean_f.astype(np.float32), comp_f.astype(np.float32)


def _rb_from_views(
    *,
    pca_in_mean: np.ndarray,
    pca_in_comp: np.ndarray,
    pca_out_mean: np.ndarray,
    pca_out_comp: np.ndarray,
    mass_nodes: np.ndarray,
    n_nodes: int,
    out_channels: int = 3,
    mass_matrix=None,
    enc_in: np.ndarray | None = None,
    enc_out: np.ndarray | None = None,
) -> PCAReducerMultiChannelMass:
    from AMOR.models.mass.mesh_mass import mass_kron_channels
    import scipy.sparse as sp

    c_in = int(pca_in_comp.shape[0])
    c_out = int(pca_out_comp.shape[0])
    evr_in = np.ones(c_in, dtype=np.float32) / max(c_in, 1)
    evr_out = np.ones(c_out, dtype=np.float32) / max(c_out, 1)
    rb = PCAReducerMultiChannelMass()
    rb.n_channels = 1
    rb.out_channels = out_channels
    rb.shape_ = (n_nodes, 1)
    rb.output_shape_ = (out_channels, n_nodes, 1)
    if mass_matrix is None:
        raise ValueError("mass_matrix (consistent sparse P1) is required; diag fallback removed")
    M_nodes = sp.csr_matrix(mass_matrix, dtype=np.float64)
    _ = mass_nodes  # legacy unused
    rb.mass_nodes_ = np.asarray(M_nodes.diagonal(), dtype=np.float32)
    rb.mass_in_ = mass_kron_channels(M_nodes, 1)
    rb.mass_out_ = mass_kron_channels(M_nodes, out_channels)
    rb.mass_diag_in_ = np.asarray(rb.mass_in_.sum(axis=1)).reshape(-1).astype(np.float32)
    rb.mass_diag_out_ = np.asarray(rb.mass_out_.sum(axis=1)).reshape(-1).astype(np.float32)
    # If enc_* is None, leave encoder_components_ unset so AMOR mass encode uses ΦᵀM.
    rb.pca_in = _MassPCAView(
        pca_in_mean.astype(np.float32),
        pca_in_comp.astype(np.float32),
        c_in,
        evr_in,
        None if enc_in is None else np.asarray(enc_in, dtype=np.float32),
    )
    rb.pca_out = _MassPCAView(
        pca_out_mean.astype(np.float32),
        pca_out_comp.astype(np.float32),
        c_out,
        evr_out,
        None if enc_out is None else np.asarray(enc_out, dtype=np.float32),
    )
    return rb


def _load_theta_only(model: torch.nn.Module, state: dict) -> Tuple[List[str], List[str]]:
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_sd and model_sd[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    learnable_mismatch = [
        k
        for k in missing
        if not k.endswith(("_mean", "_components_t", "components", "mass_diag"))
        and "coord_embed" not in k
        and "shared_fixed_basis" not in k
    ]
    return learnable_mismatch, list(unexpected)


def _infer_total_layers_from_state(state: dict, ckpt_layers: int) -> int:
    """Map checkpoint ``layers`` to current semantics (total nonlocal blocks).

    Legacy tip/training counted ``layers`` as transition + residual (excluding
    block0): block0 + block1 + blocks[0..L-2] with ``len(blocks)=layers-1``.
    Current AMOR uses ``layers`` = total blocks including block0 and the
    transition (``len(blocks)=layers-2``). Infer from state_dict so frozen-θ
    transfer does not silently drop residual blocks.
    """
    import re

    res_idx = {int(m.group(1)) for k in state if (m := re.match(r"blocks\.(\d+)\.", k))}
    n_res = (max(res_idx) + 1) if res_idx else 0
    has_block0 = any(k.startswith("block0.") for k in state)
    has_block1 = any(k.startswith("block1.") for k in state)
    if has_block0 and has_block1:
        total = 2 + n_res
        if total != int(ckpt_layers):
            print(
                f"NOTE: ckpt layers={ckpt_layers} (legacy) → total nonlocal blocks={total} "
                f"(block0+transition+{n_res} residual)",
                flush=True,
            )
        return total
    return int(ckpt_layers)


def _arch_from_state(state: dict) -> Tuple[bool, bool, bool]:
    factorized = any(k.endswith("mode_weight") or k.endswith("width_weights") for k in state)
    share = any(k.startswith("shared_fixed_basis.") for k in state)
    use_bn = any(".bn.weight" in k or k.endswith(".bn.weight") for k in state)
    return factorized, share, use_bn


def _build_model(rb, ckpt: dict, summary: dict, use_coords: bool, state: dict) -> Tuple[AMOR, int, int]:
    k_in = int(ckpt.get("k_in", summary.get("k_in", 100)))
    k_out = int(ckpt.get("k_out", summary.get("k_out", 30)))
    layers = _infer_total_layers_from_state(
        state, int(ckpt.get("layers", summary.get("layers", 5)))
    )
    factorized, share, use_bn = _arch_from_state(state)
    print(
        f"build AMOR layers={layers} factorized={factorized} share_bases={share} "
        f"bn={use_bn} coords={use_coords}",
        flush=True,
    )
    model = AMOR(
        width=int(ckpt.get("width", summary.get("width", 16))),
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=layers,
        dropout=0.0,
        init_scale=float(ckpt.get("init_scale", summary.get("init_scale", 0.7))),
        use_skip_connections=True,
        use_batch_norm=use_bn,
        use_coordinates=use_coords,
        add_constant_mode=False,
        use_factorized_weights=factorized,
        share_bases=share,
        use_film_conditioning=bool(ckpt.get("use_film_conditioning", summary.get("use_film_conditioning", True))),
        film_hidden=int(ckpt.get("film_hidden", summary.get("film_hidden", 64))),
        film_in_dim=1,
    )
    return model, k_in, k_out


def _interp_rb_from_ckpt(
    ckpt: dict,
    coarse_nodes: np.ndarray,
    coarse_elems: np.ndarray,
    fine_nodes: np.ndarray,
    fine_elems: np.ndarray,
    n_probe: int,
    *,
    dual_encoder: bool,
    dual_mode: str = "paper",
    mass_kind: str = "consistent",
) -> Tuple[PCAReducerMultiChannelMass, dict]:
    pm = ckpt.get("pca_mass")
    if pm is None:
        raise ValueError("checkpoint missing pca_mass for interpolation")
    n_c = coarse_nodes.shape[0]
    n_f = fine_nodes.shape[0]
    print(f"Building P1 prolongation P ({n_f} x {n_c})...", flush=True)
    from AMOR.models.dis_dual import m_orthonormalize_rows_general
    from AMOR.models.mass.mesh_mass import consistent_nodal_mass, lumped_nodal_mass, mass_kron_channels

    P = build_prolongation_sparse(coarse_nodes, coarse_elems, fine_nodes, n_probe)
    if mass_kind == "lumped":
        M_f = sp.diags(lumped_nodal_mass(fine_nodes, fine_elems), format="csr")
        print("Using lumped nodal mass for dual G (SI parity)", flush=True)
    else:
        M_f = consistent_nodal_mass(fine_nodes, fine_elems)
        print("Using consistent P1 mass for dual G", flush=True)
    mass_f = np.asarray(M_f.diagonal(), dtype=np.float64)

    mean_in_f, comp_in_f = _prolong_scalar_modes(
        P, np.asarray(pm["pca_in_mean"], dtype=np.float64), np.asarray(pm["pca_in_components"], dtype=np.float64)
    )
    mean_out_f, comp_out_f = _prolong_vector_modes(
        P,
        np.asarray(pm["pca_out_mean"], dtype=np.float64),
        np.asarray(pm["pca_out_components"], dtype=np.float64),
        n_c,
        3,
    )
    # B = (P Φᵀ)ᵀ; paper mode keeps B unchanged.
    B_in_ref = np.asarray(comp_in_f, dtype=np.float64).copy()
    B_out_ref = np.asarray(comp_out_f, dtype=np.float64).copy()
    diag = {
        "dual_mode": dual_mode,
        "B_in_vs_PPhi": 0.0,
        "B_out_vs_PPhi": 0.0,
        "dual_in": None,
        "dual_out": None,
    }
    enc_in = enc_out = None
    M_out = mass_kron_channels(M_f, 3)
    if dual_encoder and dual_mode == "reortho":
        # Legacy SI path: Löwdin-replace B, then encode via Φᵀ M (no separate E rows).
        print(f"Dual-encoder LEGACY reortho: B ← G^{-1/2} B ({mass_kind} M); encode=ΦᵀM", flush=True)
        M_out = mass_kron_channels(M_f, 3)
        comp_in_f = m_orthonormalize_rows_general(comp_in_f, M_f).astype(np.float32)
        comp_out_f = m_orthonormalize_rows_general(comp_out_f, M_out).astype(np.float32)
        diag["B_in_vs_PPhi"] = float(
            np.linalg.norm(comp_in_f - B_in_ref) / max(np.linalg.norm(B_in_ref), 1e-30)
        )
        diag["B_out_vs_PPhi"] = float(
            np.linalg.norm(comp_out_f - B_out_ref) / max(np.linalg.norm(B_out_ref), 1e-30)
        )
        # Leave enc_in/enc_out None → mass encode z=Φᵀ M (x−μ) matches SI table path.
        enc_in = enc_out = None
        diag["dual_in"] = {"mode": "mass_PhiTM_after_reortho", "k": int(comp_in_f.shape[0])}
        diag["dual_out"] = {"mode": "mass_PhiTM_after_reortho", "k": int(comp_out_f.shape[0])}
    elif dual_encoder:
        print(
            "Dual-encoder (paper Sec. 2.5): B = P Φ unchanged; E = G^{-1}(B M), G=B M Bᵀ",
            flush=True,
        )
        enc_in, diag["dual_in"] = _dual_encoder_rows_paper(comp_in_f, M_f, name="block_in / tip E_in")
        enc_out, diag["dual_out"] = _dual_encoder_rows_paper(
            comp_out_f, M_out, name="block_out / tip E_out"
        )
        diag["B_in_vs_PPhi"] = float(np.linalg.norm(comp_in_f - B_in_ref) / max(np.linalg.norm(B_in_ref), 1e-30))
        diag["B_out_vs_PPhi"] = float(
            np.linalg.norm(comp_out_f - B_out_ref) / max(np.linalg.norm(B_out_ref), 1e-30)
        )
        print(
            f"||B_in - (P Phi_in)^T||_rel = {diag['B_in_vs_PPhi']:.3e}; "
            f"||B_out - (P Phi_out)^T||_rel = {diag['B_out_vs_PPhi']:.3e}",
            flush=True,
        )

    rb = _rb_from_views(
        pca_in_mean=mean_in_f,
        pca_in_comp=comp_in_f,
        pca_out_mean=mean_out_f,
        pca_out_comp=comp_out_f,
        mass_nodes=mass_f,
        n_nodes=n_f,
        mass_matrix=M_f,
        enc_in=enc_in,
        enc_out=enc_out,
    )
    return rb, diag


def _fit_refit_rb(x4, y4, mass_nodes, k_in, k_out_pca, seed, nodes=None, elements=None) -> PCAReducerMultiChannelMass:
    return PCAReducerMultiChannelMass(svd_solver="randomized", random_state=seed).fit(
        x4.astype(np.float32),
        y4.astype(np.float32),
        mass_nodes=mass_nodes,
        nodes=nodes,
        elements=elements,
        k_in=k_in,
        k_out=k_out_pca,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Mass mesh transfer eval (refit or P1+dual interp)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--summary", default=None)
    p.add_argument("--target_data", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--method", choices=("refit", "interp"), required=True)
    p.add_argument("--coarse_data", default=None, help="Coarse train npz (interp); default ckpt data_path")
    p.add_argument("--use_solve_ok_only", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_probe", type=int, default=64)
    p.add_argument(
        "--dual_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="interp: dual encoder on (default: on)",
    )
    p.add_argument(
        "--dual_mode",
        choices=("paper", "reortho"),
        default="paper",
        help="paper: B=PΦ unchanged, E=G^{-1}(BM); reortho: legacy Löwdin B←G^{-1/2}B",
    )
    p.add_argument(
        "--mass_kind",
        choices=("consistent", "lumped"),
        default="consistent",
        help="fine-mesh mass for dual G=BMBᵀ (SI table used lumped)",
    )
    p.add_argument(
        "--use_coordinates_override",
        choices=["auto", "on", "off"],
        default="off",
        help="coords always off in trainers; use auto/on only for legacy tip ckpts",
    )
    args = p.parse_args()

    summary: Dict = {}
    if args.summary and os.path.isfile(args.summary):
        with open(args.summary, "r", encoding="utf-8") as f:
            summary = json.load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    _ensure_pca_mass(ckpt, state)
    coarse_for_norm = args.coarse_data or ckpt.get("data_path")
    if coarse_for_norm and os.path.isfile(str(coarse_for_norm)):
        _fill_training_norm(ckpt, str(coarse_for_norm))
    has_coords = any(k.startswith("coord_embed.") for k in state)
    if args.use_coordinates_override == "auto":
        use_coords = has_coords
    else:
        use_coords = args.use_coordinates_override == "on"
    if has_coords and not use_coords:
        print(
            "WARNING: checkpoint has coord_embed but use_coordinates=False; "
            "θ load will drop coord weights. Prefer --use_coordinates_override auto|on for legacy tip.",
            flush=True,
        )

    d = _load_dataset(args.target_data, args.use_solve_ok_only)
    nodes_f = np.asarray(d["nodes"], dtype=np.float64)
    elems_f = np.asarray(d["elements"], dtype=np.int64)
    u = np.asarray(d["u"], dtype=np.float32)
    e = np.asarray(d["E_nodal"], dtype=np.float32)
    s = u.shape[0]
    x = e[:, None, :].astype(np.float32)
    y = np.transpose(u, (0, 2, 1)).astype(np.float32)
    norm_stats = _norm_stats_from_ckpt(ckpt)
    xn, yn = _zscore_fields(x, y, norm_stats)

    k_in = int(ckpt.get("k_in", summary.get("k_in", 100)))
    k_out = int(ckpt.get("k_out", summary.get("k_out", 30)))
    k_out_pca = max(k_in, k_out)

    dual_diag = None
    if args.method == "refit":
        mass_nodes = lumped_nodal_mass(nodes_f, elems_f)  # metadata only; fit uses consistent M
        rb = _fit_refit_rb(
            xn[:, :, :, None], yn[:, :, :, None], mass_nodes, k_in, k_out_pca, args.seed,
            nodes=nodes_f, elements=elems_f,
        )
        method_name = "target_mesh_mass_pca_refit_frozen_theta"
        pca_note = "consistent-P1 M-PCA refit on z-scored target E,u; coarse norm stats"
    else:
        coarse_path = args.coarse_data or ckpt.get("data_path")
        if not coarse_path or not os.path.isfile(coarse_path):
            raise ValueError("interp requires --coarse_data or ckpt data_path")
        d_c = _load_dataset(coarse_path, False)
        nodes_c = np.asarray(d_c["nodes"], dtype=np.float64)
        elems_c = np.asarray(d_c["elements"], dtype=np.int64)
        rb, dual_diag = _interp_rb_from_ckpt(
            ckpt,
            nodes_c,
            elems_c,
            nodes_f,
            elems_f,
            args.n_probe,
            dual_encoder=args.dual_encoder,
            dual_mode=args.dual_mode,
            mass_kind=args.mass_kind,
        )
        method_name = "coarse_mass_pca_p1_dual_encoder_frozen_theta"
        pca_note = (
            "P1 prolong coarse pca_mass; paper dual: B=PΦ unchanged, E=G^{-1}(BM) "
            "with consistent P1 M on fine mesh"
        )

    model, k_in_m, k_out_m = _build_model(rb, ckpt, summary, use_coords, state)
    learnable_mismatch, unexpected = _load_theta_only(model, state)
    if unexpected:
        print(f"WARNING unexpected keys: {unexpected[:8]}...", flush=True)
    if learnable_mismatch:
        raise RuntimeError(f"missing learnable keys after theta load: {learnable_mismatch[:12]}")
    def _block_keys(sd: dict, suffix: str) -> set:
        return {
            k
            for k in sd
            if k.endswith(suffix) and k.startswith(("block0.", "block1.", "blocks."))
        }

    for suffix in (".weight", "mode_weight", "width_weights"):
        ckpt_keys = _block_keys(state, suffix)
        model_keys = _block_keys(model.state_dict(), suffix)
        if ckpt_keys != model_keys:
            raise RuntimeError(
                f"block key mismatch ({suffix}): ckpt={sorted(ckpt_keys)} model={sorted(model_keys)}"
            )
    n_blocks = model.n_nonlocal_blocks()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    factorized, share, use_bn = _arch_from_state(state)
    print(f"instantiated blocks={n_blocks} params={n_params:,}", flush=True)

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    l = np.asarray(d["load"], dtype=np.float32).reshape(-1)
    ln = ckpt.get("load_norm", {"mu": float(l.mean()), "std": float(l.std() + 1e-8)})
    l_n = ((l - float(ln["mu"])) / max(float(ln["std"]), 1e-8)).astype(np.float32)

    preds_norm = np.zeros_like(yn)
    with torch.no_grad():
        for i0 in range(0, s, args.batch_size):
            i1 = min(s, i0 + args.batch_size)
            xb = torch.from_numpy(xn[i0:i1]).float().to(device)
            lb = torch.from_numpy(l_n[i0:i1, None]).float().to(device)
            preds_norm[i0:i1] = model(xb, lb).cpu().numpy()

    y_pred_phys = _denorm_pred(preds_norm, norm_stats)
    m_norm = _metrics(preds_norm.astype(np.float64), yn.astype(np.float64))
    m_phys = _metrics(y_pred_phys.astype(np.float64), y.astype(np.float64))

    out = {
        "method": method_name,
        "transfer_method": args.method,
        "checkpoint": args.checkpoint,
        "target_data": args.target_data,
        "eval_num_samples": int(s),
        "target_nodes": int(nodes_f.shape[0]),
        "k_in": k_in_m,
        "k_out": k_out_m,
        "k_out_pca_fitted": int(rb.pca_out.n_components_),
        "use_coordinates_model": use_coords,
        "use_batch_norm": use_bn,
        "use_factorized_weights": factorized,
        "share_bases": share,
        "n_blocks": n_blocks,
        "num_params": n_params,
        "mass_kind": args.mass_kind if args.method == "interp" else "consistent",
        "dual_encoder": bool(args.dual_encoder) if args.method == "interp" else None,
        "metrics_normalized": m_norm,
        "metrics": m_phys,
        "notes": {"pca": pca_note, "theta": "shape-matched state_dict keys only"},
        "dual_residuals": dual_diag,
    }
    if args.method == "interp":
        out["coarse_data"] = coarse_path
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
