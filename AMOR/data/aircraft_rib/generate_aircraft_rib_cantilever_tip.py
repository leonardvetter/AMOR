#!/usr/bin/env python3
"""
Generate aircraft_rib_cantilever_tip dataset (marked Neumann tip load + GP-E).

Uses linear_elasticity_mesh_convergence.solve_elasticity with BCs from *_marked.npz
(produced by boundary_marking.py). Optional tip-deflection load calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any

import numpy as np


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from linear_elasticity_mesh_convergence import (  # type: ignore
    _load_mesh_entry,
    solve_elasticity,
    _sample_gp_field,
    _interp_gp_E_to_nodes,
)


def _face_set_to_array(face_set: set | None) -> np.ndarray:
    if not face_set:
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.array(sorted([tuple(sorted(k)) for k in face_set]), dtype=np.int64)
    return arr.reshape((-1, 3))


def _calibrate_load_for_target_tip_pct(
    mesh_entry: Dict[str, Any],
    solver_backend: str,
    bc_marking: str,
    max_ram_gb: float | None,
    initial_total_load: float,
    target_tip_pct: float,
    E_scalar: float,
    nu_scalar: float,
    E_field: np.ndarray | None = None,
) -> tuple[float, Dict[str, Any]]:
    """Single linear scale calibration from baseline solve."""
    base = solve_elasticity(
        mesh_entry["nodes"],
        mesh_entry["elements"],
        E=E_scalar,
        nu=nu_scalar,
        max_ram_gb=max_ram_gb,
        bc_marking=bc_marking,
        total_load=initial_total_load,
        solver_backend=solver_backend,
        E_field=E_field,
        dirichlet_face_set=mesh_entry.get("dirichlet_face_set"),
        neumann_face_set=mesh_entry.get("neumann_face_set"),
    )
    d0 = float(base.get("tip_deflection_pct", 0.0))
    if (not base.get("solve_ok", False)) or d0 <= 0.0:
        raise RuntimeError("Calibration baseline solve failed or zero tip deflection.")
    scale = target_tip_pct / d0
    scale = max(0.05, min(20.0, scale))
    return float(initial_total_load * scale), base


def _sample_scalar(
    rng: np.random.Generator,
    mode: str,
    base: float,
    vmin: float,
    vmax: float,
) -> float:
    m = mode.strip().lower()
    if m == "fixed":
        return float(base)
    if m == "uniform":
        return float(rng.uniform(vmin, vmax))
    raise ValueError(f"Unsupported mode={mode!r}; use fixed or uniform.")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate aircraft_rib_cantilever_tip NPZ")
    p.add_argument("--mesh-file", required=True, help="Path to *_marked.npz mesh")
    p.add_argument("--output", required=True, help="Output dataset .npz path")
    p.add_argument("--meta-json", default=None, help="Optional metadata JSON path")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bc-marking", choices=("marked", "geom", "surface"), default="marked")
    p.add_argument(
        "--solver-backend",
        choices=("scipy_direct", "scipy_cgilu", "petsc_lu", "petsc_cgilu", "petsc_auto"),
        default="scipy_direct",
    )
    p.add_argument("--max-ram", type=float, default=None)

    p.add_argument("--calibrate-tip-pct", type=float, default=0.1,
                   help="Calibrate baseline load so tip deflection is this %% of rib length")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--base-load", type=float, default=100.0)

    p.add_argument("--load-mode", choices=("fixed", "uniform"), default="fixed")
    p.add_argument("--load-min", type=float, default=50.0)
    p.add_argument("--load-max", type=float, default=300.0)

    p.add_argument("--E-mode", choices=("fixed", "uniform", "gp"), default="fixed")
    p.add_argument("--E", type=float, default=70e3)
    p.add_argument("--E-min", type=float, default=60e3)
    p.add_argument("--E-max", type=float, default=80e3)
    p.add_argument("--gp-std-m", type=float, default=0.4)
    p.add_argument("--gp-corr-len", type=float, default=80.0)
    p.add_argument("--gp-grid-resolution", type=int, default=64)
    p.add_argument("--gp-e-min", type=float, default=5e3,
                   help="Lower clip on GP E(x)=exp(m)+1 (same scale as --E).")
    p.add_argument("--gp-e-max", type=float, default=250e3,
                   help="Upper clip on GP E(x).")

    p.add_argument("--nu-mode", choices=("fixed", "uniform"), default="fixed")
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--nu-min", type=float, default=0.25)
    p.add_argument("--nu-max", type=float, default=0.35)
    p.add_argument("--save-every", type=int, default=100,
                   help="Checkpoint dataset to --output every N completed samples.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing --output checkpoint (uses completed_count key).")
    p.add_argument(
        "--i-start",
        type=int,
        default=0,
        help="First global sample index (inclusive). For array shards.",
    )
    p.add_argument(
        "--i-end",
        type=int,
        default=None,
        help="Last global sample index (exclusive). Default: --n-samples.",
    )
    args = p.parse_args()

    mesh_path = Path(args.mesh_file).expanduser().resolve()
    mesh_entry = _load_mesh_entry(mesh_path, args.bc_marking)

    rng = np.random.default_rng(args.seed)
    load_base = float(args.base_load)
    calibration_baseline = None
    bbox_min = np.min(mesh_entry["nodes"], axis=0)
    bbox_max = np.max(mesh_entry["nodes"], axis=0)
    calib_E_field = None
    if args.E_mode == "gp":
        gp0 = _sample_gp_field(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            mean_E=float(args.E),
            std_m=float(args.gp_std_m),
            corr_len=float(args.gp_corr_len),
            grid_resolution=int(args.gp_grid_resolution),
            seed=int(args.seed),
            e_min=float(args.gp_e_min),
            e_max=float(args.gp_e_max),
        )
        calib_E_field = _interp_gp_E_to_nodes(gp0, mesh_entry["nodes"])
    if not args.no_calibrate:
        load_base, calibration_baseline = _calibrate_load_for_target_tip_pct(
            mesh_entry=mesh_entry,
            solver_backend=args.solver_backend,
            bc_marking=args.bc_marking,
            max_ram_gb=args.max_ram,
            initial_total_load=load_base,
            target_tip_pct=float(args.calibrate_tip_pct),
            E_scalar=float(args.E),
            nu_scalar=float(args.nu),
            E_field=calib_E_field,
        )
        print(
            f"Calibrated base load: {load_base:.6g} "
            f"(target tip={args.calibrate_tip_pct:.4f}%L)",
            flush=True,
        )
    else:
        print(f"Calibration disabled; using base load={load_base:.6g}", flush=True)

    n_nodes = int(mesh_entry["n_nodes"])
    n_samples = int(args.n_samples)
    if n_samples <= 0:
        raise ValueError("--n-samples must be positive")
    i_start = int(args.i_start)
    i_end = int(args.i_end) if args.i_end is not None else n_samples
    if not (0 <= i_start < i_end <= n_samples):
        raise ValueError(f"bad range [{i_start},{i_end}) for n_samples={n_samples}")
    n_local = i_end - i_start
    print(f"Generating global samples [{i_start},{i_end}) of {n_samples} (n_local={n_local})", flush=True)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    u_all = np.zeros((n_local, n_nodes, 3), dtype=np.float32)
    umag_all = np.zeros((n_local, n_nodes), dtype=np.float32)
    force_axis_all = np.zeros((n_local, n_nodes), dtype=np.float32)
    load_all = np.zeros(n_local, dtype=np.float64)
    E_all = np.zeros(n_local, dtype=np.float64)
    E_std_all = np.zeros(n_local, dtype=np.float64)
    E_nodal_all = np.zeros((n_local, n_nodes), dtype=np.float32)
    nu_all = np.zeros(n_local, dtype=np.float64)
    tip_pct_all = np.zeros(n_local, dtype=np.float64)
    relres_all = np.zeros(n_local, dtype=np.float64)
    solve_ok_all = np.zeros(n_local, dtype=bool)
    tsolve_all = np.zeros(n_local, dtype=np.float64)
    start_local = 0

    def _save_checkpoint(completed_local: int) -> None:
        np.savez_compressed(
            out_path,
            nodes=np.asarray(mesh_entry["nodes"], dtype=np.float64),
            elements=np.asarray(mesh_entry["elements"], dtype=np.int64),
            dirichlet_tris=_face_set_to_array(mesh_entry.get("dirichlet_face_set")),
            neumann_tris=_face_set_to_array(mesh_entry.get("neumann_face_set")),
            u=u_all,
            u_magnitude=umag_all,
            force_axis=force_axis_all,
            load=load_all,
            E_mean=E_all,
            E_std=E_std_all,
            E_nodal=E_nodal_all,
            nu=nu_all,
            tip_deflection_pct=tip_pct_all,
            relative_residual=relres_all,
            solve_ok=solve_ok_all,
            t_solve=tsolve_all,
            completed_count=np.array([int(completed_local)], dtype=np.int64),
            n_samples_requested=np.array([int(n_local)], dtype=np.int64),
            i_start=np.array([int(i_start)], dtype=np.int64),
            i_end=np.array([int(i_end)], dtype=np.int64),
            n_samples_global=np.array([int(n_samples)], dtype=np.int64),
            e_layout=np.array(["vertex"], dtype=object),
        )

    if args.resume and out_path.exists():
        prev = np.load(out_path, allow_pickle=True)
        if "completed_count" not in prev.files:
            raise RuntimeError("Resume requested but output missing completed_count key.")
        start_local = int(np.asarray(prev["completed_count"]).reshape(-1)[0])
        if start_local < 0 or start_local > n_local:
            raise RuntimeError(f"Invalid completed_count={start_local} for n_local={n_local}")
        for key, arr in (
            ("u", u_all),
            ("u_magnitude", umag_all),
            ("force_axis", force_axis_all),
            ("load", load_all),
            ("E_mean", E_all),
            ("E_std", E_std_all),
            ("E_nodal", E_nodal_all),
            ("nu", nu_all),
            ("tip_deflection_pct", tip_pct_all),
            ("relative_residual", relres_all),
            ("solve_ok", solve_ok_all),
            ("t_solve", tsolve_all),
        ):
            if key in prev.files:
                src = np.asarray(prev[key])
                arr[: min(start_local, src.shape[0])] = src[: min(start_local, src.shape[0])]
        print(f"Resuming local index {start_local}/{n_local} (global {i_start+start_local})", flush=True)

    for loc in range(start_local, n_local):
        i = i_start + loc
        load_i = _sample_scalar(rng, args.load_mode, load_base, args.load_min, args.load_max)
        E_i = _sample_scalar(rng, "fixed" if args.E_mode == "gp" else args.E_mode, args.E, args.E_min, args.E_max)
        nu_i = _sample_scalar(rng, args.nu_mode, args.nu, args.nu_min, args.nu_max)
        nu_i = float(np.clip(nu_i, 1e-6, 0.499))
        E_field_i = None
        if args.E_mode == "gp":
            gp = _sample_gp_field(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                mean_E=E_i,
                std_m=float(args.gp_std_m),
                corr_len=float(args.gp_corr_len),
                grid_resolution=int(args.gp_grid_resolution),
                seed=int(args.seed + i),
                e_min=float(args.gp_e_min),
                e_max=float(args.gp_e_max),
            )
            E_field_i = _interp_gp_E_to_nodes(gp, mesh_entry["nodes"])

        res = solve_elasticity(
            mesh_entry["nodes"],
            mesh_entry["elements"],
            E=E_i,
            nu=nu_i,
            max_ram_gb=args.max_ram,
            bc_marking=args.bc_marking,
            total_load=load_i,
            solver_backend=args.solver_backend,
            E_field=E_field_i,
            dirichlet_face_set=mesh_entry.get("dirichlet_face_set"),
            neumann_face_set=mesh_entry.get("neumann_face_set"),
        )

        u_all[loc] = np.asarray(res["u_field"], dtype=np.float32)
        umag_all[loc] = np.asarray(res["u_magnitude"], dtype=np.float32)
        force_axis_all[loc] = np.asarray(res.get("force_axis", np.zeros(n_nodes)), dtype=np.float32)
        load_all[loc] = load_i
        E_all[loc] = float(res.get("E_mean", E_i))
        E_std_all[loc] = float(res.get("E_std", 0.0))
        if E_field_i is None:
            E_nodal_all[loc] = np.full(n_nodes, float(E_i), dtype=np.float32)
        else:
            E_nodal_all[loc] = np.asarray(E_field_i, dtype=np.float32)
        nu_all[loc] = nu_i
        tip_pct_all[loc] = float(res.get("tip_deflection_pct", np.nan))
        relres_all[loc] = float(res.get("relative_residual", np.nan))
        solve_ok_all[loc] = bool(res.get("solve_ok", False))
        tsolve_all[loc] = float(res.get("t_solve", np.nan))

        print(
            f"[global {i+1:04d}/{n_samples:04d} local {loc+1}/{n_local}] "
            f"ok={solve_ok_all[loc]} "
            f"tip%L={tip_pct_all[loc]:.4f} "
            f"load={load_i:.4f} Emean={E_all[loc]:.2f} Estd={E_std_all[loc]:.2f} nu={nu_i:.4f}",
            flush=True,
        )
        if args.save_every > 0 and ((loc + 1) % int(args.save_every) == 0 or (loc + 1) == n_local):
            _save_checkpoint(loc + 1)
            print(f"Checkpoint saved: {out_path} ({loc+1}/{n_local})", flush=True)

    _save_checkpoint(n_local)
    print(f"Saved dataset shard: {out_path}", flush=True)

    meta = {
        "mesh_file": str(mesh_path),
        "n_samples": n_samples,
        "i_start": i_start,
        "i_end": i_end,
        "n_local": n_local,
        "seed": int(args.seed),
        "bc_marking": args.bc_marking,
        "solver_backend": args.solver_backend,
        "load_mode": args.load_mode,
        "base_load": float(load_base),
        "calibrate_tip_pct": None if args.no_calibrate else float(args.calibrate_tip_pct),
        "E_mode": args.E_mode,
        "nu_mode": args.nu_mode,
        "mean_tip_pct": float(np.nanmean(tip_pct_all)),
        "std_tip_pct": float(np.nanstd(tip_pct_all)),
        "ok_count": int(np.count_nonzero(solve_ok_all)),
        "fail_count": int(n_local - np.count_nonzero(solve_ok_all)),
        "gp_material": bool(args.E_mode == "gp"),
        "gp_std_m": float(args.gp_std_m) if args.E_mode == "gp" else None,
        "gp_corr_len": float(args.gp_corr_len) if args.E_mode == "gp" else None,
        "gp_grid_resolution": int(args.gp_grid_resolution) if args.E_mode == "gp" else None,
        "gp_e_min": float(args.gp_e_min) if args.E_mode == "gp" else None,
        "gp_e_max": float(args.gp_e_max) if args.E_mode == "gp" else None,
        "stores_E_nodal": True,
        "e_layout": "vertex",
        "save_every": int(args.save_every),
        "resume": bool(args.resume),
    }
    if calibration_baseline is not None:
        meta["calibration_baseline_tip_pct"] = float(calibration_baseline.get("tip_deflection_pct", np.nan))
        meta["calibration_baseline_relres"] = float(calibration_baseline.get("relative_residual", np.nan))

    meta_path = Path(args.meta_json).expanduser().resolve() if args.meta_json else out_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
