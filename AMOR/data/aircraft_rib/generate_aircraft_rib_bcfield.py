#!/usr/bin/env python3
"""
BC-field generator for aircraft_rib_distributed_aero / aircraft_rib_multipatch_bracket.

Clamp BC is fixed across all samples. Load application region/type varies.
Saves both:
  - facet traction field (`traction_facet`), and
  - deterministic boundary nodal load field (`load_field_boundary`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from linear_elasticity_mesh_convergence import (  # type: ignore
    _interp_gp_E_to_nodes,
    _load_mesh_entry,
    _sample_gp_field,
    solve_elasticity,
)


def _face_set_to_array(face_set: set | None) -> np.ndarray:
    if not face_set:
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.array(sorted([tuple(sorted(k)) for k in face_set]), dtype=np.int64)
    return arr.reshape((-1, 3))


def _sample_scalar(rng: np.random.Generator, mode: str, base: float, vmin: float, vmax: float) -> float:
    m = mode.strip().lower()
    if m == "fixed":
        return float(base)
    if m == "uniform":
        return float(rng.uniform(vmin, vmax))
    raise ValueError(f"Unsupported mode={mode!r}; use fixed or uniform.")


def _sample_tip_target_pct(rng: np.random.Generator, tip_min: float, tip_max: float, uniform_mix: float) -> float:
    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    z = float(rng.random()) if rng.random() < uniform_mix else float(rng.beta(2.0, 2.0))
    return float(tip_min + z * (tip_max - tip_min))


def _sample_unit_load_direction(rng: np.random.Generator, tip_axis: np.ndarray) -> np.ndarray:
    v = rng.standard_normal(3)
    nv = float(np.linalg.norm(v))
    if nv <= 0.0:
        v = np.array([1.0, 0.0, 0.0], dtype=float)
        nv = 1.0
    v = v / nv
    tip_axis = np.asarray(tip_axis, dtype=float).reshape(3)
    tip_axis = tip_axis / (np.linalg.norm(tip_axis) + 1e-12)
    if float(np.dot(v, tip_axis)) < 0.0:
        v = -v
    return v.astype(float)


def _tet_boundary_faces(elements: np.ndarray) -> np.ndarray:
    face_count: Dict[Tuple[int, int, int], int] = {}
    e = np.asarray(elements, dtype=np.int64)
    for tet in e:
        i0, i1, i2, i3 = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
        faces = (
            tuple(sorted((i0, i1, i2))),
            tuple(sorted((i0, i1, i3))),
            tuple(sorted((i0, i2, i3))),
            tuple(sorted((i1, i2, i3))),
        )
        for f in faces:
            face_count[f] = face_count.get(f, 0) + 1
    ext = sorted([f for f, c in face_count.items() if c == 1])
    return np.asarray(ext, dtype=np.int64).reshape(-1, 3)


def _face_areas_and_centroids(nodes: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(nodes, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ctr = (a + b + c) / 3.0
    return area, ctr


def _sample_patch_faces(
    rng: np.random.Generator,
    allowed_faces_idx: np.ndarray,
    face_centroids: np.ndarray,
    face_area: np.ndarray,
    patch_area_target: float,
    min_patch_facets: int,
) -> np.ndarray:
    seed_local = int(rng.integers(0, allowed_faces_idx.shape[0]))
    seed_fi = int(allowed_faces_idx[seed_local])
    c0 = face_centroids[seed_fi]
    cand = allowed_faces_idx
    d2 = np.sum((face_centroids[cand] - c0[None, :]) ** 2, axis=1)
    order = np.argsort(d2)
    chosen: List[int] = []
    a_sum = 0.0
    for oi in order:
        fi = int(cand[int(oi)])
        chosen.append(fi)
        a_sum += float(face_area[fi])
        if len(chosen) >= int(min_patch_facets) and a_sum >= float(patch_area_target):
            break
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def _project_facet_to_boundary_nodes(
    n_nodes: int,
    boundary_faces: np.ndarray,
    boundary_node_idx: np.ndarray,
    facet_area: np.ndarray,
    traction_facet: np.ndarray,
) -> np.ndarray:
    node_wsum = np.zeros(n_nodes, dtype=np.float64)
    node_vsum = np.zeros((n_nodes, 3), dtype=np.float64)
    for fi in range(boundary_faces.shape[0]):
        a = float(facet_area[fi])
        t = np.asarray(traction_facet[fi], dtype=np.float64)
        f = boundary_faces[fi]
        for v in (int(f[0]), int(f[1]), int(f[2])):
            node_wsum[v] += a
            node_vsum[v] += a * t
    out = np.zeros((boundary_node_idx.shape[0], 3), dtype=np.float64)
    for k, v in enumerate(boundary_node_idx):
        vv = int(v)
        if node_wsum[vv] > 0.0:
            out[k] = node_vsum[vv] / node_wsum[vv]
    return out.astype(np.float32)


def _import_case_samples_from_npz(
    src_path: Path,
    import_case_id: int,
    dest: Dict[str, np.ndarray],
    n_samples: int,
) -> int:
    src = np.load(src_path, allow_pickle=True)
    if "load_case_id" not in src.files:
        raise ValueError(f"{src_path} missing load_case_id")
    n_src = int(np.asarray(src.get("completed_count", np.array([src["load_case_id"].shape[0]]))).reshape(-1)[0])
    case = np.asarray(src["load_case_id"], dtype=np.int64).reshape(-1)[:n_src]
    idx = np.where(case == int(import_case_id))[0]
    if idx.size == 0:
        raise RuntimeError(f"No samples with load_case_id={import_case_id} in {src_path}")
    n_copy = min(int(idx.size), int(n_samples))
    idx = idx[:n_copy]
    key_map = {
        "u": "u",
        "u_magnitude": "u_magnitude",
        "load": "load",
        "load_direction": "load_direction",
        "traction_facet": "traction_facet",
        "load_field_boundary": "load_field_boundary",
        "load_case_id": "load_case_id",
        "E_mean": "E_mean",
        "E_std": "E_std",
        "E_nodal": "E_nodal",
        "nu": "nu",
        "tip_target_pct": "tip_target_pct",
        "tip_deflection_pct": "tip_deflection_pct",
        "relative_residual": "relative_residual",
        "solve_ok": "solve_ok",
        "t_solve": "t_solve",
        "gp_corr_len": "gp_corr_len",
    }
    for dst_key, src_key in key_map.items():
        if src_key not in src.files:
            raise ValueError(f"{src_path} missing {src_key}")
        if dst_key not in dest:
            raise KeyError(dst_key)
        dest[dst_key][:n_copy] = np.asarray(src[src_key])[idx]
    print(
        f"Imported {n_copy} samples with load_case_id={import_case_id} from {src_path}",
        flush=True,
    )
    return n_copy


def _assign_load_case(
    rng: np.random.Generator,
    *,
    force_load_case_id: int | None,
    uniform_load_prob: float,
    allowed_faces_idx: np.ndarray,
    tip_axis: np.ndarray,
    base_total: float,
    patch_count_max: int,
    patch_area_frac_min: float,
    patch_area_frac_max: float,
    min_patch_facets: int,
    allowed_area: float,
    facet_ctr: np.ndarray,
    facet_area: np.ndarray,
) -> tuple[int, List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    if force_load_case_id is not None:
        use_uniform = int(force_load_case_id) == 0
    else:
        use_uniform = bool(rng.random() < float(uniform_load_prob))

    if use_uniform:
        load_case_id = 0
        patch_sets = [allowed_faces_idx.copy()]
        d0 = _sample_unit_load_direction(rng, tip_axis)
        patch_dirs = [d0]
        patch_loads = np.asarray([base_total], dtype=np.float64)
        load_dir_i = d0
    else:
        load_case_id = 1
        n_patch = int(rng.integers(1, int(patch_count_max) + 1))
        patch_sets = []
        patch_dirs = []
        for _ in range(n_patch):
            frac = float(np.exp(rng.uniform(np.log(patch_area_frac_min), np.log(patch_area_frac_max))))
            target_area = frac * allowed_area
            pidx = _sample_patch_faces(
                rng=rng,
                allowed_faces_idx=allowed_faces_idx,
                face_centroids=facet_ctr,
                face_area=facet_area,
                patch_area_target=target_area,
                min_patch_facets=int(min_patch_facets),
            )
            patch_sets.append(pidx)
            patch_dirs.append(_sample_unit_load_direction(rng, tip_axis))
        w = rng.random(n_patch)
        w = w / max(np.sum(w), 1e-30)
        patch_loads = w * base_total
        load_dir_i = np.sum(np.asarray(patch_dirs) * patch_loads[:, None], axis=0)
        nrm = float(np.linalg.norm(load_dir_i))
        load_dir_i = load_dir_i / nrm if nrm > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return load_case_id, patch_sets, patch_dirs, patch_loads, load_dir_i


def _combine_patch_solves_linear(
    *,
    mesh_entry: Dict[str, Any],
    solver_backend: str,
    bc_marking: str,
    max_ram_gb: float | None,
    E_scalar: float,
    nu_scalar: float,
    E_field: np.ndarray | None,
    dirichlet_face_set: set,
    boundary_faces: np.ndarray,
    facet_area: np.ndarray,
    patches_face_idx: List[np.ndarray],
    patch_dirs: List[np.ndarray],
    patch_total_loads: np.ndarray,
    tip_target_pct: float,
    load_min: float,
    load_max: float,
    primary_axis: int,
) -> Dict[str, Any]:
    n_nodes = int(np.asarray(mesh_entry["nodes"]).shape[0])
    u_sum = np.zeros((n_nodes, 3), dtype=np.float64)
    t_solve_sum = 0.0
    relres_list: List[float] = []
    solve_ok = True
    traction_facet = np.zeros((boundary_faces.shape[0], 3), dtype=np.float64)
    loaded_nodes_union: set[int] = set()

    for j, face_idx in enumerate(patches_face_idx):
        fs = {frozenset(int(v) for v in boundary_faces[int(fi)]) for fi in face_idx}
        dvec = np.asarray(patch_dirs[j], dtype=np.float64).reshape(3)
        F = float(patch_total_loads[j])
        if F <= 0.0 or len(fs) == 0:
            continue
        res = solve_elasticity(
            mesh_entry["nodes"],
            mesh_entry["elements"],
            E=E_scalar,
            nu=nu_scalar,
            max_ram_gb=max_ram_gb,
            bc_marking=bc_marking,
            total_load=F,
            load_direction=dvec,
            solver_backend=solver_backend,
            E_field=E_field,
            dirichlet_face_set=dirichlet_face_set,
            neumann_face_set=fs,
        )
        solve_ok = solve_ok and bool(res.get("solve_ok", False))
        t_solve_sum += float(res.get("t_solve", 0.0))
        rr = float(res.get("relative_residual", np.nan))
        if np.isfinite(rr):
            relres_list.append(rr)
        u_sum += np.asarray(res["u_field"], dtype=np.float64)

        area_patch = float(np.sum(facet_area[face_idx]))
        if area_patch > 0.0:
            traction_facet[face_idx, :] += (F / area_patch) * dvec[None, :]
        for fi in face_idx:
            tri = boundary_faces[int(fi)]
            loaded_nodes_union.update(int(v) for v in tri)

    nodes = np.asarray(mesh_entry["nodes"], dtype=np.float64)
    extents = nodes.max(axis=0) - nodes.min(axis=0)
    rib_len = float(extents[int(primary_axis)])
    umag = np.linalg.norm(u_sum, axis=1)
    if loaded_nodes_union:
        idx = np.asarray(sorted(loaded_nodes_union), dtype=np.int64)
        tip0 = float(np.max(umag[idx]))
    else:
        tip0 = float(np.max(umag))
    tip_pct0 = 100.0 * tip0 / max(rib_len, 1e-30)

    scale = 1.0 if tip_pct0 <= 0.0 or (not np.isfinite(tip_pct0)) else float(tip_target_pct) / float(tip_pct0)
    scale = max(0.02, min(50.0, scale))
    total_load0 = float(np.sum(patch_total_loads))
    total_load_scaled = float(np.clip(total_load0 * scale, float(load_min), float(load_max)))
    scale2 = total_load_scaled / max(total_load0, 1e-30)

    u = u_sum * scale2
    um = np.linalg.norm(u, axis=1)
    if loaded_nodes_union:
        idx = np.asarray(sorted(loaded_nodes_union), dtype=np.int64)
        tip = float(np.max(um[idx]))
    else:
        tip = float(np.max(um))
    tip_pct = 100.0 * tip / max(rib_len, 1e-30)

    return {
        "u_field": u.astype(np.float32),
        "u_magnitude": um.astype(np.float32),
        "tip_deflection_pct": float(tip_pct),
        "total_load": float(total_load_scaled),
        "patch_loads_scaled": (np.asarray(patch_total_loads, dtype=np.float64) * scale2),
        "traction_facet": (traction_facet * scale2).astype(np.float32),
        "solve_ok": bool(solve_ok),
        "relative_residual": float(np.max(relres_list)) if len(relres_list) else float("nan"),
        "t_solve": float(t_solve_sum),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate rib dataset with explicit facet+boundary load fields")
    p.add_argument(
        "--dataset",
        choices=("distributed_aero", "multipatch_bracket", "mixed"),
        default=None,
        help="Named paper datasets: distributed_aero→force case0 n=10k; "
             "multipatch_bracket→force case1 n=20k; mixed→no force (probabilistic).",
    )
    p.add_argument("--mesh-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--meta-json", default=None)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bc-marking", choices=("marked", "geom", "surface"), default="marked")
    p.add_argument("--solver-backend", choices=("scipy_direct", "scipy_cgilu", "petsc_lu", "petsc_cgilu", "petsc_auto"), default="scipy_direct")
    p.add_argument("--max-ram", type=float, default=None)

    p.add_argument("--load-mode", choices=("mixed",), default="mixed")
    p.add_argument("--base-load", type=float, default=100.0)
    p.add_argument("--load-min", type=float, default=5.0)
    p.add_argument("--load-max", type=float, default=3000.0)
    p.add_argument("--uniform-load-prob", type=float, default=0.4)
    p.add_argument("--patch-count-max", type=int, default=3)
    p.add_argument("--patch-area-frac-min", type=float, default=0.005)
    p.add_argument("--patch-area-frac-max", type=float, default=0.08)
    p.add_argument("--min-patch-facets", type=int, default=6)
    p.add_argument("--exclude-clamp-band-frac", type=float, default=0.03)

    p.add_argument("--tip-min-pct", type=float, default=0.01)
    p.add_argument("--tip-max-pct", type=float, default=0.3)
    p.add_argument("--tip-uniform-mix", type=float, default=0.6)

    p.add_argument("--E-mode", choices=("fixed", "uniform", "gp"), default="gp")
    p.add_argument("--E", type=float, default=70000.0)
    p.add_argument("--E-min", type=float, default=40000.0)
    p.add_argument("--E-max", type=float, default=120000.0)
    p.add_argument("--gp-std-m", type=float, default=0.75)
    p.add_argument("--gp-corr-len-mode", choices=("fixed", "uniform"), default="uniform")
    p.add_argument("--gp-corr-len", type=float, default=80.0)
    p.add_argument("--gp-corr-len-min", type=float, default=20.0)
    p.add_argument("--gp-corr-len-max", type=float, default=180.0)
    p.add_argument("--gp-grid-resolution", type=int, default=64)
    p.add_argument("--gp-e-min", type=float, default=5e3)
    p.add_argument("--gp-e-max", type=float, default=250e3)

    p.add_argument("--nu-mode", choices=("fixed", "uniform"), default="uniform")
    p.add_argument("--nu", type=float, default=0.3)
    p.add_argument("--nu-min", type=float, default=0.25)
    p.add_argument("--nu-max", type=float, default=0.35)

    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--force-load-case-id",
        type=int,
        choices=(0, 1),
        default=None,
        help="If set, every sample uses this load case (0=uniform boundary, 1=localized patch).",
    )
    p.add_argument(
        "--import-from-npz",
        type=str,
        default=None,
        help="Copy existing samples with --import-case-id from this NPZ into output before generating new ones.",
    )
    p.add_argument(
        "--import-case-id",
        type=int,
        choices=(0, 1),
        default=1,
        help="load_case_id to import from --import-from-npz (default 1=patch).",
    )
    args = p.parse_args()
    if args.dataset == "distributed_aero":
        args.force_load_case_id = 0
        if args.n_samples == 1000:
            args.n_samples = 10000
    elif args.dataset == "multipatch_bracket":
        args.force_load_case_id = 1
        if args.n_samples == 1000:
            args.n_samples = 20000
    # mixed: leave force_load_case_id as given (default None → probabilistic)

    rng = np.random.default_rng(args.seed)
    mesh_entry = _load_mesh_entry(Path(args.mesh_file), args.bc_marking)
    nodes = np.asarray(mesh_entry["nodes"], dtype=np.float64)
    elements = np.asarray(mesh_entry["elements"], dtype=np.int64)
    n_nodes = nodes.shape[0]
    n_samples = int(args.n_samples)

    bbox_min = nodes.min(axis=0)
    bbox_max = nodes.max(axis=0)
    ext = bbox_max - bbox_min
    primary_axis = int(np.argmax(ext))
    tip_axis = np.zeros(3, dtype=float)
    tip_axis[primary_axis] = 1.0

    dirichlet_tris = _face_set_to_array(mesh_entry.get("dirichlet_face_set"))
    neumann_tris = _face_set_to_array(mesh_entry.get("neumann_face_set"))
    boundary_faces = _tet_boundary_faces(elements)
    facet_area, facet_ctr = _face_areas_and_centroids(nodes, boundary_faces)
    boundary_node_idx = np.unique(boundary_faces.reshape(-1))
    nb = int(boundary_node_idx.shape[0])
    fb = int(boundary_faces.shape[0])

    dface_set = mesh_entry.get("dirichlet_face_set")
    if dface_set is None:
        dface_set = set()
    dirichlet_face_set = dface_set
    dir_keys = {tuple(sorted(tuple(int(v) for v in k))) for k in dirichlet_face_set}
    x_min = float(nodes[:, primary_axis].min())
    clamp_band = float(args.exclude_clamp_band_frac) * float(ext[primary_axis])
    allowed = []
    for fi in range(fb):
        tri = boundary_faces[fi]
        key = tuple(sorted((int(tri[0]), int(tri[1]), int(tri[2]))))
        if key in dir_keys:
            continue
        if float(np.min(nodes[tri, primary_axis])) <= (x_min + clamp_band):
            continue
        allowed.append(fi)
    allowed_faces_idx = np.asarray(sorted(allowed), dtype=np.int64)
    if allowed_faces_idx.size == 0:
        raise RuntimeError("No allowed boundary facets after clamp exclusion.")
    allowed_area = float(np.sum(facet_area[allowed_faces_idx]))

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    u_all = np.zeros((n_samples, n_nodes, 3), dtype=np.float32)
    umag_all = np.zeros((n_samples, n_nodes), dtype=np.float32)
    load_all = np.zeros(n_samples, dtype=np.float64)
    load_dir_all = np.zeros((n_samples, 3), dtype=np.float64)
    traction_facet_all = np.zeros((n_samples, fb, 3), dtype=np.float32)
    load_field_boundary_all = np.zeros((n_samples, nb, 3), dtype=np.float32)
    load_case_id_all = np.zeros(n_samples, dtype=np.int64)  # 0=uniform, 1=patch
    E_all = np.zeros(n_samples, dtype=np.float64)
    E_std_all = np.zeros(n_samples, dtype=np.float64)
    E_nodal_all = np.zeros((n_samples, n_nodes), dtype=np.float32)
    nu_all = np.zeros(n_samples, dtype=np.float64)
    tip_target_all = np.zeros(n_samples, dtype=np.float64)
    tip_pct_all = np.zeros(n_samples, dtype=np.float64)
    relres_all = np.zeros(n_samples, dtype=np.float64)
    solve_ok_all = np.zeros(n_samples, dtype=bool)
    tsolve_all = np.zeros(n_samples, dtype=np.float64)
    gp_corr_len_all = np.zeros(n_samples, dtype=np.float64)

    start_i = 0

    dest_arrays: Dict[str, np.ndarray] = {
        "u": u_all,
        "u_magnitude": umag_all,
        "load": load_all,
        "load_direction": load_dir_all,
        "traction_facet": traction_facet_all,
        "load_field_boundary": load_field_boundary_all,
        "load_case_id": load_case_id_all,
        "E_mean": E_all,
        "E_std": E_std_all,
        "E_nodal": E_nodal_all,
        "nu": nu_all,
        "tip_target_pct": tip_target_all,
        "tip_deflection_pct": tip_pct_all,
        "relative_residual": relres_all,
        "solve_ok": solve_ok_all,
        "t_solve": tsolve_all,
        "gp_corr_len": gp_corr_len_all,
    }

    def _save_checkpoint(completed_count: int) -> None:
        # Stage on node-local tmp to avoid 2× quota peak (main + .tmp.npz).
        stage_dir = Path(os.environ.get("SLURM_TMPDIR", out_path.parent))
        stage_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = stage_dir / (out_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            nodes=nodes,
            elements=elements,
            dirichlet_tris=dirichlet_tris,
            neumann_tris=neumann_tris,
            boundary_facet_tris=boundary_faces,
            boundary_facet_area=facet_area.astype(np.float32),
            boundary_node_idx=boundary_node_idx,
            traction_facet=traction_facet_all,
            load_field_boundary=load_field_boundary_all,
            load_case_id=load_case_id_all,
            u=u_all,
            u_magnitude=umag_all,
            load=load_all,
            load_direction=load_dir_all,
            E_mean=E_all,
            E_std=E_std_all,
            E_nodal=E_nodal_all,
            nu=nu_all,
            tip_target_pct=tip_target_all,
            tip_deflection_pct=tip_pct_all,
            relative_residual=relres_all,
            solve_ok=solve_ok_all,
            t_solve=tsolve_all,
            gp_corr_len=gp_corr_len_all,
            completed_count=np.array([int(completed_count)], dtype=np.int64),
            n_samples_requested=np.array([int(n_samples)], dtype=np.int64),
        )
        shutil.move(str(tmp_path), str(out_path))

    if args.resume and out_path.exists():
        prev = np.load(out_path, allow_pickle=True)
        start_i = int(np.asarray(prev.get("completed_count", np.array([0]))).reshape(-1)[0])
        for key, arr in (
            ("u", u_all),
            ("u_magnitude", umag_all),
            ("load", load_all),
            ("load_direction", load_dir_all),
            ("traction_facet", traction_facet_all),
            ("load_field_boundary", load_field_boundary_all),
            ("load_case_id", load_case_id_all),
            ("E_mean", E_all),
            ("E_std", E_std_all),
            ("E_nodal", E_nodal_all),
            ("nu", nu_all),
            ("tip_target_pct", tip_target_all),
            ("tip_deflection_pct", tip_pct_all),
            ("relative_residual", relres_all),
            ("solve_ok", solve_ok_all),
            ("t_solve", tsolve_all),
            ("gp_corr_len", gp_corr_len_all),
        ):
            if key in prev.files:
                src = np.asarray(prev[key])
                arr[: min(start_i, src.shape[0])] = src[: min(start_i, src.shape[0])]
        print(f"Resuming from sample {start_i}/{n_samples}", flush=True)

    if args.import_from_npz and start_i == 0:
        n_imp = _import_case_samples_from_npz(
            Path(args.import_from_npz).expanduser().resolve(),
            int(args.import_case_id),
            dest_arrays,
            n_samples,
        )
        start_i = n_imp
        if n_imp > 0:
            _save_checkpoint(n_imp)
            print(f"Import checkpoint: {n_imp}/{n_samples}", flush=True)

    if start_i >= n_samples:
        print(f"Already complete: {start_i}/{n_samples}", flush=True)
        meta_path = Path(args.meta_json).expanduser().resolve() if args.meta_json else out_path.with_suffix(".json")
        return

    for i in range(start_i, n_samples):
        # Mean E scalar always fixed/uniform; GP field built separately when E_mode=gp.
        E_i = _sample_scalar(
            rng, "fixed" if args.E_mode == "gp" else args.E_mode, args.E, args.E_min, args.E_max
        )
        nu_i = _sample_scalar(rng, args.nu_mode, args.nu, args.nu_min, args.nu_max)
        corr_len_i = _sample_scalar(rng, args.gp_corr_len_mode, args.gp_corr_len, args.gp_corr_len_min, args.gp_corr_len_max)
        gp_corr_len_all[i] = corr_len_i

        E_field_i = None
        if args.E_mode == "gp":
            gp = _sample_gp_field(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                mean_E=E_i,
                std_m=float(args.gp_std_m),
                corr_len=float(corr_len_i),
                grid_resolution=int(args.gp_grid_resolution),
                seed=int(args.seed + i),
                e_min=float(args.gp_e_min),
                e_max=float(args.gp_e_max),
            )
            E_field_i = _interp_gp_E_to_nodes(gp, nodes)

        tip_target_i = _sample_tip_target_pct(rng, args.tip_min_pct, args.tip_max_pct, args.tip_uniform_mix)
        tip_target_all[i] = tip_target_i

        base_total = float(args.base_load)
        load_case_id_i, patch_sets, patch_dirs, patch_loads, load_dir_i = _assign_load_case(
            rng,
            force_load_case_id=args.force_load_case_id,
            uniform_load_prob=float(args.uniform_load_prob),
            allowed_faces_idx=allowed_faces_idx,
            tip_axis=tip_axis,
            base_total=base_total,
            patch_count_max=int(args.patch_count_max),
            patch_area_frac_min=float(args.patch_area_frac_min),
            patch_area_frac_max=float(args.patch_area_frac_max),
            min_patch_facets=int(args.min_patch_facets),
            allowed_area=allowed_area,
            facet_ctr=facet_ctr,
            facet_area=facet_area,
        )
        load_case_id_all[i] = load_case_id_i
        load_dir_all[i] = load_dir_i

        res = _combine_patch_solves_linear(
            mesh_entry=mesh_entry,
            solver_backend=args.solver_backend,
            bc_marking=args.bc_marking,
            max_ram_gb=args.max_ram,
            E_scalar=float(E_i),
            nu_scalar=nu_i,
            E_field=E_field_i,
            dirichlet_face_set=dirichlet_face_set,
            boundary_faces=boundary_faces,
            facet_area=facet_area,
            patches_face_idx=patch_sets,
            patch_dirs=patch_dirs,
            patch_total_loads=patch_loads,
            tip_target_pct=tip_target_i,
            load_min=float(args.load_min),
            load_max=float(args.load_max),
            primary_axis=primary_axis,
        )

        u_all[i] = np.asarray(res["u_field"], dtype=np.float32)
        umag_all[i] = np.asarray(res["u_magnitude"], dtype=np.float32)
        load_all[i] = float(res["total_load"])
        traction_facet_all[i] = np.asarray(res["traction_facet"], dtype=np.float32)
        load_field_boundary_all[i] = _project_facet_to_boundary_nodes(
            n_nodes=n_nodes,
            boundary_faces=boundary_faces,
            boundary_node_idx=boundary_node_idx,
            facet_area=facet_area,
            traction_facet=traction_facet_all[i],
        )
        E_all[i] = float(np.mean(E_field_i)) if E_field_i is not None else float(E_i)
        E_std_all[i] = float(np.std(E_field_i)) if E_field_i is not None else 0.0
        E_nodal_all[i] = np.asarray(E_field_i if E_field_i is not None else np.full(n_nodes, E_i), dtype=np.float32)
        nu_all[i] = nu_i
        tip_pct_all[i] = float(res["tip_deflection_pct"])
        relres_all[i] = float(res["relative_residual"])
        solve_ok_all[i] = bool(res["solve_ok"])
        tsolve_all[i] = float(res["t_solve"])

        print(
            f"[{i+1:04d}/{n_samples:04d}] ok={solve_ok_all[i]} case={int(load_case_id_all[i])} "
            f"tip_target={tip_target_i:.4f}%L tip={tip_pct_all[i]:.4f}%L "
            f"load={load_all[i]:.4f} Emean={E_all[i]:.2f} Estd={E_std_all[i]:.2f} nu={nu_i:.4f}",
            flush=True,
        )

        if args.save_every > 0 and ((i + 1) % int(args.save_every) == 0 or (i + 1) == n_samples):
            _save_checkpoint(i + 1)
            print(f"Checkpoint saved: {out_path} ({i+1}/{n_samples})", flush=True)

    _save_checkpoint(n_samples)
    print(f"Saved dataset: {out_path}", flush=True)

    meta_path = Path(args.meta_json).expanduser().resolve() if args.meta_json else out_path.with_suffix(".json")
    meta = {
        "mesh_file": str(Path(args.mesh_file).expanduser().resolve()),
        "n_samples": int(n_samples),
        "seed": int(args.seed),
        "bc_marking": str(args.bc_marking),
        "solver_backend": str(args.solver_backend),
        "load_mode": str(args.load_mode),
        "uniform_load_prob": float(args.uniform_load_prob),
        "force_load_case_id": args.force_load_case_id,
        "import_from_npz": args.import_from_npz,
        "import_case_id": int(args.import_case_id),
        "patch_count_max": int(args.patch_count_max),
        "patch_area_frac_min": float(args.patch_area_frac_min),
        "patch_area_frac_max": float(args.patch_area_frac_max),
        "min_patch_facets": int(args.min_patch_facets),
        "exclude_clamp_band_frac": float(args.exclude_clamp_band_frac),
        "load_case_id_meaning": {"0": "uniform_allowed_boundary", "1": "localized_patch"},
        "uses_boundary_facet_traction": True,
        "uses_boundary_load_field": True,
        "traction_facet_shape": [int(n_samples), int(fb), 3],
        "boundary_load_field_shape": [int(n_samples), int(nb), 3],
        "facet_to_node_projection": "area_weighted_incident_boundary_facets_v1",
        "stores_E_nodal": True,
        "stores_nu": True,
        "stores_u": True,
        "stores_tip_deflection_pct": True,
        "save_every": int(args.save_every),
        "resume": bool(args.resume),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
