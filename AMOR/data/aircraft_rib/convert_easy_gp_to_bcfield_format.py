#!/usr/bin/env python3
"""Convert easy GP rib NPZ to bcfield-compatible layout for combined training.

Easy GP: scalar `load` on fixed marked Neumann facets, axial load direction.
Output: same keys as bcfield generator (`load_field_boundary`, `traction_facet`, …).
Uses load_case_id=2 for easy GP Neumann-tip (distinct from bcfield 0/1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EASY_LOAD_CASE_ID = 2

_MESH_KEYS = frozenset(
    {
        "nodes",
        "elements",
        "dirichlet_tris",
        "neumann_tris",
        "boundary_facet_tris",
        "boundary_facet_area",
        "boundary_node_idx",
    }
)


def _facet_keys_from_tris(tris: np.ndarray) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for tri in np.asarray(tris, dtype=np.int64).reshape(-1, 3):
        out.add(tuple(sorted(int(v) for v in tri)))
    return out


def _build_neumann_facet_mask(boundary_faces: np.ndarray, neumann_tris: np.ndarray) -> np.ndarray:
    neu_keys = _facet_keys_from_tris(neumann_tris)
    mask = np.zeros(boundary_faces.shape[0], dtype=bool)
    for fi in range(boundary_faces.shape[0]):
        tri = boundary_faces[fi]
        key = tuple(sorted(int(v) for v in tri))
        if key in neu_keys:
            mask[fi] = True
    if not np.any(mask):
        raise RuntimeError("No Neumann facets matched boundary_facet_tris")
    return mask


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
        for v in boundary_faces[fi]:
            vv = int(v)
            node_wsum[vv] += a
            node_vsum[vv] += a * t
    out = np.zeros((boundary_node_idx.shape[0], 3), dtype=np.float64)
    for k, v in enumerate(boundary_node_idx):
        vv = int(v)
        if node_wsum[vv] > 0.0:
            out[k] = node_vsum[vv] / node_wsum[vv]
    return out.astype(np.float32)


def _primary_axis(nodes: np.ndarray) -> int:
    ext = nodes.max(axis=0) - nodes.min(axis=0)
    return int(np.argmax(ext))


def convert_easy_to_bcfield(
    easy_path: Path,
    ref_bcfield_path: Path,
    *,
    use_solve_ok_only: bool = True,
) -> tuple[dict[str, np.ndarray], int]:
    ez = np.load(easy_path, allow_pickle=True)
    ref = np.load(ref_bcfield_path, allow_pickle=True)

    n_alloc = int(np.asarray(ez["u"]).shape[0])
    n_use = n_alloc
    if "completed_count" in ez.files:
        n_use = min(n_use, int(np.asarray(ez["completed_count"]).reshape(-1)[0]))
    if "n_samples_requested" in ez.files:
        n_use = min(n_use, int(np.asarray(ez["n_samples_requested"]).reshape(-1)[0]))
    if use_solve_ok_only and "solve_ok" in ez.files:
        ok = np.asarray(ez["solve_ok"], dtype=bool).reshape(-1)[:n_use]
        n_use = int(np.count_nonzero(ok))
        if n_use == 0:
            raise RuntimeError("No solve_ok samples in easy dataset")

    nodes = np.asarray(ref["nodes"], dtype=np.float64)
    elements = np.asarray(ref["elements"], dtype=np.int64)
    if not (np.array_equal(np.asarray(ez["nodes"]), nodes) and np.array_equal(np.asarray(ez["elements"]), elements)):
        raise ValueError("Easy and reference bcfield mesh (nodes/elements) differ; use same rib_h15 marked mesh")

    boundary_faces = np.asarray(ref["boundary_facet_tris"], dtype=np.int64)
    facet_area = np.asarray(ref["boundary_facet_area"], dtype=np.float32)
    boundary_node_idx = np.asarray(ref["boundary_node_idx"], dtype=np.int64)
    dirichlet_tris = np.asarray(ref["dirichlet_tris"], dtype=np.int64)
    neumann_tris = np.asarray(ez["neumann_tris"], dtype=np.int64)
    n_nodes = nodes.shape[0]
    fb = boundary_faces.shape[0]
    nb = boundary_node_idx.shape[0]

    neu_facet_mask = _build_neumann_facet_mask(boundary_faces, neumann_tris)
    neu_area = float(np.sum(facet_area[neu_facet_mask]))
    if neu_area <= 0.0:
        raise RuntimeError("Neumann facet area is zero")

    pax = _primary_axis(nodes)
    load_dir = np.zeros(3, dtype=np.float64)
    load_dir[pax] = 1.0

    def _take(k: str):
        arr = np.asarray(ez[k])
        if arr.ndim == 0 or k in _MESH_KEYS:
            return arr
        if use_solve_ok_only and "solve_ok" in ez.files:
            ok = np.asarray(ez["solve_ok"], dtype=bool).reshape(-1)
            ok = ok[: min(ok.shape[0], arr.shape[0])]
            return arr[ok]
        return arr[:n_use]

    u = _take("u").astype(np.float32)
    n_use = u.shape[0]
    umag = _take("u_magnitude").astype(np.float32)
    load_scalar = _take("load").astype(np.float64)
    E_nodal = _take("E_nodal").astype(np.float32)
    E_mean = _take("E_mean").astype(np.float64)
    E_std = _take("E_std").astype(np.float64)
    nu = _take("nu").astype(np.float64)
    tip_pct = _take("tip_deflection_pct").astype(np.float64)
    relres = _take("relative_residual").astype(np.float64)
    solve_ok = _take("solve_ok").astype(bool)
    tsolve = _take("t_solve").astype(np.float64)

    load_direction = np.tile(load_dir[None, :], (n_use, 1)).astype(np.float64)
    load_case_id = np.full(n_use, EASY_LOAD_CASE_ID, dtype=np.int64)
    gp_corr_len = np.zeros(n_use, dtype=np.float64)

    neu_idx = np.where(neu_facet_mask)[0]
    unit_facet = np.zeros((fb, 3), dtype=np.float64)
    unit_facet[neu_idx] = load_dir / neu_area
    unit_lfb = _project_facet_to_boundary_nodes(
        n_nodes, boundary_faces, boundary_node_idx, facet_area, unit_facet
    )
    traction_facet = (load_scalar[:, None, None] * unit_facet[None, :, :]).astype(np.float32)
    load_field_boundary = (load_scalar[:, None, None] * unit_lfb[None, :, :]).astype(np.float32)

    out: dict[str, np.ndarray] = {
        "nodes": nodes,
        "elements": elements,
        "dirichlet_tris": dirichlet_tris,
        "neumann_tris": neumann_tris,
        "boundary_facet_tris": boundary_faces,
        "boundary_facet_area": facet_area,
        "boundary_node_idx": boundary_node_idx,
        "traction_facet": traction_facet,
        "load_field_boundary": load_field_boundary,
        "load_case_id": load_case_id,
        "u": u,
        "u_magnitude": umag,
        "load": load_scalar,
        "load_direction": load_direction,
        "E_mean": E_mean,
        "E_std": E_std,
        "E_nodal": E_nodal,
        "nu": nu,
        "tip_target_pct": tip_pct.copy(),
        "tip_deflection_pct": tip_pct,
        "relative_residual": relres,
        "solve_ok": solve_ok,
        "t_solve": tsolve,
        "gp_corr_len": gp_corr_len,
        "completed_count": np.array([n_use], dtype=np.int64),
        "n_samples_requested": np.array([n_use], dtype=np.int64),
    }
    return out, n_use


def main() -> None:
    p = argparse.ArgumentParser(description="Convert easy GP NPZ to bcfield training format")
    p.add_argument("--easy-path", required=True)
    p.add_argument(
        "--ref-bcfield-path",
        required=True,
        help="Reference bcfield NPZ for boundary_facet_* arrays (same h15 mesh)",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--meta-json", default=None)
    p.add_argument("--include-failed", action="store_true", help="Include solve_ok=False samples")
    args = p.parse_args()

    out_dict, n_use = convert_easy_to_bcfield(
        Path(args.easy_path),
        Path(args.ref_bcfield_path),
        use_solve_ok_only=not args.include_failed,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    print(f"Wrote {out_path} n={n_use} load_case_id={EASY_LOAD_CASE_ID}", flush=True)

    meta = {
        "source_easy": str(Path(args.easy_path).resolve()),
        "ref_bcfield": str(Path(args.ref_bcfield_path).resolve()),
        "n_samples": int(n_use),
        "load_case_id": int(EASY_LOAD_CASE_ID),
        "load_case_id_meaning": {
            "0": "uniform_allowed_boundary",
            "1": "localized_patch",
            "2": "easy_gp_marked_neumann_tip",
        },
        "traction_rule": "uniform on easy neumann_tris: t = load_dir * (load / A_neumann)",
        "load_direction": "primary mesh axis (+1)",
        "compatible_with": "train_amor_aircraft_rib_bcfield.py / build_field_tensors",
    }
    meta_path = Path(args.meta_json) if args.meta_json else out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
