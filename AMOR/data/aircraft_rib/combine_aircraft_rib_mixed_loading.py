#!/usr/bin/env python3
"""Concatenate into aircraft_rib_mixed_loading_combined (paper 40k = tip10k+aero10k+patch20k).

Keeps mesh keys from the first input. Concatenates per-sample arrays along axis 0.
Writes sidecar JSON with subset paths and load_case_id counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def _n_use(z: np.lib.npyio.NpzFile) -> int:
    n = int(np.asarray(z["u"]).shape[0])
    if "completed_count" in z.files:
        n = min(n, int(np.asarray(z["completed_count"]).reshape(-1)[0]))
    if "n_samples_requested" in z.files:
        n = min(n, int(np.asarray(z["n_samples_requested"]).reshape(-1)[0]))
    return n


def combine(inputs: list[Path], out: Path, meta_json: Path | None, labels: list[str] | None) -> None:
    loaded = [np.load(p, allow_pickle=True) for p in inputs]
    ns = [_n_use(z) for z in loaded]
    n_total = int(sum(ns))
    ref = loaded[0]
    nodes = np.asarray(ref["nodes"])
    elements = np.asarray(ref["elements"])
    for i, z in enumerate(loaded[1:], start=1):
        if not (
            np.array_equal(np.asarray(z["nodes"]), nodes)
            and np.array_equal(np.asarray(z["elements"]), elements)
        ):
            raise ValueError(f"Mesh mismatch: {inputs[i]} vs {inputs[0]}")

    sample_keys = [
        k
        for k in ref.files
        if k not in _MESH_KEYS
        and k not in ("completed_count", "n_samples_requested")
        and hasattr(ref[k], "shape")
        and ref[k].ndim >= 1
        and ref[k].shape[0] in (ns[0], int(np.asarray(ref["u"]).shape[0]))
    ]
    # Prefer union of keys present in all inputs with leading sample axis.
    common = set(sample_keys)
    for z, n in zip(loaded, ns):
        keys = {
            k
            for k in z.files
            if k not in _MESH_KEYS
            and k not in ("completed_count", "n_samples_requested")
            and hasattr(z[k], "shape")
            and z[k].ndim >= 1
            and z[k].shape[0] >= n
        }
        common &= keys
    sample_keys = sorted(common)
    if "u" not in sample_keys or "load_case_id" not in sample_keys:
        raise RuntimeError(f"Missing required keys after intersect: {sample_keys}")

    out_dict: dict[str, np.ndarray] = {
        "nodes": nodes,
        "elements": elements,
    }
    for k in _MESH_KEYS:
        if k in ("nodes", "elements"):
            continue
        if k in ref.files:
            out_dict[k] = np.asarray(ref[k])

    for k in sample_keys:
        parts = [np.asarray(z[k])[:n] for z, n in zip(loaded, ns)]
        out_dict[k] = np.concatenate(parts, axis=0)

    out_dict["completed_count"] = np.array([n_total], dtype=np.int64)
    out_dict["n_samples_requested"] = np.array([n_total], dtype=np.int64)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **out_dict)

    case_ids = np.asarray(out_dict["load_case_id"]).reshape(-1)
    counts = {str(int(c)): int(np.count_nonzero(case_ids == c)) for c in sorted(set(case_ids.tolist()))}
    labels = labels or [p.name for p in inputs]
    subsets = {lab: str(p.resolve()) for lab, p in zip(labels, inputs)}
    meta = {
        "inputs": [str(p.resolve()) for p in inputs],
        "n_samples": n_total,
        "load_case_id_counts": counts,
        "subsets": subsets,
    }
    meta_path = meta_json if meta_json is not None else out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out} n={n_total} cases={counts}", flush=True)
    print(f"Wrote {meta_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Combine bcfield NPZs into mixed loading set")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Ordered NPZ paths (e.g. tip_loadfield aero multipatch)",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--meta-json", default=None)
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional subset labels matching --inputs order",
    )
    args = p.parse_args()
    inputs = [Path(x).expanduser().resolve() for x in args.inputs]
    labels = list(args.labels) if args.labels else None
    if labels is not None and len(labels) != len(inputs):
        raise SystemExit("--labels length must match --inputs")
    combine(inputs, Path(args.output).expanduser().resolve(), Path(args.meta_json) if args.meta_json else None, labels)


if __name__ == "__main__":
    main()
