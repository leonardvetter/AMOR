#!/usr/bin/env python3
"""Merge sharded rib dataset npz files into one vertex-E dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--n_shards", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta_json", default=None)
    args = ap.parse_args()
    shard_dir = Path(args.shard_dir)
    shards = []
    for i in range(args.n_shards):
        p = shard_dir / f"shard_{i:03d}.npz"
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        shards.append(np.load(p, allow_pickle=True))
    # order by i_start
    starts = [int(np.asarray(s["i_start"]).reshape(-1)[0]) for s in shards]
    order = np.argsort(starts)
    shards = [shards[j] for j in order]
    starts = [starts[j] for j in order]
    ends = [int(np.asarray(s["i_end"]).reshape(-1)[0]) for s in shards]
    n_global = int(np.asarray(shards[0]["n_samples_global"]).reshape(-1)[0])
    if starts[0] != 0 or ends[-1] != n_global:
        raise SystemExit(f"bad coverage starts={starts} ends={ends} n_global={n_global}")
    for a, b in zip(ends, starts[1:]):
        if a != b:
            raise SystemExit(f"gap/overlap at {a} vs {b}")

    def cat(key):
        return np.concatenate([np.asarray(s[key]) for s in shards], axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        nodes=np.asarray(shards[0]["nodes"]),
        elements=np.asarray(shards[0]["elements"]),
        dirichlet_tris=np.asarray(shards[0]["dirichlet_tris"]),
        neumann_tris=np.asarray(shards[0]["neumann_tris"]),
        u=cat("u"),
        u_magnitude=cat("u_magnitude"),
        force_axis=cat("force_axis"),
        load=cat("load"),
        E_mean=cat("E_mean"),
        E_std=cat("E_std"),
        E_nodal=cat("E_nodal"),
        nu=cat("nu"),
        tip_deflection_pct=cat("tip_deflection_pct"),
        relative_residual=cat("relative_residual"),
        solve_ok=cat("solve_ok"),
        t_solve=cat("t_solve"),
        completed_count=np.array([n_global], dtype=np.int64),
        n_samples_requested=np.array([n_global], dtype=np.int64),
        e_layout=np.array(["vertex"], dtype=object),
    )
    ok = int(np.count_nonzero(cat("solve_ok")))
    print(f"Merged {args.n_shards} shards → {out} n={n_global} solve_ok={ok}")
    meta = {
        "n_samples": n_global,
        "ok_count": ok,
        "e_layout": "vertex",
        "shard_dir": str(shard_dir),
        "n_shards": args.n_shards,
        "mean_tip_pct": float(np.nanmean(cat("tip_deflection_pct"))),
        "source": "merge_rib_dataset_shards.py after E vertex→DOF fix",
    }
    meta_path = Path(args.meta_json) if args.meta_json else out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
