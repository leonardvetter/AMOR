#!/usr/bin/env python3
"""Sample figures: ground truth vs AMOR prediction vs pointwise relative absolute error.

Two random test samples per dataset (fixed RandomState(0)), predictions of the chosen seed. Writes
<out-dir>/figures/{airfoil,elasticity}_sample_visualization.{png,pdf}. The error panel shows
|pred - truth| / max|truth| (maximum over the sample's nodes) pointwise.

Usage: python AMOR/scripts/make_geofno_figures.py [--seed 0] [--out-dir DIR] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from AMOR.data import geofno as gdata

AIRFOIL_CROP = (40, 20)  # drop the first/last 40 circumferential and keep 20 radial rows, as in the SI figure


def panel_data(dataset, X, Y, pred, i):
    """Coordinates and (truth, pred, rel. abs. error) of test sample i."""
    t, p = Y[i, 0], pred[i, 0]
    err = np.abs(p - t) / np.abs(t).max()
    return X[i, 0], X[i, 1], t, p, err


def draw(dataset, rows, title):
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.2 * len(rows)), squeeze=False)
    for r, (idx, (x, y, t, p, e)) in enumerate(rows):
        if dataset == "airfoil":
            nx, n = AIRFOIL_CROP
            sl = (slice(nx, -nx), slice(0, n))
            x, y, t, p, e = (a[sl] for a in (x, y, t, p, e))
            draw_one = lambda ax, v, **kw: ax.pcolormesh(x, y, v, shading="gouraud", **kw)  # noqa: E731
        else:
            x, y, t, p, e = (a.reshape(-1) for a in (x, y, t, p, e))
            draw_one = lambda ax, v, **kw: ax.scatter(x, y, c=v, s=9, **kw)  # noqa: E731
        vmin, vmax = float(t.min()), float(t.max())
        for c, (v, name, kw) in enumerate([(t, "Ground truth", dict(vmin=vmin, vmax=vmax)),
                                           (p, "AMOR prediction", dict(vmin=vmin, vmax=vmax)),
                                           (e, "Pointwise rel. abs. error", dict(cmap="magma"))]):
            ax = axes[r, c]
            im = draw_one(ax, v, **kw)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name if r == 0 else "", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        axes[r, 0].set_ylabel(f"test sample {idx}")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments_out" / "geofno")
    # Same default as train_geofno.py: --data-dir, else AMOR_DATA_DIR, else <repo>/AMOR/data.
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("AMOR_DATA_DIR", REPO_ROOT / "AMOR" / "data")))
    ap.add_argument("--dataset", nargs="+", default=["airfoil", "elasticity"])
    a = ap.parse_args()
    fig_dir = a.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ds in a.dataset:
        f = a.out_dir / "predictions" / f"{ds}_seed{a.seed}.npz"
        if not f.exists():
            print(f"skip {ds}: {f} missing")
            continue
        d = np.load(f)
        X, Y, _, te = gdata.load(ds, a.data_dir)
        assert np.array_equal(te, d["test_idx"])
        Xte, Yte, pred = X[te], Y[te], d["pred"]
        pick = np.random.RandomState(0).choice(len(te), 2, replace=False)
        rows = [(int(te[i]), panel_data(ds, Xte, Yte, pred, i)) for i in pick]
        title = f"{'Airfoil (Mach number)' if ds == 'airfoil' else 'Elasticity (stress)'} -- seed {a.seed}"
        fig = draw(ds, rows, title)
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{ds}_sample_visualization.{ext}", dpi=200)
        plt.close(fig)
        print(f"wrote {fig_dir / (ds + '_sample_visualization.png')} (test samples {[r[0] for r in rows]})")


if __name__ == "__main__":
    main()
