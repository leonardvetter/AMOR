"""AMOR inference and one FEM solve per sample, on the same node.

The historical PCA bundle for the distributed-aero checkpoint is not on disk.
Inference uses the current AMOR graph at that checkpoint's sizes: 72615 nodes,
4 input and 3 output channels, k 15/2, width 24, 5 nonlocal blocks, dense T,
FiLM on (nu, load case). Residual blocks in the June checkpoint stored an
input-side encoder; the current code encodes the output side, so this times
the current graph, not a bit-identical replay of that file.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from AMOR.models.amor import AMOR
from AMOR.training.protocol_timing import run_protocol


class _PCA:
    def __init__(self, k: int, dim: int):
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((k, dim))
        q, _ = np.linalg.qr(raw.T)
        self.components_ = q[:, :k].T.astype(np.float32)
        self.mean_ = np.zeros(dim, dtype=np.float32)
        self.n_components_ = k


class _RB:
    shape_ = (72615, 1)
    n_channels = 4
    out_channels = 3
    pca_in = _PCA(15, 4 * 72615)
    pca_out = _PCA(2, 3 * 72615)


def _time_inference(device, out_dir: Path):
    model = AMOR(
        width=24, rb=_RB(), k_in=15, k_out=2, layers=5, dropout=0.0, init_scale=0.1,
        use_skip_connections=True, use_batch_norm=False, use_coordinates=False,
        use_factorized_weights=False, share_bases=False,
        use_film_conditioning=True, film_hidden=64, film_in_dim=2,
    )
    # The protocol loader yields (x, y). FiLM is a constant batch of zeros,
    # which matches a fixed (nu, case) and does not change the matmul sizes.
    zeros = {}

    def loss_fn(m, x, y):
        b = x.shape[0]
        film = zeros.get(b)
        if film is None or film.device != x.device:
            film = torch.zeros(b, 2, device=x.device)
            zeros[b] = film
        return torch.nn.functional.mse_loss(m(x, film), y)

    class _Loader:
        def __iter__(self):
            g = torch.Generator().manual_seed(0)
            for _ in range(8):
                x = torch.randn(8, 4, 72615, generator=g)
                y = torch.randn(8, 3, 72615, generator=g)
                yield x, y

    return run_protocol(
        model, _Loader(), device, loss_fn,
        out_dir / "rib_distributed_aero_AMOR.json",
        {"entry": "rib_distributed_aero_AMOR", "dataset": "rib_distributed_aero",
         "model": "AMOR", "precision": "fp32", "batch_size_requested": 8,
         "note": "current AMOR graph at the aero checkpoint sizes; FEM timed separately"},
    )


def _time_fem(out_dir: Path, n_solves: int = 3):
    from AMOR.data.aircraft_rib.linear_elasticity_mesh_convergence import solve_elasticity

    mesh = np.load("/users/PAS3244/dvoderhol/data/meshes_3d/rib_h15.0_marked.npz")
    nodes = np.asarray(mesh["nodes"])
    elements = np.asarray(mesh["elements"])
    diri = {frozenset(int(v) for v in tri) for tri in np.asarray(mesh["dirichlet_tris"])}
    neu = {frozenset(int(v) for v in tri) for tri in np.asarray(mesh["neumann_tris"])}
    times = []
    for i in range(n_solves):
        direction = np.array([0.0, 0.0, 1.0 if i % 2 == 0 else -1.0])
        t0 = time.perf_counter()
        solve_elasticity(
            nodes, elements, E=70e3, nu=0.3, bc_marking="marked",
            total_load=100.0, load_direction=direction,
            solver_backend="scipy_direct",
            dirichlet_face_set=diri, neumann_face_set=neu,
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"fem solve {i + 1}/{n_solves} {dt:.3f}s", flush=True)
    rec = {
        "entry": "rib_distributed_aero_FEM",
        "solver_backend": "scipy_direct",
        "bc_marking": "marked",
        "n_nodes": int(nodes.shape[0]),
        "n_elements": int(elements.shape[0]),
        "solves_s": times,
        "s_per_sample": float(np.median(times[1:] if len(times) > 1 else times)),
        "first_solve_s": times[0],
        "node": socket.gethostname(),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK") or str(os.cpu_count()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    path = out_dir / "rib_distributed_aero_FEM.json"
    path.write_text(json.dumps(rec, indent=2))
    print(f"wrote {path}", flush=True)


def main():
    out = Path(os.environ.get(
        "AMOR_PROTOCOL_OUT",
        "/users/PAS3244/dvoderhol/data/followup6_20261009/results/timing",
    ))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _time_inference(device, out)
    _time_fem(out)


if __name__ == "__main__":
    main()
