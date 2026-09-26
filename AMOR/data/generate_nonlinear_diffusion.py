#!/usr/bin/env python3
"""Unified nonlinear diffusion dataset generator.

Presets match paper priors (ℓ_m, ℓ_f) / (mvar,mlen,fvar,flen):
  medium  → fvar=5,  flen=0.2,  mvar=1,   mlen=0.3   → (0.3, 0.2)
  smooth  → fvar=2,  flen=0.4,  mvar=0.5, mlen=0.5   → (0.5, 0.4)
  rough   → fvar=10, flen=0.1,  mvar=2,   mlen=0.15  → (0.15, 0.1)

Examples:
  python generate_nonlinear_diffusion.py --preset medium
  python generate_nonlinear_diffusion.py --preset medium --num_samples 20000 --suffix big
  python generate_nonlinear_diffusion.py --preset medium --N 50 --num_samples 1000 --suffix res50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve()
for _ in range(6):
    if (_REPO / "AMOR" / "paths.py").is_file():
        break
    _REPO = _REPO.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from AMOR.paths import expand  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from nonlinear_diffusion_gp import generate_nonlinear_diffusion  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PRESETS = {
    "medium": dict(fvar=5.0, flen=0.2, mvar=1.0, mlen=0.3),
    "smooth": dict(fvar=2.0, flen=0.4, mvar=0.5, mlen=0.5),
    "rough": dict(fvar=10.0, flen=0.1, mvar=2.0, mlen=0.15),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), required=True)
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--suffix", type=str, default="", help="e.g. big, res50")
    p.add_argument("--output_dir", type=str, default=expand("${AMOR_SCRATCH_DATA}"))
    args = p.parse_args()
    cfg = PRESETS[args.preset]
    generate_nonlinear_diffusion(
        **cfg,
        N=args.N,
        num_samples=args.num_samples,
        suffix=args.suffix,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
