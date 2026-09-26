#!/usr/bin/env python3
"""Shim → generate_nonlinear_diffusion.py --preset smooth (kept for old CLI paths)."""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--preset", "smooth", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parent / "generate_nonlinear_diffusion.py"), run_name="__main__")
