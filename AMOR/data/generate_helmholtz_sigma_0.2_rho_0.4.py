#!/usr/bin/env python3
"""Shim → generate_helmholtz.py --preset smooth."""
import runpy, sys
from pathlib import Path
sys.argv = [sys.argv[0], "--preset", "smooth", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parent / "generate_helmholtz.py"), run_name="__main__")
