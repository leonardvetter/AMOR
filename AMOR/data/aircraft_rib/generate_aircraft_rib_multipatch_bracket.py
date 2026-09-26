#!/usr/bin/env python3
"""Generate aircraft_rib_multipatch_bracket (localized patches, load_case_id=1)."""
import runpy, sys
from pathlib import Path
sys.argv = [sys.argv[0], "--dataset", "multipatch_bracket", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parent / "generate_aircraft_rib_bcfield.py"), run_name="__main__")
