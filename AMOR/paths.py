"""Central path helpers for AMOR. Override via env; no cluster account hardcoding."""

from __future__ import annotations

import os
import sys
from pathlib import Path

AMOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AMOR_ROOT.parent

# Defaults so os.path.expandvars("${AMOR_DATA_ROOT}/...") works in CLIs.
os.environ.setdefault("AMOR_DATA_ROOT", str(Path.home() / "data"))
os.environ.setdefault("AMOR_SCRATCH_DATA", os.environ["AMOR_DATA_ROOT"])
os.environ.setdefault("AMOR_SCRATCH", str(Path(os.environ["AMOR_SCRATCH_DATA"]).parent))
os.environ.setdefault("AMOR_REPO", str(REPO_ROOT))


def ensure_amor_on_path() -> Path:
    """Insert repo root on sys.path once (AMOR is importable as package)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def data_root() -> Path:
    return Path(os.environ["AMOR_DATA_ROOT"]).expanduser()


def scratch_data_root() -> Path:
    return Path(os.environ["AMOR_SCRATCH_DATA"]).expanduser()


def rib_datasets_dir() -> Path:
    return Path(os.environ.get("AMOR_RIB_DATASETS", data_root() / "aircraft_rib" / "datasets")).expanduser()


def meshes_dir() -> Path:
    return Path(os.environ.get("AMOR_MESHES_DIR", data_root() / "meshes_3d")).expanduser()


def expand(p: str | Path) -> str:
    """Expand ${AMOR_DATA_ROOT}, ${AMOR_SCRATCH_DATA}, ${HOME}, ${AMOR_REPO}, etc."""
    return os.path.expandvars(os.path.expanduser(str(p)))
