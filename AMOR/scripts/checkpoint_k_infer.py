"""Infer AMOR k_in/k_out from checkpoint tensors (metadata can be stale)."""

from __future__ import annotations


def strip_compile_prefix(state: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def infer_k_from_ckpt(ck: dict, default_in: int, default_out: int) -> tuple[int, int]:
    """block0: (width, width, k_in, k_in); block1: (width, width, k_in, k_out)."""
    sd = strip_compile_prefix(ck.get("model_state_dict", {}))
    b0 = sd.get("block0.weight")
    b1 = sd.get("block1.weight")
    if b0 is not None and b1 is not None:
        return int(b0.shape[2]), int(b1.shape[3])
    if b0 is not None:
        return int(b0.shape[2]), int(b0.shape[3])
    if ck.get("k_in") is not None and ck.get("k_out") is not None:
        return int(ck["k_in"]), int(ck["k_out"])
    return default_in, default_out
