"""Checkpoint I/O and relative-error helpers shared by the 2D and 3D trainers."""

from __future__ import annotations

import torch


def unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module if wrapped by torch.compile (OptimizedModule)."""
    return getattr(model, "_orig_mod", model)


def clone_state_dict(state_dict: dict) -> dict:
    """CPU-clone a state_dict; recurse through nested dicts (neuralop / compile)."""
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, dict):
            out[k] = clone_state_dict(v)
        else:
            out[k] = v
    return out


def get_model_state_dict(model: torch.nn.Module) -> dict:
    return unwrap_compiled(model).state_dict()


def strip_compile_prefix(state: dict) -> dict:
    """torch.compile checkpoints use _orig_mod.* keys; strip for eager load."""
    if not any(k.startswith("_orig_mod.") for k in state):
        return state
    return {k.removeprefix("_orig_mod."): v for k, v in state.items()}


def relative_field_errors(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Per-sample ||diff||/||target|| (Kolmogorov-style), mean over batch. Returns fractions."""
    b = pred.size(0)
    pred_flat = pred.reshape(b, -1)
    true_flat = target.reshape(b, -1)
    diff_l1 = torch.abs(pred_flat - true_flat).sum(dim=1)
    true_l1 = torch.abs(true_flat).sum(dim=1).clamp(min=1e-8)
    diff_l2 = torch.norm(pred_flat - true_flat, p=2, dim=1)
    true_l2 = torch.norm(true_flat, p=2, dim=1).clamp(min=1e-8)
    return float((diff_l1 / true_l1).mean().item()), float((diff_l2 / true_l2).mean().item())
