"""Training loop shared by the 3D trainers (FlowBench LDC, elastostatics).

AdamW + cosine annealing, relative-L1 objective, gradient clipping, early
stopping on validation relative L1. Checkpoints drop fitted PCA buffers, which
are refit from the stored pca.npz on reload.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import time

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from AMOR.training.torch_utils import (
    clone_state_dict,
    get_model_state_dict,
    relative_field_errors,
    unwrap_compiled,
)


def log_event(log_path: str | None, event: str, **fields) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    payload = {"event": event, "time": time.time(), **fields}
    with open(log_path, "a") as f:
        f.write(json.dumps(payload) + "\n")
        f.flush()
    print(f"[{event}] " + " ".join(f"{k}={v}" for k, v in fields.items()))



_PCA_BUFFER_KEYS = ("enc_components", "enc_mean", "dec_components", "dec_mean", "shared_fixed_basis")


def filter_checkpoint_state(state_dict: dict) -> dict:
    """Drop fitted PCA bases from checkpoints (reload from pca.npz; saves GB)."""
    return {k: v for k, v in state_dict.items() if not any(s in k for s in _PCA_BUFFER_KEYS)}


def atomic_torch_save(obj, path: str) -> None:
    tmp = f"{path}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def relative_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-sample relative L1 — matches training target for rel < 1."""
    b = pred.size(0)
    pred_flat = pred.reshape(b, -1)
    true_flat = target.reshape(b, -1)
    diff_l1 = torch.abs(pred_flat - true_flat).sum(dim=1)
    true_l1 = torch.abs(true_flat).sum(dim=1).clamp(min=1e-8)
    return torch.mean(diff_l1 / true_l1)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    hyperparams,
    epoch_log_path: str | None = None,
    checkpoint_path: str | None = None,
    last_checkpoint_path: str | None = None,
    checkpoint_trainable_only: bool = False,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=hyperparams["epochs"], eta_min=0.0)

    best_val_rel_l1 = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history: list[dict] = []
    train_start = time.time()
    use_bf16 = hyperparams.get("amp") == "bf16"
    if use_bf16:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 autocast requested but this GPU does not support bfloat16")
        print("AMP: bf16 autocast on the training forward (eval stays fp32)", flush=True)

    if epoch_log_path:
        os.makedirs(os.path.dirname(epoch_log_path), exist_ok=True)
        with open(epoch_log_path, "w") as f:
            f.write(json.dumps({"event": "training_start", "time": time.time()}) + "\n")

    for epoch in range(hyperparams["epochs"]):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0
        probe_batches = int(hyperparams.get("probe_batches") or 0)
        n_probe = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else contextlib.nullcontext()
            )
            with amp_ctx:
                pred = model(xb)
                loss = relative_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams["max_grad_norm"])
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)
            n_probe += 1
            if probe_batches and n_probe >= probe_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(f"PROBE batches={n_probe} peak_alloc_gib={peak:.2f}", flush=True)
                return 0.0, 0, history, time.time() - train_start
        scheduler.step()
        train_loss /= max(n_train, 1)

        train_rel_l1_acc, n_train_rel = 0.0, 0
        model.eval()
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                r1, _ = relative_field_errors(pred, yb)
                train_rel_l1_acc += r1 * xb.size(0)
                n_train_rel += xb.size(0)
        train_rel_l1 = train_rel_l1_acc / max(n_train_rel, 1)

        val_loss = 0.0
        n_val = 0
        val_rel_l1_acc, val_rel_l2_acc, n_val_rel = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += l1_loss(pred, yb).item() * xb.size(0)
                r1, r2 = relative_field_errors(pred, yb)
                val_rel_l1_acc += r1 * xb.size(0)
                val_rel_l2_acc += r2 * xb.size(0)
                n_val += xb.size(0)
                n_val_rel += xb.size(0)
        val_loss /= max(n_val, 1)
        val_rel_l1 = val_rel_l1_acc / max(n_val_rel, 1)
        val_rel_l2 = val_rel_l2_acc / max(n_val_rel, 1)

        if val_rel_l1 < best_val_rel_l1 - hyperparams["early_stopping_min_delta"]:
            best_val_rel_l1 = val_rel_l1
            best_epoch = epoch
            state = get_model_state_dict(model)
            best_state = (
                filter_checkpoint_state(state)
                if checkpoint_trainable_only
                else clone_state_dict(state)
            )
            patience = 0
            if checkpoint_path:
                atomic_torch_save(best_state, checkpoint_path)
        else:
            patience += 1

        # Rolling checkpoint: overwrite every epoch so a killed job can resume from latest.
        # Best-effort only — a failed last.pt must not kill a multi-day run (quota / NFS).
        if last_checkpoint_path:
            try:
                latest = get_model_state_dict(model)
                latest_state = (
                    filter_checkpoint_state(latest)
                    if checkpoint_trainable_only
                    else clone_state_dict(latest)
                )
                atomic_torch_save(latest_state, last_checkpoint_path)
            except Exception as exc:
                print(f"  WARNING: last checkpoint save failed: {exc}", flush=True)

        lr = scheduler.get_last_lr()[0]
        record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_rel_l1": float(train_rel_l1),
            "val_loss": float(val_loss),
            "val_rel_l1": float(val_rel_l1),
            "val_rel_l2": float(val_rel_l2),
            "lr": float(lr),
            "epoch_time_s": float(time.time() - epoch_start),
            "elapsed_s": float(time.time() - train_start),
            "best_val_rel_l1": float(best_val_rel_l1),
            "is_best": (epoch == best_epoch),
            "patience": int(patience),
        }
        history.append(record)
        print(
            f"  Epoch {epoch+1}/{hyperparams['epochs']}: "
            f"train_rel={train_rel_l1:.4f} val_rel_l1={val_rel_l1:.4f} val_l1={val_loss:.4f} "
            f"lr={lr:.2e} epoch_s={record['epoch_time_s']:.1f} elapsed={record['elapsed_s']:.0f}s"
            + (" *best*" if record["is_best"] else "")
        )
        if epoch_log_path:
            with open(epoch_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()

        if patience >= hyperparams["early_stopping_patience"]:
            print(f"  Early stop epoch {epoch+1} (best {best_epoch+1})")
            break

    total_train_s = time.time() - train_start
    if epoch_log_path:
        with open(epoch_log_path, "a") as f:
            f.write(json.dumps({
                "event": "training_end",
                "best_epoch": best_epoch + 1,
                "best_val_rel_l1": float(best_val_rel_l1),
                "total_train_s": float(total_train_s),
            }) + "\n")

    if best_state is not None:
        unwrap_compiled(model).load_state_dict(best_state, strict=False)
    return best_val_rel_l1, best_epoch, history, total_train_s
