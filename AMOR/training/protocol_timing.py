"""Wall-clock protocol for comparable train and inference times.

Training time is the mean of epochs 2 through 6 on the training loader only.
Inference time is the median batch time over 100 batches after 20 warm-up
batches, divided by the batch size. Data for inference already sits on the GPU.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch


def protocol_requested() -> bool:
    return bool(os.environ.get("AMOR_PROTOCOL_OUT"))


def maybe_run(model, loader, device, loss_fn, meta) -> bool:
    """Run the timing protocol and return True when this process should stop."""
    if not protocol_requested():
        return False
    entry = meta["entry"]
    out = Path(os.environ["AMOR_PROTOCOL_OUT"]) / f"{entry}.json"
    run_protocol(model, loader, device, loss_fn, out, meta)
    return True


def run_protocol(model, loader, device, loss_fn, out_path, meta, lr=2e-3, weight_decay=1e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_s = []
    for ep in range(1, 7):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            n_batches += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"protocol epoch {ep}/6 {dt:.3f}s batches={n_batches}", flush=True)
        if ep >= 2:
            epoch_s.append(dt)

    model.eval()
    resident = []
    for i, (xb, yb) in enumerate(loader):
        if i >= 120:
            break
        resident.append((xb.to(device), yb.to(device)))
    if not resident:
        raise RuntimeError("training loader was empty")
    with torch.no_grad():
        for i in range(20):
            xb, yb = resident[i % len(resident)]
            loss_fn(model, xb, yb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for i in range(100):
            xb, yb = resident[(20 + i) % len(resident)]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss_fn(model, xb, yb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
    med = float(np.median(samples))
    bs = int(resident[0][0].shape[0])
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    record = {
        **meta,
        "train_s_per_epoch": float(np.mean(epoch_s)),
        "train_epoch_s": [float(x) for x in epoch_s],
        "infer_ms_per_sample": med / bs * 1000.0,
        "infer_batch_s_median": med,
        "batch_size": bs,
        "resident_batches": len(resident),
        "warmup_batches": 20,
        "timed_batches": 100,
        "epochs_timed": "2-6",
        "validation_included": False,
        "gpu": gpu,
        "node": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK") or str(os.cpu_count()),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(
        f"wrote {path} train_s={record['train_s_per_epoch']:.3f} "
        f"infer_ms={record['infer_ms_per_sample']:.4f} gpu={gpu}",
        flush=True,
    )
    return record
