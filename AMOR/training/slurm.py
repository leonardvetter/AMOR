"""Wall-clock deadline for leaving time before Slurm kills the job (test + IO)."""

from __future__ import annotations

import os


def parse_slurm_timelimit_seconds(s: str) -> float | None:
    """Parse SLURM_TIMELIMIT: 'D-HH:MM:SS' or 'HH:MM:SS'."""
    s = (s or "").strip()
    if not s:
        return None
    if "-" in s:
        d_s, rest = s.split("-", 1)
        try:
            days = int(d_s)
        except ValueError:
            return None
    else:
        days = 0
        rest = s
    parts = rest.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = nums[0], nums[1], 0
    else:
        return None
    return float(days * 86400 + h * 3600 + m * 60 + sec)


def eval_deadline_walltime(*, buffer_sec: float) -> float | None:
    """
    Return unix wall time deadline for training; leaves buffer for test eval and checkpoint I/O.

    Uses SLURM_JOB_START_TIME + SLURM_TIMELIMIT when both set (typical under sbatch).
    """
    lim = os.environ.get("SLURM_TIMELIMIT")
    st = os.environ.get("SLURM_JOB_START_TIME")
    if not lim or not st:
        return None
    try:
        start = int(st)
    except ValueError:
        return None
    total = parse_slurm_timelimit_seconds(lim)
    if total is None:
        return None
    buf = float(os.environ.get("AMOR_SLURM_BUFFER_SEC", str(buffer_sec)))
    return float(start) + total - buf
