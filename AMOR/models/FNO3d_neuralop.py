"""Official 3D FNO via neuraloperator (`neuralop.models.FNO`).

Drop-in wrapper matching the old homemade ``FNO3d`` constructor so training
scripts can swap backends without API changes. Uses library defaults for
positional embedding, skips, and channel MLP (see neuraloperator docs).
"""

from __future__ import annotations

import torch.nn as nn
from neuralop.models import FNO


class FNO3d(nn.Module):
    """3D Fourier Neural Operator from the official neuraloperator package."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 24,
        modes=(12, 12, 12),
        n_layers: int = 4,
        **_ignored,
    ):
        super().__init__()
        modes = tuple(int(m) for m in modes)
        if len(modes) != 3:
            raise ValueError(f"FNO3d expects 3 modes, got {modes}")
        self.model = FNO(
            n_modes=modes,
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            hidden_channels=int(width),
            n_layers=int(n_layers),
        )

    def forward(self, x):
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
