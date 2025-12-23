"""
Fourier Neural Operator (FNO) for 2D problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer that keeps only low-frequency modes.
    """
    def __init__(self, in_channels, out_channels, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_x      = modes_x
        self.modes_y      = modes_y

        # Complex weights: (C_in, C_out, modes_x, modes_y)
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            (torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)) * scale
        )

    def forward(self, x):
        """
        x: (B, C_in, N, M) real
        returns: (B, C_out, N, M) real
        """
        B, C, N, M = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C, N, M//2+1) complex

        out_ft = torch.zeros(B, self.out_channels, N, M // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        mx = min(self.modes_x, N)
        my = min(self.modes_y, M // 2 + 1)

        # Low-frequency block multiply-accumulate
        out_ft[:, :, :mx, :my] = torch.einsum(
            "bcxy,coxy->boxy",
            x_ft[:, :, :mx, :my],
            self.weight[:, :, :mx, :my],
        )

        x_out = torch.fft.irfft2(out_ft, s=(N, M), norm="ortho").real
        return x_out


class FNOBlock2d(nn.Module):
    def __init__(self, width, modes_x, modes_y, norm="group", dropout_p=0.0, residual=True):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()
        self.residual = residual

        if norm == "group":
            # Use adaptive number of groups: at most 8, but must divide width
            num_groups = min(8, width)
            # Find largest divisor of width that is <= 8
            for g in range(min(8, width), 0, -1):
                if width % g == 0:
                    num_groups = g
                    break
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=width)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(width, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(width)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        y = self.spectral(x) + self.w(x)
        y = self.norm(y)
        y = self.act(y)              
        y = self.dropout(y)
        return x + y if self.residual else y


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D problems.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Hidden dimension width
        modes: Tuple of (modes_x, modes_y) for Fourier modes to keep
        n_layers: Number of FNO blocks
        add_coords: Whether to add coordinate channels
        norm: Normalization type ("group", "instance", "batch", or None)
        dropout_p: Dropout probability
        residual: Whether to use residual connections
    """
    def __init__(self, in_channels=3, out_channels=1, width=32, modes=(16,16), n_layers=3,
                 add_coords=False, norm="group", dropout_p=0.0, residual=True):
        super().__init__()
        self.width = width
        self.add_coords = add_coords

        lift_in = in_channels + (2 if add_coords else 0)
        self.fc0 = nn.Conv2d(lift_in, width, kernel_size=1)   # lifting
        self.blocks = nn.ModuleList([
            FNOBlock2d(width, modes[0], modes[1], norm=norm, dropout_p=dropout_p, residual=residual)
            for _ in range(n_layers)
        ])
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def _concat_coords(self, x):
        # x: (B,C,H,W)
        B, _, H, W = x.shape
        device = x.device
        ys = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H, W)
        """
        if self.add_coords:
            x = self._concat_coords(x)
        x = self.fc0(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

