"""
U-Net architecture for 2D problems
Standard encoder-decoder architecture with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concatenation, we have in_channels + out_channels channels total
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After concatenation, we have in_channels channels total
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_features: Number of features in the first layer (doubled at each level)
        depth: Number of down/up sampling levels
        bilinear: Use bilinear upsampling instead of transposed conv
    """
    
    def __init__(self, in_channels=2, out_channels=1, base_features=64, depth=4, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.depth = depth

        # Initial convolution
        self.inc = DoubleConv(in_channels, base_features)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        in_feat = base_features
        for i in range(depth):
            out_feat = in_feat * 2
            self.downs.append(Down(in_feat, out_feat))
            in_feat = out_feat
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(depth):
            out_feat = in_feat // 2
            self.ups.append(Up(in_feat, out_feat, bilinear))
            in_feat = out_feat
        
        # Output convolution
        self.outc = OutConv(base_features, out_channels)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H, W)
        """
        # Store encoder features for skip connections
        encoder_features = []
        
        # Initial conv
        x = self.inc(x)
        encoder_features.append(x)
        
        # Downsampling
        for down in self.downs:
            x = down(x)
            encoder_features.append(x)
        
        # Remove the last feature (bottom of U)
        encoder_features = encoder_features[:-1]
        
        # Upsampling with skip connections
        for i, up in enumerate(self.ups):
            skip_feat = encoder_features[-(i+1)]
            x = up(x, skip_feat)
        
        # Output
        x = self.outc(x)
        return x

