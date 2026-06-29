"""Image restoration model definitions."""

from __future__ import annotations

import torch
from torch import nn

from models.blocks import ConvBlock, DownBlock, UpBlock


class UNetRestorationModel(nn.Module):
    """U-Net for paired damaged-to-pristine image restoration.

    The network preserves spatial resolution: an input tensor shaped
    ``(N, 3, H, W)`` returns ``(N, 3, H, W)``.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        self.encoder1 = ConvBlock(in_channels, c)
        self.encoder2 = DownBlock(c, c * 2)
        self.encoder3 = DownBlock(c * 2, c * 4)
        self.encoder4 = DownBlock(c * 4, c * 8)

        self.bottleneck = DownBlock(c * 8, c * 16)

        self.decoder4 = UpBlock(c * 16, c * 8, c * 8)
        self.decoder3 = UpBlock(c * 8, c * 4, c * 4)
        self.decoder2 = UpBlock(c * 4, c * 2, c * 2)
        self.decoder1 = UpBlock(c * 2, c, c)

        self.output = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return self.output(d1)


def build_model(base_channels: int = 64) -> UNetRestorationModel:
    return UNetRestorationModel(base_channels=base_channels)
