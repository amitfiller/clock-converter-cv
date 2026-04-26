"""U-Net model for clock hand segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Apply two Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create two-layer convolution block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoded feature maps."""
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, concatenate skip, and refine."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build one decoder step."""
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Return decoded features after skip fusion."""
        x = self.up(x)
        # Align spatial size to skip connection — handles non-32-aligned inputs (e.g. 400px)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HandSegmenter(nn.Module):
    """U-Net that predicts clock hand mask logits."""

    def __init__(self) -> None:
        """Create full encoder-decoder network."""
        super().__init__()
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(128, 256)
        self.dec4 = UpBlock(256, 128)
        self.dec3 = UpBlock(128, 64)
        self.dec2 = UpBlock(64, 32)
        self.dec1 = UpBlock(32, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw mask logits with input resolution."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary masks after sigmoid threshold."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs >= 0.5).to(dtype=torch.float32)


class HandSegmenter256(nn.Module):
    """5-level U-Net for 256x256 clock hand segmentation (Task 5d).
    Adds one extra encoder/decoder level over HandSegmenter to handle higher resolution.
    """

    def __init__(self) -> None:
        """Create full 5-level encoder-decoder network."""
        super().__init__()
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.enc4 = DoubleConv(64, 128)
        self.enc5 = DoubleConv(128, 256)  # extra level for 256x256
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(256, 512)
        self.dec5 = UpBlock(512, 256)
        self.dec4 = UpBlock(256, 128)
        self.dec3 = UpBlock(128, 64)
        self.dec2 = UpBlock(64, 32)
        self.dec1 = UpBlock(32, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw mask logits with input resolution."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))
        d5 = self.dec5(b, e5)
        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary masks after sigmoid threshold."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs >= 0.5).to(dtype=torch.float32)


if __name__ == "__main__":
    model = HandSegmenter()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    print(f"Output shape: {out.shape}")
    pred = model.predict(x)
    print(f"Predict shape: {pred.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
