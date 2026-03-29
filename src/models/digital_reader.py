"""Module A: CNN reader for digital clock time."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DigitalReader(nn.Module):
    """Read digital clock image; predict 24h hour, minute, second logits."""

    def __init__(self) -> None:
        """ResNet18 backbone, dropout, three heads (24 / 60 / 60 classes)."""
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        backbone_model = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        self.dropout = nn.Dropout(p=0.3)
        self.head_h = nn.Linear(512, 24)
        self.head_m = nn.Linear(512, 60)
        self.head_s = nn.Linear(512, 60)

    def forward(self, x: torch.Tensor):
        """Return logits (out_h, out_m, out_s) for hour 0–23, min, sec."""
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        feats = self.dropout(feats)
        return self.head_h(feats), self.head_m(feats), self.head_s(feats)

    def predict(self, x: torch.Tensor):
        """Return integer predictions by argmax (hour 0–23, min, sec)."""
        self.eval()
        with torch.no_grad():
            hour_logits, minute_logits, second_logits = self.forward(x)
            hours = hour_logits.argmax(dim=1)
            minutes = minute_logits.argmax(dim=1)
            seconds = second_logits.argmax(dim=1)
        if hours.numel() == 1:
            return int(hours.item()), int(minutes.item()), int(seconds.item())
        return hours, minutes, seconds
