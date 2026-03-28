"""Module A: CNN reader for digital clock time."""

import torch
import torch.nn as nn


class DigitalReader(nn.Module):
    """Read digital clock image and predict h/m/s logits."""

    def __init__(self) -> None:
        """Build conv encoder, shared neck, and 3 heads."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.hour_head = nn.Linear(256, 24)
        self.minute_head = nn.Linear(256, 60)
        self.second_head = nn.Linear(256, 60)

    def forward(self, x: torch.Tensor):
        """Return raw logits for hour, minute, and second."""
        embedding = self.neck(self.features(x))
        hour_logits = self.hour_head(embedding)
        minute_logits = self.minute_head(embedding)
        second_logits = self.second_head(embedding)
        return hour_logits, minute_logits, second_logits

    def predict(self, x: torch.Tensor):
        """Return integer predictions by argmax from logits."""
        self.eval()
        with torch.no_grad():
            hour_logits, minute_logits, second_logits = self.forward(x)
            hours = hour_logits.argmax(dim=1)
            minutes = minute_logits.argmax(dim=1)
            seconds = second_logits.argmax(dim=1)
        if hours.numel() == 1:
            return int(hours.item()), int(minutes.item()), int(seconds.item())
        return hours, minutes, seconds
