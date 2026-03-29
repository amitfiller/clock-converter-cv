"""API: digital clock image path -> HH:MM:SS string."""

from __future__ import annotations

from src.models.digital_reader import DigitalReader
from src.predict_digital_time import load_model, predict_time_from_image

_cached_model: DigitalReader | None = None


def predict_digital_time(image_path: str) -> str:
    """Predict 24h time as HH:MM:SS from a digital clock image path."""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model()
    h, m, s = predict_time_from_image(image_path, model=_cached_model)
    return f"{h:02d}:{m:02d}:{s:02d}"
