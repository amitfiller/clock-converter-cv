"""Smoke test for predict_digital_time on local sample images."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.predict_digital_time import predict_digital_time


def _smoke_test_predict_digital_time() -> None:
    """בדיקה ידנית מהירה על כמה תמונות דיגיטליות לפני push ל-Git."""
    sample_dir = Path("samples/digital_clocks")
    paths = sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.jpg"))
    if not paths:
        print("[WARN] No sample images found in samples/digital_clocks")
        return
    print("[INFO] Running smoke test for predict_digital_time ...")
    for img_path in paths:
        try:
            result = predict_digital_time(str(img_path))
            print(f"{img_path.name} -> {result}")
        except Exception as e:
            print(f"[ERROR] Failed on {img_path.name}: {e}")


if __name__ == "__main__":
    _smoke_test_predict_digital_time()
