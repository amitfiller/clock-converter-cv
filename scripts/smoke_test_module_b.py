"""Smoke test for Module B geometry and inpainting."""

from __future__ import annotations

import random
import re
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.hand_geometry.draw_hands import draw_hand_mask
from src.inpainting.remove_hands import composite_hands, remove_hands


def _parse_hms(filename: str) -> tuple[int, int, int]:
    """Extract hour minute second from dataset filename."""
    match = re.match(r"^analog_(\d{2})_(\d{2})_(\d{2})_.+\.png$", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    """Save an RGB image using OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _load_inputs(mask_path: Path, analog_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load analog image and mask for one sample."""
    analog_path = analog_dir / mask_path.name
    analog_bgr = cv2.imread(str(analog_path), cv2.IMREAD_COLOR)
    if analog_bgr is None:
        raise FileNotFoundError(f"Missing analog image: {analog_path}")
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Missing mask image: {mask_path}")
    analog_img = cv2.cvtColor(analog_bgr, cv2.COLOR_BGR2RGB)
    mask_float = mask_gray.astype(np.float32) / 255.0
    return analog_img, mask_float


def _process_sample(mask_path: Path, analog_dir: Path, out_dir: Path) -> None:
    """Run all Module B steps for one file."""
    h, m, s = _parse_hms(mask_path.name)
    analog_img, mask_float = _load_inputs(mask_path, analog_dir)
    img_size = analog_img.shape[0]

    drawn = draw_hand_mask(h, m, s, img_size=img_size)
    draw_vis = np.repeat((drawn * 255).astype(np.uint8), 3, axis=2)
    _save_rgb(out_dir / f"draw_{mask_path.stem}.png", draw_vis)

    inpainted = remove_hands(analog_img, mask_float)
    _save_rgb(out_dir / f"inpaint_{mask_path.stem}.png", inpainted)

    composite = composite_hands(inpainted, draw_hand_mask(h, m, s, img_size=img_size))
    _save_rgb(out_dir / f"composite_{mask_path.stem}.png", composite)


def main() -> None:
    """Run a 5-sample smoke test for Module B."""
    masks_dir = Path("data/masks")
    analog_dir = Path("data/raw/analog")
    out_dir = Path("demo_outputs/smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(masks_dir.glob("*.png"))
    if len(mask_paths) < 5:
        raise RuntimeError(f"Need at least 5 masks in {masks_dir}, found {len(mask_paths)}")
    sample_paths = random.Random(42).sample(mask_paths, 5)

    passed = 0
    for mask_path in sample_paths:
        try:
            _process_sample(mask_path, analog_dir, out_dir)
            passed += 1
            print(f"✅ {mask_path.name} OK")
        except Exception as exc:
            print(f"❌ {mask_path.name} FAILED: {exc}")
            print(traceback.format_exc())
    print(f"Smoke test: {passed}/5 passed")


if __name__ == "__main__":
    main()
