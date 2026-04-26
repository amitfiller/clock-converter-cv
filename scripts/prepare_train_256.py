"""Resize raw 400x400 analog images + masks to 256x256 for higher-res U-Net training.

Saves to data/train_256/{analog,masks}/ — does NOT touch data/raw/.
Run once before train_256.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ANALOG = ROOT / "data" / "raw" / "analog"
SRC_MASKS = ROOT / "data" / "masks"
DST_ANALOG = ROOT / "data" / "train_256" / "analog"
DST_MASKS = ROOT / "data" / "train_256" / "masks"
IMG_SIZE = 256


def resize_and_save(src: Path, dst: Path, interp: int, size: int) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Cannot read {src}")
        return
    resized = cv2.resize(img, (size, size), interpolation=interp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized)


def main() -> None:
    analog_paths = sorted(SRC_ANALOG.glob("*.png"))
    if not analog_paths:
        print(f"No analog images found in {SRC_ANALOG}")
        sys.exit(1)

    matched = 0
    skipped = 0
    for analog_src in analog_paths:
        mask_src = SRC_MASKS / analog_src.name
        if not mask_src.exists():
            skipped += 1
            continue
        resize_and_save(analog_src, DST_ANALOG / analog_src.name, cv2.INTER_AREA, IMG_SIZE)
        resize_and_save(mask_src, DST_MASKS / mask_src.name, cv2.INTER_NEAREST, IMG_SIZE)
        matched += 1

    print(f"Done. Resized {matched} pairs to {IMG_SIZE}x{IMG_SIZE} → data/train_256/")
    if skipped:
        print(f"Skipped {skipped} analogs with no matching mask.")


if __name__ == "__main__":
    main()
