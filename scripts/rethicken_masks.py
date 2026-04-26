"""Thicken mask files and save a before/after preview."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def resolve_mask_dir() -> Path:
    """Pick requested mask dir, fallback to existing one."""
    requested = Path("data/raw/masks")
    fallback = Path("data/masks")
    if requested.exists():
        return requested
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No masks folder found at data/raw/masks or data/masks")


def pick_preview_files(mask_files: list[Path], k: int = 3) -> list[Path]:
    """Select k files for before/after preview."""
    if not mask_files:
        return []
    random.seed(42)
    if len(mask_files) <= k:
        return mask_files
    return random.sample(mask_files, k)


def thicken_mask(mask: np.ndarray) -> np.ndarray:
    """Dilate a binary/grayscale mask."""
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask, kernel, iterations=3)


def save_comparison(previews: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path) -> None:
    """Save before/after grid for sampled masks."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(previews)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 3 * rows))
    if rows == 1:
        axes = np.array([axes])
    for i, (name, before, after) in enumerate(previews):
        axes[i, 0].imshow(before, cmap="gray")
        axes[i, 0].set_title(f"Before: {name}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(after, cmap="gray")
        axes[i, 1].set_title(f"After: {name}")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Thicken all masks and print progress."""
    mask_dir = resolve_mask_dir()
    mask_files = sorted(mask_dir.glob("*"))
    mask_files = [p for p in mask_files if p.is_file()]
    if not mask_files:
        print("Done. Total masks thickened: 0")
        return

    preview_targets = {p.name for p in pick_preview_files(mask_files, k=3)}
    previews: list[tuple[str, np.ndarray, np.ndarray]] = []
    total = len(mask_files)
    processed = 0

    for idx, path in enumerate(mask_files, start=1):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        thick = thicken_mask(mask)
        cv2.imwrite(str(path), thick)
        processed += 1
        print(f"Processed {idx}/{total}: {path.name}")
        if path.name in preview_targets:
            previews.append((path.name, mask, thick))

    if previews:
        save_comparison(previews, Path("demo_outputs/masks_before_after.png"))
    print(f"Done. Total masks thickened: {processed}")


if __name__ == "__main__":
    main()
