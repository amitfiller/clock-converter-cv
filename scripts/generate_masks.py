"""Generate binary hand masks for analog images from filename time (geometric)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALOG_DIR = PROJECT_ROOT / "data" / "raw" / "analog"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"
IMAGE_SUFFIX = ".png"


def parse_args() -> argparse.Namespace:
    """CLI for batch mask generation."""
    parser = argparse.ArgumentParser(
        description="Build binary hand masks from analog clock PNGs (geometric).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Process only N random images (default: all).",
    )
    return parser.parse_args()


def parse_time_from_analog_stem(stem: str) -> tuple[int, int, int]:
    """Parse (H, M, S) from analog_HH_MM_SS_<style> stem."""
    parts = stem.split("_")
    if len(parts) < 5 or parts[0].lower() != "analog":
        raise ValueError(f"Bad analog stem: {stem}")
    h, m, s = int(parts[1]), int(parts[2]), int(parts[3])
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        raise ValueError(f"Time out of range: {stem}")
    return h, m, s


def generate_geometric_mask(
    image_shape: tuple[int, ...], hh: int, mm: int, ss: int
) -> np.ndarray:
    """Build single-channel hand mask from time; blur + re-binarize for U-Net."""
    h, w = image_shape[:2]
    cx, cy = w // 2, h // 2
    r = min(w, h) * 0.42
    mask = np.zeros((h, w), dtype=np.uint8)

    hour_angle = (hh % 12) * 30 + mm * 0.5 - 90
    minute_angle = mm * 6 - 90
    second_angle = ss * 6 - 90

    hands = [
        (hour_angle, 0.55, 6),
        (minute_angle, 0.80, 4),
        (second_angle, 0.90, 2),
    ]

    for angle_deg, length_ratio, thickness in hands:
        tip_x = int(cx + r * length_ratio * np.cos(np.radians(angle_deg)))
        tip_y = int(cy + r * length_ratio * np.sin(np.radians(angle_deg)))
        tip_x = max(0, min(w - 1, tip_x))
        tip_y = max(0, min(h - 1, tip_y))
        cv2.line(mask, (cx, cy), (tip_x, tip_y), 255, thickness)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask


def process_image(path: Path) -> tuple[bool, str]:
    """Build geometric mask from filename time; write to MASKS_DIR."""
    try:
        hh, mm, ss = parse_time_from_analog_stem(path.stem)
    except ValueError as exc:
        return False, f"{path.name}: {exc}"

    image = cv2.imread(str(path))
    if image is None:
        return False, f"{path.name}: could not read image"

    mask = generate_geometric_mask(image.shape, hh, mm, ss)

    out_path = MASKS_DIR / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), mask):
        return False, f"{path.name}: imwrite failed"

    return True, ""


def list_analog_images() -> list[Path]:
    """Sorted list of analog PNG paths under ANALOG_DIR."""
    paths = sorted(ANALOG_DIR.glob(f"analog_*{IMAGE_SUFFIX}"))
    return [p for p in paths if p.is_file()]


def run_generation(paths: list[Path]) -> tuple[int, list[str]]:
    """Process paths; return success count and error lines for logging."""
    ok = 0
    failed: list[str] = []
    for i, path in enumerate(paths):
        success, err = process_image(path)
        if success:
            ok += 1
        else:
            failed.append(err or path.name)
        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"Progress: {i + 1}/{len(paths)}", flush=True)
    return ok, failed


def main() -> None:
    """Generate masks for all or a random sample; print progress and failures."""
    args = parse_args()
    all_paths = list_analog_images()
    if not all_paths:
        print(f"[ERROR] No analog images under {ANALOG_DIR}")
        return

    if args.sample is not None:
        n = min(args.sample, len(all_paths))
        paths = random.sample(all_paths, n)
    else:
        paths = all_paths

    ok, failed = run_generation(paths)
    print(f"\n[OK] Generated {ok} masks | [WARN] Failed {len(failed)} images")
    if failed:
        print("Failed:")
        for line in failed:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
