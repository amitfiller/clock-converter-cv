"""Visual QA: overlay binary masks on analog images."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALOG_DIR = PROJECT_ROOT / "data" / "raw" / "analog"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"
VERIFY_BASE = PROJECT_ROOT / "demo_outputs" / "mask_verify"
LEGACY_DIR_NAME = "_legacy_unsorted"


def parse_args() -> argparse.Namespace:
    """CLI for mask verification."""
    parser = argparse.ArgumentParser(
        description="Overlay hand masks on analog images for QA.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        metavar="N",
        help="Number of random masks to verify (default: 20).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show cv2 windows (key to advance).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write overlays (default if --show omitted).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save PNGs here; overrides run_* timestamp folder.",
    )
    parsed = parser.parse_args()
    if parsed.show and parsed.save:
        parser.error("Use either --show or --save, not both.")
    return parsed


def mask_to_2d(mask: np.ndarray) -> np.ndarray:
    """Return single-channel uint8 mask (H,W)."""
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    raise ValueError("Unsupported mask shape")


def red_overlay(bgr: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    """Blend red at half opacity where mask > 128."""
    m = mask_to_2d(mask_2d) > 128
    base = bgr.astype(np.float32)
    red = np.zeros_like(base)
    red[:, :, 2] = 255.0
    out = base.copy()
    out[m] = 0.5 * base[m] + 0.5 * red[m]
    return out.astype(np.uint8)


def side_by_side(left_bgr: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    """Concatenate overlay (left) and raw mask as BGR (right)."""
    m = mask_to_2d(mask_2d)
    right = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    h = min(left_bgr.shape[0], right.shape[0])
    l = cv2.resize(left_bgr, (int(left_bgr.shape[1] * h / left_bgr.shape[0]), h))
    r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    w_l, w_r = l.shape[1], r.shape[1]
    if w_l != w_r:
        target_w = max(w_l, w_r)
        if w_l < target_w:
            pad = target_w - w_l
            l = cv2.copyMakeBorder(l, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if w_r < target_w:
            pad = target_w - w_r
            r = cv2.copyMakeBorder(r, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.hstack([l, r])


def list_mask_files() -> list[Path]:
    """All PNG masks under MASKS_DIR."""
    return sorted(p for p in MASKS_DIR.glob("*.png") if p.is_file())


def archive_loose_pngs_in_verify_root() -> None:
    """Move stray *.png from mask_verify/ root into _legacy_unsorted/."""
    if not VERIFY_BASE.is_dir():
        VERIFY_BASE.mkdir(parents=True, exist_ok=True)
        return
    loose = [
        p for p in VERIFY_BASE.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ]
    if not loose:
        return
    dest = VERIFY_BASE / LEGACY_DIR_NAME
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Archiving {len(loose)} loose PNG(s) -> {LEGACY_DIR_NAME}/")
    for src in loose:
        target = dest / src.name
        n = 1
        while target.exists():
            target = dest / f"{src.stem}_{n}{src.suffix}"
            n += 1
        src.rename(target)


def resolve_save_dir(output_dir: Path | None) -> Path:
    """Pick save directory: explicit path or run_<timestamp> under VERIFY_BASE."""
    if output_dir is not None:
        return output_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return VERIFY_BASE / f"run_{stamp}"


def verify_one(mask_path: Path, show: bool, save_dir: Path) -> str | None:
    """Load pair, build composite; show or save. Return error message or None."""
    analog_path = ANALOG_DIR / mask_path.name
    if not analog_path.is_file():
        return f"missing analog: {analog_path.name}"

    bgr = cv2.imread(str(analog_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        return f"cannot read analog: {analog_path.name}"
    if mask is None:
        return f"cannot read mask: {mask_path.name}"

    mask2 = mask_to_2d(mask)
    overlay = red_overlay(bgr, mask2)
    combined = side_by_side(overlay, mask2)

    stem = mask_path.stem
    if show:
        cv2.imshow("overlay | raw mask", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{stem}_verify.png"
        if not cv2.imwrite(str(out_path), combined):
            return f"imwrite failed: {out_path.name}"
    return None


def main() -> None:
    """Verify random or all masks; print errors."""
    args = parse_args()
    use_show = args.show
    save_dir: Path | None = None
    if not use_show:
        archive_loose_pngs_in_verify_root()
        save_dir = resolve_save_dir(args.output_dir)

    masks = list_mask_files()
    if not masks:
        print(f"[ERROR] No masks under {MASKS_DIR}")
        return

    n = min(args.sample, len(masks))
    chosen = random.sample(masks, n)
    errors: list[str] = []

    for mp in chosen:
        err = verify_one(mp, use_show, save_dir or VERIFY_BASE)
        if err:
            errors.append(err)

    if use_show:
        mode = "show (cv2 windows)"
    else:
        mode = f"saved under {save_dir}"
    print(f"Verified {len(chosen)} image(s) ({mode}).")
    if errors:
        print("Issues:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
