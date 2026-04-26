"""Generate binary hand masks for analog images from filename time (geometric).

Regeneration (full pseudo-label refresh):
  1) Backup or delete: data/masks/*.png (PowerShell: Remove-Item data/masks/*.png)
  2) Run: python scripts/generate_masks.py
     (omit --sample so every analog_* file gets a mask from the current code)

GO / NO-GO before U-Net (heuristic):
  - Regenerate all masks, then open data/mask_debug/run_*.csv and demo_outputs/mask_debug/run_*/*_debug.png
  - Run: python scripts/verify_masks.py --sample 80 --save --seed 42
  - If ~70-80%% of the verify sample looks aligned AND no single style is mostly broken,
    OK to plan a small U-Net dataset on a clean subset. Otherwise tune geometry first.

Assumptions:
  - Filenames: analog_HH_MM_SS_<style>.png with a single style token (or joined tail if extended).
  - Hand length: default r = 0.42*min(w,h); style "simple" may scale from Hough/Otsu radius (heuristic).
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALOG_DIR = PROJECT_ROOT / "data" / "raw" / "analog"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"
CSV_DEBUG_DIR = PROJECT_ROOT / "data" / "mask_debug"
DEBUG_IMG_DIR = PROJECT_ROOT / "demo_outputs" / "mask_debug"
IMAGE_SUFFIX = ".png"

# orange/vue/atc specific: rendered hands are shorter than default 0.42*min(w,h) reach.
ORANGE_VUE_ATC_HAND_LENGTH_SCALE: dict[str, float] = {
    "orange": 0.62,
    "vue": 0.65,
    "atc": 0.65,
}


@dataclass
class DialEstimate:
    """Dial pivot; hough_radius None when no circle estimate (see dial_source)."""

    cx: int
    cy: int
    used_hough: bool
    hough_radius: float | None
    dial_source: str = "hough"


@dataclass
class DebugConfig:
    """Limit how many geometric overlay PNGs are written per run."""

    out_dir: Path
    max_count: int
    written: int = 0


def parse_args() -> argparse.Namespace:
    """CLI for batch mask generation and optional debug artifacts."""
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save geometric ray overlays (center + 3 colored hands, pre-blur).",
    )
    parser.add_argument(
        "--debug-max",
        type=int,
        default=100,
        metavar="N",
        help="Max debug overlay PNGs when --debug (default: 100).",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        metavar="NAME",
        help="Process only images whose style token matches (e.g. simple).",
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


def parse_style_from_stem(stem: str) -> str:
    """Style token(s) after analog_HH_MM_SS_."""
    parts = stem.split("_")
    if len(parts) < 5:
        return ""
    return "_".join(parts[4:])


def _hough_best_circle(
    gray: np.ndarray,
    *,
    blur_ksize: int,
    dp: float,
    min_dist_frac: float,
    param1: int,
    param2: int,
    min_r_frac: float,
    max_r_frac: float,
) -> tuple[int, int, float] | None:
    """Choose Hough circle closest to image center; None if detector finds none."""
    k = blur_ksize | 1
    h, w = gray.shape
    mnh = min(w, h)
    min_r, max_r = max(int(mnh * min_r_frac), 20), int(mnh * max_r_frac)
    if max_r <= min_r + 4:
        return None
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    md = max(int(mnh * min_dist_frac), 32)
    circ = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp, md, param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circ is None:
        return None
    icx, icy = w // 2, h // 2
    b = min(circ[0], key=lambda c: (c[0] - icx) ** 2 + (c[1] - icy) ** 2)
    return int(round(b[0])), int(round(b[1])), float(b[2])


def _hough_all_circles(
    gray: np.ndarray,
    *,
    blur_ksize: int,
    dp: float,
    min_dist_frac: float,
    param1: int,
    param2: int,
    min_r_frac: float,
    max_r_frac: float,
) -> list[tuple[float, float, float]]:
    """All circles from one Hough run (empty list if none)."""
    k = blur_ksize | 1
    h, w = gray.shape
    mnh = min(w, h)
    min_r, max_r = max(int(mnh * min_r_frac), 20), int(mnh * max_r_frac)
    if max_r <= min_r + 4:
        return []
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    md = max(int(mnh * min_dist_frac), 32)
    circ = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp, md, param1=param1, param2=param2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circ is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in circ[0]]


def _simple_merged_hough_circles(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Deduped union of two Hough parameter sets (simple style)."""
    specs = [
        (7, 1.25, 0.2, 70, 16, 0.22, 0.56),
        (5, 1.4, 0.14, 50, 11, 0.18, 0.58),
    ]
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[float, float, float]] = []
    for bk, dp, mdf, p1, p2, rf1, rf2 in specs:
        for c in _hough_all_circles(
            gray, blur_ksize=bk, dp=dp, min_dist_frac=mdf, param1=p1,
            param2=p2, min_r_frac=rf1, max_r_frac=rf2,
        ):
            key = (int(round(c[0])), int(round(c[1])), int(round(c[2])))
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


def _simple_pratt_algebra(xy: np.ndarray) -> tuple[float, float, float] | None:
    """Pratt circle fit for Nx2 points; None if degenerate."""
    x, y = xy[:, 0], xy[:, 1]
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    suu, svv, suv = (u * u).sum(), (v * v).sum(), (u * v).sum()
    suuu, svvv = (u * u * u).sum(), (v * v * v).sum()
    suvv, svuu = (u * v * v).sum(), (v * u * u).sum()
    a_mat = np.array([[suu, suv], [suv, svv]], dtype=np.float64)
    b_vec = 0.5 * np.array([suuu + suvv, svvv + svuu])
    try:
        uc, vc = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        return None
    cx, cy = xm + float(uc), ym + float(vc)
    r = float(np.sqrt(float(uc) ** 2 + float(vc) ** 2 + (u * u + v * v).mean()))
    return cx, cy, r


def _simple_pratt_pivot(bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Circle fit on subsampled Canny edges when dial edges are weak for Hough."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ed = cv2.Canny(blur, 30, 90)
    ys, xs = np.where(ed > 0)
    if len(xs) < 40:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    m = int(min(w, h) * 0.06)
    sel = (pts[:, 0] >= m) & (pts[:, 0] < w - m) & (pts[:, 1] >= m) & (pts[:, 1] < h - m)
    pts = pts[sel]
    if len(pts) < 40:
        return None
    rng = np.random.default_rng(42)
    if len(pts) > 1200:
        pts = pts[rng.choice(len(pts), 1200, replace=False)]
    fit = _simple_pratt_algebra(pts)
    if fit is None:
        return None
    cx, cy, r = fit
    mnh = min(w, h)
    if r < mnh * 0.15 or r > mnh * 0.55:
        return None
    return cx, cy, r


def _simple_pick_hough_with_pratt_hint(
    bgr: np.ndarray, gray: np.ndarray, h: int, w: int
) -> DialEstimate | None:
    """Use Pratt as pivot hint; prefer Hough circle near hint, else Pratt alone."""
    mnh = min(w, h)
    pr = _simple_pratt_pivot(bgr)
    cands = _simple_merged_hough_circles(gray)
    hx, hy = (pr[0], pr[1]) if pr else (w / 2.0, h / 2.0)
    hr = pr[2] if pr else mnh * 0.37
    if not cands:
        if pr is not None:
            return DialEstimate(
                int(round(pr[0])), int(round(pr[1])), False, pr[2], "simple_pratt"
            )
        return None

    def cost(c: tuple[float, float, float]) -> float:
        return (c[0] - hx) ** 2 + (c[1] - hy) ** 2 + 0.28 * (c[2] - hr) ** 2

    best = min(cands, key=cost)
    if pr is not None:
        dxy = float(np.hypot(best[0] - pr[0], best[1] - pr[1]))
        if dxy > 0.027 * mnh:
            return DialEstimate(
                int(round(pr[0])), int(round(pr[1])), False, pr[2], "simple_pratt"
            )
    src = "simple_loose_hough" if len(cands) >= 4 else "hough"
    return DialEstimate(
        int(round(best[0])), int(round(best[1])), True, best[2], src,
    )


def _estimate_dial_center_default(bgr: np.ndarray) -> DialEstimate:
    """Standard Hough dial; image center if Hough fails (all non-simple styles)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    got = _hough_best_circle(
        gray,
        blur_ksize=9,
        dp=1.2,
        min_dist_frac=0.25,
        param1=80,
        param2=22,
        min_r_frac=0.28,
        max_r_frac=0.52,
    )
    if got:
        return DialEstimate(got[0], got[1], True, got[2], "hough")
    return DialEstimate(w // 2, h // 2, False, None, "image_center")


def _simple_largest_blob_centroid(roi_gray: np.ndarray) -> tuple[int, int, float] | None:
    """Otsu + largest mid-size contour; centroid and disk radius (ROI coordinates)."""
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rh, rw = roi_gray.shape
    lo, hi = rh * rw * 0.07, rh * rw * 0.93
    for inv in (False, True):
        t = cv2.bitwise_not(th) if inv else th
        cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_a, best_c = 0, None
        for c in cnts:
            a = float(cv2.contourArea(c))
            if not (lo <= a <= hi):
                continue
            if a > best_a:
                best_a, best_c = a, c
        if best_c is None:
            continue
        m = cv2.moments(best_c)
        if m["m00"] <= 1e-6:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        r = float(np.sqrt(best_a / np.pi))
        return cx, cy, r
    return None


def _simple_pivot_otsu(bgr: np.ndarray) -> tuple[int, int, float] | None:
    """Map Otsu blob centroid from a margin-cropped ROI to full-image coordinates."""
    h, w = bgr.shape[:2]
    m = int(min(w, h) * 0.08)
    roi = bgr[m : h - m, m : w - m]
    if roi.size < 400:
        return None
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    got = _simple_largest_blob_centroid(gray_roi)
    if got is None:
        return None
    cx_r, cy_r, r = got
    return cx_r + m, cy_r + m, r


def _simple_pivot_edge_weighted(bgr: np.ndarray) -> tuple[int, int] | None:
    """Gaussian-weighted mean of Canny edge pixels in central ROI (simple fallback)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m = int(min(w, h) * 0.1)
    roi = gray[m : h - m, m : w - m]
    rh, rw = roi.shape
    if rh < 8 or rw < 8:
        return None
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    ed = cv2.Canny(blur, 35, 100)
    yy, xx = np.indices(ed.shape)
    cxr, cyr = (rw - 1) / 2.0, (rh - 1) / 2.0
    sig = min(rw, rh) * 0.32
    wg = np.exp(-((xx - cxr) ** 2 + (yy - cyr) ** 2) / (2 * sig * sig))
    wt = ed.astype(np.float64) * wg
    s = float(wt.sum())
    if s < 1e-3:
        return None
    sx = float((wt * xx).sum() / s)
    sy = float((wt * yy).sum() / s)
    return int(round(sx + m)), int(round(sy + m))


def _estimate_dial_center_simple(bgr: np.ndarray) -> DialEstimate:
    """Pratt-guided merged Hough, then Otsu blob, edge-weighted mean, center."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    picked = _simple_pick_hough_with_pratt_hint(bgr, gray, h, w)
    if picked is not None:
        return picked
    ots = _simple_pivot_otsu(bgr)
    if ots:
        cx, cy, r_est = ots
        return DialEstimate(cx, cy, False, r_est, "simple_otsu")
    ew = _simple_pivot_edge_weighted(bgr)
    if ew:
        return DialEstimate(ew[0], ew[1], False, None, "simple_edges")
    return DialEstimate(w // 2, h // 2, False, None, "image_center")


def estimate_dial_center(bgr: np.ndarray, style: str = "") -> DialEstimate:
    """Hand pivot: style 'simple' uses extra Hough tuning and non-Hough fallbacks."""
    if style == "simple":
        return _estimate_dial_center_simple(bgr)
    return _estimate_dial_center_default(bgr)


def simple_fallback_csv(style: str, dial_source: str) -> str:
    """Non-empty only when simple pivot is not Hough circles (heuristic fallbacks)."""
    if style != "simple":
        return ""
    if dial_source in ("hough", "simple_loose_hough"):
        return ""
    if dial_source == "simple_otsu":
        return "otsu"
    if dial_source == "simple_pratt":
        return "pratt"
    if dial_source == "simple_edges":
        return "edges"
    if dial_source == "image_center":
        return "center"
    return ""


def hand_extent_radius(w: int, h: int, style: str, dial: DialEstimate) -> float:
    """Ray length scale; simple may follow Hough/Otsu disk radius (heuristic clamp)."""
    base = min(w, h) * 0.42
    # orange/vue/atc specific
    if style in ORANGE_VUE_ATC_HAND_LENGTH_SCALE:
        base *= ORANGE_VUE_ATC_HAND_LENGTH_SCALE[style]
    if style != "simple":
        return base
    hr = dial.hough_radius
    if hr is None or hr < 8.0:
        return base
    r = hr / 0.88
    lo, hi = min(w, h) * 0.30, min(w, h) * 0.50
    return max(lo, min(hi, r))


def generate_geometric_mask(
    bgr: np.ndarray,
    hh: int,
    mm: int,
    ss: int,
    dial: DialEstimate,
    hand_r: float,
) -> np.ndarray:
    """Build single-channel hand mask; blur + re-binarize for U-Net."""
    h, w = bgr.shape[:2]
    cx, cy = dial.cx, dial.cy
    r = hand_r
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
        layer = np.zeros((h, w), dtype=np.uint8)
        tip_x = int(cx + r * length_ratio * np.cos(np.radians(angle_deg)))
        tip_y = int(cy + r * length_ratio * np.sin(np.radians(angle_deg)))
        tip_x = max(0, min(w - 1, tip_x))
        tip_y = max(0, min(h - 1, tip_y))
        cv2.line(layer, (cx, cy), (tip_x, tip_y), 255, thickness)
        mask = np.maximum(mask, layer)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask


def mask_frac_beyond_dial_ring(
    mask: np.ndarray, cx: int, cy: int, w: int, h: int
) -> float:
    """Share of mask pixels farther than 0.52*min(w,h) from pivot (crude broken-mask flag)."""
    ys, xs = np.where(mask > 128)
    if len(xs) == 0:
        return 1.0
    max_d = min(w, h) * 0.52
    dist = np.hypot(xs.astype(np.float64) - cx, ys.astype(np.float64) - cy)
    return float(np.mean(dist > max_d))


def draw_geometry_debug_overlay(
    bgr: np.ndarray,
    dial: DialEstimate,
    hh: int,
    mm: int,
    ss: int,
    hand_r: float,
) -> np.ndarray:
    """BGR image: cross at pivot + 3 colored rays (pre-blur geometry)."""
    out = bgr.copy()
    hi, w = out.shape[:2]
    cx, cy = dial.cx, dial.cy
    r = hand_r
    cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

    specs = [
        ((hh % 12) * 30 + mm * 0.5 - 90, 0.55, (0, 255, 0)),
        (mm * 6 - 90, 0.80, (255, 0, 0)),
        (ss * 6 - 90, 0.90, (255, 0, 255)),
    ]
    for ang, lr, bgr_color in specs:
        tx = int(cx + r * lr * np.cos(np.radians(ang)))
        ty = int(cy + r * lr * np.sin(np.radians(ang)))
        tx = max(0, min(w - 1, tx))
        ty = max(0, min(hi - 1, ty))
        cv2.line(out, (cx, cy), (tx, ty), bgr_color, 2, lineType=cv2.LINE_AA)
    return out


def maybe_save_debug_overlay(
    bgr: np.ndarray,
    dial: DialEstimate,
    hh: int,
    mm: int,
    ss: int,
    stem: str,
    cfg: DebugConfig | None,
    hand_r: float,
) -> None:
    """Write one debug PNG if cfg allows."""
    if cfg is None or cfg.written >= cfg.max_count:
        return
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    vis = draw_geometry_debug_overlay(bgr, dial, hh, mm, ss, hand_r)
    path = cfg.out_dir / f"{stem}_debug.png"
    if cv2.imwrite(str(path), vis):
        cfg.written += 1


def process_image(
    path: Path,
    debug_cfg: DebugConfig | None,
) -> tuple[bool, str, dict | None]:
    """Build mask + optional debug; return CSV row dict on success."""
    try:
        hh, mm, ss = parse_time_from_analog_stem(path.stem)
    except ValueError as exc:
        return False, f"{path.name}: {exc}", None

    image = cv2.imread(str(path))
    if image is None:
        return False, f"{path.name}: could not read image", None

    h, w = image.shape[:2]
    style = parse_style_from_stem(path.stem)
    dial = estimate_dial_center(image, style)
    hand_r = hand_extent_radius(w, h, style, dial)
    # orange/vue/atc specific
    hls = ORANGE_VUE_ATC_HAND_LENGTH_SCALE.get(style, 1.0)
    mask = generate_geometric_mask(image, hh, mm, ss, dial, hand_r)
    frac = mask_frac_beyond_dial_ring(mask, dial.cx, dial.cy, w, h)

    out_path = MASKS_DIR / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), mask):
        return False, f"{path.name}: imwrite failed", None

    maybe_save_debug_overlay(
        image, dial, hh, mm, ss, path.stem, debug_cfg, hand_r
    )

    row = {
        "filename": path.name,
        "hh": hh,
        "mm": mm,
        "ss": ss,
        "style": style,
        "cx": dial.cx,
        "cy": dial.cy,
        "used_hough": dial.used_hough,
        "hough_radius": dial.hough_radius,
        "dial_source": dial.dial_source,
        "simple_fallback": simple_fallback_csv(style, dial.dial_source),
        "frac_mask_beyond_dial": round(frac, 4),
        "hand_length_scale": hls,
        "hand_r": round(hand_r, 2),
        "w": w,
        "h": h,
    }
    return True, "", row


def list_analog_images() -> list[Path]:
    """Sorted list of analog PNG paths under ANALOG_DIR."""
    paths = sorted(ANALOG_DIR.glob(f"analog_*{IMAGE_SUFFIX}"))
    return [p for p in paths if p.is_file()]


def write_csv(rows: list[dict], csv_path: Path) -> None:
    """Write mask generation log CSV."""
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "hh",
        "mm",
        "ss",
        "style",
        "cx",
        "cy",
        "used_hough",
        "hough_radius",
        "dial_source",
        "simple_fallback",
        "frac_mask_beyond_dial",
        "hand_length_scale",
        "hand_r",
        "w",
        "h",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fieldnames}
            if flat.get("hough_radius") is None:
                flat["hough_radius"] = ""
            writer.writerow(flat)


def run_generation(
    paths: list[Path],
    debug_cfg: DebugConfig | None,
) -> tuple[int, list[str], list[dict]]:
    """Process paths; collect CSV rows."""
    ok = 0
    failed: list[str] = []
    rows: list[dict] = []
    for i, path in enumerate(paths):
        success, err, row = process_image(path, debug_cfg)
        if success and row is not None:
            ok += 1
            rows.append(row)
        else:
            failed.append(err or path.name)
        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"Progress: {i + 1}/{len(paths)}", flush=True)
    return ok, failed, rows


def main() -> None:
    """Generate masks, CSV log, optional geometric debug PNGs."""
    args = parse_args()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = CSV_DEBUG_DIR / f"run_{run_tag}.csv"

    all_paths = list_analog_images()
    if not all_paths:
        print(f"[ERROR] No analog images under {ANALOG_DIR}")
        return

    if args.sample is not None:
        n = min(args.sample, len(all_paths))
        paths = random.sample(all_paths, n)
    else:
        paths = all_paths

    if args.style is not None:
        paths = [p for p in paths if parse_style_from_stem(p.stem) == args.style]
        if not paths:
            print(f"[ERROR] No images with style={args.style!r}")
            return
        print(f"[INFO] Filter style={args.style!r} -> {len(paths)} image(s)")

    debug_cfg: DebugConfig | None = None
    if args.debug:
        debug_cfg = DebugConfig(
            out_dir=DEBUG_IMG_DIR / f"run_{run_tag}",
            max_count=max(1, args.debug_max),
        )
        print(f"[INFO] Debug overlays -> {debug_cfg.out_dir} (max {debug_cfg.max_count})")

    ok, failed, rows = run_generation(paths, debug_cfg)
    write_csv(rows, csv_path)

    print(f"\n[OK] Generated {ok} masks | [WARN] Failed {len(failed)} images")
    print(f"[INFO] CSV log -> {csv_path}")
    if failed:
        print("Failed:")
        for line in failed:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
