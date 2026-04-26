"""Hand removal and optional mask compositing (legacy) utilities."""

from __future__ import annotations

import cv2
import numpy as np


def _clock_face_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Estimate inner clock-face circle and return binary mask."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    min_r = int(min(h, w) * 0.22)
    max_r = int(min(h, w) * 0.48)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=120,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )
    cx, cy, r = w // 2, h // 2, int(min(h, w) * 0.34)
    if circles is not None and len(circles[0]) > 0:
        x, y, rad = max(circles[0], key=lambda c: c[2])
        cx, cy, r = int(x), int(y), int(rad * 0.84)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(face_mask, (cx, cy), max(8, r), 255, -1)
    return face_mask


def _clock_center_radius_from_mask(face_mask: np.ndarray) -> tuple[tuple[int, int], int]:
    """Get clock center and radius from binary mask."""
    contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = face_mask.shape[:2]
        return (w // 2, h // 2), min(h, w) // 2
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return (int(cx), int(cy)), max(4, int(r))


def _line_cleanup_mask(img_bgr: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """Detect hand-like center lines for extra inpaint coverage."""
    h, w = face_mask.shape[:2]
    cx, cy = w // 2, h // 2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.bitwise_and(edges, face_mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=min(h, w) // 8, maxLineGap=10)
    extra = np.zeros((h, w), dtype=np.uint8)
    if lines is None:
        return extra
    r = min(h, w) // 2
    for seg in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in seg]
        d1 = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5
        d2 = ((x2 - cx) ** 2 + (y2 - cy) ** 2) ** 0.5
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if min(d1, d2) > r * 0.22 or max(d1, d2) > r * 0.88 or length < r * 0.22:
            continue
        cv2.line(extra, (x1, y1), (x2, y2), 255, 5, cv2.LINE_AA)
    return cv2.bitwise_and(extra, face_mask)


def _detect_red_hand_mask(img_bgr: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """Detect red second-hand residues inside clock face."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    low1 = np.array([0, 60, 60], dtype=np.uint8)
    high1 = np.array([12, 255, 255], dtype=np.uint8)
    low2 = np.array([168, 60, 60], dtype=np.uint8)
    high2 = np.array([179, 255, 255], dtype=np.uint8)
    red = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)
    red = cv2.bitwise_and(red, face_mask)
    kernel = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.dilate(red, kernel, iterations=2)


def _endpoint_cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Boost mask coverage at hand tips to remove endpoint smears."""
    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extra = np.zeros_like(mask)
    r_tip = max(4, min(h, w) // 44)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        pts = cnt.reshape(-1, 2)
        d = ((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) ** 0.5
        tip = pts[int(np.argmax(d))]
        tx, ty = int(tip[0]), int(tip[1])
        cv2.circle(extra, (tx, ty), r_tip, 255, -1)
    return extra


def _extract_hand_specs(
    original_bgr: np.ndarray,
    binary_mask: np.ndarray,
    clock_center: tuple[int, int] | None = None,
) -> dict | None:
    """Estimate per-hand style from masked line segments near center (Fix 2).
    Uses clock_center for accurate pivot distance and cv2.minAreaRect for thickness.
    """
    h, w = binary_mask.shape[:2]
    # Fix 2: use actual clock center instead of always w//2, h//2
    # cx, cy = w // 2, h // 2
    cx, cy = clock_center if clock_center is not None else (w // 2, h // 2)
    radius = max(1, min(h, w) // 2)
    masked = cv2.bitwise_and(original_bgr, original_bgr, mask=binary_mask)
    edges = cv2.Canny(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=radius // 5, maxLineGap=8)
    if lines is None:
        return None
    specs = []
    for seg in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in seg]
        d1 = float(np.hypot(x1 - cx, y1 - cy))
        d2 = float(np.hypot(x2 - cx, y2 - cy))
        length = float(np.hypot(x2 - x1, y2 - y1))
        if min(d1, d2) > radius * 0.28 or length < radius * 0.2:
            continue
        tx, ty = (x1, y1) if d1 >= d2 else (x2, y2)
        line_mask = np.zeros_like(binary_mask)
        cv2.line(line_mask, (cx, cy), (tx, ty), 255, 3, cv2.LINE_AA)
        line_mask = cv2.bitwise_and(line_mask, binary_mask)
        px = original_bgr[line_mask > 0]
        if len(px) < 8:
            continue
        color = tuple(int(x) for x in np.median(px, axis=0))
        # Fix 2: use minAreaRect for thickness instead of distance transform
        # thick = max(2, int(np.median(dist[line_mask > 0]) * 2.0))
        hand_cnts, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hand_cnts:
            rect = cv2.minAreaRect(max(hand_cnts, key=cv2.contourArea))
            thick = max(2, int(min(rect[1])))
        else:
            thick = max(2, int(radius * 0.03))
        specs.append({"length_ratio": length / radius, "color": color, "thickness": thick})
    if len(specs) < 2:
        return None
    specs = sorted(specs, key=lambda s: s["length_ratio"])
    hour = specs[0]
    minute = specs[min(1, len(specs) - 1)]
    red_idx = int(np.argmax([s["color"][2] - max(s["color"][0], s["color"][1]) for s in specs]))
    second = specs[red_idx] if (specs[red_idx]["color"][2] - max(specs[red_idx]["color"][0], specs[red_idx]["color"][1])) > 15 else specs[-1]
    return {"hour": hour, "minute": minute, "second": second}


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert any mask format to uint8 0/255."""
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[..., 0]
    if mask.dtype == np.uint8:
        if mask.max() <= 1:
            return (mask.astype(np.float32) * 255.0).astype(np.uint8)
        return mask
    mask_float = mask.astype(np.float32)
    if mask_float.max() <= 1.0:
        mask_float *= 255.0
    return np.clip(mask_float, 0, 255).astype(np.uint8)


def sample_hand_style(
    original_bgr: np.ndarray,
    binary_mask: np.ndarray,
    clock_center: tuple[int, int] | None = None,
) -> dict | None:
    """Sample hand color and thickness from masked hand pixels (Fix 2).
    Uses HSV masking to isolate actual hand pixels from clock face background.
    """
    # Fix 2: detect hand pixels via HSV — exclude bright desaturated background pixels
    # (old approach sampled bg from image corners which may not match clock face bg)
    hsv_full = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    bright_desaturated = cv2.inRange(hsv_full, (0, 0, 200), (180, 30, 255))
    hand_mask = cv2.bitwise_and(binary_mask, cv2.bitwise_not(bright_desaturated))
    mask_pixels = original_bgr[hand_mask > 0]
    if len(mask_pixels) < 10:
        mask_pixels = original_bgr[binary_mask > 0]
    if len(mask_pixels) < 10:
        return None
    # Fix 2: color = mean BGR of isolated hand pixels
    color = tuple(int(x) for x in np.median(mask_pixels, axis=0))
    clock_radius = min(original_bgr.shape[:2]) // 2
    thickness = max(2, clock_radius // 24)
    # Fix 2: pass clock_center so _extract_hand_specs uses correct pivot
    per_hand = _extract_hand_specs(original_bgr, binary_mask, clock_center=clock_center)
    return {"color": color, "thickness": thickness, "per_hand": per_hand}


def sample_style_geometric(img_bgr: np.ndarray, cx: int, cy: int, radius: int) -> dict:
    """Sample a fallback hand style from a circular ring around the pivot."""
    import math

    colors: list[list[int]] = []
    for angle_deg in range(0, 360, 5):
        ang = math.radians(angle_deg)
        for r_ratio in (0.3, 0.4, 0.5, 0.6):
            x = int(cx + r_ratio * radius * math.sin(ang))
            y = int(cy - r_ratio * radius * math.cos(ang))
            if 0 <= x < img_bgr.shape[1] and 0 <= y < img_bgr.shape[0]:
                b, g, r = img_bgr[y, x].astype(int)
                is_red = (r > 120) and (r > 2 * b) and (r > 2 * g)
                is_bright = (b + g + r) > 500
                if not is_red and not is_bright:
                    colors.append([b, g, r])
    if len(colors) > 10:
        color_arr = np.array(colors, dtype=np.uint8)
        brightness = color_arr.mean(axis=1)
        dark_idx = np.argsort(brightness)[: max(10, len(color_arr) // 5)]
        hand_color = tuple(np.median(color_arr[dark_idx], axis=0).astype(int).tolist())
    else:
        hand_color = (30, 25, 20)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = red_mask1 | red_mask2
    if int(np.count_nonzero(red_mask)) > 100:
        red_pixels = img_bgr[red_mask > 0]
        second_color = tuple(np.median(red_pixels, axis=0).astype(int).tolist())
    else:
        second_color = (0, 0, 200)
    return {
        "color": hand_color,
        "thickness": max(3, int(radius * 0.03)),
        "per_hand": {
            "hour": {"length_ratio": 0.50, "color": hand_color, "thickness": max(5, int(radius * 0.038))},
            "minute": {"length_ratio": 0.75, "color": hand_color, "thickness": max(3, int(radius * 0.026))},
            "second": {"length_ratio": 0.88, "color": second_color, "thickness": max(2, int(radius * 0.014))},
        },
    }


def reconstruct_clock_face(
    original_bgr: np.ndarray,
    binary_mask: np.ndarray,
    clock_center: tuple[int, int],
    clock_radius: int,
) -> np.ndarray:
    """Rebuild masked pixels using radial polynomial fit instead of cv2.inpaint."""
    h, w = original_bgr.shape[:2]
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    cx, cy = clock_center
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / float(max(clock_radius, 1))
    inside_clock = dist <= 0.95
    unmasked = inside_clock & (binary_mask == 0)
    to_fill = inside_clock & (binary_mask > 0)
    if not np.any(to_fill):
        return original_bgr.copy()
    result = original_bgr.copy().astype(np.float32)
    d_unmasked = dist[unmasked]
    d_fill = dist[to_fill]
    for c in range(3):
        channel = original_bgr[:, :, c].astype(np.float64)
        v_unmasked = channel[unmasked]
        if v_unmasked.size < 8:
            continue
        try:
            coeffs = np.polyfit(d_unmasked, v_unmasked, deg=2)
        except Exception:
            coeffs = np.polyfit(d_unmasked, v_unmasked, deg=1)
        filled = np.polyval(coeffs, d_fill)
        filled = np.clip(filled, 0, 255)
        result[:, :, c][to_fill] = filled
    return result.astype(np.uint8)


def remove_hands(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    clock_center: tuple[int, int] | None = None,
    clock_radius: int | None = None,
) -> tuple[np.ndarray, dict | None]:
    """Remove hands via mask cleanup and radial clock-face reconstruction."""
    mask_uint8 = _normalize_mask(mask)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    face_mask = _clock_face_mask(img_bgr)
    closed = cv2.bitwise_and(closed, face_mask)
    h, w = closed.shape[:2]
    cx, cy = w // 2, h // 2
    center_seed = np.zeros_like(closed)
    cv2.circle(center_seed, (cx, cy), max(6, min(h, w) // 28), 255, -1)
    bridge = cv2.dilate(closed, kernel, iterations=3)
    num_labels, labels = cv2.connectedComponents((bridge > 0).astype(np.uint8))
    center_ids = np.unique(labels[center_seed > 0])
    keep = np.zeros_like(bridge)
    for label_id in center_ids.tolist():
        if label_id == 0:
            continue
        keep[labels == label_id] = 255
    if np.count_nonzero(keep) > 0:
        closed = cv2.bitwise_and(bridge, keep)
    line_extra = _line_cleanup_mask(img_bgr, face_mask)
    closed = cv2.max(closed, line_extra)
    closed = cv2.max(closed, _detect_red_hand_mask(img_bgr, face_mask))
    closed = cv2.max(closed, _endpoint_cleanup_mask(closed))
    # Fix 2: derive center before sample_hand_style so the sampler has the real pivot
    # (previously center was derived after, so sample_hand_style got w//2,h//2)
    if clock_center is None or clock_radius is None:
        clock_center, clock_radius = _clock_center_radius_from_mask(face_mask)

    hand_style = sample_hand_style(img_bgr, closed, clock_center=clock_center)

    # Reconstruct the clock face using radial model, no cv2.inpaint
    binary_mask = (closed > 0).astype(np.uint8)
    inpainted = reconstruct_clock_face(img_bgr, binary_mask, clock_center, clock_radius)

    return inpainted, hand_style


def composite_hands(
    clean_img: np.ndarray, hand_mask: np.ndarray, hand_color: tuple[int, int, int] = (40, 40, 40)
) -> np.ndarray:
    """Overlay synthetic hands on a clean image."""
    if hand_mask.ndim == 2:
        hand_mask = hand_mask[..., None]
    alpha = np.clip(hand_mask.astype(np.float32), 0.0, 1.0)
    color_layer = np.full_like(clean_img, hand_color, dtype=np.uint8)
    result = clean_img.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
