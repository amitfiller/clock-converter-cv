"""Clock hand drawing for pipeline (BGR) and pseudo-label masks for training."""

import cv2
import numpy as np


def _clip_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Clamp BGR values to valid uint8 range."""
    return tuple(int(np.clip(v, 0, 255)) for v in color)


def _ensure_min_contrast(color: tuple[int, int, int], bg: np.ndarray) -> tuple[int, int, int]:
    """Nudge color if too close to local background."""
    c = np.array(color, dtype=np.float32)
    b = np.array(bg, dtype=np.float32)
    if np.max(np.abs(c - b)) < 35:
        c = np.clip(b - np.array([70, 70, 70], dtype=np.float32), 0, 255)
    return _clip_color(tuple(int(v) for v in c))


def draw_hand_polygon(
    img: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float,
    length: float,
    base_width: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a triangular clock hand using filled polygon from center."""
    import math

    # Angles are given in clock coordinates: 0° at 12 o'clock, clockwise.
    angle_rad = math.radians(angle_deg)
    # Image coordinates: x = cx + L * sin(theta), y = cy - L * cos(theta)
    tip_x = int(cx + length * math.sin(angle_rad))
    tip_y = int(cy - length * math.cos(angle_rad))
    # Perpendicular direction for base width
    perp = angle_rad + math.pi / 2.0
    bw = base_width / 2.0
    base1_x = int(cx + bw * math.sin(perp))
    base1_y = int(cy - bw * math.cos(perp))
    base2_x = int(cx - bw * math.sin(perp))
    base2_y = int(cy + bw * math.cos(perp))
    pts = np.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]], np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)


def _draw_hand_line(
    img: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float,
    length: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a clock hand as a straight line from center to tip (Fix 4).
    Angle is in clock coordinates: 0° at 12 o'clock, clockwise.
    """
    import math
    angle_rad = math.radians(angle_deg)
    tip_x = int(cx + length * math.sin(angle_rad))
    tip_y = int(cy - length * math.cos(angle_rad))
    cv2.line(img, (cx, cy), (tip_x, tip_y), color, max(1, thickness), cv2.LINE_AA)


def _hand_segment_pivot_to_tip(
    cx: int, cy: int, angle_deg: float, r_tip: float, thickness: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Inner and outer endpoints so thick LINE_AA does not extend backward past pivot."""
    rad = np.radians(angle_deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    r0 = max(thickness * 0.65 + 2.0, 4.0)
    p0 = (int(cx + c * r0), int(cy + s * r0))
    p1 = (int(cx + c * r_tip), int(cy + s * r_tip))
    return p0, p1


def draw_hands_on_image(
    img_bgr: np.ndarray,
    h: int,
    m: int,
    s: int,
    hand_style: dict | None = None,
    clock_center: tuple[int, int] | None = None,  # Fix 3a: accept detected clock center
    clock_radius: int | None = None,
) -> np.ndarray:
    """
    Draw clock hands directly on img_bgr using straight lines.
    Works at ANY resolution. Returns a new image with hands drawn.
    """
    out = img_bgr.copy()
    H, W = out.shape[:2]
    # Fix 3b: use detected center if provided, fall back to image center
    # cx, cy = W // 2, H // 2
    cx, cy = clock_center if clock_center is not None else (W // 2, H // 2)
    R = min(H, W) * 0.38

    # Angles in clock coordinates: 0° at 12 o'clock, clockwise.
    hour_angle = (h % 12) * 30 + m * 0.5
    minute_angle = m * 6 + s * 0.1
    second_angle = s * 6

    if clock_radius is not None and clock_radius > 10:
        radius = int(clock_radius * 0.90)  # 90% of detected radius — safety margin
    else:
        radius = min(out.shape[:2]) // 2
    if hand_style is not None:
        hand_color = _clip_color(tuple(hand_style["color"]))
        color_hour = hand_color
        color_minute = hand_color
        # Allow optional thickness_ratio override from style
        if "thickness_ratio" in hand_style and "thickness" not in hand_style:
            thickness = max(2, int(radius * float(hand_style["thickness_ratio"])))
        else:
            thickness = int(hand_style.get("thickness", max(3, int(R * 0.07))))
        hour_len = radius * 0.50
        minute_len = radius * 0.75
        second_len = radius * 0.85
        second_color = (0, 0, 220)
        if isinstance(hand_style.get("per_hand"), dict):
            ph = hand_style["per_hand"]
            if isinstance(ph.get("hour"), dict):
                color_hour = _clip_color(tuple(ph["hour"].get("color", color_hour)))
                thickness = max(2, int(ph["hour"].get("thickness", thickness)))
                hour_len = radius * float(np.clip(ph["hour"].get("length_ratio", 0.50), 0.35, 0.70))
            if isinstance(ph.get("minute"), dict):
                color_minute = _clip_color(tuple(ph["minute"].get("color", color_minute)))
                minute_len = radius * float(np.clip(ph["minute"].get("length_ratio", 0.75), 0.45, 0.92))
            if isinstance(ph.get("second"), dict):
                c2 = _clip_color(tuple(ph["second"].get("color", second_color)))
                second_color = c2 if (c2[2] - max(c2[0], c2[1])) > 10 else second_color
                second_len = radius * float(np.clip(ph["second"].get("length_ratio", 0.85), 0.60, 0.98))
    else:
        # Detect background brightness for adaptive color fallback
        gray_mean = np.mean(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
        if gray_mean < 128:
            hand_color = (230, 230, 230)  # white-ish
            color_minute = (200, 200, 200)
        else:
            hand_color = (30, 30, 30)  # dark
            color_minute = (50, 50, 50)
        color_hour = hand_color
        thickness = max(3, int(R * 0.07))
        hour_len = radius * 0.50
        minute_len = radius * 0.75
        second_len = radius * 0.85
        second_color = (0, 0, 220)
    # Enforce minimum thicknesses — prefer per-hand sampled values when available (Fix 1)
    # hour_thickness = max(int(radius * 0.045), 5)
    # minute_thickness = max(int(radius * 0.032), 4)
    # second_thickness = max(int(radius * 0.018), 2)
    if hand_style is not None and isinstance(hand_style.get("per_hand"), dict):
        ph = hand_style["per_hand"]
        hour_thickness   = max(2, int(ph.get("hour",   {}).get("thickness", int(radius * 0.045))))
        minute_thickness = max(2, int(ph.get("minute", {}).get("thickness", int(radius * 0.032))))
        second_thickness = max(2, int(ph.get("second", {}).get("thickness", int(radius * 0.018))))
    else:
        hour_thickness   = max(int(radius * 0.045), 5)
        minute_thickness = max(int(radius * 0.032), 4)
        second_thickness = max(int(radius * 0.018), 2)

    local_bg = np.median(out[max(0, cy - 24) : min(H, cy + 24), max(0, cx - 24) : min(W, cx + 24)], axis=(0, 1))
    hand_color = _ensure_min_contrast(hand_color, local_bg)
    color_hour = _ensure_min_contrast(color_hour, local_bg)
    color_minute = _ensure_min_contrast(
        _clip_color(tuple(int(v) for v in np.array(color_minute) * 0.95)),
        local_bg,
    )

    # If sampled colors are still too close to background, force dark hands (keep red seconds)
    def _max_dist(c, bg):
        c_arr = np.array(c, dtype=np.float32)
        b_arr = np.array(bg, dtype=np.float32)
        return float(np.max(np.abs(c_arr - b_arr)))

    if _max_dist(color_hour, local_bg) < 30.0:
        color_hour = (40, 40, 40)
    if _max_dist(color_minute, local_bg) < 30.0:
        color_minute = (40, 40, 40)

    color_second = second_color

    # Debug prints for angle and style
    print(
        f"[DRAW] time={h:02d}:{m:02d}:{s:02d} -> "
        f"hour_angle={hour_angle:.1f}deg minute_angle={minute_angle:.1f}deg second_angle={second_angle:.1f}deg"
    )
    print(
        "[draw_hands] center=({},{}) radius={} thicknesses(h,m,s)=({},{},{}) colors(h,m,s)={} {} {}".format(
            cx,
            cy,
            radius,
            hour_thickness,
            minute_thickness,
            second_thickness,
            color_hour,
            color_minute,
            color_second,
        )
    )

    # Fix 4: draw hands as straight lines instead of filled triangles
    # draw_hand_polygon(out, cx, cy, hour_angle, hour_len, hour_thickness, color_hour)
    # draw_hand_polygon(out, cx, cy, minute_angle, minute_len, minute_thickness, color_minute)
    # draw_hand_polygon(out, cx, cy, second_angle, second_len, second_thickness, color_second)
    _draw_hand_line(out, cx, cy, hour_angle, hour_len, color_hour, hour_thickness)
    _draw_hand_line(out, cx, cy, minute_angle, minute_len, color_minute, minute_thickness)
    _draw_hand_line(out, cx, cy, second_angle, second_len, color_second, second_thickness)

    # Center cap circle
    center_r = max(2, int(radius * 0.03))
    cv2.circle(out, (cx, cy), center_r, (30, 30, 30), -1, cv2.LINE_AA)

    return out


def draw_hand_mask(h: int, m: int, s: int, img_size: int = 64) -> np.ndarray:
    """
    Returns a binary mask (img_size, img_size, 1) float32 marking hand pixels.
    Used by the U-Net training pipeline — keep this unchanged.
    """
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    cx = cy = img_size // 2
    R = img_size * 0.38

    def ep(angle_deg, length):
        rad = np.radians(angle_deg)
        return (int(cx + np.cos(rad) * length), int(cy + np.sin(rad) * length))

    hour_angle = (h % 12) * 30 + m * 0.5 - 90
    minute_angle = m * 6 - 90
    second_angle = s * 6 - 90

    cv2.line(canvas, (cx, cy), ep(hour_angle, R * 0.55), 255, max(2, int(R * 0.10)))
    cv2.line(canvas, (cx, cy), ep(minute_angle, R * 0.80), 255, max(1, int(R * 0.06)))
    cv2.line(canvas, (cx, cy), ep(second_angle, R * 0.90), 255, max(1, int(R * 0.04)))
    cv2.circle(canvas, (cx, cy), max(2, int(R * 0.06)), 255, -1)

    return (canvas.astype(np.float32) / 255.0)[..., np.newaxis]
