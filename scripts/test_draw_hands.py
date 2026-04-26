"""Isolated clock hand drawing test on one analog image."""

from __future__ import annotations

import math

import cv2
import numpy as np


def draw_single_hand(
    img: object,
    cx: int,
    cy: int,
    angle_deg: float,
    length_ratio: float,
    radius: int,
    thickness: int,
    color: tuple[int, int, int],
) -> object:
    """Draw one clock hand from center using clock-angle coordinates."""
    length = length_ratio * radius
    rad = math.radians(angle_deg)
    ex = int(cx + length * math.sin(rad))
    ey = int(cy - length * math.cos(rad))
    cv2.line(img, (cx, cy), (ex, ey), color, thickness, cv2.LINE_AA)
    return img


def sample_hand_colors(
    img: np.ndarray, cx: int, cy: int, radius: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Sample dark hand color and red second-hand color from source image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    red_mask = red1 | red2

    colors: list[list[int]] = []
    for angle in range(0, 360, 3):
        for ratio in (0.25, 0.35, 0.45, 0.55, 0.65):
            x = int(cx + ratio * radius * math.sin(math.radians(angle)))
            y = int(cy - ratio * radius * math.cos(math.radians(angle)))
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                if red_mask[y, x] == 0:
                    b, g, r = img[y, x].astype(int)
                    brightness = b + g + r
                    if brightness < 300:
                        colors.append([b, g, r])

    if len(colors) >= 10:
        arr = np.array(colors, dtype=np.uint8)
        brightness = arr.mean(axis=1)
        n = max(5, len(arr) // 3)
        dark = arr[np.argsort(brightness)[:n]]
        hand_color = tuple(np.median(dark, axis=0).astype(int).tolist())
    else:
        hand_color = (35, 30, 25)

    if int(np.count_nonzero(red_mask)) > 200:
        red_pixels = img[red_mask > 0]
        second_color = tuple(np.median(red_pixels, axis=0).astype(int).tolist())
    else:
        second_color = (30, 30, 180)

    return hand_color, second_color


def main() -> None:
    """Load one clock and draw target time hands for visual inspection."""
    img = cv2.imread("data/raw/analog/analog_09_15_03_wall.png")
    if img is None:
        raise FileNotFoundError("Could not read analog_09_15_03_wall.png")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=int(min(w, h) * 0.25),
        maxRadius=int(min(w, h) * 0.55),
    )
    if circles is not None:
        c = np.round(circles[0][0]).astype(int)
        cx, cy, radius = int(c[0]), int(c[1]), int(c[2])
        print(f"Detected circle: center=({cx},{cy}), radius={radius}")
    else:
        cx, cy = w // 2, h // 2
        radius = int(min(w, h) * 0.38)
        print(f"HoughCircles failed, using fallback radius={radius}")
    radius = int(radius * 0.85)
    print(f"Adjusted radius: {radius}")

    print(f"Image size: {w}x{h}")
    print(f"Clock center: ({cx}, {cy}), radius: {radius}")
    hand_color, second_color = sample_hand_colors(img, cx, cy, radius)
    print(f"Sampled hand_color: {hand_color}")
    print(f"Sampled second_color: {second_color}")

    hour_angle = (7 % 12) / 12 * 360 + (23 / 60) * 30
    minute_angle = 23 / 60 * 360 + (18 / 60) * 6
    second_angle = 18 / 60 * 360

    print(
        f"Angles — Hour: {hour_angle:.1f}°, "
        f"Minute: {minute_angle:.1f}°, Second: {second_angle:.1f}°"
    )

    result = img.copy()
    result = draw_single_hand(result, cx, cy, hour_angle, 0.50, radius, 6, hand_color)
    result = draw_single_hand(result, cx, cy, minute_angle, 0.75, radius, 4, hand_color)
    result = draw_single_hand(result, cx, cy, second_angle, 0.85, radius, 2, second_color)

    out_path = "demo_outputs/debug/test_draw_hands_isolated.png"
    if not cv2.imwrite(out_path, result):
        raise RuntimeError("Failed to save test_draw_hands_isolated.png")
    print("Saved: test_draw_hands_isolated.png")


if __name__ == "__main__":
    main()
