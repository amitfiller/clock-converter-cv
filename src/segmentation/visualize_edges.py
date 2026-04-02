from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "analog"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo_outputs" / "segmentation"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolate analog clock hands with Canny edges and Hough line detection."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a single analog clock image. Defaults to the first image in data/raw/analog.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the hand image. Defaults to demo_outputs/segmentation/<name>_hands.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the edge image in an OpenCV window after saving.",
    )
    return parser.parse_args()


def find_default_image() -> Path:
    images = sorted(
        path for path in DEFAULT_INPUT_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No image files found in {DEFAULT_INPUT_DIR}")
    return images[0]


def build_output_path(image_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{image_path.stem}_hands.png"


def point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def distance_to_center(
    line: tuple[int, int, int, int], center: tuple[float, float]
) -> float:
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return float("inf")

    numerator = abs(dy * center[0] - dx * center[1] + x2 * y1 - y2 * x1)
    denominator = math.hypot(dx, dy)
    return numerator / denominator


def line_length(line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def angle_gap(angle_a: float, angle_b: float) -> float:
    gap = abs(angle_a - angle_b)
    return min(gap, 180.0 - gap)


def select_tip(
    line: tuple[int, int, int, int], center: tuple[float, float]
) -> tuple[int, int]:
    endpoint_a = (line[0], line[1])
    endpoint_b = (line[2], line[3])
    if point_distance(endpoint_a, center) >= point_distance(endpoint_b, center):
        return endpoint_a
    return endpoint_b


def ray_angle(center: tuple[float, float], tip: tuple[int, int]) -> float:
    angle = math.degrees(math.atan2(tip[1] - center[1], tip[0] - center[0]))
    return (angle + 180.0) % 180.0


def detect_hand_lines(image: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(15, min(image.shape[:2]) // 8),
        maxLineGap=18,
    )
    if detected is None:
        return edges, []

    height, width = grayscale.shape
    center = (width / 2.0, height / 2.0)
    clock_radius = min(width, height) * 0.48
    min_length = clock_radius * 0.18
    max_center_distance = clock_radius * 0.18
    rim_margin = clock_radius * 0.1

    candidates: list[tuple[float, float, tuple[int, int]]] = []
    for raw_line in detected[:, 0, :]:
        line = tuple(int(value) for value in raw_line)
        length = line_length(line)
        if length < min_length:
            continue

        center_distance = distance_to_center(line, center)
        if center_distance > max_center_distance:
            continue

        endpoint_a = (line[0], line[1])
        endpoint_b = (line[2], line[3])
        radial_a = point_distance(endpoint_a, center)
        radial_b = point_distance(endpoint_b, center)

        # Hands usually start near the center and extend outward; rim fragments do not.
        if min(radial_a, radial_b) > clock_radius * 0.45:
            continue
        if max(radial_a, radial_b) >= clock_radius + rim_margin:
            continue

        tip = select_tip(line, center)
        tip_length = point_distance(tip, center)
        if tip_length < clock_radius * 0.18:
            continue

        score = tip_length - (center_distance * 2.0)
        candidates.append((score, ray_angle(center, tip), tip))

    candidates.sort(key=lambda item: item[0], reverse=True)

    unique_candidates: list[tuple[float, float, tuple[int, int]]] = []
    for score, angle, tip in candidates:
        tip_length = point_distance(tip, center)
        duplicate = False
        for kept_score, kept_angle, kept_tip in unique_candidates:
            kept_length = point_distance(kept_tip, center)
            same_angle = angle_gap(angle, kept_angle) < 8.0
            same_length = abs(tip_length - kept_length) < clock_radius * 0.08
            if same_angle and same_length:
                duplicate = True
                if score > kept_score:
                    unique_candidates.remove((kept_score, kept_angle, kept_tip))
                    unique_candidates.append((score, angle, tip))
                break
        if not duplicate:
            unique_candidates.append((score, angle, tip))

    unique_candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, int, int, int]] = []
    selected_meta: list[tuple[float, float]] = []
    center_point = (int(round(center[0])), int(round(center[1])))
    for _, angle, tip in unique_candidates:
        tip_length = point_distance(tip, center)
        if any(
            angle_gap(angle, kept_angle) < 8.0 and abs(tip_length - kept_length) < clock_radius * 0.14
            for kept_angle, kept_length in selected_meta
        ):
            continue
        selected.append((center_point[0], center_point[1], tip[0], tip[1]))
        selected_meta.append((angle, tip_length))
        if len(selected) == 2:
            break

    if len(selected) < 2:
        selected = [
            (center_point[0], center_point[1], tip[0], tip[1])
            for _, _, tip in unique_candidates[:2]
        ]

    return edges, selected


def render_lines(shape: tuple[int, int, int], lines: list[tuple[int, int, int, int]]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    ranked_lines = sorted(lines, key=line_length)
    intensities = [180, 255]
    for index, (x1, y1, x2, y2) in enumerate(ranked_lines):
        intensity = intensities[min(index, len(intensities) - 1)]
        cv2.line(canvas, (x1, y1), (x2, y2), (intensity, intensity, intensity), thickness=3)
    return canvas


def main() -> None:
    args = parse_args()
    image_path = args.input or find_default_image()
    output_path = build_output_path(image_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    edges, hand_lines = detect_hand_lines(image)
    result = render_lines(image.shape, hand_lines)

    if not hand_lines:
        raise RuntimeError("No clock-hand lines were detected in the image.")
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to save hand image to {output_path}")

    print(f"Input image: {image_path}")
    print("1. Converted the image to grayscale and blurred it slightly to stabilize the edges.")
    print("2. Ran Canny edge detection to keep only strong intensity changes.")
    print("3. Used Hough Line Transform to find straight-line candidates in the edge map.")
    print("4. Filtered out short lines and lines that do not pass near the clock center.")
    print("5. Kept the two strongest line candidates as the clock hands.")
    print(f"Saved hand-only image: {output_path}")

    if args.show:
        cv2.imshow("Detected Clock Hands", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()