"""Live demo entry point for ClockWise pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.pipeline import ClockConverterPipeline


def configure_console_encoding() -> None:
    """Ensure console can print UTF-8 text on Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for live demo."""
    parser = argparse.ArgumentParser(description="ClockWise live demo.")
    parser.add_argument("--digital", required=True, help="Path to digital clock image.")
    parser.add_argument("--analog", required=True, help="Path to analog clock image.")
    parser.add_argument("--no-display", action="store_true", help="Skip cv2.imshow window.")
    return parser.parse_args()


def validate_input_paths(digital_path: Path, analog_path: Path) -> None:
    """Validate that both image paths exist."""
    if not digital_path.exists():
        raise FileNotFoundError(f"Digital image not found: {digital_path}")
    if not analog_path.exists():
        raise FileNotFoundError(f"Analog image not found: {analog_path}")


def create_pipeline() -> ClockConverterPipeline:
    """Create pipeline using fixed checkpoint paths."""
    return ClockConverterPipeline(
        module_a_path="models/checkpoints/digital_reader_best.pth",
        module_b_path="checkpoints/unet_256.pth",
    )


def get_output_image(result: dict) -> object:
    """Return output image from known pipeline keys."""
    if "output_display" in result:
        return result["output_display"]
    if "output_image" in result:
        return result["output_image"]
    raise KeyError("Pipeline result has no output image key.")


def format_time(predicted_time: tuple[int, int, int]) -> str:
    """Format predicted tuple into HH:MM:SS."""
    h, m, s = predicted_time
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_output_image(output_image: object, out_path: Path) -> None:
    """Save output image and ensure parent folder exists."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), output_image):
        raise RuntimeError(f"Failed to save output image to: {out_path}")


def load_bgr_image(image_path: Path, label: str) -> np.ndarray:
    """Read image from disk as BGR array."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read {label} image: {image_path}")
    return image


def resize_to_height(image: np.ndarray, target_height: int = 400) -> np.ndarray:
    """Resize image while preserving aspect ratio."""
    height, width = image.shape[:2]
    if height == target_height:
        return image.copy()
    target_width = max(1, int(width * (target_height / height)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def add_panel_label(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Draw label text on top-left corner of panel."""
    labeled = image.copy()
    cv2.putText(
        labeled,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return labeled


def maybe_show_output(
    digital_path: Path,
    analog_path: Path,
    output_image: np.ndarray,
    time_text: str,
    no_display: bool,
) -> None:
    """Display combined preview unless --no-display was set."""
    if no_display:
        return
    digital_img = load_bgr_image(digital_path, "digital")
    analog_img = load_bgr_image(analog_path, "analog")

    left = add_panel_label(
        resize_to_height(digital_img),
        "INPUT: Digital",
        (0, 255, 0),
    )
    center = add_panel_label(
        resize_to_height(analog_img),
        "INPUT: Analog",
        (0, 165, 255),
    )
    right = add_panel_label(
        resize_to_height(output_image),
        f"OUTPUT: {time_text}",
        (255, 255, 0),
    )

    combined = np.hstack([left, center, right])
    cv2.imshow("ClockWise — Digital to Analog Converter", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    """Run live demo pipeline and return process exit code."""
    configure_console_encoding()
    args = parse_args()
    digital_path = Path(args.digital)
    analog_path = Path(args.analog)
    out_path = Path("demo_outputs/live_result.png")

    try:
        validate_input_paths(digital_path, analog_path)
        pipeline = create_pipeline()
        result = pipeline.run(digital_path, analog_path)
        predicted_time = result["predicted_time"]
        time_text = format_time(predicted_time)
        output_image = get_output_image(result)
        print(f"✓ Detected time: {time_text}")
        save_output_image(output_image, out_path)
        maybe_show_output(digital_path, analog_path, output_image, time_text, args.no_display)
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
# Phase 4 - Entry point
