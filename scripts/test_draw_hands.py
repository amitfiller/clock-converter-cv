from __future__ import annotations

from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hand_geometry.draw_hands import draw_hands_on_image

INPUT_IMAGE = PROJECT_ROOT / "data" / "raw" / "analog" / "analog_03_15_15_orange.png"
OUTPUT_IMAGE = PROJECT_ROOT / "demo_outputs" / "segmentation" / "test_output.png"


def main() -> None:
    image = cv2.imread(str(INPUT_IMAGE))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {INPUT_IMAGE}")

    output = draw_hands_on_image(image, hour=3, minute=15, second=30)
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_IMAGE), output):
        raise RuntimeError(f"Failed to save output image: {OUTPUT_IMAGE}")

    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()