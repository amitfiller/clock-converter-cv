"""Gradio demo for Digital → Analog Clock Converter."""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path when launched from demo/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import gradio as gr

from src.pipeline import ClockConverterPipeline
from src.hand_geometry.draw_hands import draw_hands_on_image
from src.inpainting.remove_hands import _clock_face_mask, sample_style_geometric

# ── Checkpoint paths (relative to project root) ──────────────────────────────
_MODULE_A = str(_root / "models" / "checkpoints" / "digital_reader_best.pth")
_MODULE_B = str(_root / "checkpoints" / "unet_256.pth")

_pipeline: ClockConverterPipeline | None = None


def _get_pipeline() -> ClockConverterPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ClockConverterPipeline(
            module_a_path=_MODULE_A,
            module_b_path=_MODULE_B,
            device="cpu",
        )
    return _pipeline


def _numpy_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Gradio gives RGB numpy arrays; cv2 needs BGR."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ── Task 3: Convert tab ───────────────────────────────────────────────────────

def convert(digital_img: np.ndarray | None, analog_img: np.ndarray | None):
    """Run pipeline on uploaded images, return (output_rgb, time_string)."""
    if digital_img is None or analog_img is None:
        return None, "Please upload both images."

    pipeline = _get_pipeline()

    # Save temp files so pipeline can read them (it uses cv2.imread internally)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as df:
        digital_path = df.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as af:
        analog_path = af.name

    try:
        cv2.imwrite(digital_path, _numpy_to_bgr(digital_img))
        cv2.imwrite(analog_path, _numpy_to_bgr(analog_img))

        result = pipeline.run(digital_path, analog_path)
        h, m, s = result["predicted_time"]
        output_rgb = _bgr_to_rgb(result["output_image"])
        time_str = f"Detected: {h:02d}:{m:02d}:{s:02d}"
        return output_rgb, time_str
    except Exception as exc:
        return None, f"Error: {exc}"
    finally:
        os.unlink(digital_path)
        os.unlink(analog_path)


# ── Task 4: Live Clock tab ────────────────────────────────────────────────────

def _detect_clock_center(img_bgr: np.ndarray):
    """Return (cx, cy, radius) from the clock face mask."""
    h, w = img_bgr.shape[:2]
    mask = _clock_face_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        return int(cx), int(cy), max(4, int(r))
    return w // 2, h // 2, min(h, w) // 2


def live_clock(analog_img: np.ndarray | None):
    """Draw current real time on analog clock image."""
    if analog_img is None:
        return None, "Upload an analog clock image."

    now = datetime.datetime.now()
    h, m, s = now.hour % 12, now.minute, now.second

    img_bgr = _numpy_to_bgr(analog_img)
    cx, cy, radius = _detect_clock_center(img_bgr)
    hand_style = sample_style_geometric(img_bgr, cx, cy, radius)
    output_bgr = draw_hands_on_image(img_bgr, h, m, s, hand_style=hand_style, clock_center=(cx, cy))
    return _bgr_to_rgb(output_bgr), now.strftime("%H:%M:%S")


# ── Build UI ──────────────────────────────────────────────────────────────────

_data_root = _root / "data" / "raw"
_example_pairs = [
    [
        str(_data_root / "digital" / "digital_00_19_18.png"),
        str(_data_root / "analog" / "analog_00_00_00_orange.png"),
    ],
    [
        str(_data_root / "digital" / "digital_00_46_56.png"),
        str(_data_root / "analog" / "analog_00_00_00_vue.png"),
    ],
]

with gr.Blocks(title="Digital → Analog Clock Converter") as demo:
    gr.Markdown("# Digital → Analog Clock Converter")
    gr.Markdown(
        "Upload a **digital clock** and an **analog clock**. "
        "The system will redraw the analog clock hands to match the digital time."
    )

    with gr.Tab("Convert"):
        with gr.Row():
            with gr.Column():
                digital_input = gr.Image(label="Digital Clock", type="numpy")
            with gr.Column():
                analog_input = gr.Image(label="Analog Clock (style reference)", type="numpy")
            with gr.Column():
                output_image = gr.Image(label="Output: Analog with new time", type="numpy")

        convert_btn = gr.Button("Convert", variant="primary")
        time_label = gr.Textbox(label="Detected Time", interactive=False)

        convert_btn.click(
            fn=convert,
            inputs=[digital_input, analog_input],
            outputs=[output_image, time_label],
        )

        gr.Examples(
            examples=_example_pairs,
            inputs=[digital_input, analog_input],
            label="Example inputs (click to load)",
        )

    with gr.Tab("Live Clock"):
        gr.Markdown(
            "Upload an analog clock image. The display will update every second "
            "to show the **current real time**."
        )
        with gr.Row():
            with gr.Column():
                live_analog_input = gr.Image(label="Analog Clock (style reference)", type="numpy")
                live_btn = gr.Button("Start / Refresh", variant="secondary")
            with gr.Column():
                live_output = gr.Image(label="Live Clock", type="numpy")
                live_time = gr.Textbox(label="Current Time", interactive=False)

        live_btn.click(
            fn=live_clock,
            inputs=[live_analog_input],
            outputs=[live_output, live_time],
        )

        # Auto-refresh every second using gr.Timer
        timer = gr.Timer(value=1)
        timer.tick(
            fn=live_clock,
            inputs=[live_analog_input],
            outputs=[live_output, live_time],
        )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
