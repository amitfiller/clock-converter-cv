"""End-to-end pipeline: digital time → segment hands → inpaint → draw hands at full res."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from src.hand_geometry.draw_hands import draw_hands_on_image
from src.inpainting.remove_hands import _clock_face_mask, remove_hands, sample_style_geometric
from src.models.digital_reader import DigitalReader
from src.models.hand_segmenter import HandSegmenter, HandSegmenter256


class ClockConverterPipeline:
    """Digital clock → time read → analog hands replaced at predicted time."""

    def __init__(self, module_a_path: str | Path, module_b_path: str | Path, device: str = "cpu") -> None:
        """Load Module A/B checkpoints, auto-detect HandSegmenter vs HandSegmenter256."""
        self.device = torch.device(device)
        self.reader = DigitalReader().to(self.device)
        a_state = torch.load(module_a_path, map_location=self.device, weights_only=False)
        # Module A checkpoint may be raw state_dict or wrapped dict
        if isinstance(a_state, dict) and "model_state_dict" in a_state:
            a_state = a_state["model_state_dict"]
        self.reader.load_state_dict(a_state)
        b_ck = torch.load(module_b_path, map_location=self.device, weights_only=False)
        ms = b_ck.get("model_state_dict", b_ck) if isinstance(b_ck, dict) else b_ck
        if any("enc5" in k for k in ms.keys()):
            self.segmenter = HandSegmenter256().to(self.device)
        else:
            self.segmenter = HandSegmenter().to(self.device)
        self.segmenter.load_state_dict(ms)
        self.reader.eval()
        self.segmenter.eval()

    def _preprocess_digital(self, img_path: str | Path) -> torch.Tensor:
        """RGB 224×224 + ImageNet norm for DigitalReader (matches Module A training)."""
        pil = Image.open(img_path).convert("RGB")
        pil = TF.resize(pil, (224, 224), interpolation=InterpolationMode.BILINEAR)
        t = TF.to_tensor(pil)
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return t.unsqueeze(0).to(self.device)

    def _analog_batch_64(self, img_path: str | Path) -> torch.Tensor:
        """Normalized 64² tensor for HandSegmenter (training resolution)."""
        pil = Image.open(img_path).convert("RGB")
        pil64 = TF.resize(pil, [64, 64], interpolation=InterpolationMode.BILINEAR)
        t = TF.to_tensor(pil64)
        t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return t.unsqueeze(0).to(self.device)

    def _analog_batch_256(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Normalized 256² tensor for HandSegmenter256 (Task 5e).
        Takes BGR numpy array (already loaded) to avoid double file read.
        """
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        pil256 = TF.resize(pil, [256, 256], interpolation=InterpolationMode.BILINEAR)
        t = TF.to_tensor(pil256)
        t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return t.unsqueeze(0).to(self.device)

    def run(self, digital_img_path: str | Path, analog_img_path: str | Path) -> dict:
        """Segment at 64², inpaint and draw hands at original analog resolution (BGR)."""
        BYPASS_INPAINT = True
        analog_bgr = cv2.imread(str(analog_img_path), cv2.IMREAD_COLOR)
        if analog_bgr is None:
            raise FileNotFoundError(f"Cannot read analog image: {analog_img_path}")
        debug_dir = Path("demo_outputs/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "debug_01_analog_input.png"), analog_bgr)
        orig_h, orig_w = analog_bgr.shape[:2]
        clock_mask = _clock_face_mask(analog_bgr)
        # Clock center and radius from mask contour
        contours, _ = cv2.findContours(clock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            (cx_f, cy_f), r_f = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
            clock_center = (int(cx_f), int(cy_f))
            clock_radius = max(4, int(r_f))
        else:
            clock_center = (orig_w // 2, orig_h // 2)
            clock_radius = min(orig_h, orig_w) // 2
        digital_tensor = self._preprocess_digital(digital_img_path)
        h, m, s = self.reader.predict(digital_tensor)
        # Task 5e: resize input to 256x256 for HandSegmenter256 if segmenter supports it,
        # otherwise fall back to legacy 64x64 path.
        # analog_tensor = self._analog_batch_64(analog_img_path)
        if hasattr(self.segmenter, "enc5"):
            analog_tensor = self._analog_batch_256(analog_bgr)
        else:
            analog_tensor = self._analog_batch_64(analog_img_path)
        hand_mask_tensor = self.segmenter.predict(analog_tensor)
        hand_mask_np = hand_mask_tensor[0, 0].detach().cpu().numpy()
        mask_raw_vis = (np.clip(hand_mask_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "debug_02_mask_raw.png"), mask_raw_vis)
        mask_2d = hand_mask_np[:, :] if hand_mask_np.ndim == 2 else hand_mask_np[:, :, 0]
        mask_full = cv2.resize(
            mask_2d.astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_full = cv2.GaussianBlur(mask_full, (0, 0), sigmaX=1.5, sigmaY=1.5)
        mask_full = (mask_full > 0.20).astype(np.float32)
        mask_upscaled_vis = (np.clip(mask_full, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "debug_03_mask_upscaled.png"), mask_upscaled_vis)
        # Stage 1: close small gaps in mask
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_vis = cv2.morphologyEx(mask_upscaled_vis, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        # Stage 2: expand mask to cover hand edges fully
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_vis = cv2.dilate(closed_vis, kernel_dilate, iterations=3)

        # Mask sanity check before inpainting: excessive coverage triggers softer dilation
        clock_area = int(np.count_nonzero(clock_mask))
        dilated_area = int(np.count_nonzero(dilated_vis))
        coverage = (dilated_area / clock_area) if clock_area > 0 else 0.0
        if coverage > 0.35:
            print(
                f"[WARNING] Mask covers {coverage:.1%} of clock face — reducing dilation to (3,3)×1"
            )
            small_kernel = np.ones((3, 3), np.uint8)
            dilated_vis = cv2.dilate(closed_vis, small_kernel, iterations=1)

        cv2.imwrite(str(debug_dir / "debug_04_mask_dilated.png"), dilated_vis)

        # Use the (possibly adjusted) dilated mask as input mask for remove_hands
        mask_for_inpaint = (dilated_vis > 0).astype(np.float32)
        if BYPASS_INPAINT:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thin_mask = cv2.erode(mask_for_inpaint.astype(np.uint8), kernel_erode, iterations=1)
            thin_mask = cv2.dilate(thin_mask, np.ones((3, 3), np.uint8), iterations=1)
            thin_mask = (thin_mask > 0).astype(np.uint8) * 255
            # Color fallback: catch warm/golden hand pixels the segmenter missed (e.g. vue style)
            hsv_orig = cv2.cvtColor(analog_bgr, cv2.COLOR_BGR2HSV)
            golden_mask = cv2.inRange(hsv_orig, np.array([10, 40, 60], dtype=np.uint8),
                                      np.array([40, 255, 255], dtype=np.uint8))
            golden_mask = cv2.bitwise_and(golden_mask, clock_mask)
            kernel_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            golden_mask = cv2.dilate(golden_mask, kernel_g, iterations=2)
            thin_mask = cv2.bitwise_or(thin_mask, golden_mask)
            clean_analog = cv2.inpaint(analog_bgr, thin_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
            hand_style = sample_style_geometric(analog_bgr, clock_center[0], clock_center[1], clock_radius)
        else:
            clean_analog, hand_style = remove_hands(
                analog_bgr,
                mask_for_inpaint,
                clock_center=clock_center,
                clock_radius=clock_radius,
            )
        cv2.imwrite(str(debug_dir / "debug_05_inpainted.png"), clean_analog)

        # Check inpainting brightness shift inside clock circle
        face_idx = clock_mask > 0
        if np.any(face_idx):
            orig_face = cv2.cvtColor(analog_bgr, cv2.COLOR_BGR2GRAY)[face_idx]
            inpaint_face = cv2.cvtColor(clean_analog, cv2.COLOR_BGR2GRAY)[face_idx]
            delta_mean = float(np.mean(inpaint_face) - np.mean(orig_face))
            if delta_mean > 40.0:
                print(
                    f"[WARNING] Inpainting brightened clock face by {delta_mean:.1f} levels "
                    f"(too aggressive)."
                )

        # Ensure we always provide a usable hand_style
        if not hand_style:
            hand_style = {"color": (50, 50, 50), "thickness_ratio": 0.04}

        # Fix 3c: pass detected clock_center so hands radiate from the real pivot
        # output_image = draw_hands_on_image(clean_analog, h, m, s, hand_style=hand_style)
        output_image = draw_hands_on_image(clean_analog, h, m, s, hand_style=hand_style, clock_center=clock_center, clock_radius=clock_radius)
        cv2.imwrite(str(debug_dir / "debug_06_final_output.png"), output_image)
        # Task 5e: resize output back to original input resolution
        if output_image.shape[:2] != (orig_h, orig_w):
            output_image = cv2.resize(output_image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_vis = (np.clip(mask_full, 0.0, 1.0) * 255.0).astype(np.uint8)
        result = {
            "predicted_time": (h, m, s),
            "hand_mask": hand_mask_np,
            "clean_analog": clean_analog,
            "output_image": output_image,
            "clean_analog_display": clean_analog,
            "output_display": output_image,
            "hand_mask_display": mask_vis,
            "original_size": (orig_h, orig_w),
        }
        return result

    def run_batch(self, pairs_list: list[tuple[str | Path, str | Path]]) -> list[dict]:
        """Run pipeline on a list of (digital_path, analog_path) pairs."""
        return [self.run(d, a) for d, a in pairs_list]


_default_pipeline: ClockConverterPipeline | None = None


def _get_default_pipeline(device: str = "cpu") -> ClockConverterPipeline:
    """Return a cached pipeline using default checkpoint paths."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ClockConverterPipeline(
            module_a_path="models/checkpoints/digital_reader_best.pth",
            module_b_path="checkpoints/unet_256.pth",
            device=device,
        )
    return _default_pipeline


def run_pipeline(
    digital_path: str | Path,
    analog_path: str | Path,
    device: str = "cpu",
) -> np.ndarray:
    """Convenience wrapper: run end-to-end pipeline and return output BGR image."""
    pipeline = _get_default_pipeline(device)
    result = pipeline.run(digital_path, analog_path)
    return result["output_image"]
