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
from src.inpainting.remove_hands import remove_hands
from src.models.digital_reader import DigitalReader
from src.models.hand_segmenter import HandSegmenter


class ClockConverterPipeline:
    """Digital clock → time read → analog hands replaced at predicted time."""

    def __init__(self, module_a_path: str | Path, module_b_path: str | Path, device: str = "cpu") -> None:
        """Load Module A/B checkpoints and move models to device."""
        self.device = torch.device(device)
        self.reader = DigitalReader().to(self.device)
        a_state = torch.load(module_a_path, map_location=self.device, weights_only=False)
        self.reader.load_state_dict(a_state)
        self.segmenter = HandSegmenter().to(self.device)
        b_ck = torch.load(module_b_path, map_location=self.device, weights_only=False)
        self.segmenter.load_state_dict(b_ck["model_state_dict"])
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

    def run(self, digital_img_path: str | Path, analog_img_path: str | Path) -> dict:
        """Segment at 64², inpaint and draw hands at original analog resolution (BGR)."""
        analog_bgr = cv2.imread(str(analog_img_path), cv2.IMREAD_COLOR)
        if analog_bgr is None:
            raise FileNotFoundError(f"Cannot read analog image: {analog_img_path}")
        orig_h, orig_w = analog_bgr.shape[:2]
        digital_tensor = self._preprocess_digital(digital_img_path)
        h, m, s = self.reader.predict(digital_tensor)
        analog_tensor = self._analog_batch_64(analog_img_path)
        hand_mask_tensor = self.segmenter.predict(analog_tensor)
        hand_mask_np = hand_mask_tensor[0, 0].detach().cpu().numpy()
        mask_2d = hand_mask_np[:, :] if hand_mask_np.ndim == 2 else hand_mask_np[:, :, 0]
        mask_full = cv2.resize(
            mask_2d.astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        clean_analog = remove_hands(analog_bgr, mask_full)
        output_image = draw_hands_on_image(clean_analog, h, m, s)
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
