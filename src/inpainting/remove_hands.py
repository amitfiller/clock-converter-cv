"""Hand removal and optional mask compositing (legacy) utilities."""

from __future__ import annotations

import cv2
import numpy as np


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


def remove_hands(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint hand regions on a BGR image; dilate mask then TELEA."""
    mask_uint8 = _normalize_mask(mask)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=3)
    return cv2.inpaint(img_bgr, dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


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
