"""Dataset for analog clock hand segmentation."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

MULTI_SCALE_DEFAULT = [64, 128, 192, 256, 320, 400]


class SegmentationDataset(Dataset):
    """Load paired analog images and hand masks."""

    def __init__(
        self,
        analog_dir: str,
        mask_dir: str,
        mode: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
        img_size: int = 64,       # kept for backward compat; ignored when scales is set
        scales: list[int] | None = None,   # Task 1a: multi-scale list
        augment: bool = True,              # Task 2b: toggle augmentation
    ) -> None:
        """Build split and augmentation setup."""
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        self.mode = mode
        # Task 1a: scales overrides img_size; fall back to single-scale when scales not given
        # self.img_size = img_size
        self.scales = scales if scales is not None else [img_size]
        self.augment = augment  # Task 2b
        self.analog_dir = Path(analog_dir)
        self.mask_dir = Path(mask_dir)
        # Task 2a: stronger ColorJitter (was brightness=0.2, contrast=0.2)
        # self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2)
        self.color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        self.samples = self._build_split(val_split=val_split, seed=seed)

    def _build_split(self, val_split: float, seed: int) -> list[tuple[Path, Path]]:
        """Collect pairs and split to train or val."""
        pairs: list[tuple[Path, Path]] = []
        for analog_path in sorted(self.analog_dir.glob("*.png")):
            mask_path = self.mask_dir / analog_path.name
            if mask_path.exists():
                pairs.append((analog_path, mask_path))
        if not pairs:
            raise FileNotFoundError("No paired analog/mask PNG files found.")
        rng = random.Random(seed)
        rng.shuffle(pairs)
        split_idx = int(len(pairs) * (1.0 - val_split))
        if self.mode == "train":
            return pairs[:split_idx]
        return pairs[split_idx:]

    def _load_pair(self, idx: int) -> tuple[Image.Image, Image.Image]:
        """Read one RGB image and one grayscale mask."""
        analog_path, mask_path = self.samples[idx]
        analog_img = Image.open(analog_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        return analog_img, mask_img

    def _apply_spatial_aug(self, analog_img: Image.Image, mask_img: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Apply synchronized spatial transforms to both image and mask (Task 2a)."""
        if random.random() < 0.5:
            analog_img = TF.hflip(analog_img)
            mask_img = TF.hflip(mask_img)
        if random.random() < 0.3:
            analog_img = TF.vflip(analog_img)
            mask_img = TF.vflip(mask_img)
        # Task 2a: random rotation +-5 degrees
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            analog_img = TF.rotate(analog_img, angle)
            mask_img = TF.rotate(mask_img, angle)
        # Task 2a: random translation +-5% of image size
        if random.random() > 0.5:
            translate_x = random.uniform(-0.05, 0.05) * analog_img.width
            translate_y = random.uniform(-0.05, 0.05) * analog_img.height
            analog_img = TF.affine(analog_img, angle=0, translate=[translate_x, translate_y], scale=1, shear=0)
            mask_img = TF.affine(mask_img, angle=0, translate=[translate_x, translate_y], scale=1, shear=0)
        return analog_img, mask_img

    def _apply_color_aug(self, analog_img: Image.Image) -> Image.Image:
        """Apply image-only color and blur augmentation (Task 2a)."""
        analog_img = self.color_jitter(analog_img)
        sigma = random.uniform(0.1, 1.5)
        analog_img = TF.gaussian_blur(analog_img, kernel_size=3, sigma=sigma)
        return analog_img

    # Task 2a: old method kept for reference
    # def _apply_train_aug(self, analog_img, mask_img):
    #     if random.random() < 0.5: analog_img = TF.hflip(analog_img); mask_img = TF.hflip(mask_img)
    #     if random.random() < 0.3: analog_img = TF.vflip(analog_img); mask_img = TF.vflip(mask_img)
    #     analog_img = self.color_jitter(analog_img)
    #     return analog_img, mask_img

    def _to_tensors(self, analog_img: Image.Image, mask_img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert already-resized PIL images to normalized tensors.
        Resize is now done in __getitem__ per-sampled scale (Task 1a).
        """
        # Task 1a: resize moved to __getitem__; kept here commented for reference
        # analog_img = TF.resize(analog_img, [self.img_size, self.img_size], ...)
        # mask_img   = TF.resize(mask_img,   [self.img_size, self.img_size], ...)
        analog_tensor = TF.to_tensor(analog_img)
        analog_tensor = TF.normalize(analog_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        mask_tensor = TF.to_tensor(mask_img)
        mask_tensor = (mask_tensor >= 0.5).to(dtype=torch.float32)
        return analog_tensor, mask_tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one image tensor and one mask tensor at a randomly sampled scale."""
        analog_img, mask_img = self._load_pair(idx)
        # Task 1a: pick random scale per sample
        scale = random.choice(self.scales)
        analog_img = TF.resize(analog_img, [scale, scale], interpolation=InterpolationMode.BILINEAR)
        mask_img = TF.resize(mask_img, [scale, scale], interpolation=InterpolationMode.NEAREST)
        # Task 2b: apply augmentation only when self.augment=True (training set)
        # Old: if self.mode == "train": self._apply_train_aug(...)
        if self.augment:
            analog_img, mask_img = self._apply_spatial_aug(analog_img, mask_img)
            analog_img = self._apply_color_aug(analog_img)
        return self._to_tensors(analog_img, mask_img)

    def __len__(self) -> int:
        """Return split size."""
        return len(self.samples)


if __name__ == "__main__":
    ds = SegmentationDataset("data/raw/analog", "data/masks", mode="train")
    print(f"Train samples: {len(ds)}")
    img, mask = ds[0]
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Mask shape:  {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask unique values: {mask.unique()}")
