"""Dataset loader for paired digital and analog clocks."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _digital_train_transform() -> transforms.Compose:
    """Augmented pipeline for training digital crops (ImageNet normalize)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2
            ),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _digital_val_transform() -> transforms.Compose:
    """Deterministic pipeline for validation digital crops."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class ClockDataset(Dataset):
    """Load paired clock images with split by timestamps."""

    STYLE_LABELS = [
        "orange",
        "wall",
        "design3",
        "watch",
        "atc",
        "sweet",
        "vue",
        "js30",
        "simple",
        "3d",
    ]

    def __init__(
        self,
        root_dir: str = "data",
        mode: str = "train",
        digital_transform: Optional[Callable] = None,
    ) -> None:
        """Init dataset; optional digital_transform overrides train/val pipes."""
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        self.root_dir = Path(root_dir)
        self.mode = mode
        default_root = Path("data").resolve()
        if self.root_dir.resolve() == default_root:
            self.digital_dir = Path("data/raw/digital")
            self.analog_dir = Path("data/raw/analog")
        else:
            self.digital_dir = self.root_dir / "digital"
            self.analog_dir = self.root_dir / "analog"
        self._pattern = re.compile(r"^digital_(\d{2})_(\d{2})_(\d{2})\.png$")

        self.all_samples = self._collect_paired_samples()
        self.train_samples, self.val_samples = self._split_samples(self.all_samples)
        self.samples = self.train_samples if mode == "train" else self.val_samples

        if digital_transform is not None:
            self.digital_transform = digital_transform
        else:
            self.digital_transform = (
                _digital_train_transform()
                if mode == "train"
                else _digital_val_transform()
            )
        self.analog_transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )

    def _collect_paired_samples(self) -> List[Dict]:
        """Parse digital filenames and keep only paired timestamps."""
        samples = []
        for file_path in sorted(self.digital_dir.glob("digital_*.png")):
            match = self._pattern.match(file_path.name)
            if not match:
                continue
            hour, minute, second = map(int, match.groups())
            stamp = f"{hour:02d}_{minute:02d}_{second:02d}"
            analog_candidates = [
                self.analog_dir / f"analog_{stamp}_{self.STYLE_LABELS[0]}.png",
                self.analog_dir / f"analog_{stamp}.png",
            ]
            analog_path = next((p for p in analog_candidates if p.exists()), None)
            if analog_path is not None:
                samples.append(
                    {
                        "digital_path": file_path,
                        "analog_path": analog_path,
                        "label": (hour, minute, second),
                    }
                )
        if not samples:
            raise FileNotFoundError("No paired clock images were found.")
        return samples

    def _split_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split unique timestamps into train and validation sets."""
        rng = random.Random(42)
        shuffled = samples[:]
        rng.shuffle(shuffled)
        split_idx = int(0.8 * len(shuffled))
        return shuffled[:split_idx], shuffled[split_idx:]

    def __len__(self) -> int:
        """Return number of samples in selected mode."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return digital tensor, time label tuple, and analog tensor."""
        sample = self.samples[idx]
        hour, minute, second = sample["label"]
        tag = f"{hour:02d}_{minute:02d}_{second:02d}"
        with Image.open(sample["digital_path"]) as digital_img:
            digital_tensor = self.digital_transform(digital_img.convert("RGB"))
        if self.mode == "train":
            style = random.choice(self.STYLE_LABELS)
        else:
            style = self.STYLE_LABELS[0]
        analog_path = self.analog_dir / f"analog_{tag}_{style}.png"
        if not analog_path.exists():
            analog_path = sample["analog_path"]
        with Image.open(analog_path) as analog_img:
            analog_tensor = self.analog_transform(analog_img.convert("RGB"))
        return digital_tensor, sample["label"], analog_tensor

    def get_stats(self) -> None:
        """Print dataset sizes and time ranges."""
        timestamps = [s["label"] for s in self.all_samples]
        hours = [t[0] for t in timestamps]
        minutes = [t[1] for t in timestamps]
        seconds = [t[2] for t in timestamps]

        print(f"Total timestamps: {len(self.all_samples)}")
        print(f"Train count: {len(self.train_samples)}")
        print(f"Val count: {len(self.val_samples)}")
        print(f"Hour range: {min(hours)}-{max(hours)}")
        print(f"Minute range: {min(minutes)}-{max(minutes)}")
        print(f"Second range: {min(seconds)}-{max(seconds)}")


if __name__ == "__main__":
    train_ds = ClockDataset(mode="train")
    val_ds = ClockDataset(mode="val")

    print("Train dataset stats:")
    train_ds.get_stats()
    print("\nValidation dataset stats:")
    val_ds.get_stats()

    digital_tensor, label, analog_tensor = train_ds[0]
    print("\nOne train sample:")
    print(f"Digital shape: {digital_tensor.shape}")
    print(f"Label: {label}")
    print(f"Analog shape: {analog_tensor.shape}")
