"""Offline evaluation for Module A (DigitalReader) on digital clock images."""

import argparse
import glob
import os
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from src.models.digital_reader import DigitalReader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIGITAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@dataclass
class EvalResult:
    """Counts for per-head and full-time matches."""

    total: int
    correct_h: int
    correct_m: int
    correct_s: int
    correct_full: int

    @property
    def acc_h(self) -> float:
        return self.correct_h / self.total if self.total > 0 else 0.0

    @property
    def acc_m(self) -> float:
        return self.correct_m / self.total if self.total > 0 else 0.0

    @property
    def acc_s(self) -> float:
        return self.correct_s / self.total if self.total > 0 else 0.0

    @property
    def acc_full(self) -> float:
        return self.correct_full / self.total if self.total > 0 else 0.0


def load_model(checkpoint_path: str = "models/checkpoints/digital_reader_best.pth") -> DigitalReader:
    """Load checkpoint and return model in eval mode on DEVICE."""
    model = DigitalReader().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_time_from_filename(path: str):
    """
    Parse (h, m, s) from basename.
    Supports digital_HH_MM_SS.png or HH_MM_SS.* with _ - : separators.
    """
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    for sep in ["_", "-", ":"]:
        parts = name.split(sep)
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            h, m, s = map(int, parts)
            return h, m, s
        if (
            len(parts) == 4
            and parts[0].lower() == "digital"
            and all(p.isdigit() for p in parts[1:4])
        ):
            h, m, s = map(int, parts[1:4])
            return h, m, s
    raise ValueError(f"Could not parse time from filename: {path}")


@torch.no_grad()
def evaluate_on_images(model: DigitalReader, image_paths) -> EvalResult:
    """Run model on paths; skip files with unparsable names."""
    result = EvalResult(
        total=0,
        correct_h=0,
        correct_m=0,
        correct_s=0,
        correct_full=0,
    )
    for path in image_paths:
        try:
            true_h, true_m, true_s = parse_time_from_filename(path)
        except ValueError:
            continue
        img = Image.open(path).convert("RGB")
        x = DIGITAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        out_h, out_m, out_s = model(x)
        pred_h = out_h.argmax(dim=1).item()
        pred_m = out_m.argmax(dim=1).item()
        pred_s = out_s.argmax(dim=1).item()
        result.total += 1
        if pred_h == true_h:
            result.correct_h += 1
        if pred_m == true_m:
            result.correct_m += 1
        if pred_s == true_s:
            result.correct_s += 1
        if pred_h == true_h and pred_m == true_m and pred_s == true_s:
            result.correct_full += 1
    return result


def find_digital_images(root: str, limit: int | None = None):
    """Return sorted image paths under root, optionally truncated."""
    patterns = [
        os.path.join(root, "**", "*.png"),
        os.path.join(root, "**", "*.jpg"),
        os.path.join(root, "**", "*.jpeg"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    paths = sorted(paths)
    if limit is not None:
        paths = paths[:limit]
    return paths


def _print_eval_summary(result: EvalResult) -> None:
    """Print accuracy table for an EvalResult."""
    print("\n=== Evaluation Results ===")
    print(f"Total evaluated: {result.total}")
    if result.total == 0:
        print("No valid images with parsable HH_MM_SS names were found.")
        return
    print(
        f"H accuracy:    {result.acc_h * 100:.2f}%  "
        f"({result.correct_h}/{result.total})"
    )
    print(
        f"M accuracy:    {result.acc_m * 100:.2f}%  "
        f"({result.correct_m}/{result.total})"
    )
    print(
        f"S accuracy:    {result.acc_s * 100:.2f}%  "
        f"({result.correct_s}/{result.total})"
    )
    print(
        f"Full accuracy: {result.acc_full * 100:.2f}%  "
        f"({result.correct_full}/{result.total})"
    )


def main() -> None:
    """CLI: load checkpoint, scan images, print metrics."""
    parser = argparse.ArgumentParser(
        description="Evaluate DigitalReader on digital clock images.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw/digital",
        help="Folder with digital clock images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/digital_reader_best.pth",
        help="Trained DigitalReader checkpoint.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=200,
        help="Max images to evaluate (sorted order).",
    )
    args = parser.parse_args()
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)
    print(f"[INFO] Scanning images under: {args.data_root}")
    image_paths = find_digital_images(args.data_root, limit=args.num_images)
    print(
        f"[INFO] Found {len(image_paths)} image files (before filename filtering).",
    )
    if not image_paths:
        print("[ERROR] No images found. Adjust --data-root.")
        return
    result = evaluate_on_images(model, image_paths)
    _print_eval_summary(result)


if __name__ == "__main__":
    main()
