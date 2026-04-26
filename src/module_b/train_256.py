"""Training script for HandSegmenter256 at 256x256 resolution (Task 5d).

Run prepare_train_256.py first to build data/train_256/.
Saves best checkpoint to checkpoints/unet_256.pth.
Logs per-epoch metrics to logs/train_256.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import SegmentationDataset, MULTI_SCALE_DEFAULT
from src.models.hand_segmenter import HandSegmenter256

# Task 1b: removed hardcoded IMG_SIZE = 256 — resolution is now sampled per-batch
# IMG_SIZE = 256
SCALES = MULTI_SCALE_DEFAULT  # [64, 128, 192, 256, 320, 400]


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Return one minus mean dice score."""
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.shape[0], -1)
        target_flat = targets.view(targets.shape[0], -1)
        intersection = (probs_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + eps) / (probs_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
        return 1 - dice.mean()


def compute_iou(pred_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU metric after sigmoid threshold."""
    pred_binary = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred_binary * targets).sum()
    union = (pred_binary + targets).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()


def compute_dice(pred_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice metric after sigmoid threshold."""
    pred_binary = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred_binary * targets).sum()
    return (2 * intersection / (pred_binary.sum() + targets.sum() + 1e-6)).item()


def main() -> None:
    """Train HandSegmenter256 on multi-scale pairs with augmentation and validation tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    analog_dir = project_root / "data" / "train_256" / "analog"
    mask_dir = project_root / "data" / "train_256" / "masks"

    if not analog_dir.exists():
        raise FileNotFoundError(
            f"{analog_dir} not found — run scripts/prepare_train_256.py first"
        )

    # Task 1b + 2c: multi-scale + augment flag; batch_size=1 required for variable resolution
    # train_ds = SegmentationDataset(analog_dir, mask_dir, mode="train", img_size=IMG_SIZE)
    # val_ds   = SegmentationDataset(analog_dir, mask_dir, mode="val",   img_size=IMG_SIZE)
    train_ds = SegmentationDataset(analog_dir, mask_dir, mode="train", scales=SCALES, augment=True)
    val_ds   = SegmentationDataset(analog_dir, mask_dir, mode="val",   scales=SCALES, augment=False)
    # Task 1b: batch_size MUST be 1 — different scales cannot be batched together
    # train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    # val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    model = HandSegmenter256().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    save_path = project_root / "checkpoints" / "unet_256.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = project_root / "logs" / "train_256.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    patience_counter = 0
    early_stop_patience = 3
    early_stop_threshold = 0.90
    max_epochs = 100

    with open(log_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_iou", "val_iou", "train_dice", "val_dice"])

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss_sum = train_iou_sum = train_dice_sum = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = bce_loss(logits, masks) + dice_loss(logits, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
                train_iou_sum += compute_iou(logits.detach(), masks)
                train_dice_sum += compute_dice(logits.detach(), masks)

            train_loss = train_loss_sum / len(train_loader)
            train_iou = train_iou_sum / len(train_loader)
            train_dice = train_dice_sum / len(train_loader)

            model.eval()
            val_loss_sum = val_iou_sum = val_dice_sum = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    loss = bce_loss(logits, masks) + dice_loss(logits, masks)
                    val_loss_sum += loss.item()
                    val_iou_sum += compute_iou(logits, masks)
                    val_dice_sum += compute_dice(logits, masks)

            val_loss = val_loss_sum / len(val_loader)
            val_iou = val_iou_sum / len(val_loader)
            val_dice = val_dice_sum / len(val_loader)

            print(
                f"Epoch {epoch}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | "
                f"Train IoU: {train_iou:.4f} Val IoU: {val_iou:.4f}"
            )
            writer.writerow([epoch, train_loss, val_loss, train_iou, val_iou, train_dice, val_dice])
            csv_file.flush()

            if val_iou > best_iou:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_iou": val_iou,
                        "val_dice": val_dice,
                        "scales": SCALES,  # Task 1b: replaced img_size with scales list
                    },
                    save_path,
                )
                best_iou = val_iou

            if val_iou >= early_stop_threshold:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stop at epoch {epoch}: IoU {val_iou:.4f} >= {early_stop_threshold}")
                    break
            else:
                patience_counter = 0

            scheduler.step(val_iou)

    print(f"Training complete. Best Val IoU: {best_iou:.4f}  checkpoint: {save_path}")


if __name__ == "__main__":
    main()
