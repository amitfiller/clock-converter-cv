"""Training script for hand segmentation model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import SegmentationDataset
from src.models.hand_segmenter import HandSegmenter


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
    """Train U-Net on analog/mask pairs with validation tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    analog_dir = project_root / "data" / "raw" / "analog"
    mask_dir = project_root / "data" / "masks"

    train_ds = SegmentationDataset(analog_dir, mask_dir, mode="train")
    val_ds = SegmentationDataset(analog_dir, mask_dir, mode="val")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = HandSegmenter().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    save_path = project_root / "models" / "checkpoints" / "hand_segmenter_best.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    patience_counter = 0
    early_stop_patience = 3
    early_stop_threshold = 0.90
    max_epochs = 100

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0
        train_dice_sum = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
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
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_dice_sum = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = bce_loss(logits, masks) + dice_loss(logits, masks)
                val_loss_sum += loss.item()
                val_iou_sum += compute_iou(logits, masks)
                val_dice_sum += compute_dice(logits, masks)

        val_loss = val_loss_sum / len(val_loader)
        val_iou = val_iou_sum / len(val_loader)
        val_dice = val_dice_sum / len(val_loader)

        print(f"Epoch {epoch}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics | Train IoU: {train_iou:.4f} Dice: {train_dice:.4f} | Val IoU: {val_iou:.4f} Dice: {val_dice:.4f}")

        if val_iou > best_iou:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_dice": val_dice,
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

    print(f"Training complete. Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
