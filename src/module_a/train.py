"""Train Module A: digital clock reader only."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import ClockDataset
from src.models.digital_reader import DigitalReader


def unpack_batch(batch):
    """Extract digital tensor and labels; hour is raw 0–23."""
    digital_tensor, labels, _analog_tensor = batch
    hour, minute, second = labels
    return digital_tensor, hour.long(), minute.long(), second.long()


def batch_full_match(hour_logits, minute_logits, second_logits, targets):
    """Check strict full-time match for each sample (24h hour)."""
    hour_t, minute_t, second_t = targets
    hour_ok = hour_logits.argmax(dim=1).eq(hour_t)
    minute_ok = minute_logits.argmax(dim=1).eq(minute_t)
    second_ok = second_logits.argmax(dim=1).eq(second_t)
    return hour_ok & minute_ok & second_ok


def _eval_accumulate_batch(
    model, digital, hour_t, minute_t, second_t, criterion, device
):
    """Forward one batch; return loss sum and per-head correct counts."""
    digital = digital.to(device)
    hour_t = hour_t.to(device)
    minute_t = minute_t.to(device)
    second_t = second_t.to(device)
    hour_l, minute_l, second_l = model(digital)
    loss = criterion(hour_l, hour_t)
    loss = loss + criterion(minute_l, minute_t) + criterion(second_l, second_t)
    bsz = digital.size(0)
    h_ok = hour_l.argmax(dim=1).eq(hour_t).sum().item()
    m_ok = minute_l.argmax(dim=1).eq(minute_t).sum().item()
    s_ok = second_l.argmax(dim=1).eq(second_t).sum().item()
    full_ok = batch_full_match(
        hour_l, minute_l, second_l, (hour_t, minute_t, second_t)
    ).sum().item()
    return loss.item() * bsz, bsz, h_ok, m_ok, s_ok, full_ok


def evaluate(model, loader, device, criterion):
    """Val loss, per-head accuracies, and full-time accuracy."""
    model.eval()
    total_loss = 0.0
    total = 0
    h_ok = m_ok = s_ok = full_ok = 0
    with torch.no_grad():
        for batch in loader:
            digital, hour_t, minute_t, second_t = unpack_batch(batch)
            l_sum, n, hc, mc, sc, fc = _eval_accumulate_batch(
                model, digital, hour_t, minute_t, second_t, criterion, device
            )
            total_loss += l_sum
            total += n
            h_ok += hc
            m_ok += mc
            s_ok += sc
            full_ok += fc
    n = max(total, 1)
    val_loss = total_loss / n
    hour_acc = h_ok / n
    minute_acc = m_ok / n
    second_acc = s_ok / n
    full_acc = full_ok / n
    return val_loss, hour_acc, minute_acc, second_acc, full_acc


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        digital, hour_t, minute_t, second_t = unpack_batch(batch)
        digital = digital.to(device)
        hour_t = hour_t.to(device)
        minute_t = minute_t.to(device)
        second_t = second_t.to(device)
        optimizer.zero_grad()
        hour_l, minute_l, second_l = model(digital)
        loss = criterion(hour_l, hour_t)
        loss = loss + criterion(minute_l, minute_t) + criterion(second_l, second_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * digital.size(0)
        total_samples += digital.size(0)
    return total_loss / max(total_samples, 1)


def save_model(model, save_path):
    """Save trained weights to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def _current_lr(optimizer):
    """Return LR from the first param group (after scheduler updates)."""
    return optimizer.param_groups[0]["lr"]


def main():
    """Train with AdamW, cosine LR, label smoothing, early stopping on full-acc."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"
    train_ds = ClockDataset(root_dir=str(data_root), mode="train")
    val_ds = ClockDataset(root_dir=str(data_root), mode="val")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = DigitalReader().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    save_path = (
        project_root / "models" / "checkpoints" / "digital_reader_best.pth"
    )
    best_val_acc = -1.0
    epochs_no_improve = 0
    patience = 30
    max_epochs = 500

    for epoch in range(1, max_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, h_acc, m_acc, s_acc, full_acc = evaluate(
            model, val_loader, device, criterion
        )
        scheduler.step()
        lr = _current_lr(optimizer)
        loss = avg_loss
        acc_h = h_acc * 100
        acc_m = m_acc * 100
        acc_s = s_acc * 100
        acc_full = full_acc * 100
        print(
            f"Epoch {epoch:02d} | Loss: {loss:.4f} | "
            f"H: {acc_h:.2f}% | M: {acc_m:.2f}% | S: {acc_s:.2f}% | "
            f"Full: {acc_full:.2f}% | LR: {lr:.2e}",
            flush=True,
        )
        if full_acc > best_val_acc:
            best_val_acc = full_acc
            epochs_no_improve = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_model(model, save_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(
                f"Early stop: no val full-acc gain for {patience} epochs.",
                flush=True,
            )
            break

    print(f"Done. Best val full-acc: {best_val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
