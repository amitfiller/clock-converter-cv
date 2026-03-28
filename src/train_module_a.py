"""Train Module A: digital clock reader only."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from data.dataset import ClockDataset
    from models.digital_reader import DigitalReader
except ModuleNotFoundError:
    from src.data.dataset import ClockDataset
    from src.models.digital_reader import DigitalReader


def unpack_batch(batch):
    """Extract digital tensor and label tuple; ignore analog."""
    digital_tensor, labels, _analog_tensor = batch
    hour, minute, second = labels
    return digital_tensor, hour.long(), minute.long(), second.long()


def batch_full_match(hour_logits, minute_logits, second_logits, targets):
    """Check strict full-time match for each sample."""
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


def main():
    """Run Module A training with scheduler and best-checkpoint saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "raw"
    train_ds = ClockDataset(root_dir=str(data_root), mode="train")
    val_ds = ClockDataset(root_dir=str(data_root), mode="val")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = DigitalReader().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    save_path = project_root / "models" / "digital_reader.pth"
    best_val_acc = -1.0
    max_epochs = 150

    for epoch in range(1, max_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, h_acc, m_acc, s_acc, full_acc = evaluate(
            model, val_loader, device, criterion
        )
        old_lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler.step(val_loss)
        new_lrs = [g["lr"] for g in optimizer.param_groups]
        if new_lrs != old_lrs:
            print(f"  LR reduced: {old_lrs[0]:.2e} -> {new_lrs[0]:.2e}", flush=True)
        print(
            f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | "
            f"H: {h_acc:.2f} | M: {m_acc:.2f} | S: {s_acc:.2f} | "
            f"Full: {full_acc:.4f}",
            flush=True,
        )
        if full_acc > best_val_acc:
            best_val_acc = full_acc
            save_model(model, save_path)
        if full_acc >= 0.95:
            print("✅ Target reached! Model saved.")
            return

    print(f"Training done after {max_epochs} epochs. Best val full-acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
