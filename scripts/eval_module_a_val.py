"""Run Module A on the val set; print per-head and full accuracy."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ClockDataset
from src.models.digital_reader import DigitalReader
from src.module_a.train import batch_full_match, unpack_batch


def _accumulate_batch(model, batch, device):
    """Return correct counts for one batch (h, m, s, full) and sample count."""
    digital, hour_t, minute_t, second_t = unpack_batch(batch)
    digital = digital.to(device)
    hour_t = hour_t.to(device)
    minute_t = minute_t.to(device)
    second_t = second_t.to(device)
    hour_l, minute_l, second_l = model(digital)
    h_ok = hour_l.argmax(dim=1).eq(hour_t).sum().item()
    m_ok = minute_l.argmax(dim=1).eq(minute_t).sum().item()
    s_ok = second_l.argmax(dim=1).eq(second_t).sum().item()
    full_ok = batch_full_match(
        hour_l, minute_l, second_l, (hour_t, minute_t, second_t)
    ).sum().item()
    return h_ok, m_ok, s_ok, full_ok, digital.size(0)


def eval_val_heads(model, loader, device):
    """Aggregate H/M/S/full accuracy on the loader."""
    model.eval()
    h_ok = m_ok = s_ok = full_ok = total = 0
    with torch.no_grad():
        for batch in loader:
            hc, mc, sc, fc, n = _accumulate_batch(model, batch, device)
            h_ok += hc
            m_ok += mc
            s_ok += sc
            full_ok += fc
            total += n
    n = max(total, 1)
    return h_ok / n, m_ok / n, s_ok / n, full_ok / n


def main():
    """Load best checkpoint and print val accuracies with threshold hints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = _ROOT
    ckpt = root / "models" / "checkpoints" / "digital_reader_best.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    data_root = root / "data" / "raw"
    val_ds = ClockDataset(root_dir=str(data_root), mode="val")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = DigitalReader().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    h_acc, m_acc, s_acc, full_acc = eval_val_heads(model, val_loader, device)

    print("Module A — validation (per-head + full)")
    print(f"  H_acc: {h_acc * 100:.2f}%")
    print(f"  M_acc: {m_acc * 100:.2f}%")
    print(f"  S_acc: {s_acc * 100:.2f}%")
    print(f"  Full:  {full_acc * 100:.2f}%")
    print()
    if h_acc >= 0.90 and m_acc >= 0.85:
        print("Ready for Module B (H>=90%, M>=85%).")
    elif h_acc >= 0.88 and m_acc >= 0.88:
        print("Strong heads — reasonable to start Module B.")
    elif h_acc < 0.80:
        print("Hour head <80% — consider more training or a small tweak.")
    else:
        print("Intermediate — tune or train longer before Module B if needed.")


if __name__ == "__main__":
    main()
