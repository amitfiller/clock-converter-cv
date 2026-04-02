# Module A — Digital Clock Reader (Technical Summary)

This document summarizes the **Digital Clock Reader** (Module A): architecture, data pipeline, training configuration, recorded results, and key design decisions. It reflects the current implementation in `src/models/digital_reader.py`, `src/data/dataset.py`, and `src/module_a/train.py`.

**Checkpoint:** `models/checkpoints/digital_reader_best.pth` — **present** on disk (verified).

---

## 1. Architecture (Module A)

### Backbone

- **ResNet18** loaded with **ImageNet** pretrained weights (`torchvision.models.resnet18(pretrained=True)`).

### Modifications to vanilla ResNet18

- The original **fully connected classifier** is **removed** by wrapping all modules **except the last child** in a sequential backbone:
  - `self.backbone = nn.Sequential(*list(resnet.children())[:-1])`
- This keeps conv layers through **global average pooling**, producing a **512-dimensional** feature map per sample (spatially `1×1` before flatten).

### Classification heads

Shared features pass through **dropout**, then three independent linear heads:

| Head       | Layer              | Output size | Semantics        |
| ---------- | ------------------ | ----------- | ---------------- |
| `head_h`   | `Linear(512, 24)`  | 24 logits   | Hour **0–23**    |
| `head_m`   | `Linear(512, 60)`  | 60 logits   | Minute **0–59**  |
| `head_s`   | `Linear(512, 60)`  | 60 logits   | Second **0–59**  |

### Dropout

- **`nn.Dropout(p=0.3)`** applied on the flattened 512-d features **before** the three heads.

### Forward pass

- Input → backbone → `flatten` → dropout → `(out_h, out_m, out_s)` (three logit tensors).

---

## 2. Dataset & Transforms

### Data locations

- **Digital images:** `data/raw/digital/` (pattern `digital_HH_MM_SS.png`).
- **Analog images:** `data/raw/analog/` (paired by timestamp; used for the paired dataset; Module A training uses the digital tensor and labels only).
- Training script sets `root_dir` to **`data/raw`**, so digital dir resolves to **`data/raw/digital`**.

### Train vs validation split

- **80% / 20%** split over **unique timestamp samples**, with **shuffle seed 42** (`random.Random(42)`).
- `mode="train"` → train subset; `mode="val"` → validation subset.

### Train transforms (digital, augmentation)

Applied when `mode == "train"` (unless a custom `digital_transform` is passed):

- `Resize((224, 224))`
- `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)`
- `RandomRotation(degrees=5)`
- `RandomAffine(degrees=0, translate=(0.05, 0.05))`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (ImageNet statistics)

### Validation transforms (digital, no augmentation)

- `Resize((224, 224))`
- `ToTensor()`
- Same **ImageNet** `Normalize` as training.

### Labels

- Tuple **`(H, M, S)`** parsed from the digital filename.
- **H:** **0–23** (24-hour, no modulo on hour in the dataset or in `unpack_batch` for training).
- **M, S:** **0–59** each.

### Optional override

- `ClockDataset(..., digital_transform=...)` can replace the default train/val pipelines (e.g. for tests or ablations).

---

## 3. Training Setup

| Item | Configuration |
| ---- | --------------- |
| **Optimizer** | **AdamW** — `lr=1e-4`, `weight_decay=1e-4` |
| **Scheduler** | **CosineAnnealingLR** — `T_max=100`, `eta_min=1e-6`; **`scheduler.step()` once per epoch after validation** |
| **Loss** | **`nn.CrossEntropyLoss(label_smoothing=0.1)`** — same criterion summed over all three heads |
| **Batch size** | 32 (train & val loaders) |
| **Max epochs** | 500 (upper cap) |
| **Early stopping** | **Patience = 30** epochs **without improvement** in validation **full accuracy** (H, M, and S all correct) |
| **Checkpoint** | Best weights saved only when val **full accuracy** improves → **`models/checkpoints/digital_reader_best.pth`** |

Per-epoch log format (validation metrics as percentages):  
`Epoch XX | Loss: … | H: …% | M: …% | S: …% | Full: …% | LR: …`

---

## 4. Training Results

Training ran for **112 epochs** before early stopping.

| Metric | Value |
| ------ | ----- |
| **Best checkpoint** | `models/checkpoints/digital_reader_best.pth` |
| **Best validation full accuracy** | **80.00%** |
| **Early stop** | No validation full-accuracy improvement for **30** consecutive epochs |

### Final epoch (Epoch 112)

| Metric | Value |
| ------ | ----- |
| Loss (train avg) | 2.2983 |
| H Acc (val) | 100.00% |
| M Acc (val) | 86.25% |
| S Acc (val) | 90.00% |
| Full (val) | 76.25% |

---

## 5. Key Decisions & Changes (This Branch)

- **Backbone:** Replaced the earlier **custom CNN** with **ResNet18** pretrained on ImageNet for stronger visual features on 224×224 RGB inputs.
- **Hour head:** Corrected from **12** to **24** classes for true **0–23** hours; removed any **`hour % 12`** usage in the training label path.
- **Regularization:** **`Dropout(0.3)`** before the classification heads.
- **Optimization:** Switched to **AdamW** with **`weight_decay=1e-4`**.
- **Learning rate:** **Cosine annealing** over a 100-epoch period (`T_max=100`), with floor **`eta_min=1e-6`**.
- **Loss:** **Label smoothing `0.1`** on cross-entropy for each head (stabilizes logits on small datasets).

---

*Generated from the current Module A source files and the training run described above.*
