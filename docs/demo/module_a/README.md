# Module A — Digital clock reader / קורא שעון דיגיטלי

**EN:** Trains a **ResNet18** backbone with **three classification heads** (hour 0–23, minute 0–59, second 0–59) on **400** digital crops from `data/raw/digital/`.

**HE:** מאמן **ResNet18** עם **שלושה ראשי סיווג** (שעה 0–23, דקה ושנייה 0–59) על **400** תמונות דיגיטליות מתוך `data/raw/digital/`.

## Where to look / איפה לראות

| Resource | Path |
|----------|------|
| Model | `src/models/digital_reader.py` |
| Training | `src/module_a/train.py` |
| Val eval script | `scripts/eval_module_a_val.py` |
| Long-form summary | `docs/module_a_summary.md` |
| Results folder (screenshots) | [results/](results/README.md) |
| Experiments log | [experiments.md](experiments.md) |
