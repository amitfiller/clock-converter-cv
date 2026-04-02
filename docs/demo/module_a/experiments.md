# Module A — experiments / ניסויים

## 1. Model / מודל

**EN:** **ResNet18** (ImageNet weights) as backbone; global-pooled features → **Dropout(0.3)** → three linear heads: **24** (hour), **60** (minute), **60** (second).

**HE:** **ResNet18** (משקולות ImageNet) כגב; אחרי pooling גלובלי → **Dropout(0.3)** → שלושה ראשים לינאריים: **24**, **60**, **60**.

---

## 2. Training setup / הגדרות אימון

**EN:**

- Optimizer: **AdamW** (`lr=1e-4`, `weight_decay=1e-4`)
- Scheduler: **CosineAnnealingLR** (`T_max=100`, `eta_min=1e-6`), stepped once per epoch
- Loss: **CrossEntropyLoss** with **label smoothing 0.1** (summed over the three heads)
- **Early stopping:** patience **30** epochs without improvement in validation **full** accuracy (H, M, S all correct)
- Batch size **32**; max epochs **500** (cap)

**HE:** AdamW, קוסינוס, CrossEntropy עם label smoothing 0.1, עצירה מוקדמת לפי שיפור ב־full accuracy על הולידציה, batch 32.

---

## 3. Dataset / נתונים

**EN:** **400** digital images (`digital_HH_MM_SS.png`). **80/20** train/val split over unique timestamps, seed **42** (`ClockDataset`).

**HE:** 400 תמונות דיגיטליות, חלוקה 80/20 לפי חותמת זמן עם seed 42.

---

## 4. `eval_module_a_val.py` output / פלט הערכה

**EN:** Run from repo root (with `venv` activated and `pip install -r requirements.txt`):

```bash
python scripts/eval_module_a_val.py
```

**Paste terminal output below** (placeholder until you run locally):

```
(Module A — validation: run manually and paste output here)
```

**HE:** אם הסביבה חסרה חבילות (`torchvision` וכו'), התקינו מ־`requirements.txt` והריצו את הפקודה למעלה, והדביקו את הפלט כאן.

### Reference metrics (training doc) / ערכי עזר מתוך תיעוד האימון

**EN:** For comparison, `docs/module_a_summary.md` records one full training run: **best validation full accuracy 80%**; final logged epoch showed **H 100%**, **M 86.25%**, **S 90%** on val (epoch 112). Your `eval_module_a_val.py` numbers should match the **saved best checkpoint** on the same split.

**HE:** לצורך השוואה, ב־`docs/module_a_summary.md` מתועד אימון מלא: full accuracy טובה ביותר **80%**; באפוק האחרון: **H 100%**, **M 86.25%**, **S 90%**. פלט הסקריפט אמור להתאים לצ'קפוינט השמור על אותה חלוקה.

---

## 5. Issues encountered / בעיות שעלו

**EN:**

- **Small label space:** Only **400** unique times; minute/second heads are harder than hour; **label smoothing** and **augmentation** (color jitter, light affine/rotation on digital crops) help but do not remove ambiguity on some fonts.
- **Minute accuracy bottleneck:** Training log in `module_a_summary.md` shows minute head lagging hour; full-time accuracy requires all three heads correct.
- **Environment:** Eval requires **torchvision** aligned with your **torch** build; use the project `requirements.txt` inside a venv.

**HE:** מרחב קטן של 400 זמנים; ראש הדקות הוא צוואר בקבוק; נדרשות סביבת Python תקינה עם torchvision.
