# Project demo documentation / תיעוד הדגמה לפרויקט

**EN:** This folder is the defense-ready story of ClockWise: what we built, what we tried and dropped, and how data and models evolved.

**HE:** תיקייה זו מסכמת את מסע הפיתוח של ClockWise עבור ההגנה: מה בנינו, מה ניסינו וויתרנו עליו, ואיך התפתחו הנתונים והמודלים.

---

## Project overview / סקירת הפרויקט

**EN:** ClockWise is a PyTorch + OpenCV pipeline that reads the time from a **digital** clock image and (in the full design) renders that time on an **analog** clock. The training set is capped at **400 unique (hour, minute, second)** triplets; each triplet appears with multiple analog styles for richer supervision downstream.

**HE:** ClockWise הוא צינור עיבוד (PyTorch + OpenCV) שקורא את השעה מתמונת שעון **דיגיטלי** ובעיצוב המלא מציג אותה על שעון **אנלוגי**. קבוצת האימון מוגבלת ל־**400 שלשות (שעה, דקה, שנייה)** ייחודיות; לכל שלשה יש מספר סגנונות אנלוגיים לצורך למידה עתידית.

---

## Pipeline (target architecture) / צינור (ארכיטקטורת יעד)

```
INPUT: [Digital Clock Image] + [Analog Clock Image]
         |                          |
         v                          v
   [Network 1]                 [Network 2]
   ResNet18                    U-Net Segmentation
   Reads digits                Finds hand pixels
         |                          |
         v                          v
      (H,M,S)               [Binary Hand Mask]
         |                          |
         +----------+---------------+
                    v
             [Network 3]
         Inpainting + Geometric Draw
                    |
                    v
         OUTPUT: [Analog Clock with correct time]
```

**HE:** רשת 1 קוראת זמן מהדיגיטלי; רשת 2 מפיקה מסכת ידיים; רשת 3 משלימה ציור/השלמה לתמונה אנלוגית סופית (חלק מהצינור עדיין בפיתוח).

---

## Module status / סטטוס מודולים

| Module | EN | HE |
|--------|----|----|
| **Data generation** | Selenium scraper, 400 digital + 4000 analog (10 styles), manifest + validation scripts | איסוף Selenium, 400 דיגיטלי ו־4000 אנלוגי, מניפסט וסקריפטי אימות |
| **Module A** | ResNet18, 3 heads; trained with AdamW, cosine LR, label smoothing, early stopping | ResNet18 עם שלושה ראשים; אימון עם AdamW, קוסינוס, label smoothing, עצירה מוקדמת |
| **Module B (hands)** | Attempt 1: Canny + Hough (fragile). Attempt 2: geometric masks from filename time for U-Net targets | ניסוי 1: Canny+Hough. ניסוי 2: מסכות גיאומטריות לפי שם הקובץ כ־GT לאימון U-Net |
| **Module C (full render)** | Inpainting / composite pipeline stubbed in repo | השלמת תמונה — עדיין שכבת ממשק/מקום בקוד |

---

## Folder index / אינדקס תיקיות

| Path | Content |
|------|---------|
| [module_a/](module_a/README.md) | Digital reader journey |
| [module_b/](module_b/README.md) | Hand detection / masks journey |
| [data_generation/](data_generation/README.md) | Dataset scraping and fixes |

---

## Git History Summary

*EN + HE — סיכום מ־`git log --oneline --all` ו־`--name-status` (דגימה אחרונה).*

**EN (summary of `git log --oneline --all` and recent `--name-status`):**

- **Bootstrap:** Initial project setup, `.gitignore`, Cursor rules, then Phase 1 **data generation**: scraper, validation pipeline, HTML clocks under `src/clocks/`.
- **Dataset refresh:** Commits aligned the scraper (headless, skip existing, raw paths) and **regenerated** analog/digital assets; large batches of `data/raw/analog/*.png` show as **deleted in Git history** when the set was replaced (e.g. after **24h format** alignment and manifest fixes)—the history reflects **do-overs**, not missing work.
- **Module A:** Added ResNet-based `DigitalReader`, training/eval/predict paths, checkpoint `digital_reader_best.pth`, and docs; duplicate commit messages appear around a merge/rebase window. Training code moved from `src/train_module_a.py` into **`src/module_a/train.py`** with API paths grouped under `src/module_a/`.
- **Module B exploration:** Introduced **`src/segmentation/visualize_edges.py`** (Canny + **Hough** hand isolation) and scraper/digital HTML tweaks in the same phase; later commits **save edges + hand masks** for debugging and evolved visualization.
- **Noise / cleanup:** Minor chores (line endings, `.gitkeep` on raw folders). Early file **`data/validation_sample.png`** was removed when structure moved to `data/raw/`.

**HE:** ההיסטוריה מראה הקמה, איסוף נתונים בשלב 1, החלפות מלאות של תמונות האנלוגי (מחיקות המוניות בקומיטים = דטה סט חדש), הוספת מודול A (ResNet + אימון), ואז ניסוי סגמנטציה עם Canny/Hough וקובץ `visualize_edges.py` שעודכן לשמירת מסכות לדיבוג.

---

*Last doc update: demo folder scaffold for project defense.*
