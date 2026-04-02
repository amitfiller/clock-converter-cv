# Attempt 01 — Hough Transform / ניסוי 1 — הופ

## English

### What Hough does here

We run **Canny** edge detection, then the **Probabilistic Hough Transform** to detect **line segments** in the edge image. Long segments passing near the **clock center** are treated as candidate **hands**.

### Why we tried it first

- No manual labeling: fully **classical CV**.
- Fast feedback on a few analog styles.

### What failed

- **Spurious lines** follow the **dial rim**, **ticks**, or **image border**, so detected lines often **shoot to the edge** instead of stopping at hand tips.
- **Hand confusion:** similar-length segments make **hour vs minute** assignment unstable.
- **Style sensitivity:** **Vue** and **3D / dark** themes add **noise, gradients, and shadows**, weakening edges and breaking the line model.

### Samples / דוגמות

See [samples/](samples/README.md). Copy **3–5 bad** verify PNGs from `demo_outputs/mask_verify/` (e.g. `*_vue_*`, `*_3d_*`) into `samples/` for the defense deck.

---

## עברית

### מה עושה הופ כאן

מזהים קנטים ב־**Canny** ואז מחפשים **קטעי קו** עם **Hough הסתברותי**. קטעים ארוכים שעוברים ליד **מרכז השעון** נחשבים מועמדים ל**ידיים**.

### למה ניסינו קודם

- בלי תיוג ידני — CV קלאסי בלבד.
- משוב מהיר על כמה סגנונות.

### מה נכשל

- **קווים מזויפים** על **היקף**, **סימוני דקות**, או **גבול התמונה** — הקווים “בורחים” לשולי התמונה.
- **בלבול בין ידיים** (שעה מול דקה).
- **רגישות לסגנון:** **Vue** ו־**3D** מוסיפים רעש וצללים שמקלקלים את הקנטים.

### דוגמות

ראו [samples/](samples/README.md) — יש להעתיק לכאן תמונות אימות שמדגימות כשל (למשל קבצים עם `vue` או `3d` בשם).
