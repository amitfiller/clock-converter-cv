# Attempt 02 — Geometric masks / ניסוי 2 — מסכות גיאומטריות

## English

### Why we switched

Hough on raw pixels was **not stable** across **10 analog themes**. We already know the **ground-truth time** for every training PNG from its filename (`analog_HH_MM_SS_<style>.png`), so we can **synthesize** consistent masks.

### How it works

For an image of size `H×W`, let the center be `(cx, cy)` and a radius fraction `r`. Hand angles (degrees, **0° = 3 o’clock** style math with −90° offset to match clock convention in code):

- **Hour:** `(hh % 12) * 30 + mm * 0.5 - 90`
- **Minute:** `mm * 6 - 90`
- **Second:** `ss * 6 - 90`

Each hand is drawn as a **thick line** from center toward a tip at a **length ratio** (hour shorter than minute, second longest). Then **Gaussian blur + threshold** softens edges to better match anti-aliased renders.

Implementation: `scripts/generate_masks.py` → `generate_geometric_mask`.

### Why this is “perfect GT” for our CSS clocks

The dataset analog faces are **programmatically rendered** with predictable geometry. The mask generator uses the **same angular model**, so labels are **consistent** across all styles for U-Net training.

### Samples / דוגמות

See [samples/](samples/README.md). Prefer copies from the latest `demo_outputs/mask_verify/run_<timestamp>/` after regenerating masks; if you only have loose PNGs in `mask_verify/`, copy **clean** overlays from simpler styles (**simple**, **wall**).

---

## עברית

### למה עברנו גישה

הופ על פיקסלים גולמיים היה **לא יציב** בין עשרת הסגנונות. הזמן **האמיתי** ידוע משם הקובץ, ולכן אפשר **לסנתז** מסכות עקביות.

### איך זה עובד

זוויות אנליטיות לשעה/דקה/שנייה, קווים עבים מהמרכז החוצה, יחסי אורך שונים לכל יד, ואז **טשטוש גאוסי + סף** כדי לדמות קצוות רכים. הקוד ב־`scripts/generate_masks.py`.

### למה זה GT “מושלם” לשעוני ה־CSS שלנו

התמונות נוצרות **באופן פרוגרמטי** — המודל הגיאומטרי תואם את איך שהזמן מוצג בדאטה.

### דוגמות

ראו [samples/](samples/README.md) — מומלץ להעתיק מ־`demo_outputs/mask_verify/run_<timestamp>/` אחרי הרצת `verify_masks.py --save`; אם אין תיקיית `run_*`, השתמשו בתמונות “נקיות” מסגנונות פשוטים.
