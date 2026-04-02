# Module B — experiments / ניסויים

## Timeline / ציר זמן

**EN:**

1. **Hough attempt (`9c70d79`, `visualize_edges.py`):** After Canny edge detection, **Probabilistic Hough** finds line segments; heuristics pick segments near the clock center and map them to hands. Fast to prototype, no labels needed.
2. **Pain points:** Rim ticks, shadows, and long **border lines** produce segments that **reach the image edge**; **Vue / 3D** styles add clutter so **hour vs minute** lines are confused. Not reliable enough for 4000-style training labels.
3. **Geometric attempt (`generate_masks.py`):** Our analog clocks are **rendered in HTML/CSS**; the **true time** is encoded in `analog_HH_MM_SS_<style>.png`. We draw three thick rays from the image center at analytic angles (hour includes minute drift), **blur + re-threshold** to mimic soft strokes, and save masks to `data/masks/`.
4. **Verification:** `scripts/verify_masks.py` saves side-by-side overlays under `demo_outputs/mask_verify/` (optionally `run_<timestamp>/`).

**HE:** קודם ניסינו Hough על קנטים — מהיר אך שביר. עברנו למסכות גיאומטריות לפי שם הקובץ, מתאים לשעוני CSS עם זמן ידוע.

## Why geometric is “perfect GT” here / למה GT הגיאומטרי מתאים כאן

**EN:** The training analog images are **synthetic from the same clock geometry** we encode in the mask script (center, radius ratios, hand thickness). The model learns to match **our** render, not arbitrary real-world clocks.

**HE:** התמונות נוצרות מהדפדפן באותה לוגיקה גיאומטרית שבה מחושבים הזוויות — לכן המסכה מתאימה כ־GT לפרויקט הזה.

## Open risks / סיכונים

**EN:** Geometric masks assume the **visual hand** aligns with **ideal** angles; decorative hands or non-standard CSS could add **domain gap**. U-Net must still **generalize** to styles with texture and lighting.

**HE:** אם סגנון ויזואלי סוטה מהאידיאל, עשוי להיווצר פער בין המסכה לתמונה.
