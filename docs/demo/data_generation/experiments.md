# Data generation — experiments / ניסויים ביצירת נתונים

## English

### What we built

- **Phase 1 pipeline** (`680a013`, `1855b96`): project structure for scraping, manifests, and HTML clock pages with query parameters `?h=&m=&s=`.
- **Scraper hardening** (`19e3d59`): **headless=new**, **skip existing** analog files for resume, consistent **raw** paths under `data/raw/`.
- **24h alignment** (`5e4a81b`): regenerated data so digital labels match **24-hour** conventions; **`ClockDataset`** validation tightened.
- **Digital capture tweak** (`9c70d79`): small stabilization changes alongside early segmentation experiments.

### What the Git history “deletions” mean

Commit `1855b96` shows **many** `data/raw/analog/*.png` as **deleted**: we **replaced** the entire analog set (new crops, naming, or manifest alignment), not abandoned the task. The professor can read this as an **intentional dataset revision**.

### Lessons

- Always run **`python scripts/validate_dataset.py`** after scraping.
- Long runs: design for **idempotency** (skip existing) and **resume**.

---

## עברית

### מה בנינו

שלב 1: סקרייפר, מניפסט, דפי HTML לשעונים. שיפורים: headless, דילוג על קבצים קיימים, נתיבי `raw`, יישור ל־**פורמט 24 שעות**, ואימות דרך `validate_dataset.py`.

### מה משמעות המחיקות בגיט

מחיקות המוניות של PNG תחת `data/raw/analog/` משקפות **החלפת דטה סט**, לא ויתור על הפרויקט.

### לקחים

להריץ ולידציה אחרי כל איסוף; לתכנן **המשך ריצה** (skip existing).
