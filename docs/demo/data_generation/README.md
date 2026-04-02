# Data generation / יצירת הדאטהסט

**EN:** The training corpus is produced by **Selenium + Chrome**: for each of **400** unique times we capture **one digital** screenshot and **ten analog** crops (one per HTML style), with filenames derived from the time and style.

**HE:** קבוצת האימון נוצרת ב־**Selenium + Chrome**: לכל אחת מ־**400** השלשות זמן יש **תמונה דיגיטלית אחת** ו־**עשר תמונות אנלוגיות** (סגנון אחד לכל HTML), עם שמות קבצים לפי זמן וסגנון.

## Key paths / נתיבים

| Item | Path |
|------|------|
| Scraper | `scripts/scrape_clocks.py` |
| Validation | `scripts/validate_dataset.py` |
| Time manifest | `data/times_manifest.json` |
| HTML clocks | `src/clocks/` |

## Narrative / סיפור מפורט

See [experiments.md](experiments.md).
