# Module B — Analog hands / ידיים אנלוגיות

**EN:** We need **binary hand masks** (or equivalent supervision) for a **U-Net** style segmenter. This folder documents two approaches: classical **Hough** lines on edges, then **geometric** masks derived from the known time in the filename.

**HE:** נדרשות **מסכות ידיים בינאריות** לאימון סגמנטציה (U-Net). כאן מתועדים שני מסלולים: **Hough** על קנטים, ואחר כך **מסכות גיאומטריות** לפי הזמן בשם הקובץ.

## Subfolders / תתי־תיקיות

| Path | Topic |
|------|--------|
| [attempt_01_hough/](attempt_01_hough/README.md) | Canny + Hough (first attempt, brittle) |
| [attempt_02_geometric/](attempt_02_geometric/README.md) | Filename-time geometry (stable GT for CSS clocks) |
| [experiments.md](experiments.md) | Timeline and decisions |

## Code pointers / מצביעי קוד

| Role | Path |
|------|------|
| Hough demo / edges | `src/segmentation/visualize_edges.py` |
| Geometric mask batch | `scripts/generate_masks.py` |
| Overlay QA | `scripts/verify_masks.py` |
