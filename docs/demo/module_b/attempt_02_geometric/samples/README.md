# Geometric masks — sample images / דוגמאות מסכה טובות

**EN:** Place **3–5** verify PNGs where the **red overlay** aligns with the **true hands** (geometric targets). Prefer the newest batch under `demo_outputs/mask_verify/run_<timestamp>/` created by:

```bash
python scripts/verify_masks.py --save --sample 20
```

**HE:** יש להעתיק **3–5** תמונות אימות עם **התאמה טובה** בין המסכה לידיים. אם נוצרה תיקיית `run_*`, העדיפו אותה.

## Suggested filenames (simple / wall styles) / שמות מומלצים

If `run_*` is empty, these usually show **clean** geometric alignment:

- `analog_06_27_50_simple_verify.png`
- `analog_06_00_00_wall_verify.png`
- `analog_21_40_39_simple_verify.png`
- `analog_13_38_04_simple_verify.png`
- `analog_12_24_38_wall_verify.png`

**PowerShell (run from repo root):**

```powershell
$dst = "docs\demo\module_b\attempt_02_geometric\samples"
$src = "demo_outputs\mask_verify"
# Prefer latest run folder if present:
$run = Get-ChildItem "$src\run_*" -Directory | Sort-Object Name -Descending | Select-Object -First 1
if ($run) { $src = $run.FullName }
@(
  "analog_06_27_50_simple_verify.png",
  "analog_06_00_00_wall_verify.png",
  "analog_21_40_39_simple_verify.png",
  "analog_13_38_04_simple_verify.png",
  "analog_12_24_38_wall_verify.png"
) | ForEach-Object { Copy-Item "$src\$_" "$dst\$_" -ErrorAction SilentlyContinue }
```
