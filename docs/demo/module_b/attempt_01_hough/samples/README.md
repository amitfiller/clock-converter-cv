# Hough attempt — sample images / דוגמאות לכשל

**EN:** This folder should contain **3–5** verification PNGs showing **bad** Hough-style overlays (misaligned lines, edge-chasing segments). If your machine already generated them, copy **(not move)** from `demo_outputs/mask_verify/`.

**HE:** יש למלא **3–5** תמונות אימות שמראות **כשל** של זיהוי מבוסס קווים. להעתיק בלבד מ־`demo_outputs/mask_verify/`.

## Suggested filenames / שמות מומלצים

From a typical verify run, prefer styles that were problematic for edges:

- `analog_05_34_49_3d_verify.png`
- `analog_03_15_15_vue_verify.png`
- `analog_04_20_20_vue_verify.png`
- `analog_08_49_49_vue_verify.png`
- `analog_17_48_46_vue_verify.png`

**PowerShell (run from repo root):**

```powershell
$dst = "docs\demo\module_b\attempt_01_hough\samples"
$src = "demo_outputs\mask_verify"
@(
  "analog_05_34_49_3d_verify.png",
  "analog_03_15_15_vue_verify.png",
  "analog_04_20_20_vue_verify.png",
  "analog_08_49_49_vue_verify.png",
  "analog_17_48_46_vue_verify.png"
) | ForEach-Object { Copy-Item "$src\$_" "$dst\$_" -ErrorAction SilentlyContinue }
```

If files live under `demo_outputs/mask_verify/run_<timestamp>/`, adjust `$src` accordingly.
