"""Validate ClockWise raw dataset quality and pairing."""

import random
import re
from pathlib import Path

from PIL import Image, ImageStat


TIME_RE = re.compile(r"^(digital|analog)_(\d{2})_(\d{2})_(\d{2})\.png$")


# אוסף רק קבצי PNG מתיקייה / Collect only PNG files from a folder.
def list_png_files(folder):
    return sorted([p for p in folder.glob("*.png") if p.is_file()])


# מחלץ חותמת זמן משם קובץ / Extract HH_MM_SS timestamp from file name.
def parse_timestamp(file_path):
    match = TIME_RE.match(file_path.name)
    if not match:
        return None
    _, hh, mm, ss = match.groups()
    return f"{hh}_{mm}_{ss}"


# בודק ספירה מדויקת של 400 / Verify exact file count requirement.
def check_counts(digital_files, analog_files, issues):
    ok_digital = len(digital_files) == 400
    ok_analog = len(analog_files) == 400
    if not ok_digital:
        issues.append(f"digital count is {len(digital_files)} (expected 400)")
    if not ok_analog:
        issues.append(f"analog count is {len(analog_files)} (expected 400)")
    return ok_digital and ok_analog


# בודק זמנים ייחודיים ומזהה כפילויות / Verify unique timestamps and duplicates.
def check_unique_times(files, label, issues):
    seen = set()
    duplicates = set()
    parsed = []
    for file_path in files:
        ts = parse_timestamp(file_path)
        if ts is None:
            issues.append(f"{label} bad filename format: {file_path.name}")
            continue
        if ts in seen:
            duplicates.add(ts)
        seen.add(ts)
        parsed.append(ts)
    if duplicates:
        issues.append(f"{label} duplicate timestamps: {sorted(duplicates)}")
    if len(seen) != 400:
        issues.append(f"{label} unique timestamps count is {len(seen)} (expected 400)")
    return len(duplicates) == 0 and len(seen) == 400, seen, parsed


# בודק תקינות תמונה וגודל/בהירות / Validate image load, size, and non-blank intensity.
def inspect_image(file_path, min_mean=5):
    with Image.open(file_path) as img:
        rgb = img.convert("RGB")
        size_ok = rgb.size == (300, 300)
        mean_val = ImageStat.Stat(rgb.convert("L")).mean[0]
    not_white = mean_val <= 250
    not_black = mean_val >= min_mean
    return size_ok, mean_val, not_white and not_black


# עובר על כל התמונות ומדווח בעיות / Check all images for corruption or suspicious content.
def check_image_quality(files, label, issues):
    ok = True
    min_mean = 2 if label == "digital" else 5
    for file_path in files:
        try:
            size_ok, mean_val, intensity_ok = inspect_image(file_path, min_mean=min_mean)
        except Exception as exc:  # pylint: disable=broad-except
            ok = False
            issues.append(f"{label} corrupted image: {file_path.name} ({exc})")
            continue
        if not size_ok:
            ok = False
            issues.append(f"{label} wrong size: {file_path.name}")
        if not intensity_ok:
            ok = False
            issues.append(f"{label} suspicious mean {mean_val:.2f}: {file_path.name}")
    return ok


# בודק התאמה מלאה בין זוגות analog/digital / Ensure every timestamp has both file types.
def check_pairs(digital_ts, analog_ts, issues):
    missing_digital = sorted(list(analog_ts - digital_ts))
    missing_analog = sorted(list(digital_ts - analog_ts))
    for ts in missing_digital:
        issues.append(f"missing digital pair for timestamp {ts}")
    for ts in missing_analog:
        issues.append(f"missing analog pair for timestamp {ts}")
    return len(missing_digital) == 0 and len(missing_analog) == 0


# מדפיס טווחים של שעה/דקה/שנייה מתוך חותמות זמן / Print hour/minute/second ranges from timestamps.
def print_time_ranges(timestamps):
    parsed = [tuple(map(int, ts.split("_"))) for ts in timestamps]
    hours = [h for h, _, _ in parsed]
    minutes = [m for _, m, _ in parsed]
    seconds = [s for _, _, s in parsed]
    print(f"Hour range: {min(hours)}-{max(hours)}")
    print(f"Minute range: {min(minutes)}-{max(minutes)}")
    print(f"Second range: {min(seconds)}-{max(seconds)}")


# שומר דוגמת השוואה ויזואלית של 5 זוגות / Save 5 random [digital | analog] sample rows.
def save_visual_sample(common_timestamps, digital_dir, analog_dir, output_path, issues):
    if len(common_timestamps) < 5:
        issues.append("not enough pairs for 5-row validation sample")
        return False
    sample_ts = random.sample(sorted(common_timestamps), 5)
    canvas = Image.new("RGB", (600, 1500), color=(240, 240, 240))
    for row, ts in enumerate(sample_ts):
        digital_img = Image.open(digital_dir / f"digital_{ts}.png").convert("RGB")
        analog_img = Image.open(analog_dir / f"analog_{ts}.png").convert("RGB")
        y = row * 300
        canvas.paste(digital_img, (0, y))
        canvas.paste(analog_img, (300, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


# מפעיל את כל בדיקות האיכות ומדפיס דוח מלא / Run full dataset validation and print report.
def main():
    random.seed(42)
    root = Path(__file__).resolve().parents[1]
    digital_dir = root / "data" / "raw" / "digital"
    analog_dir = root / "data" / "raw" / "analog"
    sample_out = root / "data" / "validation_sample.png"
    issues = []

    digital_files = list_png_files(digital_dir)
    analog_files = list_png_files(analog_dir)
    count_ok = check_counts(digital_files, analog_files, issues)
    unique_d_ok, digital_ts, _ = check_unique_times(digital_files, "digital", issues)
    unique_a_ok, analog_ts, _ = check_unique_times(analog_files, "analog", issues)
    quality_d_ok = check_image_quality(digital_files, "digital", issues)
    quality_a_ok = check_image_quality(analog_files, "analog", issues)
    pairs_ok = check_pairs(digital_ts, analog_ts, issues)
    sample_ok = save_visual_sample(digital_ts & analog_ts, digital_dir, analog_dir, sample_out, issues)

    print("\n=== DATASET VALIDATION REPORT ===")
    print(f"Count check: {'✅' if count_ok else '❌'}")
    print(f"Unique times check (digital): {'✅' if unique_d_ok else '❌'}")
    print(f"Unique times check (analog): {'✅' if unique_a_ok else '❌'}")
    print(f"Image quality check (digital): {'✅' if quality_d_ok else '❌'}")
    print(f"Image quality check (analog): {'✅' if quality_a_ok else '❌'}")
    print(f"Pairs check: {'✅' if pairs_ok else '❌'}")
    print(f"Visual sample check: {'✅' if sample_ok else '❌'}")
    print(f"Validation sample output: {sample_out}")
    if digital_ts:
        print_time_ranges(digital_ts)

    if issues:
        print("\nIssues:")
        for idx, issue in enumerate(issues, start=1):
            print(f"{idx}. {issue}")
    print(f"\nTotal issues found: {len(issues)}")


if __name__ == "__main__":
    main()
