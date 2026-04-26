import argparse
import json
import pathlib
import random
import time
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

CLOCKS_DIR = pathlib.Path("src/clocks")
OUT_DIGITAL = pathlib.Path("data/raw/digital")
OUT_ANALOG = pathlib.Path("data/raw/analog")
DIGITAL_HTML = CLOCKS_DIR / "digital_clock.html"
SEED = 42
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

ANALOG_STYLES = [
    ("clock_01_orange.html", "orange"),
    ("clock_02_wall.html", "wall"),
    ("clock_03_design3.html", "design3"),
    ("clock_04_watch.html", "watch"),
    ("clock_05_atc_vintage.html", "atc"),
    ("clock_06_sweet.html", "sweet"),
    ("clock_07_vue_hybrid.html", "vue"),
    ("clock_08_js30.html", "js30"),
    ("clock_09_simple.html", "simple"),
    ("clock_10_3d_dark.html", "3d"),
]

EDGE_CASES = list(dict.fromkeys([
    # CATEGORY 1 — Cardinal angles (0° / 90° / 180° / 270°)
    (12, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0),

    # CATEGORY 2 — Intermediate: minute+second hand on every clock number
    (1, 5, 5), (2, 10, 10), (4, 20, 20), (5, 25, 25),
    (7, 35, 35), (8, 40, 40), (10, 50, 50), (11, 55, 55),

    # CATEGORY 3 — Overlapping / near-overlapping hands
    (1, 1, 1),      # near-total overlap of all 3 hands
    (6, 30, 30),    # hour+minute+second close together
    (9, 45, 45),    # minute == second
    (3, 15, 15),    # minute == second at quarter
    (12, 30, 0),    # MAX separation between hour and minute

    # CATEGORY 4 — Non-round minutes/seconds (arbitrary angles)
    (2, 13, 42), (4, 47, 18), (7, 9, 56),
    (10, 22, 34), (5, 51, 7), (8, 38, 29),

    # CATEGORY 5 — 24-hour format (tests digital reader for hours > 12)
    (13, 15, 0), (15, 45, 30), (18, 0, 0),
    (20, 20, 20), (22, 10, 5), (23, 59, 59), (0, 0, 0),

    # BONUS — All 24 hour-transition moments
    *[(h, 0, 0) for h in range(24)],

    # BONUS — AM/PM boundary conditions
    (0, 0, 1), (11, 59, 59), (12, 0, 1), (23, 59, 59),
]))


def build_400_times(edge_cases, seed=SEED):
    """Build list of 400 unique (h, m, s) tuples from edge cases then random fill."""
    times = list(edge_cases[:400])
    seen = set(times)
    rng = random.Random(seed)
    while len(times) < 400:
        t = (rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
        if t not in seen:
            times.append(t)
            seen.add(t)
    return times[:400]


def crop_clock_from_page(driver, screenshot_path: Path) -> None:
    """Crop around .clock bbox with safe padding, save 400x400."""
    rect = driver.execute_script(
        """
        const el = document.querySelector('.clock');
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return {x: r.x, y: r.y, w: r.width, h: r.height, dpr: window.devicePixelRatio || 1};
        """
    )
    img = Image.open(str(screenshot_path)).convert("RGB")
    if not rect:
        w, h = img.size
        left = max((w - 400) // 2, 0)
        top = max((h - 400) // 2, 0)
        img.crop((left, top, left + 400, top + 400)).save(str(screenshot_path))
        return
    dpr = float(rect["dpr"])
    cx = (float(rect["x"]) + float(rect["w"]) / 2.0) * dpr
    cy = (float(rect["y"]) + float(rect["h"]) / 2.0) * dpr
    side = max(float(rect["w"]), float(rect["h"])) * 1.30 * dpr
    half = side / 2.0
    left = max(int(cx - half), 0)
    top = max(int(cy - half), 0)
    right = min(int(cx + half), img.size[0])
    bottom = min(int(cy + half), img.size[1])
    cropped = img.crop((left, top, right, bottom))
    cropped.resize((400, 400), resample=Image.Resampling.LANCZOS).save(
        str(screenshot_path)
    )


def capture_digital_clock(driver, screenshot_path: Path) -> None:
    """Capture the padded digital clock container after forcing a centered layout."""
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    container = WebDriverWait(driver, 5).until(
        lambda d: d.find_element(By.ID, "clock-capture")
    )
    driver.execute_script(
        """
        const el = arguments[0];
        document.documentElement.style.width = '100%';
        document.documentElement.style.height = '100%';
        document.body.style.width = '100%';
        document.body.style.height = '100%';
        document.body.style.margin = '0';
        document.body.style.display = 'grid';
        document.body.style.placeItems = 'center';
        el.scrollIntoView({block: 'center', inline: 'center'});
        """,
        container,
    )
    time.sleep(0.05)
    container.screenshot(str(screenshot_path))


def make_driver():
    """Create headless Chrome with a fixed window size."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}")
    opts.add_argument("--hide-scrollbars")
    return webdriver.Chrome(options=opts)


def shot(driver, url, path, style_label=None):
    """Load URL and capture a stable image for each clock style."""
    is_digital = "digital_clock.html" in str(url)
    wait = 0.1 if is_digital else 0.3
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    driver.get(url)
    time.sleep(wait)
    if is_digital:
        capture_digital_clock(driver, pathlib.Path(path))
        return
    _ = style_label
    driver.save_screenshot(str(path))
    crop_clock_from_page(driver, pathlib.Path(path))


def furl(p):
    """Build file:// URL for a local path."""
    return "file://" + str(pathlib.Path(p).resolve())


def parse_args() -> argparse.Namespace:
    """CLI: optional subset of analog styles (e.g. rescrape orange atc only)."""
    p = argparse.ArgumentParser(description="Scrape digital + analog clock PNGs.")
    p.add_argument(
        "--styles",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Analog style tokens to capture (default: all). Example: orange atc",
    )
    return p.parse_args()


def analog_styles_subset(names: list[str] | None) -> list[tuple[str, str]]:
    """Filter ANALOG_STYLES by label; exit if none match."""
    if not names:
        return ANALOG_STYLES
    want = set(names)
    out = [x for x in ANALOG_STYLES if x[1] in want]
    if not out:
        print(f"[ERROR] No styles in {want!r}. Valid: {[s[1] for s in ANALOG_STYLES]}")
        raise SystemExit(1)
    unknown = want - {x[1] for x in out}
    if unknown:
        print(f"[WARN] Unknown style(s) ignored: {sorted(unknown)}")
    return out


if __name__ == "__main__":
    args = parse_args()
    analog_only = analog_styles_subset(args.styles)

    OUT_DIGITAL.mkdir(parents=True, exist_ok=True)
    OUT_ANALOG.mkdir(parents=True, exist_ok=True)

    times = build_400_times(EDGE_CASES)
    n_edge = min(len(EDGE_CASES), 400)
    print(f"Times: 400 | Edge cases: {n_edge} | Random fill: {400 - n_edge}")
    print(
        f"Analog styles this run: {[s[1] for s in analog_only]} "
        f"({len(analog_only)} of {len(ANALOG_STYLES)})\n"
    )

    pathlib.Path("data").mkdir(exist_ok=True)
    with open("data/times_manifest.json", "w") as f:
        json.dump(
            [
                {"h": h, "m": m, "s": s, "is_edge": i < n_edge}
                for i, (h, m, s) in enumerate(times)
            ],
            f,
            indent=2,
        )

    driver = make_driver()
    try:
        for idx, (h, m, s) in enumerate(times, 1):
            tag = f"{h:02d}_{m:02d}_{s:02d}"

            dp = OUT_DIGITAL / f"digital_{tag}.png"
            if not dp.exists():
                shot(driver, f"{furl(DIGITAL_HTML)}?h={h}&m={m}&s={s}", dp)

            for style_file, style_label in analog_only:
                ap = OUT_ANALOG / f"analog_{tag}_{style_label}.png"
                if not ap.exists():
                    shot(
                        driver,
                        f"{furl(CLOCKS_DIR / style_file)}?h={h}&m={m}&s={s}",
                        ap,
                        style_label=style_label,
                    )

            if idx % 50 == 0 or idx == 400:
                print(f"  [{idx:>3}/400] {h:02d}:{m:02d}:{s:02d}")

        d = len(list(OUT_DIGITAL.glob("*.png")))
        a = len(list(OUT_ANALOG.glob("*.png")))
        print(f"\n[OK] DONE — Digital: {d}  Analog: {a}  Total: {d + a}")
    finally:
        driver.quit()
