"""Phase 1.1 - Selenium scraper for offline clock dataset generation."""

import random
import tempfile
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


DIGITAL_HTML = """<!doctype html>
<html lang="en"><head><meta charset="UTF-8"><title>Digital Clock</title>
<style>
body { margin: 0; width: 100vw; height: 100vh; display: grid; place-items: center; background: #000; }
#clock { color: #00ff66; font: 700 52px/1.1 monospace; letter-spacing: 4px; white-space: nowrap; }
</style></head>
<body>
  <div id="clock">00:00:00</div>
  <script>
  function pad(n) { return String(n).padStart(2, '0'); }
  function set_time(h, m, s) {
    document.getElementById('clock').textContent = pad(h) + ':' + pad(m) + ':' + pad(s);
  }
  </script>
</body></html>"""


# יוצר רשימה אקראית של זמנים ייחודיים / Generate unique random time tuples.
def generate_time_combinations(n=400):
    all_times = [(h, m, s) for h in range(1, 13) for m in range(60) for s in range(60)]
    if n > len(all_times):
        raise ValueError(f"Requested {n} times, but only {len(all_times)} are possible.")
    return random.sample(all_times, n)


# מגדיר דרייבר כרום ב-headless / Configure headless Chrome Selenium driver.
def setup_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=600,600")
    options.add_argument("--disable-gpu")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=1")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


# טוען HTML מקומי, מגדיר שעה, ושומר צילום מרכזי 300x300 / Set time via JS and save center crop screenshot.
def capture_clock(driver, html_path, hour, minute, second, save_path):
    driver.get(Path(html_path).resolve().as_uri())
    driver.execute_script("set_time(arguments[0], arguments[1], arguments[2]);", hour, minute, second)
    time.sleep(0.5)
    png_data = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_data)).convert("RGB")
    width, height = image.size
    left = max((width - 300) // 2, 0)
    top = max((height - 300) // 2, 0)
    cropped = image.crop((left, top, left + 300, top + 300))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cropped.save(save_path)


# מצייר שעון אנלוגי ב-Pillow ללא Selenium / Draw analog clock with Pillow only.
def draw_analog_clock(h, m, s, size=300):
    from PIL import Image, ImageDraw
    import math

    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = int(size * 0.44)  # clock radius with padding

    # Clock border circle
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline='black', width=6)

    # Tick marks for 12/3/6/9
    for angle_deg in [0, 90, 180, 270]:
        angle = math.radians(angle_deg)
        x1 = cx + int((r-18) * math.sin(angle))
        y1 = cy - int((r-18) * math.cos(angle))
        x2 = cx + int(r * math.sin(angle))
        y2 = cy - int(r * math.cos(angle))
        draw.line([x1, y1, x2, y2], fill='black', width=5)

    # Hour hand
    hour_angle = math.radians((h % 12) * 30 + m * 0.5)
    hx = cx + int(r * 0.5 * math.sin(hour_angle))
    hy = cy - int(r * 0.5 * math.cos(hour_angle))
    draw.line([cx, cy, hx, hy], fill='#111111', width=8)

    # Minute hand
    min_angle = math.radians(m * 6 + s * 0.1)
    mx2 = cx + int(r * 0.75 * math.sin(min_angle))
    my2 = cy - int(r * 0.75 * math.cos(min_angle))
    draw.line([cx, cy, mx2, my2], fill='#333333', width=5)

    # Second hand
    sec_angle = math.radians(s * 6)
    sx2 = cx + int(r * 0.82 * math.sin(sec_angle))
    sy2 = cy - int(r * 0.82 * math.cos(sec_angle))
    draw.line([cx, cy, sx2, sy2], fill='#cc0000', width=2)

    # Center dot
    draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill='black')

    return img


# מפעיל את כל תהליך יצירת הדאטה / Orchestrate full screenshot generation pipeline.
def main():
    random.seed(42)
    times = generate_time_combinations(n=400)
    base_dir = Path(__file__).resolve().parents[1]
    digital_dir = base_dir / "data" / "raw" / "digital"
    analog_dir = base_dir / "data" / "raw" / "analog"
    digital_dir.mkdir(parents=True, exist_ok=True)
    analog_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        digital_html = Path(tmp) / "digital_clock.html"
        digital_html.write_text(DIGITAL_HTML, encoding="utf-8")
        driver = setup_driver()
        try:
            for idx, (h, m, s) in enumerate(times, start=1):
                stamp = f"{h:02d}_{m:02d}_{s:02d}"
                capture_clock(driver, digital_html, h, m, s, digital_dir / f"digital_{stamp}.png")
                analog_img = draw_analog_clock(h, m, s)
                analog_img.save(analog_dir / f"analog_{stamp}.png")
                if idx % 50 == 0:
                    print(f"Progress: {idx}/400 time combinations captured.")
        finally:
            driver.quit()
    print("Done: created 400 digital + 400 analog screenshots.")


if __name__ == "__main__":
    main()
