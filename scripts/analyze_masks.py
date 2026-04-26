"""Aggregate mask generation CSV: per-style heuristics and Hough usage."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI for mask CSV summary."""
    p = argparse.ArgumentParser(
        description="Summarize data/mask_debug/run_*.csv by style.",
    )
    p.add_argument(
        "csv_path",
        type=Path,
        help="Path to run_*.csv from generate_masks.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV for aggregated rows.",
    )
    return p.parse_args()


def main() -> None:
    """Print and optionally save per-style aggregates."""
    args = parse_args()
    path = args.csv_path
    if not path.is_file():
        print(f"[ERROR] Not found: {path}")
        return

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_style: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "sum_frac": 0.0, "hough": 0.0}
    )
    total = 0
    hough_total = 0
    for r in rows:
        st = r.get("style") or "unknown"
        by_style[st]["n"] += 1
        by_style[st]["sum_frac"] += float(r.get("frac_mask_beyond_dial", 1))
        if r.get("used_hough") == "True":
            by_style[st]["hough"] += 1
        total += 1
        if r.get("used_hough") == "True":
            hough_total += 1

    print(f"Rows: {total} | used_hough: {hough_total} ({100*hough_total/max(total,1):.1f}%)")
    print("\nPer style: n | mean(frac_beyond_dial) | frac_hough")
    agg_rows: list[list] = []
    for st in sorted(by_style.keys(), key=lambda s: (-by_style[s]["n"], s)):
        d = by_style[st]
        n = int(d["n"])
        mean_f = d["sum_frac"] / max(n, 1)
        fh = d["hough"] / max(n, 1)
        print(f"  {st}: {n} | {mean_f:.4f} | {fh:.2f}")
        agg_rows.append([st, n, round(mean_f, 4), round(fh, 4)])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["style", "n", "mean_frac_mask_beyond_dial", "frac_used_hough"])
            w.writerows(agg_rows)
        print(f"\n[INFO] Wrote {args.out}")


if __name__ == "__main__":
    main()
