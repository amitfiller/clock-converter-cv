import pathlib

OUT_DIGITAL  = pathlib.Path("data/raw/digital")
OUT_ANALOG   = pathlib.Path("data/raw/analog")
STYLE_LABELS = ["orange","wall","design3","watch","atc","sweet","vue","js30","simple","3d"]

EDGE_CATEGORIES = {
    "1_cardinal":     [(12,0,0),(3,0,0),(6,0,0),(9,0,0)],
    "2_intermediate": [(1,5,5),(2,10,10),(4,20,20),(5,25,25),
                       (7,35,35),(8,40,40),(10,50,50),(11,55,55)],
    "3_overlap":      [(1,1,1),(6,30,30),(9,45,45),(3,15,15),(12,30,0)],
    "4_non_round":    [(2,13,42),(4,47,18),(7,9,56),(10,22,34),(5,51,7),(8,38,29)],
    "5_24h_format":   [(13,15,0),(15,45,30),(18,0,0),(20,20,20),
                       (22,10,5),(23,59,59),(0,0,0)],
}

def check():
    print("=" * 52)
    print("DATASET VALIDATION")
    print("=" * 52)

    d_files = list(OUT_DIGITAL.glob("digital_*.png"))
    a_files = list(OUT_ANALOG.glob("analog_*.png"))
    print(f"Digital : {len(d_files):>4}  (expected 400)")
    print(f"Analog  : {len(a_files):>4}  (expected 4000)")

    times = set()
    for f in d_files:
        p = f.stem.split("_")
        if len(p) == 4:
            times.add((int(p[1]), int(p[2]), int(p[3])))

    missing = []
    for h, m, s in times:
        tag = f"{h:02d}_{m:02d}_{s:02d}"
        for label in STYLE_LABELS:
            if not (OUT_ANALOG / f"analog_{tag}_{label}.png").exists():
                missing.append(f"analog_{tag}_{label}.png")

    print(f"\nStyle completeness:")
    if missing:
        print(f"  ✗ Missing {len(missing)} files. First 5:")
        for x in missing[:5]: print(f"    - {x}")
    else:
        print(f"  ✓ All 10 styles present for all {len(times)} times")

    print(f"\nEdge case coverage:")
    all_ok = True
    for cat, cases in EDGE_CATEGORIES.items():
        found = sum(1 for t in cases if t in times)
        ok = found == len(cases)
        if not ok: all_ok = False
        print(f"  {'✓' if ok else '✗'} {cat}: {found}/{len(cases)}")
        if not ok:
            for t in cases:
                if t not in times:
                    print(f"      MISSING {t[0]:02d}:{t[1]:02d}:{t[2]:02d}")

    if times:
        hours = [t[0] for t in times]
        print(f"\nHour range : {min(hours):02d} → {max(hours):02d}  (expected 00→23)")
    print(f"Unique times: {len(times)}")
    print("=" * 52)
    verdict = len(d_files)==400 and len(a_files)==4000 and all_ok
    print("✓ ALL PASSED — ready for training" if verdict else "✗ FAILED — re-run scrape_clocks.py")
    print("=" * 52)

if __name__ == "__main__":
    check()
