import argparse, os, re, math, csv
from pathlib import Path

PATTERN = re.compile(r"(?P<sec>\d{9})\.(?P<nsec>\d{9})\.(png|jpg|jpeg)$", re.IGNORECASE)

def parse_ts_ms(name: str):
    m = PATTERN.search(name)
    if not m: return None
    sec = int(m.group("sec"))
    nsec = int(m.group("nsec"))
    ts_ms = sec*1000 + nsec//1_000_000
    return ts_ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="frames.csv")
    args = ap.parse_args()

    imgs = [f for f in os.listdir(args.images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    rows = []
    for name in imgs:
        ts = parse_ts_ms(name)
        if ts is not None:
            rows.append((name, ts))

    rows.sort(key=lambda x: x[1])  # ordina per timestamp

    outp = Path(args.out)
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx","frame_name","ts_ms"])
        for i,(name,ts) in enumerate(rows):
            w.writerow([i, name, ts])

    print(f"Scritti {len(rows)} frame in {outp}")

if __name__ == "__main__":
    main()
