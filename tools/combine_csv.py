#!/usr/bin/env python3
import argparse, os, sys, glob
import pandas as pd
import numpy as np

def find_path(p):
    """Se p esiste lo restituisce; altrimenti cerca per basename in . e outputs/**."""
    if p and os.path.exists(p):
        return p
    if not p:
        return None
    base = os.path.basename(p)
    candidates = glob.glob(base) + glob.glob(f"outputs/**/{base}", recursive=True)
    return candidates[0] if candidates else None

def err(msg, code=2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def safe_read_csv(path, required_cols=None, label="file"):
    p = find_path(path)
    if not p:
        err(f"{label}: non trovato -> '{path}'. "
            f"Prova a passare un path valido o metti il file in ./ o outputs/**")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        err(f"{label}: errore lettura '{p}': {e}")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            err(f"{label}: colonne mancanti {missing} in '{p}'")
    return df, p

def load_frames(path):
    df, real = safe_read_csv(path, required_cols=["ts_ms","frame_idx","frame_name"], label="frames.csv")
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df, real

def load_gps(path):
    # ts_ms + almeno lat/lon
    df, real = safe_read_csv(path, required_cols=["ts_ms","lat_deg","lon_deg"], label="gps.csv")
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df, real

def aggregate_detections(path):
    df, real = safe_read_csv(path, None, label="detections.csv")
    # accetta sia 'frame' sia 'frame_idx'
    frame_col = "frame_idx" if "frame_idx" in df.columns else ("frame" if "frame" in df.columns else None)
    if frame_col is None:
        err("detections.csv: colonna 'frame_idx' o 'frame' mancante")
    cnt_total = df.groupby(frame_col).size().rename("oysters_total")
    agg = cnt_total.reset_index().rename(columns={frame_col:"frame_idx"})
    # opzionale: side
    if "side" in df.columns:
        pivot = df.pivot_table(index=frame_col, columns="side", aggfunc="size", fill_value=0)
        pivot.columns = [f"oysters_{c}" for c in pivot.columns]  # oysters_left/right
        pivot = pivot.reset_index().rename(columns={frame_col:"frame_idx"})
        agg = pd.merge(agg, pivot, on="frame_idx", how="left")
    return agg, real

def interpolate_time(df, tcol="ts_ms", cols=("lat_deg","lon_deg","alt_m","rel_alt_m","hdg_deg","vx_mps","vy_mps","vz_mps")):
    df2 = df.copy()
    # crea indice datetime solo per l'interpolazione
    dt = pd.to_datetime(df2[tcol], unit='ms')
    df2 = df2.set_index(dt)
    cols = [c for c in cols if c in df2.columns]
    if cols:
        df2[cols] = df2[cols].interpolate(method="time").ffill().bfill()
    return df2.reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--gps", required=True)
    ap.add_argument("--detections")
    ap.add_argument("--out", default="final.csv")
    ap.add_argument("--method", choices=["nearest","interp"], default="nearest")
    ap.add_argument("--tolerance-ms", type=int, default=500)
    ap.add_argument("--offset-ms", type=int, default=0)
    args = ap.parse_args()

    frames, frames_path = load_frames(args.frames)
    gps, gps_path = load_gps(args.gps)

    # offset sui frame se richiesto
    if args.offset_ms:
        frames["ts_ms"] = frames["ts_ms"].astype("int64") + int(args.offset_ms)

    # eventualmente densifica/interpola i valori GPS (non i timestamp req/resp)
    gpsi = interpolate_time(gps) if args.method == "interp" else gps.copy()

    # merge_asof su ts_ms numerico -> tolerance DEVE essere numerica
    frames = frames.sort_values("ts_ms")
    gpsi = gpsi.sort_values("ts_ms")

    try:
        merged = pd.merge_asof(
            frames, gpsi,
            on="ts_ms", direction="nearest",
            tolerance=args.tolerance_ms  # <-- numerico, compatibile con int
        )
    except Exception as e:
        err(f"merge_asof fallito: {e}")

    # aggrega detection se presente
    if args.detections:
        dets, det_path = aggregate_detections(args.detections)
        merged = merged.merge(dets, how="left", on="frame_idx")
        for c in [c for c in merged.columns if c.startswith("oysters_")]:
            merged[c] = merged[c].fillna(0).astype(int)

    # ordina colonne “map-ready” + resto
    prefer = [c for c in ["frame_idx","frame_name","ts_ms","lat_deg","lon_deg","alt_m",
                          "rel_alt_m","hdg_deg","vx_mps","vy_mps","vz_mps",
                          "oysters_total","oysters_left","oysters_right"] if c in merged.columns]
    other = [c for c in merged.columns if c not in prefer]
    merged = merged[prefer + other]

    try:
        merged.to_csv(args.out, index=False)
    except Exception as e:
        err(f"errore scrittura '{args.out}': {e}")

    print(f"Salvato {args.out} con {len(merged)} righe.")

if __name__ == "__main__":
    main()
