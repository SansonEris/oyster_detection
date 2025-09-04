import argparse, pandas as pd, numpy as np

def load_frames(path):
    df = pd.read_csv(path)
    # frame_idx, frame_name, ts_ms
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df

def load_gps(path):
    df = pd.read_csv(path)
    # ts_ms, lat_deg, lon_deg, alt_m, rel_alt_m, hdg_deg, vx_mps, vy_mps, vz_mps ...
    if "ts_ms" not in df.columns:
        # fallback se il campo si chiama diverso
        raise ValueError("gps.csv deve contenere la colonna ts_ms")
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df

def aggregate_detections(path):
    det = pd.read_csv(path)
    # atteso: per-detection: frame, side, class_id, ... (dal tuo pipeline)
    # facciamo un conteggio per frame e per side
    if "frame" not in det.columns:
        raise ValueError("detections.csv deve avere la colonna 'frame' (indice frame)")
    # totale per frame
    cnt_total = det.groupby("frame").size().rename("oysters_total")
    # per side se presente
    if "side" in det.columns:
        pivot = det.pivot_table(index="frame", columns="side", aggfunc="size", fill_value=0)
        pivot.columns = [f"oysters_{c}" for c in pivot.columns]  # oysters_left/right
        agg = pd.concat([cnt_total, pivot], axis=1).reset_index()
    else:
        agg = cnt_total.reset_index()
    agg = agg.rename(columns={"frame":"frame_idx"})
    return agg
def interpolate_time(df, tcol="ts_ms", cols=("lat_deg","lon_deg","alt_m","rel_alt_m","hdg_deg","vx_mps","vy_mps","vz_mps")):
    df2 = df.set_index(pd.to_datetime(df[tcol], unit='ms'))
    # interpolazione time-based solo sulle colonne numeriche presenti
    cols = [c for c in cols if c in df2.columns]
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
    ap.add_argument("--offset-ms", type=int, default=0, help="Offset da aggiungere ai ts dei frame (+ in ms)")
    args = ap.parse_args()

    frames = load_frames(args.frames)
    gps = load_gps(args.gps)

    # applica offset ai frame se necessario
    if args.offset_ms != 0:
        frames["ts_ms"] = frames["ts_ms"] + args.offset_ms

    # opzionale: interpolazione per rendere il GPS “denso”
    if args.method == "interp":
        gpsi = interpolate_time(gps)
    else:
        gpsi = gps.copy()

    # merge_asof (nearest) con tolleranza
    frames = frames.sort_values("ts_ms")
    gpsi = gpsi.sort_values("ts_ms")

    merged = pd.merge_asof(
        frames, gpsi,
        on="ts_ms", direction="nearest", tolerance=pd.Timedelta(milliseconds=args.tolerance_ms)
    )

    # Se alcuni frame rimangono senza gps (NaN) e stai usando interp, ripeti senza tolleranza per spalmare la stima
    if args.method == "interp":
        na = merged["lat_deg"].isna()
        if na.any():
            merged.loc[na, ["lat_deg","lon_deg","alt_m","rel_alt_m","hdg_deg","vx_mps","vy_mps","vz_mps"]] = \
                pd.merge_asof(frames[na], gpsi, on="ts_ms", direction="nearest")[["lat_deg","lon_deg","alt_m","rel_alt_m","hdg_deg","vx_mps","vy_mps","vz_mps"]].values

    # aggiungi conteggi ostriche se disponibile
    if args.detections:
        det = aggregate_detections(args.detections)
        merged = merged.merge(det, how="left", on="frame_idx")
        for c in merged.columns:
            if c.startswith("oysters_"):
                merged[c] = merged[c].fillna(0).astype(int)

    # ordina colonne in modo carino
    cols_order = [c for c in ["frame_idx","frame_name","ts_ms","lat_deg","lon_deg","alt_m","rel_alt_m","hdg_deg","vx_mps","vy_mps","vz_mps","oysters_total","oysters_left","oysters_right"] if c in merged.columns]
    rest = [c for c in merged.columns if c not in cols_order]
    merged = merged[cols_order + rest]

    merged.to_csv(args.out, index=False)
    print(f"Salvato {args.out} con {len(merged)} righe.")

if __name__ == "__main__":
    main()
