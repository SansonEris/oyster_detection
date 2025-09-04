#!/usr/bin/env python3
import argparse, os, sys, glob
import pandas as pd
import numpy as np

def robust_read_csv(path):
    """Legge CSV provando separatori comuni."""
    import pandas as pd
    last_err = None
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            # Se ha 1 colonna e il nome contiene separatori, riprova
            if df.shape[1] == 1 and any(ch in str(df.columns[0]) for ch in [";","\t","|"]):
                continue
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Impossibile leggere CSV '{path}': {last_err}")

def normalize_detections(df):
    """
    Rende 'dets' compatibile con il parser:
      - rinomina alias: frame->frame_idx, class_id->class, confidence->conf
      - garantisce colonne minime: frame_idx, class, conf, x1,y1,x2,y2, side
      - se side manca o è vuoto, prova a inferirlo da 'frame_name/path/filename', altrimenti 'mono'
      - NON rimuove nessuna colonna extra (tutte le colonne del CSV reale restano)
    """
    import numpy as np
    
    # Rinominare alias comuni
    ren = {}
    if "frame" in df.columns and "frame_idx" not in df.columns: 
        ren["frame"] = "frame_idx"
    if "class_id" in df.columns and "class" not in df.columns: 
        ren["class_id"] = "class"
    if "confidence" in df.columns and "conf" not in df.columns: 
        ren["confidence"] = "conf"
    
    df = df.rename(columns=ren)

    # Colonne minime
    for col in ["frame_idx","class","conf","x1","y1","x2","y2"]:
        if col not in df.columns:
            df[col] = np.nan

    # side: se non c'è, creala; se vuota, inferisci
    if "side" not in df.columns:
        df["side"] = np.nan

    if df["side"].isna().any():
        # Cerca colonne che potrebbero contenere info sul side
        side_cols = [c for c in df.columns if c.lower() in ("frame_name","path","file","filename")]
        
        def infer_side(row):
            txts = []
            for c in side_cols:
                try:
                    v = row[c]
                except Exception:
                    v = None
                if pd.notna(v):
                    txts.append(str(v))
            txt = " ".join(txts).lower()
            if "left" in txt or "_l" in txt or "-l" in txt:  
                return "left"
            if "right" in txt or "_r" in txt or "-r" in txt: 
                return "right"
            return np.nan
        
        mask = df["side"].isna()
        if side_cols:
            df.loc[mask,"side"] = df[mask].apply(infer_side, axis=1)
        df["side"] = df["side"].fillna("mono")

    return df

# --- FINE: helper robusti per CSV detection eterogenei ---

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
        df = robust_read_csv(p)  # Usa il parser robusto
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
    
    # Normalizza le colonne per renderle compatibili
    df = normalize_detections(df)
    
    # Usa sempre 'frame_idx' dopo la normalizzazione
    frame_col = "frame_idx"
    
    # Conta totale per frame
    cnt_total = df.groupby(frame_col).size().rename("oysters_total")
    agg = cnt_total.reset_index()
    
    # Aggrega per side se presente
    if "side" in df.columns and df["side"].notna().any():
        # Crea pivot table per side
        side_counts = df.groupby([frame_col, "side"]).size().unstack(fill_value=0)
        
        # Rinomina colonne con prefisso oysters_
        side_counts.columns = [f"oysters_{col}" for col in side_counts.columns]
        side_counts = side_counts.reset_index()
        
        # Merge con il totale
        agg = pd.merge(agg, side_counts, on=frame_col, how="left")
        
        # Riempi NaN con 0 per le colonne numeriche
        for col in agg.columns:
            if col.startswith("oysters_") and col != "oysters_total":
                agg[col] = agg[col].fillna(0).astype(int)
    
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

    # ordina colonne "map-ready" + resto
    prefer = [c for c in ["frame_idx","frame_name","ts_ms","lat_deg","lon_deg","alt_m",
                          "rel_alt_m","hdg_deg","vx_mps","vy_mps","vz_mps",
                          "oysters_total","oysters_left","oysters_right","oysters_mono"] if c in merged.columns]
    other = [c for c in merged.columns if c not in prefer]
    merged = merged[prefer + other]

    try:
        merged.to_csv(args.out, index=False)
    except Exception as e:
        err(f"errore scrittura '{args.out}': {e}")

    print(f"Salvato {args.out} con {len(merged)} righe.")

if __name__ == "__main__":
    main()
