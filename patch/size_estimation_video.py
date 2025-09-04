#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo size estimation (video-enabled)
- Carica calibrazione da 'stereo_calibration_data.pkl' (chiavi in stile CalibrationCamera.py)
- Usa le mappe di rettifica salvate (map1_l/map2_l/map1_r/map2_r) se presenti
- Stima disparità -> Z via reprojectImageTo3D(Q)
- Stima dimensioni reali (in unità di T/Q: nel tuo dataset sono centimetri)
- INPUT: due video stereo (o due device index), processati frame-by-frame

Nota: ho mantenuto la struttura e la logica del tuo script; ho sostituito soltanto
la lettura di immagini statiche con la lettura di due sorgenti video.
"""

import os, sys, pickle
import numpy as np
import cv2 as cv

# --------------------------
# Utility per caricamento pickle robusto (compat "numpy._core")
# --------------------------
def load_pickle_compat(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        # shim per vecchi pickle che puntano a numpy._core
        import types, numpy as _np
        core_mod = types.ModuleType("numpy._core")
        core_mod.__dict__.update(_np.core.__dict__)
        sys.modules["numpy._core"] = core_mod
        sys.modules["numpy._core.multiarray"] = _np.core.multiarray
        sys.modules["numpy._core.numerictypes"] = _np.core.numerictypes
        sys.modules["numpy._core.overrides"] = _np.core.overrides
        sys.modules["numpy._core.fromnumeric"] = _np.core.fromnumeric
        try:
            sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath  # può non esistere in alcune build
        except Exception:
            pass
        with open(path, "rb") as f:
            return pickle.load(f)

# --------------------------
# Helper per recuperare la prima chiave disponibile
# --------------------------
def get_any(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def to_mat(m, shape=None, dtype=np.float64):
    if m is None: return None
    arr = np.array(m, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr

def ensure_1d(a):
    if a is None: return None
    a = np.asarray(a, dtype=np.float64)
    return a.ravel()

# --------------------------
# 0) Parametri / Percorsi
# --------------------------
PKL_PATH = "stereo_calibration_data.pkl"

# Imposta qui le sorgenti video (file o device index)
# Esempi:
#   LEFT_SRC = 0 ; RIGHT_SRC = 1         # due webcam
#   LEFT_SRC = "left_video.mp4"          # file video sinistro
#   RIGHT_SRC = "right_video.mp4"        # file video destro
LEFT_SRC = "left_test_2.mp4"
RIGHT_SRC = "right_test_2.mp4"

# Modello YOLO (facoltativo). Se non presente, lo script funziona ma senza detection.
YOLO_MODEL = "models/best_yolo8.pt"  # usa il tuo modello; in fallback prova yolo11n.pt

# --------------------------
# 1) Caricamento calibrazione
# --------------------------
if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"File di calibrazione non trovato: {PKL_PATH}")

data = load_pickle_compat(PKL_PATH)

# Supporta entrambe le convenzioni di chiavi
KL = to_mat(get_any(data, "KL", "camera_matrix_l"), (3,3))
KR = to_mat(get_any(data, "KR", "camera_matrix_r"), (3,3))
DL = ensure_1d(get_any(data, "DL", "dist_coeffs_l"))
DR = ensure_1d(get_any(data, "DR", "dist_coeffs_r"))
R  = to_mat(get_any(data, "R"), (3,3))
T  = ensure_1d(get_any(data, "T"))
# Rettifica/proiezioni/Q
RL = to_mat(get_any(data, "RL", "R1"))
RR = to_mat(get_any(data, "RR", "R2"))
PL = to_mat(get_any(data, "PL", "P1"))
PR = to_mat(get_any(data, "PR", "P2"))
Q  = to_mat(get_any(data, "Q"), (4,4))

# Mappe
map1_l = get_any(data, "map1_l", "mapLx")
map2_l = get_any(data, "map2_l", "mapLy")
map1_r = get_any(data, "map1_r", "mapRx")
map2_r = get_any(data, "map2_r", "mapRy")

# Size (fallback se necessario, ma per video non è strettamente richiesto)
size = data.get("size")
if size is None:
    if map1_l is not None and hasattr(map1_l, "shape"):
        h, w = map1_l.shape[:2]
        size = (w, h)
    else:
        size = (640, 480)
else:
    size = tuple(map(int, size))

# Validazioni minime
for name, m, shp in [
    ("KL", KL, (3,3)),
    ("KR", KR, (3,3)),
    ("DL", DL, None),
    ("DR", DR, None),
    ("R",  R,  (3,3)),
    ("T",  T,  (3,)),
    ("Q",  Q,  (4,4))
]:
    if m is None:
        raise ValueError(f"Parametro '{name}' assente nel pickle.")
    if shp is not None and tuple(m.shape) != shp:
        raise ValueError(f"Parametro '{name}' ha shape {m.shape} ma atteso {shp}")

# Assicura rettifica/mappe se assenti
have_maps = (map1_l is not None and map2_l is not None and map1_r is not None and map2_r is not None)
if not have_maps:
    flags = cv.CALIB_ZERO_DISPARITY
    RL, RR, PL, PR, Q, roiL, roiR = cv.stereoRectify(
        KL, DL, KR, DR, size, R, T, flags=flags, alpha=0
    )
    map1_l, map2_l = cv.initUndistortRectifyMap(KL, DL, RL, PL, size, cv.CV_32FC1)
    map1_r, map2_r = cv.initUndistortRectifyMap(KR, DR, RR, PR, size, cv.CV_32FC1)

# Info
fx_rect = float(PL[0,0])
fy_rect = float(PL[1,1])
baseline_units = float(np.linalg.norm(T))
units_label = "cm" if baseline_units > 1.0 else "m"
print(f"[INFO] size={size}, fx={fx_rect:.2f}, fy={fy_rect:.2f}, baseline≈{baseline_units:.2f} {units_label}")

# --------------------------
# 2) Apertura sorgenti video
# --------------------------
def open_cap(src):
    # consente sia index int sia stringa path
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv.VideoCapture(src)
    return cap

capL = open_cap(LEFT_SRC)
capR = open_cap(RIGHT_SRC)
if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError(f"Impossibile aprire le sorgenti video: {LEFT_SRC} / {RIGHT_SRC}")

# --------------------------
# 3) Stereo matcher
# --------------------------
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,    # multiplo di 16
    blockSize=5,
    P1=8*3*5*5,
    P2=32*3*5*5,
    speckleWindowSize=50,
    speckleRange=32,
    disp12MaxDiff=1,
    uniquenessRatio=10
)

# --------------------------
# 4) YOLO (opzionale)
# --------------------------
yolo_model = None
try:
    from ultralytics import YOLO
    if os.path.exists(YOLO_MODEL):
        yolo_model = YOLO(YOLO_MODEL)
        print(f"[INFO] Caricato modello YOLO: {YOLO_MODEL}")
    else:
        print("[WARN] Modello YOLO personalizzato non trovato; uso 'yolo11n.pt' predefinito.")
        yolo_model = YOLO("yolo11n.pt")
except Exception as e:
    print(f"[WARN] YOLO non disponibile ({e}). Procedo senza detection.")

def median_depth_from_mask(mask_u8, Z):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0: return None
    vals = Z[ys, xs]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else None

# --------------------------
# 5) Loop frame-by-frame
# --------------------------
while True:
    okL, imgL = capL.read()
    okR, imgR = capR.read()
    if not okL or not okR:
        break

    # Rettifica
    rectL = cv.remap(imgL, np.asarray(map1_l), np.asarray(map2_l), cv.INTER_LINEAR)
    rectR = cv.remap(imgR, np.asarray(map1_r), np.asarray(map2_r), cv.INTER_LINEAR)

    # Disparità -> Z
    grayL = cv.cvtColor(rectL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(rectR, cv.COLOR_BGR2GRAY)
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    points3D = cv.reprojectImageTo3D(disp, Q)  # X,Y,Z nelle unità di T/Q
    Z = points3D[..., 2]

    out = rectL.copy()

    # Detection (se YOLO è disponibile)
    if yolo_model is not None:
        try:
            results = yolo_model.predict(rectL, verbose=False)
            for r in results:
                if r.masks is not None:
                    for mask, box in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                        mask_resized = cv.resize((mask*255).astype(np.uint8),
                                                 (rectL.shape[1], rectL.shape[0]))
                        Z_med = median_depth_from_mask(mask_resized, Z)
                        if Z_med is None: 
                            continue
                        ys, xs = np.where(mask_resized > 0)
                        w_px = xs.max() - xs.min(); h_px = ys.max() - ys.min()
                        W_u = (w_px * Z_med) / fx_rect
                        H_u = (h_px * Z_med) / fy_rect

                        x1,y1,x2,y2 = map(int, box)
                        cv.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv.putText(out, f"Z~{Z_med:.2f}{units_label} W~{W_u:.2f}{units_label} H~{H_u:.2f}{units_label}",
                                   (x1, max(0,y1-8)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                else:
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1,y1,x2,y2 = map(int, box)
                        roiZ = Z[y1:y2, x1:x2]; roiZ = roiZ[np.isfinite(roiZ)]
                        if roiZ.size == 0: 
                            continue
                        Z_med = float(np.median(roiZ))
                        w_px = (x2-x1); h_px = (y2-y1)
                        W_u = (w_px * Z_med) / fx_rect
                        H_u = (h_px * Z_med) / fy_rect
                        cv.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv.putText(out, f"Z~{Z_med:.2f}{units_label} W~{W_u:.2f}{units_label} H~{H_u:.2f}{units_label}",
                                   (x1, max(0,y1-8)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        except Exception as e:
            cv.putText(out, f"YOLO error: {e}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Mostra
    cv.imshow("Stereo Left (sizes)", out)
    if cv.waitKey(1) & 0xFF == 27:  # ESC per uscire
        break

# Cleanup
capL.release()
capR.release()
cv.destroyAllWindows()
