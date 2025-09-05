from ultralytics import YOLO
import cv2
import math
import time
import os
import threading
import csv
import pickle
import sys
import numpy as np
from datetime import datetime


def _make_unique_run_dir(base_out):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(base_out, f"run_{ts}")
    path = base
    idx = 1
    while os.path.exists(path):
        path = f"{base}_{idx}"
        idx += 1
    os.makedirs(path, exist_ok=True)
    return path


def load_pickle_compat(path):
    """Utility per caricamento pickle robusto (compat numpy._core)"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        import types, numpy as _np
        core_mod = types.ModuleType("numpy._core")
        core_mod.__dict__.update(_np.core.__dict__)
        sys.modules["numpy._core"] = core_mod
        sys.modules["numpy._core.multiarray"] = _np.core.multiarray
        sys.modules["numpy._core.numerictypes"] = _np.core.numerictypes
        sys.modules["numpy._core.overrides"] = _np.core.overrides
        sys.modules["numpy._core.fromnumeric"] = _np.core.fromnumeric
        try:
            sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath
        except Exception:
            pass
        with open(path, "rb") as f:
            return pickle.load(f)


def get_any(d, *keys, default=None):
    """Helper per recuperare la prima chiave disponibile"""
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


class OysterDetector:
    def __init__(self, model_path, mode, left_video, right_video=None, 
                 confidence=0.25, iou=0.45, output_dir="outputs/", stereo_calibration_path=None):
        self.model = YOLO(model_path)
        self.mode = mode
        self.left_video = left_video
        self.right_video = right_video
        self.confidence = confidence
        self.iou = iou
        self.output_dir = output_dir
        self.stereo_calibration_path = stereo_calibration_path
        if mode == "stereo" and not stereo_calibration_path:
            for cand in (
                "stereo_calibration_data.pkl",
                "calibration/stereo_calibration_data.pkl",
                "Calibration/stereo_calibration_data.pkl",
            ):
                if os.path.exists(cand):
                    self.stereo_calibration_path = cand
                    break
        # Create unique run directory for this session
        self.output_dir = _make_unique_run_dir(self.output_dir)
        self.run_started_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Status tracking
        self.running = False
        self.frames_processed = 0
        self.total_frames = 0
        self.fps = 0.0
        self.current_oyster_count = 0
        self.lock = threading.Lock()
        
        # Class names for oysters
        self.classNames = ["Oyster- Indeterminate", "Oyster-Closed", "Oyster-Open"]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize stereo calibration if provided and in stereo mode
        self.stereo_enabled = False

        if mode == "stereo" and self.stereo_calibration_path and os.path.exists(self.stereo_calibration_path):
            self._load_stereo_calibration(self.stereo_calibration_path)
        # Initialize video capture
        self._init_video_captures()
        
        # Create log file
        self.log_file = os.path.join(self.output_dir, "detection_log.txt")
        with open(self.log_file, 'w') as f:
            f.write("Frame,Oysters_Total,Oysters_Left,Oysters_Right,FPS,Timestamp\n")
        
        # Per-detection CSV - aggiungo colonne size solo se stereo abilitato
        self.detections_csv = os.path.join(self.output_dir, 'detections.csv')
        with open(self.detections_csv, 'w', newline='') as cf:
            cw = csv.writer(cf)
            if self.stereo_enabled:
                cw.writerow(['frame','side','class_id','class_name','confidence','x1','y1','x2','y2','cx','cy','w','h','width_cm','height_cm','depth_cm','timestamp'])
            else:
                cw.writerow(['frame','side','class_id','class_name','confidence','x1','y1','x2','y2','cx','cy','w','h','timestamp'])
    
    def _load_stereo_calibration(self, calibration_path):
        """Carica parametri di calibrazione stereo dal file del CalibrationCamera.py"""
        try:
            data = load_pickle_compat(calibration_path)
            
            # Supporta entrambe le convenzioni di chiavi dal CalibrationCamera.py
            KL = to_mat(get_any(data, "KL", "camera_matrix_l"), (3,3))
            KR = to_mat(get_any(data, "KR", "camera_matrix_r"), (3,3))
            DL = ensure_1d(get_any(data, "DL", "dist_coeffs_l"))
            DR = ensure_1d(get_any(data, "DR", "dist_coeffs_r"))
            R  = to_mat(get_any(data, "R"), (3,3))
            T  = ensure_1d(get_any(data, "T"))
            Q  = to_mat(get_any(data, "Q"), (4,4))
            
            # Mappe di rettifica (se già calcolate)
            map1_l = get_any(data, "map1_l", "mapLx")
            map2_l = get_any(data, "map2_l", "mapLy")
            map1_r = get_any(data, "map1_r", "mapRx")
            map2_r = get_any(data, "map2_r", "mapRy")
            
            # Parametri di proiezione rettificati (se presenti)
            PL = to_mat(get_any(data, "PL", "P1"))
            PR = to_mat(get_any(data, "PR", "P2"))

            # Validazioni minime
            if any(p is None for p in [KL, KR, DL, DR, R, T]):
                print("ERRORE: Parametri di calibrazione mancanti")
                return
            
            # Leggi la size reale dei video correnti
            cap_temp = cv2.VideoCapture(self.left_video)
            ret, frame0 = cap_temp.read()
            cap_temp.release()
            if not ret:
                print("ERRORE: impossibile leggere un frame per determinare la size")
                return
            h0, w0 = frame0.shape[:2]
            cur_size = (w0, h0)

            # Controllo esistenza mappe e loro size
            have_maps = all(m is not None for m in [map1_l, map2_l, map1_r, map2_r])

            if have_maps:
                saved_size = (map1_l.shape[1], map1_l.shape[0])
                if saved_size != cur_size:
                    # rigenera mappe per la size dei video correnti
                    RL = to_mat(get_any(data, "RL", "R1"))
                    RR = to_mat(get_any(data, "RR", "R2"))
                    if RL is None or RR is None or PL is None or PR is None or Q is None:
                        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
                            KL, DL, KR, DR, cur_size, R, T,
                            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
                        )
                    map1_l, map2_l = cv2.initUndistortRectifyMap(KL, DL, RL, PL, cur_size, cv2.CV_32FC1)
                    map1_r, map2_r = cv2.initUndistortRectifyMap(KR, DR, RR, PR, cur_size, cv2.CV_32FC1)
                    size = cur_size
                else:
                    size = saved_size
            else:
                # Mappe assenti: ricavo/impongo size e genero mappe
                size = cur_size
                RL = to_mat(get_any(data, "RL", "R1"))
                RR = to_mat(get_any(data, "RR", "R2"))
                if RL is None or RR is None or PL is None or PR is None or Q is None:
                    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
                        KL, DL, KR, DR, size, R, T,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
                    )
                map1_l, map2_l = cv2.initUndistortRectifyMap(KL, DL, RL, PL, size, cv2.CV_32FC1)
                map1_r, map2_r = cv2.initUndistortRectifyMap(KR, DR, RR, PR, size, cv2.CV_32FC1)

            # Salva parametri per uso successivo (size è SEMPRE definita e coerente ai video)
            self.stereo_params = {
                'map1_l': map1_l, 'map2_l': map2_l,
                'map1_r': map1_r, 'map2_r': map2_r,
                'Q': Q,
                'size': size
            }

            # Info focale e baseline
            fx_rect = float(PL[0,0]) if PL is not None else float(KL[0,0])
            fy_rect = float(PL[1,1]) if PL is not None else float(KL[1,1])
            baseline_units = float(np.linalg.norm(T))
            self.stereo_params['fx'] = fx_rect
            self.stereo_params['fy'] = fy_rect
            self.stereo_params['baseline'] = baseline_units

            print(f"Calibrazione stereo caricata: fx={fx_rect:.2f}, fy={fy_rect:.2f}, baseline={baseline_units:.2f}cm")

            # Matcher robusto (grayscale)
            bs = 5
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=128,          # multiplo di 16: range ampio
                blockSize=bs,                # kernel più stabile di 3
                P1=8*1*bs*bs,                # channels=1 (gray)
                P2=32*1*bs*bs,               # channels=1 (gray)
                speckleWindowSize=120,
                speckleRange=24,
                disp12MaxDiff=1,
                uniquenessRatio=10
            )
            self.stereo_enabled = True
            print("Size estimation stereo abilitata")

        except Exception as e:
            print(f"Errore caricamento calibrazione stereo: {e}")
            self.stereo_enabled = False

    def _init_video_captures(self):
        """Initialize video captures"""
        self.cap_left = cv2.VideoCapture(self.left_video)
        self.cap_right = cv2.VideoCapture(self.right_video) if self.right_video else None
        
        # Get total frames
        if self.cap_left.isOpened():
            self.total_frames = int(self.cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.mode == "stereo" and self.cap_right and not self.cap_right.isOpened():
            raise ValueError("Cannot open right video file")
        
        if not self.cap_left.isOpened():
            raise ValueError("Cannot open left video file")
    
    def draw_rectangle(self, img, x1, y1, x2, y2, thickness=2):
        import numpy as np
        import cv2

        H, W = img.shape[:2]
        x1 = int(np.clip(round(x1), 0, W - 1))
        y1 = int(np.clip(round(y1), 0, H - 1))
        x2 = int(np.clip(round(x2), 0, W - 1))
        y2 = int(np.clip(round(y2), 0, H - 1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1

        color = (0, 0, 255)
        line_type = cv2.LINE_AA  

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=line_type)
        
        # DISATTIVATO
        corner_size = 0  
        if corner_size > 0:
            cs = int(min(corner_size, x2 - x1, y2 - y1))
            if cs > 0:
                t2 = max(1, thickness + 1)
                # Top-left
                cv2.line(img, (x1, y1), (x1 + cs, y1), color, t2, line_type)
                cv2.line(img, (x1, y1), (x1, y1 + cs), color, t2, line_type)
                # Top-right
                cv2.line(img, (x2, y1), (x2 - cs, y1), color, t2, line_type)
                cv2.line(img, (x2, y1), (x2, y1 + cs), color, t2, line_type)
                # Bottom-left
                cv2.line(img, (x1, y2), (x1 + cs, y2), color, t2, line_type)
                cv2.line(img, (x1, y2), (x1, y2 - cs), color, t2, line_type)
                # Bottom-right
                cv2.line(img, (x2, y2), (x2 - cs, y2), color, t2, line_type)
                cv2.line(img, (x2, y2), (x2, y2 - cs), color, t2, line_type)

    def overlay_oyster_count(self, img, count, margin=12):
        """Scrive 'Ostriche: N' in alto a sinistra, con sfondo sfocato (modifica img in-place)."""
        import cv2, numpy as np
        txt = f"Oysters: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (tw, th), base = cv2.getTextSize(txt, font, font_scale, thickness)
        pad_x, pad_y = 10, 8
        x1, y1 = margin, margin
        x2, y2 = x1 + tw + pad_x*2, y1 + th + base + pad_y*2

        H, W = img.shape[:2]
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))

        roi = img[y1:y2, x1:x2]
        if roi.size:
            blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=9, sigmaY=9)
            cv2.addWeighted(blur, 0.85, roi, 0.15, 0.0, dst=roi)

        org = (x1 + pad_x, y1 + pad_y + th)
        # bordo nero + testo bianco
        cv2.putText(img, txt, org, font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, txt, org, font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        
    def put_text_with_background(self, img, text, pos, font_scale=0.6, thickness=2):
        """Put white text with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = pos
        
        # Ensure text doesn't go out of bounds
        x = max(0, min(x, img.shape[1] - text_width - 10))
        y = max(text_height + 5, min(y, img.shape[0] - 5))
        
        # Draw background rectangle
        cv2.rectangle(img, 
                     (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), 
                     (0, 0, 255), -1)
        
        # Draw white text
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    def calculate_size_from_stereo(self, x1, y1, x2, y2, rect_l, rect_r):
        """Calcola dimensioni usando stereo vision come in size_estimation_video.py"""
        if not self.stereo_enabled:
            print("DEBUG: Stereo non abilitato")
            return None, None, None
    
        print(f"DEBUG: Calcolo stereo per bbox ({x1},{y1},{x2},{y2})")
        
        try:
            # Converti in scala di grigi
            grayL = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
            
            # Calcola disparità
            disp = self.stereo_matcher.compute(grayL, grayR).astype(np.float32) / 16.0
            disp[disp <= 0] = np.nan
            
            # Riproietta in 3D
            points3D = cv2.reprojectImageTo3D(disp, self.stereo_params['Q'])
            Z = points3D[..., 2]
            
            # Estrai ROI della detection
            roiZ = Z[y1:y2, x1:x2]
            roiZ = roiZ[np.isfinite(roiZ)]
            
            if roiZ.size == 0:
                print("DEBUG: ROI depth vuota")
                return None, None, None
            
            Z_med = float(np.median(roiZ))
            print(f"DEBUG: Depth mediano: {Z_med}")    
            
            # Calcola dimensioni reali dalle dimensioni pixel
            w_px = (x2 - x1)
            h_px = (y2 - y1)
            
            W_real = (w_px * Z_med) / self.stereo_params['fx']
            H_real = (h_px * Z_med) / self.stereo_params['fy']
            
            print(f"DEBUG: Dimensioni calcolate W:{W_real:.2f}, H:{H_real:.2f}, D:{Z_med:.2f}")
            return W_real, H_real, Z_med
            
        except Exception as e:
            print(f"Errore calcolo stereo size: {e}")
            return None, None, None
    
    def process_frame(self, img, side_label='mono', rect_l=None, rect_r=None):
        """
        Process a single frame and return:
          - immagine annotata
          - conteggio ostriche
          - lista detections nel formato atteso da _append_detections_csv:
            (side, cls, class_name, conf_txt, x1, y1, x2, y2, cx, cy, w, h, width_cm, height_cm, depth_cm)
        L’overlay mostra SOLO W/H/D (in cm) quando disponibili.
        """
        if side_label == 'left' and self.stereo_enabled:
            print(f"DEBUG: Processing {side_label}, stereo_enabled={self.stereo_enabled}")
            print(f"DEBUG: rect_l is not None: {rect_l is not None}")
            print(f"DEBUG: rect_r is not None: {rect_r is not None}")

        results = self.model(img, stream=True, conf=self.confidence, iou=self.iou)
        oyster_count = 0
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # bbox rossa
                self.draw_rectangle(img, x1, y1, x2, y2)

                # info base (le lasciamo per compatibilità CSV; non verranno mostrate in overlay)
                conf = float(box.conf[0])
                conf_txt = math.ceil(conf * 100) / 100
                cls = int(box.cls[0])
                cls_name = self.classNames[cls] if 0 <= cls < len(self.classNames) else f'class_{cls}'

                # size estimation (solo se stereo abilitato; usiamo i frame rettificati)
                width_cm = height_cm = depth_cm = None
                if self.stereo_enabled and side_label == 'left' and rect_l is not None and rect_r is not None:
                    width_cm, height_cm, depth_cm = self.calculate_size_from_stereo(
                        x1, y1, x2, y2, rect_l, rect_r
                    )

                # overlay: SOLO dimensioni se disponibili
                if width_cm is not None and height_cm is not None and depth_cm is not None:
                    label = f"W:{width_cm:.1f}cm  H:{height_cm:.1f}cm  D:{depth_cm:.1f}cm"
                    self.put_text_with_background(img, label, (x1, max(0, y1 - 8)))
                # altrimenti non scriviamo testo

                oyster_count += 1
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                # per il CSV manteniamo il formato completo già previsto da _append_detections_csv
                detections.append((
                    side_label, cls, cls_name, conf_txt,
                    x1, y1, x2, y2, cx, cy, w, h,
                    width_cm, height_cm, depth_cm
                ))

        return img, oyster_count, detections
    
    def _append_detections_csv(self, frame_idx, detections):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        if not detections:
            return
        
        with open(self.detections_csv, 'a', newline='') as cf:
            cw = csv.writer(cf)
            for d in detections:
                if self.stereo_enabled and len(d) >= 15:
                    side, cls, cls_name, conf_txt, x1, y1, x2, y2, cx, cy, w, h, width_cm, height_cm, depth_cm = d
                    cw.writerow([frame_idx, side, cls, cls_name, conf_txt, x1, y1, x2, y2, cx, cy, w, h, width_cm, height_cm, depth_cm, ts])
                else:
                    side, cls, cls_name, conf_txt, x1, y1, x2, y2, cx, cy, w, h = d[:12]
                    cw.writerow([frame_idx, side, cls, cls_name, conf_txt, x1, y1, x2, y2, cx, cy, w, h, ts])

    def log_detection_data(self, frame_num, total_oysters, left_oysters=0, right_oysters=0, fps=0.0):
        """Log detection data to file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{frame_num},{total_oysters},{left_oysters},{right_oysters},{fps},{timestamp}\n")
    
    def run(self):
        """Main detection loop"""
        self.running = True
        start_time = time.time()
        
        # Determina se serve ridimensionamento
        need_resize = False
        target_size = None
        
        if self.stereo_enabled:
            target_size = self.stereo_params['size']  # (640, 480) dalla calibrazione
            
            # Verifica dimensioni del video
            ret, test_frame = self.cap_left.read()
            if ret:
                self.cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Riavvolgi
                h, w = test_frame.shape[:2]
                current_size = (w, h)
                
                if current_size != target_size:
                    need_resize = True
                    print(f"Video: {current_size}, Calibrazione: {target_size}")
                    print(f"Ridimensionerò tutti i frame da {current_size} a {target_size}")
        
        try:
            while self.running:
                # Read frames
                ret_left, frame_left = self.cap_left.read()
                
                if not ret_left:
                    break
                
                if self.mode == "stereo" and self.cap_right:
                    ret_right, frame_right = self.cap_right.read()
                    if not ret_right:
                        break
                
                # Ridimensiona se necessario PRIMA di qualsiasi processing
                if need_resize and target_size:
                    frame_left = cv2.resize(frame_left, target_size)
                    if self.mode == "stereo":
                        frame_right = cv2.resize(frame_right, target_size)
                
                # Process frames
                if self.mode == "mono":
                    processed_frame, oyster_count, dets = self.process_frame(frame_left, 'mono')
                    
                    self.overlay_oyster_count(processed_frame, oyster_count)
                    # Save frame
                    output_path = os.path.join(self.output_dir, f"frame_{self.frames_processed:06d}.jpg")
                    cv2.imwrite(output_path, processed_frame)
                    
                    with self.lock:
                        self.current_oyster_count = oyster_count
                        self.frames_processed += 1
                        elapsed = time.time() - start_time
                        self.fps = round(self.frames_processed / elapsed, 1) if elapsed > 0 else 0.0
                    
                    # Log data
                    self._append_detections_csv(self.frames_processed, dets)
                    self.log_detection_data(self.frames_processed, oyster_count, oyster_count, 0, self.fps)
                
                elif self.mode == "stereo":
                    # Rettifica frames se stereo abilitato
                    if self.stereo_enabled:
                        rect_left = cv2.remap(frame_left, np.asarray(self.stereo_params['map1_l']), 
                                            np.asarray(self.stereo_params['map2_l']), cv2.INTER_LINEAR)
                        rect_right = cv2.remap(frame_right, np.asarray(self.stereo_params['map1_r']), 
                                             np.asarray(self.stereo_params['map2_r']), cv2.INTER_LINEAR)
                    else:
                        rect_left = frame_left
                        rect_right = frame_right
                    
                    # Process both frames
                    processed_left, oyster_count_left, dets_left = self.process_frame(
                        rect_left, 'left', rect_left, rect_right)
                    processed_right, oyster_count_right, dets_right = self.process_frame(
                        rect_right, 'right', rect_left, rect_right)
                    
                    # Combine frames side by side
                    h1, w1 = processed_left.shape[:2]
                    h2, w2 = processed_right.shape[:2]
                    
                    if h1 != h2:
                        target_height = min(h1, h2)
                        processed_left = cv2.resize(processed_left, (int(w1 * target_height / h1), target_height))
                        processed_right = cv2.resize(processed_right, (int(w2 * target_height / h2), target_height))
                    
                    combined_frame = cv2.hconcat([processed_left, processed_right])

                    # Save combined frame
                    output_path = os.path.join(self.output_dir, f"frame_{self.frames_processed:06d}.jpg")
                    total_oysters = oyster_count_left + oyster_count_right
                    self.overlay_oyster_count(combined_frame, total_oysters)
                    cv2.imwrite(output_path, combined_frame)
                    with self.lock:
                        self.current_oyster_count = total_oysters
                        self.frames_processed += 1
                        elapsed = time.time() - start_time
                        self.fps = round(self.frames_processed / elapsed, 1) if elapsed > 0 else 0.0
                    
                    # Log data
                    self._append_detections_csv(self.frames_processed, dets_left + dets_right)
                    self.log_detection_data(self.frames_processed, total_oysters, oyster_count_left, oyster_count_right, self.fps)
            
        except Exception as e:
            print(f"Detection error: {e}")
        
        finally:
            self.running = False
            self._cleanup()
            self._create_output_video()
    
    def _create_output_video(self):
        """Create output video from saved frames"""
        try:
            frame_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.jpg')])
            
            if not frame_files:
                print("No frames found to create video")
                return
            
            # Get video properties from first frame
            first_frame = cv2.imread(os.path.join(self.output_dir, frame_files[0]))
            if first_frame is None:
                return
            
            height, width = first_frame.shape[:2]
            
            # Create video writer
            output_video_path = os.path.join(self.output_dir, "oyster_detection_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
            
            # Write frames to video
            for frame_file in frame_files:
                frame_path = os.path.join(self.output_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
            
            out.release()
            print(f"Output video created: {output_video_path}")
            
            # Create summary log
            self._create_summary_log()
            
        except Exception as e:
            print(f"Error creating output video: {e}")
    
    def _create_summary_log(self):
        """Create summary statistics log"""
        try:
            summary_path = os.path.join(self.output_dir, "detection_summary.txt")
            
            total_oysters = 0
            total_frames = self.frames_processed
            avg_oysters_per_frame = 0
            
            # Read log file to calculate statistics
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    if lines:
                        oyster_counts = [int(line.split(',')[1]) for line in lines if line.strip()]
                        total_oysters = sum(oyster_counts)
                        avg_oysters_per_frame = total_oysters / len(oyster_counts) if oyster_counts else 0
                        max_oysters = max(oyster_counts) if oyster_counts else 0
                        min_oysters = min(oyster_counts) if oyster_counts else 0
            
            with open(summary_path, 'w') as f:
                f.write("=== OYSTER DETECTION SUMMARY ===\n")
                f.write(f"Mode: {self.mode}\n")
                f.write(f"Run Directory: {self.output_dir}\n")
                f.write(f"Started At: {self.run_started_at}\n")
                f.write(f"Model: {os.path.basename(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else os.path.basename(getattr(self.model, 'model', 'Unknown'))}\n")
                f.write(f"Total Frames Processed: {total_frames}\n")
                f.write(f"Total Oysters Detected: {total_oysters}\n")
                f.write(f"Average Oysters per Frame: {avg_oysters_per_frame:.2f}\n")
                f.write(f"Max Oysters in Single Frame: {max_oysters}\n")
                f.write(f"Min Oysters in Single Frame: {min_oysters}\n")
                f.write(f"Average FPS: {self.fps}\n")
                f.write(f"Confidence Threshold: {self.confidence}\n")
                f.write(f"IoU Threshold: {self.iou}\n")
                f.write(f"Processing Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                if self.stereo_enabled:
                    f.write(f"Stereo Size Estimation: Enabled\n")
                    f.write(f"Baseline: {self.stereo_params['baseline']:.2f}cm\n")
                    f.write(f"Focal Length (fx): {self.stereo_params['fx']:.2f}\n")
                    f.write(f"Focal Length (fy): {self.stereo_params['fy']:.2f}\n")
                else:
                    f.write(f"Stereo Size Estimation: Disabled\n")
                
        except Exception as e:
            print(f"Error creating summary log: {e}")
    
    def _cleanup(self):
        """Release video captures"""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        """Stop detection"""
        self.running = False
    
    def is_running(self):
        """Check if detection is running"""
        return self.running
    
    def get_status(self):
        """Get current status"""
        with self.lock:
            return {
                'running': self.running,
                'frames': self.frames_processed,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'oyster_count': self.current_oyster_count
            }
