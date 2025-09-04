#!/usr/bin/env python3
"""
Sistema Completo per Rilevamento Distanza April Tag con Telecamera Stereo
Autore: Sistema di Calibrazione e Misurazione Automatica
Versione: 2.1

Requisiti:
pip install opencv-python numpy matplotlib

Uso:
1. python stereo_apriltag_system.py --calibrate           # Per calibrare in tempo reale
2. python stereo_apriltag_system.py --calibrate-from-files # Per calibrare da file salvati
3. python stereo_apriltag_system.py --detect              # Per rilevare distanze
4. python stereo_apriltag_system.py --create-chessboard   # Crea scacchiera ottimale
5. python stereo_apriltag_system.py --tips                # Consigli per calibrazione
6. python stereo_apriltag_system.py --help                # Per aiuto completo

IMPORTANTE: Prepara una scacchiera 7x10 con quadrati di 2.5-3cm
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import sys
import argparse
import time
import glob
from datetime import datetime
import pickle
import re

# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

class Config:
    # Parametri stream video
    #STREAM_URL = "http://raspberrypi:8080/video_feed"
    STREAM_URL = "http://192.168.52.50:8081/front/stereo/feed"
    
    # Parametri scacchiera per calibrazione
    CHESSBOARD_SIZE = (7, 10)  # Numero di angoli interni (width, height)
    SQUARE_SIZE = 3.6  # cm - MISURA QUESTO VALORE PRECISAMENTE!
    
    # Parametri April Tag
    ARUCO_DICT = aruco.DICT_7X7_250
    TAG_SIZE_REAL = 5.0  # cm - MISURA IL QUADRATO INTERNO NERO!
    
    # File di salvataggio
    CALIBRATION_FILE = "stereo_calibration_data.pkl"
    LOG_FILE = "calibration_log.txt"
    
    # Parametri di qualità
    MIN_CALIBRATION_IMAGES = 15
    MAX_REPROJECTION_ERROR = 1.0  # pixel
    MIN_DISPARITY = 1.0  # pixel
    
    # Pattern per file immagini
    IMAGE_PATTERN_LEFT = "calib_img_*_left.jpg"
    IMAGE_PATTERN_RIGHT = "calib_img_*_right.jpg"

# =============================================================================
# UTILITÀ E LOGGING
# =============================================================================

class Logger:
    def __init__(self, filename=Config.LOG_FILE):
        self.filename = filename
        self.log("="*50)
        self.log(f"Sessione iniziata: {datetime.now()}")
        self.log("="*50)
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

logger = Logger()

def create_optimal_chessboard():
    """
    Crea una scacchiera ottimale per la calibrazione stereo
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Parametri ottimali
    pattern_size = (7, 10)  # 8x11 quadrati totali
    square_size_cm = 3.0
    
    # Dimensioni totali
    width_cm = (pattern_size[0] + 1) * square_size_cm
    height_cm = (pattern_size[1] + 1) * square_size_cm
    
    print(f"SCACCHIERA OTTIMALE PER CALIBRAZIONE STEREO")
    print(f"="*50)
    print(f"Dimensioni pattern: {pattern_size[0]}x{pattern_size[1]} angoli interni")
    print(f"Quadrati totali: {pattern_size[0]+1}x{pattern_size[1]+1}")
    print(f"Dimensione quadrato: {square_size_cm}cm")
    print(f"Dimensioni totali: {width_cm}cm x {height_cm}cm")
    print(f"")
    print(f"ISTRUZIONI:")
    print(f"1. Stampa questa scacchiera su cartoncino rigido A2")
    print(f"2. Assicurati che sia perfettamente piatta")
    print(f"3. Usa buona illuminazione uniforme")
    print(f"4. Modifica il codice con questi parametri:")
    print(f"   CHESSBOARD_SIZE = {pattern_size}")
    print(f"   SQUARE_SIZE = {square_size_cm}")
    
    # Crea la scacchiera
    fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54), dpi=150)
    
    for i in range(pattern_size[0] + 1):
        for j in range(pattern_size[1] + 1):
            if (i + j) % 2 == 0:
                color = 'black'
            else:
                color = 'white'
            
            square = patches.Rectangle((i * square_size_cm, j * square_size_cm), 
                                     square_size_cm, square_size_cm, 
                                     facecolor=color, edgecolor='none')
            ax.add_patch(square)
    
    ax.set_xlim(0, width_cm)
    ax.set_ylim(0, height_cm)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('scacchiera_ottimale.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig('scacchiera_ottimale.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
    
    print(f"Scacchiera salvata come 'scacchiera_ottimale.pdf' e 'scacchiera_ottimale.png'")

def print_calibration_tips():
    """
    Stampa consigli per una buona calibrazione
    """
    print("\n" + "="*60)
    print("CONSIGLI PER UNA CALIBRAZIONE STEREO PERFETTA")
    print("="*60)
    print()
    print("📐 SCACCHIERA:")
    print("   - Usa pattern 7x10 (raccomandato per distanze lunghe)")
    print("   - Quadrati di almeno 2.5-3cm")
    print("   - Stampa su cartoncino rigido A2")
    print("   - Bordi netti, contrasto alto")
    print()
    print("📷 TELECAMERE:")
    print("   - Devono essere PERFETTAMENTE sincronizzate")
    print("   - Stesso frame rate e risoluzione")
    print("   - Evita auto-focus durante calibrazione")
    print("   - Distanza fissa tra le telecamere")
    print()
    print("💡 ILLUMINAZIONE:")
    print("   - Luce uniforme, evita ombre")
    print("   - Non usare flash")
    print("   - Evita riflessi sulla scacchiera")
    print()
    print("🎯 CATTURA IMMAGINI:")
    print("   - Almeno 20-30 immagini buone")
    print("   - Varie posizioni: centro, angoli, vicino, lontano")
    print("   - Varie inclinazioni: 0°, ±15°, ±30°")
    print("   - La scacchiera deve essere SEMPRE ferma quando catturi")
    print("   - Copri almeno 70% dell'area dell'immagine")
    print()
    print("💾 CALIBRAZIONE DA FILE:")
    print("   - Nomina i file: calib_img_XX_timestamp_left.jpg")
    print("   - Nomina i file: calib_img_XX_timestamp_right.jpg")
    print("   - Assicurati che left e right abbiano stesso timestamp")
    print("   - Usa --calibrate-from-files per caricare da disco")
    print()
    print("❌ ERRORI COMUNI:")
    print("   - Scacchiera mossa durante cattura")
    print("   - Pattern troppo piccolo o troppo grande")
    print("   - Poche variazioni di posizione")
    print("   - Telecamere non allineate")
    print("   - File left/right non corrispondenti")
    print("=" * 60)

def find_calibration_image_pairs(directory="."):
    """
    Trova coppie di immagini di calibrazione left/right
    Formato: calib_img_XX_timestamp_left.jpg e calib_img_XX_timestamp_right.jpg
    """
    left_pattern = os.path.join(directory, Config.IMAGE_PATTERN_LEFT)
    right_pattern = os.path.join(directory, Config.IMAGE_PATTERN_RIGHT)
    
    left_files = glob.glob(left_pattern)
    right_files = glob.glob(right_pattern)
    
    # Estrai timestamp da nomi file
    def extract_info(filename):
        basename = os.path.basename(filename)
        # Pattern: calib_img_XX_timestamp_side.jpg
        match = re.match(r'calib_img_(\d+)_(\d+)_(left|right)\.jpg', basename)
        if match:
            img_num, timestamp, side = match.groups()
            return int(img_num), int(timestamp), side
        return None
    
    # Organizza per timestamp
    images = {}
    
    for filename in left_files:
        info = extract_info(filename)
        if info:
            img_num, timestamp, side = info
            if timestamp not in images:
                images[timestamp] = {}
            images[timestamp]['left'] = filename
            images[timestamp]['img_num'] = img_num
    
    for filename in right_files:
        info = extract_info(filename)
        if info:
            img_num, timestamp, side = info
            if timestamp in images:
                images[timestamp]['right'] = filename
    
    # Filtra solo coppie complete
    pairs = []
    for timestamp, data in images.items():
        if 'left' in data and 'right' in data:
            pairs.append((data['left'], data['right'], data['img_num']))
    
    # Ordina per numero immagine
    pairs.sort(key=lambda x: x[2])
    
    return pairs

def validate_chessboard_image(gray_l, gray_r, chessboard_size):
    """
    Valida la qualità dell'immagine della scacchiera
    """
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                cv2.CALIB_CB_FILTER_QUADS)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                cv2.CALIB_CB_FILTER_QUADS)
    
    if not (ret_l and ret_r):
        return False, "Scacchiera non trovata in entrambe le immagini"
    
    # Controlla che gli angoli siano ben distribuiti
    if len(corners_l) != chessboard_size[0] * chessboard_size[1]:
        return False, "Numero angoli non corretto"
    
    # Controlla la qualità degli angoli
    corners_l_flat = corners_l.reshape(-1, 2)
    corners_r_flat = corners_r.reshape(-1, 2)
    
    # Calcola l'area coperta dalla scacchiera
    min_x_l, min_y_l = corners_l_flat.min(axis=0)
    max_x_l, max_y_l = corners_l_flat.max(axis=0)
    area_l = (max_x_l - min_x_l) * (max_y_l - min_y_l)
    
    min_x_r, min_y_r = corners_r_flat.min(axis=0)
    max_x_r, max_y_r = corners_r_flat.max(axis=0)
    area_r = (max_x_r - min_x_r) * (max_y_r - min_y_r)
    
    # Area minima
    min_area = 15000  # pixel quadrati
    
    if area_l < min_area or area_r < min_area:
        return False, f"Scacchiera troppo piccola (area: {area_l:.0f}, {area_r:.0f})"
    
    # Controlla che non sia troppo vicina ai bordi
    img_h, img_w = gray_l.shape
    border_margin = 30
    
    if (min_x_l < border_margin or min_y_l < border_margin or 
        max_x_l > img_w - border_margin or max_y_l > img_h - border_margin):
        return False, "Scacchiera troppo vicina ai bordi"
    
    return True, "OK"

# =============================================================================
# CALIBRATORE STEREO MIGLIORATO
# =============================================================================

class AdvancedStereoCalibrator:
    def __init__(self, config=Config):
        self.config = config
        self.logger = logger
        
        # Criteri di ottimizzazione
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
        
        # Array per i punti
        self.objpoints = []
        self.imgpoints_l = []
        self.imgpoints_r = []
        
        # Punti 3D della scacchiera
        self.objp = np.zeros((config.CHESSBOARD_SIZE[0] * config.CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:config.CHESSBOARD_SIZE[0], 0:config.CHESSBOARD_SIZE[1]].T.reshape(-1,2)
        self.objp *= config.SQUARE_SIZE
        
        # Parametri di calibrazione
        self.camera_matrix_l = None
        self.camera_matrix_r = None
        self.dist_coeffs_l = None
        self.dist_coeffs_r = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.calibration_quality = {}
        
        # Mappe di rettificazione
        self.map1_l = None
        self.map2_l = None
        self.map1_r = None
        self.map2_r = None
        self.Q = None
        
        self.logger.log("Calibratore stereo inizializzato")
    
    def connect_camera(self):
        """Connessione alla telecamera con retry"""
        for attempt in range(3):
            try:
                cap = cv2.VideoCapture(self.config.STREAM_URL)
                if cap.isOpened():
                    self.logger.log(f"Connesso alla telecamera: {self.config.STREAM_URL}")
                    return cap
                else:
                    self.logger.log(f"Tentativo {attempt+1} fallito")
                    time.sleep(2)
            except Exception as e:
                self.logger.log(f"Errore connessione: {e}")
        
        self.logger.log("ERRORE: Impossibile connettersi alla telecamera")
        return None
    
    def load_images_from_files(self, directory="."):
        """
        Carica immagini di calibrazione da file salvati
        """
        self.logger.log(f"Caricamento immagini da directory: {directory}")
        
        # Trova coppie di immagini
        image_pairs = find_calibration_image_pairs(directory)
        
        if not image_pairs:
            self.logger.log("ERRORE: Nessuna coppia di immagini trovata!")
            self.logger.log(f"Cerca file con pattern: {Config.IMAGE_PATTERN_LEFT} e {Config.IMAGE_PATTERN_RIGHT}")
            return False
        
        self.logger.log(f"Trovate {len(image_pairs)} coppie di immagini")
        
        processed = 0
        skipped = 0
        
        for left_file, right_file, img_num in image_pairs:
            self.logger.log(f"Processando coppia {img_num}: {os.path.basename(left_file)} + {os.path.basename(right_file)}")
            
            # Carica immagini
            try:
                img_l = cv2.imread(left_file)
                img_r = cv2.imread(right_file)
                
                if img_l is None or img_r is None:
                    self.logger.log(f"ERRORE: Impossibile caricare immagini {left_file} o {right_file}")
                    skipped += 1
                    continue
                
                # Verifica che abbiano stesse dimensioni
                if img_l.shape != img_r.shape:
                    self.logger.log(f"ERRORE: Immagini con dimensioni diverse: {img_l.shape} vs {img_r.shape}")
                    skipped += 1
                    continue
                
                # Converti in scala di grigi
                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
                
                # Valida immagini
                is_valid, message = validate_chessboard_image(gray_l, gray_r, self.config.CHESSBOARD_SIZE)
                
                if not is_valid:
                    self.logger.log(f"Immagine saltata: {message}")
                    skipped += 1
                    continue
                
                # Trova angoli
                flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                        cv2.CALIB_CB_NORMALIZE_IMAGE + 
                        cv2.CALIB_CB_FILTER_QUADS)
                
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.config.CHESSBOARD_SIZE, flags)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.config.CHESSBOARD_SIZE, flags)
                
                if ret_l and ret_r:
                    # Refina angoli
                    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), self.criteria)
                    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), self.criteria)
                    
                    # Salva punti
                    self.objpoints.append(self.objp)
                    self.imgpoints_l.append(corners_l)
                    self.imgpoints_r.append(corners_r)
                    
                    processed += 1
                    self.logger.log(f"✓ Immagine {processed} processata con successo")
                    
                    # Salva immagine di debug con angoli disegnati
                    debug_l = img_l.copy()
                    debug_r = img_r.copy()
                    #cv2.drawChessboardCorners(debug_l, self.config.CHESSBOARD_SIZE, corners_l, ret_l)
                    #cv2.drawChessboardCorners(debug_r, self.config.CHESSBOARD_SIZE, corners_r, ret_r)
                    
                    debug_combined = np.hstack([debug_l, debug_r])
                    debug_filename = f"debug_processed_{processed:02d}_{img_num}.jpg"
                    cv2.imwrite(debug_filename, debug_combined)
                    
                else:
                    self.logger.log(f"Angoli non trovati in {left_file} o {right_file}")
                    skipped += 1
                    
            except Exception as e:
                self.logger.log(f"ERRORE processando {left_file}: {e}")
                skipped += 1
                continue
        
        self.logger.log(f"Caricamento completato: {processed} immagini processate, {skipped} saltate")
        
        if processed < self.config.MIN_CALIBRATION_IMAGES:
            self.logger.log(f"ERRORE: Servono almeno {self.config.MIN_CALIBRATION_IMAGES} immagini valide")
            return False
        
        return True
    
    def capture_calibration_images(self, target_images=35):
        """
        Cattura immagini di calibrazione con validazione avanzata
        """
        cap = self.connect_camera()
        if not cap:
            return False
        
        self.logger.log(f"Inizio cattura {target_images} immagini di calibrazione")
        self.logger.log("ISTRUZIONI:")
        self.logger.log("- Muovi la scacchiera lentamente in diverse posizioni")
        self.logger.log("- Inclina la scacchiera in vari angoli")
        self.logger.log("- Assicurati che sia ben illuminata e piatta")
        self.logger.log("- SPAZIO = cattura, ESC = esci, S = salta immagine")
        
        captured = 0
        rejected = 0
        
        while captured < target_images:
            ret, frame = cap.read()
            if not ret:
                self.logger.log("Errore lettura frame")
                continue
            
            # Dividi il frame
            h, w = frame.shape[:2]
            half_w = w // 2
            frame_l = frame[:, :half_w]
            frame_r = frame[:, half_w:]
            
            # Converti in scala di grigi
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            # Valida immagine
            is_valid, message = validate_chessboard_image(gray_l, gray_r, self.config.CHESSBOARD_SIZE)
            
            # Trova angoli con parametri migliorati
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE + 
                    cv2.CALIB_CB_FILTER_QUADS)
            
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.config.CHESSBOARD_SIZE, flags)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.config.CHESSBOARD_SIZE, flags)
            
            if ret_l and ret_r and is_valid:
                # Disegna angoli
                #cv2.drawChessboardCorners(frame_l, self.config.CHESSBOARD_SIZE, corners_l, ret_l)
                #cv2.drawChessboardCorners(frame_r, self.config.CHESSBOARD_SIZE, corners_r, ret_r)
                
                # Status buono
                status_color = (0, 255, 0)
                status_text = f"PRONTO - {message}"
            else:
                status_color = (0, 0, 255)
                status_text = f"NON VALIDO - {message}"
            
            # Prepara display
            combined = np.hstack([frame_l, frame_r])
            
            # Aggiungi informazioni
            cv2.putText(combined, f"Catturate: {captured}/{target_images} | Rifiutate: {rejected}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(combined, "SPAZIO=Cattura | ESC=Esci | S=Salta", 
                       (10, combined.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(combined, f"Orario: {time.strftime("%Y-%m-%d %H:%M:%S")}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Aggiungi separatore
            cv2.line(combined, (half_w, 0), (half_w, h), (255, 255, 255), 2)
            cv2.putText(combined, "SINISTRA", (50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "DESTRA", (half_w + 50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Calibrazione Stereo - Cattura Immagini', combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Cattura
                if ret_l and ret_r and is_valid:
                    # Refina angoli
                    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), self.criteria)
                    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), self.criteria)
                    
                    # Salva punti
                    self.objpoints.append(self.objp)
                    self.imgpoints_l.append(corners_l)
                    self.imgpoints_r.append(corners_r)
                    
                    captured += 1
                    self.logger.log(f"Immagine {captured} catturata con successo")
                    
                    # Salva immagine per debug
                    timestamp = int(time.time())
                    filename_combined = f"calib_img_{captured:02d}_{timestamp}_combined.jpg"
                    filename_left = f"calib_img_{captured:02d}_{timestamp}_left.jpg"
                    filename_right = f"calib_img_{captured:02d}_{timestamp}_right.jpg"
                    
                    cv2.imwrite(filename_combined, combined)
                    cv2.imwrite(filename_left, frame_l)
                    cv2.imwrite(filename_right, frame_r)
                    
                    self.logger.log(f"Immagini salvate: {filename_combined}, {filename_left}, {filename_right}")
                    
                else:
                    rejected += 1
                    self.logger.log(f"Immagine rifiutata: {message}")
            
            elif key == 27:  # ESC
                break
            elif key == ord('s'):  # Skip/Salta
                self.logger.log("Immagine saltata dall'utente")
        
        cap.release()
        cv2.destroyAllWindows()
        
        success = captured >= self.config.MIN_CALIBRATION_IMAGES
        self.logger.log(f"Cattura completata: {captured} immagini valide, {rejected} rifiutate")
        return success
    
    def calibrate_individual_cameras(self, img_size):
        """
        Calibra le singole telecamere
        """
        self.logger.log("Calibrazione telecamere individuali...")
        
        # Telecamera sinistra
        ret_l, self.camera_matrix_l, self.dist_coeffs_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_size, None, None, criteria=self.criteria
        )
        
        # Telecamera destra
        ret_r, self.camera_matrix_r, self.dist_coeffs_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_size, None, None, criteria=self.criteria
        )
        
        self.calibration_quality['left_rms'] = ret_l
        self.calibration_quality['right_rms'] = ret_r
        
        self.logger.log(f"Calibrazione sinistra RMS: {ret_l:.3f}")
        self.logger.log(f"Calibrazione destra RMS: {ret_r:.3f}")
        
        return ret_l < 1.0 and ret_r < 1.0
    
    def calibrate_stereo_system(self, img_size):
        """
        Calibra il sistema stereo
        """
        self.logger.log("Calibrazione sistema stereo...")
        
        # Flags per calibrazione stereo
        flags = (cv2.CALIB_FIX_INTRINSIC +
                cv2.CALIB_RATIONAL_MODEL +
                cv2.CALIB_FIX_PRINCIPAL_POINT)
        
        # Calibrazione stereo
        ret, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.camera_matrix_l, self.dist_coeffs_l,
            self.camera_matrix_r, self.dist_coeffs_r,
            img_size, criteria=self.criteria_stereo, flags=flags
        )
        
        self.calibration_quality['stereo_rms'] = ret
        
        # Calcola baseline
        baseline = np.linalg.norm(self.T)
        self.calibration_quality['baseline'] = baseline
        
        self.logger.log(f"Calibrazione stereo RMS: {ret:.3f}")
        self.logger.log(f"Baseline calcolata: {baseline:.2f} cm")
        
        # Rettificazione stereo
        self.R1, self.R2, self.P1, self.P2, self.Q, roi_l, roi_r = cv2.stereoRectify(
            self.camera_matrix_l, self.dist_coeffs_l,
            self.camera_matrix_r, self.dist_coeffs_r,
            img_size, self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY
        )
        
        # Calcola mappe di rettificazione
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.camera_matrix_l, self.dist_coeffs_l, self.R1, self.P1, img_size, cv2.CV_16SC2
        )
        
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.camera_matrix_r, self.dist_coeffs_r, self.R2, self.P2, img_size, cv2.CV_16SC2
        )
        
        self.logger.log("Mappe di rettificazione calcolate")
        
        return ret < self.config.MAX_REPROJECTION_ERROR
    
    def save_calibration(self):
        """
        Salva tutti i parametri di calibrazione
        """
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'chessboard_size': self.config.CHESSBOARD_SIZE,
                'square_size': self.config.SQUARE_SIZE,
                'tag_size_real': self.config.TAG_SIZE_REAL
            },
            'quality': self.calibration_quality,
            'camera_matrix_l': self.camera_matrix_l,
            'camera_matrix_r': self.camera_matrix_r,
            'dist_coeffs_l': self.dist_coeffs_l,
            'dist_coeffs_r': self.dist_coeffs_r,
            'R': self.R,
            'T': self.T,
            'E': self.E,
            'F': self.F,
            'R1': self.R1,
            'R2': self.R2,
            'P1': self.P1,
            'P2': self.P2,
            'Q': self.Q,
            'map1_l': self.map1_l,
            'map2_l': self.map2_l,
            'map1_r': self.map1_r,
            'map2_r': self.map2_r
        }
        
        with open(self.config.CALIBRATION_FILE, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        self.logger.log(f"Calibrazione salvata in {self.config.CALIBRATION_FILE}")
        
        # Salva anche report testuale
        report_file = self.config.CALIBRATION_FILE.replace('.pkl', '_report.txt')
        with open(report_file, 'w') as f:
            f.write("REPORT CALIBRAZIONE STEREO\n")
            f.write("="*40 + "\n")
            f.write(f"Data: {calibration_data['timestamp']}\n")
            f.write(f"RMS Errore Sinistro: {self.calibration_quality['left_rms']:.3f}\n")
            f.write(f"RMS Errore Destro: {self.calibration_quality['right_rms']:.3f}\n")
            f.write(f"RMS Errore Stereo: {self.calibration_quality['stereo_rms']:.3f}\n")
            f.write(f"Baseline: {self.calibration_quality['baseline']:.2f} cm\n")
            f.write(f"Dimensione Scacchiera: {self.config.CHESSBOARD_SIZE}\n")
            f.write(f"Dimensione Quadrato: {self.config.SQUARE_SIZE} cm\n")
            f.write(f"Numero immagini usate: {len(self.objpoints)}\n")
        
        self.logger.log(f"Report salvato in {report_file}")
    
    def load_calibration(self):
        """
        Carica parametri di calibrazione esistenti
        """
        if not os.path.exists(self.config.CALIBRATION_FILE):
            return False
        
        try:
            with open(self.config.CALIBRATION_FILE, 'rb') as f:
                data = pickle.load(f)
            
            self.camera_matrix_l = data['camera_matrix_l']
            self.camera_matrix_r = data['camera_matrix_r']
            self.dist_coeffs_l = data['dist_coeffs_l']
            self.dist_coeffs_r = data['dist_coeffs_r']
            self.R = data['R']
            self.T = data['T']
            self.E = data['E']
            self.F = data['F']
            self.R1 = data['R1']
            self.R2 = data['R2']
            self.P1 = data['P1']
            self.P2 = data['P2']
            self.Q = data['Q']
            self.map1_l = data['map1_l']
            self.map2_l = data['map2_l']
            self.map1_r = data['map1_r']
            self.map2_r = data['map2_r']
            self.calibration_quality = data['quality']
            
            self.logger.log(f"Calibrazione caricata da {self.config.CALIBRATION_FILE}")
            self.logger.log(f"RMS Stereo: {self.calibration_quality['stereo_rms']:.3f}")
            self.logger.log(f"Baseline: {self.calibration_quality['baseline']:.2f} cm")
            
            return True
            
        except Exception as e:
            self.logger.log(f"Errore caricamento calibrazione: {e}")
            return False
    
    def run_full_calibration(self, from_files=False, directory="."):
        """
        Esegue il processo completo di calibrazione
        """
        self.logger.log("INIZIO CALIBRAZIONE COMPLETA")
        
        # Step 1: Cattura o carica immagini
        if from_files:
            self.logger.log(f"Caricamento immagini da directory: {directory}")
            if not self.load_images_from_files(directory):
                self.logger.log("ERRORE: Calibrazione fallita - caricamento immagini fallito")
                return False
            
            # Per file, determina dimensioni dalla prima immagine
            pairs = find_calibration_image_pairs(directory)
            if not pairs:
                return False
            
            first_img = cv2.imread(pairs[0][0])  # Prima immagine left
            if first_img is None:
                self.logger.log("ERRORE: Impossibile leggere prima immagine per dimensioni")
                return False
            
            h, w = first_img.shape[:2]
            img_size = (w, h)
            
        else:
            self.logger.log("Cattura immagini dal live stream")
            if not self.capture_calibration_images():
                self.logger.log("ERRORE: Calibrazione fallita - immagini insufficienti")
                return False
            
            # Per live stream, determina dimensioni dalla camera
            cap = self.connect_camera()
            if not cap:
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.logger.log("ERRORE: Impossibile ottenere dimensioni immagine")
                return False
            
            h, w = frame.shape[:2]
            img_size = (w // 2, h)  # Metà larghezza per singola camera
        
        self.logger.log(f"Dimensioni immagine per calibrazione: {img_size}")
        
        # Step 2: Calibra telecamere individuali
        if not self.calibrate_individual_cameras(img_size):
            self.logger.log("ERRORE: Calibrazione telecamere individuali fallita")
            return False
        
        # Step 3: Calibra sistema stereo
        if not self.calibrate_stereo_system(img_size):
            self.logger.log("ERRORE: Calibrazione stereo fallita")
            return False
        
        # Step 4: Salva risultati
        self.save_calibration()
        
        self.logger.log("CALIBRAZIONE COMPLETATA CON SUCCESSO!")
        return True

# =============================================================================
# RILEVATORE DISTANZE
# =============================================================================

class AprilTagDistanceDetector:
    def __init__(self, config=Config):
        self.config = config
        self.logger = logger
        self.calibrator = AdvancedStereoCalibrator(config)
        
        # Setup ArUco
        self.aruco_dict = aruco.getPredefinedDictionary(config.ARUCO_DICT)
        self.aruco_params = aruco.DetectorParameters()
        
        # Filtri per smoothing
        self.distance_filters = {}
        
        self.logger.log("Rilevatore distanze inizializzato")
    
    def get_smoothed_distance(self, tag_id, distance):
        """
        Applica filtro passa-basso per smooth delle misurazioni
        """
        if tag_id not in self.distance_filters:
            self.distance_filters[tag_id] = []
        
        self.distance_filters[tag_id].append(distance)
        
        # Mantieni solo le ultime 5 misurazioni
        if len(self.distance_filters[tag_id]) > 5:
            self.distance_filters[tag_id].pop(0)
        
        # Calcola media pesata (più peso alle misurazioni recenti)
        weights = np.linspace(0.5, 1.0, len(self.distance_filters[tag_id]))
        weighted_avg = np.average(self.distance_filters[tag_id], weights=weights)
        
        return weighted_avg
    
    def detect_and_measure(self):
        """
        Loop principale di rilevamento e misurazione
        """
        # Carica calibrazione
        if not self.calibrator.load_calibration():
            self.logger.log("ERRORE: Nessuna calibrazione trovata!")
            self.logger.log("Esegui prima: python stereo_apriltag_system.py --calibrate")
            return False
        
        # Connetti alla camera
        cap = self.calibrator.connect_camera()
        if not cap:
            return False
        
        self.logger.log("SISTEMA DI RILEVAMENTO AVVIATO")
        self.logger.log("Comandi: Q=Esci, S=Screenshot, R=Reset filtri, D=Debug")
        
        show_debug = False
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Dividi frame
                h, w = frame.shape[:2]
                half_w = w // 2
                frame_l = frame[:, :half_w]
                frame_r = frame[:, half_w:]
                
                # Rettifica usando calibrazione
                rect_l = cv2.remap(frame_l, self.calibrator.map1_l, self.calibrator.map2_l, cv2.INTER_LINEAR)
                rect_r = cv2.remap(frame_r, self.calibrator.map1_r, self.calibrator.map2_r, cv2.INTER_LINEAR)
                
                # Rileva April Tags
                corners_l, ids_l, _ = aruco.detectMarkers(rect_l, self.aruco_dict, parameters=self.aruco_params)
                corners_r, ids_r, _ = aruco.detectMarkers(rect_r, self.aruco_dict, parameters=self.aruco_params)
                
                # Disegna markers rilevati
                if ids_l is not None:
                    aruco.drawDetectedMarkers(rect_l, corners_l, ids_l)
                if ids_r is not None:
                    aruco.drawDetectedMarkers(rect_r, corners_r, ids_r)
                
                # Calcola distanze per tag comuni
                distances = {}
                if ids_l is not None and ids_r is not None:
                    ids_l_flat = ids_l.flatten()
                    ids_r_flat = ids_r.flatten()
                    common_ids = set(ids_l_flat) & set(ids_r_flat)
                    
                    for tag_id in common_ids:
                        # Trova indici
                        idx_l = np.where(ids_l_flat == tag_id)[0][0]
                        idx_r = np.where(ids_r_flat == tag_id)[0][0]
                        
                        # Calcola centri
                        center_l = corners_l[idx_l][0].mean(axis=0)
                        center_r = corners_r[idx_r][0].mean(axis=0)
                        
                        # Calcola disparità
                        disparity = abs(center_l[0] - center_r[0])
                        
                        if disparity > self.config.MIN_DISPARITY:
                            # Usa parametri di calibrazione per calcolo preciso
                            focal_length = self.calibrator.P1[0, 0]
                            baseline = self.calibrator.calibration_quality['baseline']
                            
                            # Calcola distanza
                            distance_raw = (focal_length * baseline) / disparity
                            distance_smooth = self.get_smoothed_distance(tag_id, distance_raw)
                            
                            distances[tag_id] = {
                                'raw': distance_raw,
                                'smooth': distance_smooth,
                                'disparity': disparity,
                                'center_l': center_l,
                                'center_r': center_r
                            }
                
                # Disegna informazioni
                display_l = rect_l.copy()
                display_r = rect_r.copy()
                
                for tag_id, data in distances.items():
                    center_l = data['center_l'].astype(int)
                    center_r = data['center_r'].astype(int)
                    
                    # Disegna centro e info su sinistra
                    cv2.circle(display_l, tuple(center_l), 5, (0, 255, 0), -1)
                    text = f"ID {tag_id}: {data['smooth']:.1f}cm"
                    cv2.putText(display_l, text, (center_l[0]-60, center_l[1]-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Disegna centro su destra
                    cv2.circle(display_r, tuple(center_r), 5, (0, 255, 0), -1)
                    
                    # Log periodico
                    if frame_count % 30 == 0:  # Ogni secondo circa
                        self.logger.log(f"Tag {tag_id}: {data['smooth']:.1f}cm (raw: {data['raw']:.1f}cm, disp: {data['disparity']:.1f}px)")
                
                # Disegna linee epipolari per debug
                if show_debug:
                    for i in range(0, h, 50):
                        cv2.line(display_l, (0, i), (half_w, i), (255, 255, 0), 1)
                        cv2.line(display_r, (0, i), (half_w, i), (255, 255, 0), 1)
                
                # Combina display
                combined = np.hstack([display_l, display_r])
                
                # Aggiungi info globali
                info_y = 30
                cv2.putText(combined, f"Tags rilevati: {len(distances)}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, f"Orario: {time.strftime("%Y-%m-%d %H:%M:%S")}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if distances:
                    info_y += 25
                    avg_distance = np.mean([d['smooth'] for d in distances.values()])
                    cv2.putText(combined, f"Distanza media: {avg_distance:.1f}cm", (10, info_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                
                # Info calibrazione
                info_y = combined.shape[0] - 60
                rms = self.calibrator.calibration_quality.get('stereo_rms', 0)
                baseline = self.calibrator.calibration_quality.get('baseline', 0)
                cv2.putText(combined, f"RMS: {rms:.2f}px | Baseline: {baseline:.1f}cm", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Separatore
                cv2.line(combined, (half_w, 0), (half_w, h), (255, 255, 255), 2)
                cv2.putText(combined, "SINISTRA", (50, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(combined, "DESTRA", (half_w + 50, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow('April Tag Distance Detection', combined)
                
                # Gestione comandi
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, combined)
                    self.logger.log(f"Screenshot salvato: {filename}")
                elif key == ord('r'):
                    self.distance_filters.clear()
                    self.logger.log("Filtri distanza resettati")
                elif key == ord('d'):
                    show_debug = not show_debug
                    self.logger.log(f"Debug mode: {'ON' if show_debug else 'OFF'}")
        
        except KeyboardInterrupt:
            self.logger.log("Rilevamento interrotto dall'utente")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return True

# =============================================================================
# FUNZIONE MAIN E ARGOMENTI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sistema Completo Stereo April Tag', 
                                   formatter_class=argparse.RawDescriptionHelpFormatter,
                                   epilog="""
ESEMPI DI USO:
  %(prog)s --create-chessboard        Crea scacchiera ottimale per calibrazione
  %(prog)s --calibrate                Calibra da stream video in tempo reale
  %(prog)s --calibrate-from-files     Calibra da immagini salvate su disco
  %(prog)s --detect                   Rileva distanze April Tag
  %(prog)s --tips                     Mostra consigli per calibrazione

FORMATI FILE PER CALIBRAZIONE DA DISCO:
  Le immagini devono seguire il formato:
  - calib_img_01_1234567890_left.jpg
  - calib_img_01_1234567890_right.jpg
  - calib_img_02_1234567891_left.jpg
  - calib_img_02_1234567891_right.jpg
  
  Dove il timestamp deve corrispondere tra left e right della stessa immagine.
                                   """)
    
    parser.add_argument('--calibrate', action='store_true', 
                       help='Esegui calibrazione stereo da stream video')
    parser.add_argument('--calibrate-from-files', action='store_true',
                       help='Esegui calibrazione da immagini salvate su disco')
    parser.add_argument('--detect', action='store_true', 
                       help='Rileva distanze April Tag')
    parser.add_argument('--create-chessboard', action='store_true', 
                       help='Crea scacchiera ottimale')
    parser.add_argument('--tips', action='store_true', 
                       help='Mostra consigli calibrazione')
    parser.add_argument('--config', help='File di configurazione personalizzato')
    parser.add_argument('--directory', default='.', 
                       help='Directory contenente le immagini di calibrazione (default: directory corrente)')
    
    args = parser.parse_args()
    
    # Mostra help se nessun argomento
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    if args.tips:
        print_calibration_tips()
        return
    
    if args.create_chessboard:
        try:
            create_optimal_chessboard()
        except ImportError:
            print("Per creare la scacchiera serve matplotlib:")
            print("pip install matplotlib")
        return
    
    if args.calibrate or args.calibrate_from_files:
        logger.log("AVVIO PROCESSO DI CALIBRAZIONE")
        calibrator = AdvancedStereoCalibrator()
        
        # Verifica configurazione
        print(f"\nCONFIGURAZIONE ATTUALE:")
        print(f"Scacchiera: {Config.CHESSBOARD_SIZE} angoli interni")
        print(f"Dimensione quadrato: {Config.SQUARE_SIZE}cm")
        if not args.calibrate_from_files:
            print(f"URL stream: {Config.STREAM_URL}")
        else:
            print(f"Directory immagini: {args.directory}")
            # Verifica che ci siano immagini nella directory
            pairs = find_calibration_image_pairs(args.directory)
            print(f"Coppie di immagini trovate: {len(pairs)}")
            if pairs:
                print("Prime 3 coppie:")
                for i, (left, right, num) in enumerate(pairs[:3]):
                    print(f"  {i+1}. {os.path.basename(left)} + {os.path.basename(right)}")
            else:
                print("⚠️  ATTENZIONE: Nessuna coppia di immagini trovata!")
                print(f"   Cerca file con pattern: {Config.IMAGE_PATTERN_LEFT} e {Config.IMAGE_PATTERN_RIGHT}")
        
        response = input("\nConfermi questa configurazione? (y/n): ")
        if response.lower() != 'y':
            print("Modifica i parametri nella classe Config e riprova.")
            return
        
        # Esegui calibrazione
        if args.calibrate_from_files:
            success = calibrator.run_full_calibration(from_files=True, directory=args.directory)
        else:
            success = calibrator.run_full_calibration(from_files=False)
        
        if success:
            print("\n✅ CALIBRAZIONE COMPLETATA CON SUCCESSO!")
            print("Ora puoi usare: python stereo_apriltag_system.py --detect")
        else:
            print("\n❌ CALIBRAZIONE FALLITA!")
            print("Controlla i log e riprova con immagini migliori.")
    
    elif args.detect:
        logger.log("AVVIO RILEVAMENTO DISTANZE")
        detector = AprilTagDistanceDetector()
        
        print(f"\nCONFIGURAZIONE APRIL TAG:")
        print(f"Dizionario: {Config.ARUCO_DICT}")
        print(f"Dimensione tag: {Config.TAG_SIZE_REAL}cm")
        
        if detector.detect_and_measure():
            print("\n✅ RILEVAMENTO COMPLETATO")
        else:
            print("\n❌ ERRORE DURANTE IL RILEVAMENTO")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramma interrotto dall'utente")
    except Exception as e:
        logger.log(f"ERRORE CRITICO: {e}")
        print(f"Errore critico: {e}")
        import traceback
        traceback.print_exc()
