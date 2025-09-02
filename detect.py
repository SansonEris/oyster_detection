from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os
import argparse

def main():
    # Argument parser per parametri configurabili
    parser = argparse.ArgumentParser(description='Oyster Detection with YOLO')
    parser.add_argument('--model', default='models/best_yolo8.onnx', help='Path to YOLO model')
    parser.add_argument('--mode', choices=['mono', 'stereo'], default='mono', help='Detection mode')
    parser.add_argument('--left', default='videos/left.mp4', help='Left video source')
    parser.add_argument('--right', help='Right video source (required for stereo mode)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--output', default='outputs/', help='Output directory')
    
    args = parser.parse_args()
    
    # Validazione parametri
    if args.mode == 'stereo' and not args.right:
        print("Error: Right video source required for stereo mode")
        return
    
    # Carica il modello YOLO
    model = YOLO(args.model)
    
    # Inizializza video captures
    cap_left = cv2.VideoCapture(args.left)
    cap_right = cv2.VideoCapture(args.right) if args.right else None
    
    if not cap_left.isOpened():
        print(f"Error: Cannot open left video: {args.left}")
        return
    
    if args.mode == 'stereo' and (not cap_right or not cap_right.isOpened()):
        print(f"Error: Cannot open right video: {args.right}")
        return
    
    # Crea cartella di output
    os.makedirs(args.output, exist_ok=True)
    
    # Classi del modello
    classNames = ["Oyster- Indeterminate", "Oyster-Closed", "Oyster-Open"]
    
    frame_count = 0
    start_time = time.time()
    
    def process_frame(img, prefix=""):
        """Processa un singolo frame e restituisce l'immagine annotata e il conteggio ostriche"""
        # Inferenza YOLO con parametri configurabili
        results = model(img, stream=True, conf=args.conf, iou=args.iou)
        
        oyster_count = 0
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))

                    # Confidence
                    conf = float(box.conf[0])
                    conf_txt = math.ceil(conf * 100) / 100

                    # Classe
                    cls = int(box.cls[0])
                    if 0 <= cls < len(classNames):
                        label = f'{classNames[cls]} {conf_txt}'
                    else:
                        label = f'class_{cls} {conf_txt}'

                    cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), 
                                     scale=1, thickness=1)
                    
                    oyster_count += 1
        
        # Aggiungi contatore frame e ostriche
        info_text = f"{prefix}Frame: {frame_count} | Oysters: {oyster_count}"
        cvzone.putTextRect(img, info_text, (10, 30), scale=1.5, thickness=2)
        
        return img, oyster_count

    # Loop principale
    while True:
        # Leggi i frame
        ret_left, frame_left = cap_left.read()
        
        if not ret_left:
            break  # Fine del video
        
        if args.mode == 'stereo':
            ret_right, frame_right = cap_right.read()
            if not ret_right:
                break
        
        # Processa i frame
        if args.mode == 'mono':
            processed_frame, oyster_count = process_frame(frame_left)
            
            # Salva il frame
            output_path = os.path.join(args.output, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, processed_frame)
            
            # Mostra il frame
            ccv2.imshow("Oyster Detection", processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC o 'q' per fermare
                self.running = False
            
        elif args.mode == 'stereo':
            # Processa entrambi i frame
            processed_left, oyster_count_left = process_frame(frame_left, "L-")
            processed_right, oyster_count_right = process_frame(frame_right, "R-")
            
            # Combina i frame side-by-side
            h1, w1 = processed_left.shape[:2]
            h2, w2 = processed_right.shape[:2]
            
            # Ridimensiona se necessario per altezza uniforme
            if h1 != h2:
                target_height = min(h1, h2)
                processed_left = cv2.resize(processed_left, (int(w1 * target_height / h1), target_height))
                processed_right = cv2.resize(processed_right, (int(w2 * target_height / h2), target_height))
            
            combined_frame = cv2.hconcat([processed_left, processed_right])
            
            # Salva il frame combinato
            output_path = os.path.join(args.output, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, combined_frame)
            
            # Mostra il frame combinato
            cv2.imshow("Stereo Oyster Detection", combined_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC o 'q' per fermare
                self.running = False
                break
            
            oyster_count = oyster_count_left + oyster_count_right
        
        # Calcola e mostra FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"Frame {frame_count} | Oysters: {oyster_count} | FPS: {fps:.1f}", end='\r')
        
        # Esci con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Pulizia
    cap_left.release()
    if cap_right:
        cap_right.release()
    cv2.destroyAllWindows()

    # Crea video di output dai frame salvati
    print(f"\nCreating output video...")
    frame_files = sorted([f for f in os.listdir(args.output) if f.endswith('.jpg')])
    
    if frame_files:
        # Leggi il primo frame per ottenere dimensioni
        first_frame = cv2.imread(os.path.join(args.output, frame_files[0]))
        if first_frame is not None:
            height, width = first_frame.shape[:2]
            
            # Crea video writer
            output_video_path = os.path.join(args.output, "oyster_detection_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
            
            # Scrivi tutti i frame
            for frame_file in frame_files:
                frame_path = os.path.join(args.output, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
            
            out.release()
            print(f"Video created successfully: {output_video_path}")
        else:
            print("Error reading frames for video creation")
    else:
        print("No frames found to create video")

if __name__ == "__main__":
