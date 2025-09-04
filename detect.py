from ultralytics import YOLO
import cv2
import math
import time
import os
import argparse

def draw_red_rectangle(img, x1, y1, x2, y2, thickness=2):
    """Draw red rectangle with rounded corners effect"""
    # Main rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    
    # Corner decorations for better visibility
    corner_size = 10
    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + corner_size, y1), (0, 0, 255), thickness + 1)
    cv2.line(img, (x1, y1), (x1, y1 + corner_size), (0, 0, 255), thickness + 1)
    
    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - corner_size, y1), (0, 0, 255), thickness + 1)
    cv2.line(img, (x2, y1), (x2, y1 + corner_size), (0, 0, 255), thickness + 1)
    
    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + corner_size, y2), (0, 0, 255), thickness + 1)
    cv2.line(img, (x1, y2), (x1, y2 - corner_size), (0, 0, 255), thickness + 1)
    
    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - corner_size, y2), (0, 0, 255), thickness + 1)
    cv2.line(img, (x2, y2), (x2, y2 - corner_size), (0, 0, 255), thickness + 1)

def put_text_with_background(img, text, pos, font_scale=0.6, thickness=2):
    """Put white text with red background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    
    # Ensure text doesn't go out of bounds
    x = max(0, min(x, img.shape[1] - text_width - 10))
    y = max(text_height + 5, min(y, img.shape[0] - 5))
    
    # Draw red background rectangle
    cv2.rectangle(img, 
                 (x - 5, y - text_height - 5), 
                 (x + text_width + 5, y + baseline + 5), 
                 (0, 0, 255), -1)
    
    # Draw white text
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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
    
    # Initialize logging
    log_file = os.path.join(args.output, "detection_log.txt")
    with open(log_file, 'w') as f:
        f.write("Frame,Oysters_Total,Oysters_Left,Oysters_Right,FPS,Timestamp\n")
    
    frame_count = 0
    start_time = time.time()
    running = True
    
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
                    
                    # Draw red bounding box
                    draw_red_rectangle(img, x1, y1, x2, y2)

                    # Confidence
                    conf = float(box.conf[0])
                    conf_txt = math.ceil(conf * 100) / 100

                    # Classe
                    cls = int(box.cls[0])
                    if 0 <= cls < len(classNames):
                        label = f'{classNames[cls]} {conf_txt}'
                    else:
                        label = f'class_{cls} {conf_txt}'

                    # Draw white text with red background
                    put_text_with_background(img, label, (x1, y1 - 10))
                    
                    oyster_count += 1
        
        return img, oyster_count

    def log_detection_data(frame_num, total_oysters, left_oysters=0, right_oysters=0, fps=0.0):
        """Log detection data to file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"{frame_num},{total_oysters},{left_oysters},{right_oysters},{fps},{timestamp}\n")

    # Loop principale
    while running:
        # Leggi i frame
        ret_left, frame_left = cap_left.read()
        
        if not ret_left:
            break  # Fine del video
        
        if args.mode == 'stereo':
            ret_right, frame_right = cap_right.read()
            if not ret_right:
                break
        
        frame_count += 1
        
        # Processa i frame
        if args.mode == 'mono':
            processed_frame, oyster_count = process_frame(frame_left)
            
            # Salva il frame
            output_path = os.path.join(args.output, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, processed_frame)
            
            # Calcola FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Log data
            log_detection_data(frame_count, oyster_count, oyster_count, 0, round(fps, 1))
            
            # Mostra il frame
            cv2.imshow("Oyster Detection", processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC o 'q' per fermare
                running = False
            
        elif args.mode == 'stereo':
            # Processa entrambi i frame
            processed_left, oyster_count_left = process_frame(frame_left)
            processed_right, oyster_count_right = process_frame(frame_right)
            
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
            
            oyster_count = oyster_count_left + oyster_count_right
            
            # Calcola FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Log data
            log_detection_data(frame_count, oyster_count, oyster_count_left, oyster_count_right, round(fps, 1))
            
            # Mostra il frame combinato
            cv2.imshow("Stereo Oyster Detection", combined_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC o 'q' per fermare
                running = False
        
        # Stampa progresso senza sovrascrivere
        print(f"Frame {frame_count} | Oysters: {oyster_count} | FPS: {fps:.1f}")

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
            
            # Create summary log
            create_summary_log(args.output, log_file, args, frame_count, fps)
        else:
            print("Error reading frames for video creation")
    else:
        print("No frames found to create video")

def create_summary_log(output_dir, log_file, args, total_frames, avg_fps):
    """Create summary statistics log"""
    try:
        summary_path = os.path.join(output_dir, "detection_summary.txt")
        
        total_oysters = 0
        avg_oysters_per_frame = 0
        max_oysters = 0
        min_oysters = 0
        
        # Read log file to calculate statistics
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                if lines:
                    oyster_counts = [int(line.split(',')[1]) for line in lines if line.strip()]
                    if oyster_counts:
                        total_oysters = sum(oyster_counts)
                        avg_oysters_per_frame = total_oysters / len(oyster_counts)
                        max_oysters = max(oyster_counts)
                        min_oysters = min(oyster_counts)
        
        with open(summary_path, 'w') as f:
            f.write("=== OYSTER DETECTION SUMMARY ===\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Model: {os.path.basename(args.model)}\n")
            f.write(f"Total Frames Processed: {total_frames}\n")
            f.write(f"Total Oysters Detected: {total_oysters}\n")
            f.write(f"Average Oysters per Frame: {avg_oysters_per_frame:.2f}\n")
            f.write(f"Max Oysters in Single Frame: {max_oysters}\n")
            f.write(f"Min Oysters in Single Frame: {min_oysters}\n")
            f.write(f"Average FPS: {avg_fps:.1f}\n")
            f.write(f"Confidence Threshold: {args.conf}\n")
            f.write(f"IoU Threshold: {args.iou}\n")
            f.write(f"Processing Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        print(f"Summary log created: {summary_path}")
            
    except Exception as e:
        print(f"Error creating summary log: {e}")

if __name__ == "__main__":
    main()
