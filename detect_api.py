from ultralytics import YOLO
import cv2
import math
import time
import os
import threading
import csv
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
class OysterDetector:
    def __init__(self, model_path, mode, left_video, right_video=None, 
                 confidence=0.25, iou=0.45, output_dir="outputs/"):
        self.model = YOLO(model_path)
        self.mode = mode
        self.left_video = left_video
        self.right_video = right_video
        self.confidence = confidence
        self.iou = iou
        self.output_dir = output_dir
        
        
        # Create unique run directory for this session
        self.output_dir = _make_unique_run_dir(self.output_dir)
        self.run_started_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')# Status tracking
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
        
        # Initialize video capture
        self._init_video_captures()
        
        # Create log file
        self.log_file = os.path.join(self.output_dir, "detection_log.txt")
        with open(self.log_file, 'w') as f:
            f.write("Frame,Oysters_Total,Oysters_Left,Oysters_Right,FPS,Timestamp\n")
        # Per-detection CSV
        self.detections_csv = os.path.join(self.output_dir, 'detections.csv')
        with open(self.detections_csv, 'w', newline='') as cf:
            cw = csv.writer(cf)
            cw.writerow(['frame','side','class_id','class_name','confidence','x1','y1','x2','y2','cx','cy','w','h','timestamp'])
    
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
    
    def draw_red_rectangle(self, img, x1, y1, x2, y2, thickness=2):
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
    
    def put_text_with_background(self, img, text, pos, font_scale=0.6, thickness=2):
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
    
    def process_frame(self, img, side_label='mono'):
        """Process a single frame and return annotated image, oyster count, and detections list"""
        results = self.model(img, stream=True, conf=self.confidence, iou=self.iou)
        oyster_count = 0
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    self.draw_red_rectangle(img, x1, y1, x2, y2)
                    conf = float(box.conf[0])
                    conf_txt = math.ceil(conf * 100) / 100
                    cls = int(box.cls[0])
                    cls_name = self.classNames[cls] if 0 <= cls < len(self.classNames) else f'class_{cls}'
                    label = f'{cls_name} {conf_txt}'
                    self.put_text_with_background(img, label, (x1, y1 - 10))
                    oyster_count += 1
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w/2
                    cy = y1 + h/2
                    detections.append((side_label, cls, cls_name, conf_txt, x1, y1, x2, y2, cx, cy, w, h))
        return img, oyster_count, detections
    
    def _append_detections_csv(self, frame_idx, detections):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        if not detections:
            return
        with open(self.detections_csv, 'a', newline='') as cf:
            cw = csv.writer(cf)
            for d in detections:
                side, cls, cls_name, conf_txt, x1, y1, x2, y2, cx, cy, w, h = d
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
                
                # Process frames
                if self.mode == "mono":
                    processed_frame, oyster_count, dets = self.process_frame(frame_left, 'mono')
                    
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
                    # Process both frames
                    processed_left, oyster_count_left, dets_left = self.process_frame(frame_left, 'left')
                    processed_right, oyster_count_right, dets_right = self.process_frame(frame_right, 'right')
                    
                    # Combine frames side by side
                    # Resize frames to same height if needed
                    h1, w1 = processed_left.shape[:2]
                    h2, w2 = processed_right.shape[:2]
                    
                    if h1 != h2:
                        target_height = min(h1, h2)
                        processed_left = cv2.resize(processed_left, (int(w1 * target_height / h1), target_height))
                        processed_right = cv2.resize(processed_right, (int(w2 * target_height / h2), target_height))
                    
                    combined_frame = cv2.hconcat([processed_left, processed_right])
                    
                    # Save combined frame
                    output_path = os.path.join(self.output_dir, f"frame_{self.frames_processed:06d}.jpg")
                    cv2.imwrite(output_path, combined_frame)
                    
                    total_oysters = oyster_count_left + oyster_count_right
                    
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
