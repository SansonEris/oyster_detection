from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os
import threading

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
        
        # Initialize video capture
        self._init_video_captures()
    
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
    
    def process_frame(self, img, prefix=""):
        """Process a single frame and return annotated image with oyster count"""
        # Configure YOLO parameters
        results = self.model(img, stream=True, conf=self.confidence, iou=self.iou)
        
        oyster_count = 0
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    
                    # Draw bounding box
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    
                    # Confidence
                    conf = float(box.conf[0])
                    conf_txt = math.ceil(conf * 100) / 100
                    
                    # Class
                    cls = int(box.cls[0])
                    if 0 <= cls < len(self.classNames):
                        label = f'{self.classNames[cls]} {conf_txt}'
                    else:
                        label = f'class_{cls} {conf_txt}'
                    
                    # Draw label
                    cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), 
                                     scale=1, thickness=1)
                    
                    oyster_count += 1
        
        # Add frame counter and oyster count to image
        info_text = f"{prefix}Frame: {self.frames_processed} | Oysters: {oyster_count}"
        cvzone.putTextRect(img, info_text, (10, 30), scale=1.5, thickness=2)
        
        return img, oyster_count
    
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
                    processed_frame, oyster_count = self.process_frame(frame_left)
                    
                    # Save frame
                    output_path = os.path.join(self.output_dir, f"frame_{self.frames_processed:06d}.jpg")
                    cv2.imwrite(output_path, processed_frame)
                    
                    with self.lock:
                        self.current_oyster_count = oyster_count
                
                elif self.mode == "stereo":
                    # Process both frames
                    processed_left, oyster_count_left = self.process_frame(frame_left, "L-")
                    processed_right, oyster_count_right = self.process_frame(frame_right, "R-")
                    
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
                    with self.lock:
                        self.current_oyster_count = oyster_count_left + oyster_count_right
                
                # Update status
                with self.lock:
                    self.frames_processed += 1
                    elapsed = time.time() - start_time
                    self.fps = round(self.frames_processed / elapsed, 1) if elapsed > 0 else 0.0
            
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
            
        except Exception as e:
            print(f"Error creating output video: {e}")
    
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
