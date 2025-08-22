#!/usr/bin/env python3
"""
Stereo Oyster Detection Script
Compatible with Docker container and Jetson Nano (CPU only)
Supports both calibrated and uncalibrated stereo setups
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import threading
import queue
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass (frozen=True)
class Detection:
    """Single detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    center: Tuple[int, int]

@dataclass
class StereoDetection:
    """Matched stereo detection"""
    left_det: Detection
    right_det: Detection
    disparity: float
    depth: Optional[float] = None
    real_size: Optional[float] = None

class CameraCalibration:
    """Camera calibration data handler"""
    def __init__(self, calibration_file: Optional[str] = None):
        self.calibrated = False
        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)

    def load_calibration(self, calibration_file: str):
        """Load camera calibration from JSON file"""
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)

            self.K1 = np.array(data['left_camera_matrix'], dtype=np.float32)
            self.K2 = np.array(data['right_camera_matrix'], dtype=np.float32)
            self.D1 = np.array(data['left_distortion'], dtype=np.float32)
            self.D2 = np.array(data['right_distortion'], dtype=np.float32)
            self.R = np.array(data['rotation_matrix'], dtype=np.float32)
            self.T = np.array(data['translation_vector'], dtype=np.float32)
            self.baseline = data.get('baseline', 100.0)  # mm

            self.calibrated = True
            logger.info("Camera calibration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.calibrated = False

    def rectify_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply stereo rectification if calibrated"""
        if not self.calibrated:
            return left_frame, right_frame

        try:
            h, w = left_frame.shape[:2]

            # Compute rectification maps
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                    self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T
                    )

            map1_left, map2_left = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, (w, h), cv2.CV_32FC1)
            map1_right, map2_right = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, (w, h), cv2.CV_32FC1)

            # Apply rectification
            left_rect = cv2.remap(left_frame, map1_left, map2_left, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_frame, map1_right, map2_right, cv2.INTER_LINEAR)

            return left_rect, right_rect

        except Exception as e:
            logger.error(f"Rectification failed: {e}")
            return left_frame, right_frame

class ONNXDetector:
    """ONNX-based YOLOv5 detector"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.4, input_size: int = 640):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Initialize ONNX Runtime session
        providers = ['CPUExecutionProvider']  # CPU only as requested
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get model input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"ONNX model loaded: {model_path}")
        logger.info(f"Input: {self.input_name}")
        logger.info(f"Input shape: {self.session.get_inputs()[0].shape}")
        logger.info(f"Outputs: {self.output_names}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for inference - YOLOv5 style"""
        original_shape = image.shape[:2]  # H, W

        # Resize while maintaining aspect ratio (letterbox)
        scale = min(self.input_size / original_shape[0], self.input_size / original_shape[1])
        new_shape = (int(original_shape[1] * scale), int(original_shape[0] * scale))  # W, H

        resized = cv2.resize(image, new_shape)

        # Calculate padding
        pad_w = self.input_size - new_shape[0]
        pad_h = self.input_size - new_shape[1]

        # Pad to center the image
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)

        # Pad with gray color (114, 114, 114)
        padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )

        # Normalize to 0-1 and convert to RGB
        input_array = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Transpose from HWC to CHW
        input_array = np.transpose(input_array, (2, 0, 1))

        # Add batch dimension
        input_array = np.expand_dims(input_array, axis=0)

        return input_array, scale, (left, top)

    def postprocess(self, outputs: List[np.ndarray], scale: float, pads: Tuple[int, int]) -> List[Detection]:
        """Post-process ONNX outputs to get detections"""
        predictions = outputs[0]  # Shape should be [batch, num_detections, 6] for 1 class

        detections = []

        # Handle different output shapes
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension

        logger.debug(f"Predictions shape: {predictions.shape}")

        for detection in predictions:
            # YOLOv5 format: [x_center, y_center, width, height, confidence, class_score]
            if len(detection) < 5:
                continue

            x_center, y_center, width, height = detection[:4]
            confidence = detection[4]

            # For single class model, confidence might be the final score
            # or there might be a separate class score
            if len(detection) > 5:
                class_score = detection[5]
                final_conf = confidence * class_score
            else:
                final_conf = confidence

            if final_conf < self.conf_threshold:
                continue

            # Convert from relative to absolute coordinates if needed
            # Check if coordinates are normalized (0-1)
            if x_center <= 1.0 and y_center <= 1.0:
                x_center *= self.input_size
                y_center *= self.input_size
                width *= self.input_size
                height *= self.input_size

            # Remove padding and scale back to original image coordinates
            x_center = (x_center - pads[0]) / scale
            y_center = (y_center - pads[1]) / scale
            width = width / scale
            height = height / scale

            # Convert to corner coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Ensure coordinates are positive
            x1, y1 = max(0, x1), max(0, y1)

            center = (int(x_center), int(y_center))

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(final_conf),
                class_id=0,  # Single class: oyster
                center=center
                ))

        # Apply NMS
        return self.apply_nms(detections)

    def apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []

        boxes = np.array([det.bbox for det in detections], dtype=np.float32)
        scores = np.array([det.confidence for det in detections], dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), 
                self.conf_threshold, self.iou_threshold
                )

        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            return [detections[i] for i in indices]
        return []

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on image"""
        try:
            input_array, scale, pads = self.preprocess(image)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_array})

            # Post-process
            detections = self.postprocess(outputs, scale, pads)

            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

class StereoMatcher:
    """Match detections between stereo pair"""

    def __init__(self, max_disparity: int = 100, y_tolerance: int = 20):
        self.max_disparity = max_disparity
        self.y_tolerance = y_tolerance

    def match_detections(self, left_dets: List[Detection], right_dets: List[Detection], 
                         calibration: CameraCalibration) -> List[StereoDetection]:
        """Match detections between left and right cameras"""
        stereo_detections = []
        used_right = set()

        for left_det in left_dets:
            best_match = None
            best_score = float('inf')

            for i, right_det in enumerate(right_dets):
                if i in used_right:
                    continue

                # Check if detections could be the same object
                if self._is_valid_match(left_det, right_det):
                    score = self._compute_match_score(left_det, right_det)
                    if score < best_score:
                        best_score = score
                        best_match = (i, right_det)

            if best_match is not None:
                right_idx, right_det = best_match
                used_right.add(right_idx)

                # Calculate disparity
                disparity = left_det.center[0] - right_det.center[0]

                # Create stereo detection
                stereo_det = StereoDetection(
                        left_det=left_det,
                        right_det=right_det,
                        disparity=disparity
                        )

                # Calculate depth if calibrated
                if calibration.calibrated and disparity > 0:
                    # Simple depth calculation: depth = (focal_length * baseline) / disparity
                    focal_length = calibration.K1[0, 0]  # fx from left camera
                    depth = (focal_length * calibration.baseline) / disparity
                    stereo_det.depth = depth

                    # Estimate real size (assuming oysters are roughly circular)
                    bbox_width = left_det.bbox[2] - left_det.bbox[0]
                    real_size = (bbox_width * depth) / focal_length
                    stereo_det.real_size = real_size

                stereo_detections.append(stereo_det)

        return stereo_detections

    def _is_valid_match(self, left_det: Detection, right_det: Detection) -> bool:
        """Check if two detections could be a valid stereo match"""
        # Y coordinates should be similar (epipolar constraint for rectified images)
        y_diff = abs(left_det.center[1] - right_det.center[1])
        if y_diff > self.y_tolerance:
            return False

        # Right detection should be to the left of left detection (positive disparity)
        if left_det.center[0] <= right_det.center[0]:
            return False

        # Disparity should be reasonable
        disparity = left_det.center[0] - right_det.center[0]
        if disparity > self.max_disparity:
            return False

        # Size should be similar
        left_area = (left_det.bbox[2] - left_det.bbox[0]) * (left_det.bbox[3] - left_det.bbox[1])
        right_area = (right_det.bbox[2] - right_det.bbox[0]) * (right_det.bbox[3] - right_det.bbox[1])

        if left_area == 0 or right_area == 0:
            return False

        area_ratio = min(left_area, right_area) / max(left_area, right_area)
        if area_ratio < 0.3:  # Areas shouldn't differ by more than 3x
            return False

        return True

    def _compute_match_score(self, left_det: Detection, right_det: Detection) -> float:
        """Compute matching score between two detections"""
        # Combine multiple factors
        y_diff = abs(left_det.center[1] - right_det.center[1])
        size_diff = abs((left_det.bbox[2] - left_det.bbox[0]) - (right_det.bbox[2] - right_det.bbox[0]))
        conf_diff = abs(left_det.confidence - right_det.confidence)

        # Weighted combination
        score = y_diff * 2.0 + size_diff * 0.1 + conf_diff * 10.0
        return score

class VideoStreamer:
    """Handle video streaming from files or HTTP streams"""

    def __init__(self, left_source: str, right_source: str):
        self.left_source = left_source
        self.right_source = right_source
        self.left_cap = None
        self.right_cap = None

    def initialize(self) -> bool:
        """Initialize video captures"""
        try:
            # Set up video captures
            self.left_cap = cv2.VideoCapture(self.left_source)
            self.right_cap = cv2.VideoCapture(self.right_source)

            # Configure for better streaming performance
            if self.left_source.startswith('http') or self.left_source.startswith('rtsp'):
                self.left_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.left_cap.set(cv2.CAP_PROP_FPS, 30)
            if self.right_source.startswith('http') or self.right_source.startswith('rtsp'):
                self.right_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.right_cap.set(cv2.CAP_PROP_FPS, 30)

            # Check if both captures are working
            ret1, _ = self.left_cap.read()
            ret2, _ = self.right_cap.read()

            if not ret1 or not ret2:
                logger.error("Failed to read from video sources")
                return False

            logger.info("Video streams initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize video streams: {e}")
            return False

    def read_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Read synchronized frames from both cameras"""
        if self.left_cap is None or self.right_cap is None:
            return None, None

        ret1, left_frame = self.left_cap.read()
        ret2, right_frame = self.right_cap.read()

        if not ret1 or not ret2:
            return None, None

        return left_frame, right_frame

    def release(self):
        """Release video captures"""
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()

class HTTPStreamer:
    """Simple HTTP streamer for real-time visualization"""

    def __init__(self, port: int = 5000):
        self.port = port
        self.latest_frame = None
        self.running = False

    def update_frame(self, frame: np.ndarray):
        """Update the frame to be streamed"""
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.latest_frame = buffer.tobytes()

    def get_frame(self):
        """Get the latest frame for streaming"""
        return self.latest_frame

class StereoOysterCounter:
    """Main stereo oyster detection and counting system"""

    def __init__(self, model_path: str, calibration_file: Optional[str] = None, 
                 conf_threshold: float = 0.5, iou_threshold: float = 0.4, input_size: int = 640):
        self.detector = ONNXDetector(model_path, conf_threshold, iou_threshold, input_size)
        self.calibration = CameraCalibration(calibration_file)
        self.matcher = StereoMatcher()

        self.total_count = 0
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0

        # HTTP streaming
        self.http_streamer = HTTPStreamer()

    def process_frame_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[List[StereoDetection], np.ndarray, np.ndarray]:
        """Process a stereo frame pair"""
        # Apply rectification if calibrated
        if self.calibration.calibrated:
            left_frame, right_frame = self.calibration.rectify_frames(left_frame, right_frame)

        # Run detection on both frames
        left_detections = self.detector.detect(left_frame)
        right_detections = self.detector.detect(right_frame)

        logger.debug(f"Left detections: {len(left_detections)}, Right detections: {len(right_detections)}")

        # Match detections
        stereo_detections = self.matcher.match_detections(left_detections, right_detections, self.calibration)

        # If not calibrated, also count unmatched detections from both sides
        if not self.calibration.calibrated:
            # Add unmatched left detections
            matched_left = {sd.left_det for sd in stereo_detections}
            unmatched_left = [det for det in left_detections if det not in matched_left]

            # Add unmatched right detections  
            matched_right = {sd.right_det for sd in stereo_detections}
            unmatched_right = [det for det in right_detections if det not in matched_right]

            # Create pseudo stereo detections for unmatched ones
            for det in unmatched_left:
                stereo_detections.append(StereoDetection(det, det, 0))
            for det in unmatched_right:
                # Avoid double counting by checking overlap with left detections
                if not self._overlaps_with_left_detections(det, left_detections):
                    stereo_detections.append(StereoDetection(det, det, 0))

        # Draw results
        left_vis = self._draw_detections(left_frame.copy(), 
                                         [sd.left_det for sd in stereo_detections], 
                                         f"LEFT - {len(left_detections)} detected")
        right_vis = self._draw_detections(right_frame.copy(), 
                                          [sd.right_det for sd in stereo_detections], 
                                          f"RIGHT - {len(right_detections)} detected")

        return stereo_detections, left_vis, right_vis

    def _overlaps_with_left_detections(self, right_det: Detection, left_detections: List[Detection]) -> bool:
        """Check if right detection overlaps significantly with any left detection"""
        for left_det in left_detections:
            # Simple overlap check based on center distance
            dist = euclidean(right_det.center, left_det.center)
            avg_size = ((right_det.bbox[2] - right_det.bbox[0]) + (left_det.bbox[2] - left_det.bbox[0])) / 2
            if avg_size > 0 and dist < avg_size * 0.5:  # If centers are close relative to size
                return True
        return False

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection], camera_label: str) -> np.ndarray:
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            label = f"Oyster: {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add info overlay
        info_y = 30
        cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, camera_label, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def run(self, left_source: str, right_source: str, output_dir: Optional[str] = None, 
            save_video: bool = True, headless: bool = False):
        """Run the stereo detection system"""
        streamer = VideoStreamer(left_source, right_source)

        if not streamer.initialize():
            logger.error("Failed to initialize video streams")
            return

        logger.info("Starting stereo oyster detection...")
        logger.info(f"Calibrated: {'Yes' if self.calibration.calibrated else 'No'}")
        logger.info(f"Headless mode: {headless}")
        logger.info(f"Save video: {save_video}")

        writer = None
        output_path = None

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if save_video:
                output_path = str(Path(output_dir) / "stereo_detection.mp4")

        try:
            while True:
                start_time = time.time()

                left_frame, right_frame = streamer.read_frames()
                if left_frame is None or right_frame is None:
                    logger.info("End of video or stream disconnected")
                    break

                stereo_detections, left_vis, right_vis = self.process_frame_pair(left_frame, right_frame)
                current_count = len(stereo_detections)
                self.total_count = max(self.total_count, current_count)
                self.frame_count += 1
                self.fps_counter += 1

                # Calculate FPS
                current_time = time.time()
                if current_time - self.fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_time)
                    self.fps_counter = 0
                    self.fps_time = current_time
                    logger.info(f"Frame {self.frame_count}: {current_count} oysters, FPS: {self.current_fps:.1f}")

                # Create combined view
                combined = np.hstack([left_vis, right_vis])

                # Update HTTP stream
                self.http_streamer.update_frame(combined)

                # Save video if requested
                if save_video and output_path:
                    if writer is None:
                        height, width, _ = combined.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                        logger.info(f"Video writer initialized: {output_path}")

                    if writer:
                        writer.write(combined)

                # Display if not headless
                if not headless:
                    cv2.imshow("Stereo Oyster Detection", combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and output_dir:
                        # Save current frame
                        frame_path = str(Path(output_dir) / f"frame_{self.frame_count:06d}.jpg")
                        cv2.imwrite(frame_path, combined)
                        logger.info(f"Frame saved: {frame_path}")

                # Small delay to prevent overwhelming
                processing_time = time.time() - start_time
                if processing_time < 0.033:  # Target ~30 FPS
                    time.sleep(0.033 - processing_time)

        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            streamer.release()
            if writer:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()

        logger.info(f"Detection completed. Total frames: {self.frame_count}, Max oysters: {self.total_count}")

    # -------------------- MONO MODE (aggiunta minima) --------------------
    def run_mono(self, source: str, output_dir: Optional[str] = None,
                 save_video: bool = True, headless: bool = False):
        """Run single-video (mono) detection with same HTTP streamer."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Failed to open video source")
            return

        logger.info("Starting MONO oyster detection...")
        logger.info(f"Headless mode: {headless}")
        logger.info(f"Save video: {save_video}")

        writer = None
        output_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if save_video:
                output_path = str(Path(output_dir) / "mono_detection.mp4")

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video or stream disconnected")
                    break

                detections = self.detector.detect(frame)

                vis = self._draw_detections(frame.copy(), detections, f"MONO - {len(detections)} detected")

                self.frame_count += 1
                self.fps_counter += 1
                now = time.time()
                if now - self.fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (now - self.fps_time)
                    self.fps_counter = 0
                    self.fps_time = now
                    logger.info(f"Frame {self.frame_count}: {len(detections)} oysters, FPS: {self.current_fps:.1f}")

                # HTTP stream (single frame)
                self.http_streamer.update_frame(vis)

                # Save video
                if save_video and output_path:
                    if writer is None:
                        h, w = vis.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
                        logger.info(f"Video writer initialized: {output_path}")
                    writer.write(vis)

                if not headless:
                    cv2.imshow("Mono Oyster Detection", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and output_dir:
                        frame_path = str(Path(output_dir) / f"frame_{self.frame_count:06d}.jpg")
                        cv2.imwrite(frame_path, vis)
                        logger.info(f"Frame saved: {frame_path}")

                # Throttle leggero per target ~30 FPS
                proc_t = time.time() - start_time
                if proc_t < 0.033:
                    time.sleep(0.033 - proc_t)

        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            cap.release()
            if writer:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()
        logger.info(f"Detection completed. Total frames: {self.frame_count}")
    # --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Stereo Oyster Detection with ONNX')

    # Model parameters
    parser.add_argument('--model', required=True, help='Path to ONNX model file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--size', type=int, default=640, help='Input image size (default: 640)')

    # Input sources
    parser.add_argument('--left', required=True, help='Left camera source (file or stream)')
    parser.add_argument('--right', required=False, help='Right camera source (file or stream)')  # <- reso opzionale
    parser.add_argument('--mono', action='store_true', help='Enable single-video (mono) detection')  # <- nuova flag

    # Calibration
    parser.add_argument('--calibration', help='Camera calibration JSON file (optional)')

    # Output
    parser.add_argument('--output', help='Output directory for saving results')
    parser.add_argument('--save-video', action='store_true', help='Save detection video')
    parser.add_argument('--no-display', action='store_true', help='Run without GUI display (headless)')

    # HTTP streaming
    parser.add_argument('--http', action='store_true', help='Enable HTTP streaming')
    parser.add_argument('--http-port', type=int, default=5000, help='HTTP streaming port')

    # Legacy compatibility
    parser.add_argument('--videos', nargs=2, help='[Deprecated] Use --left and --right instead')
    parser.add_argument('--single', help='[Deprecated] Single video mode not supported')
    parser.add_argument('--streams', nargs=2, help='[Deprecated] Use --left and --right instead')
    parser.add_argument('--imgsz', type=int, help='[Deprecated] Use --size instead')
    parser.add_argument('--device', type=str, default='cpu', help='[Deprecated] Always uses CPU')
    parser.add_argument('--engine', type=str, default='onnx', help='[Deprecated] Always uses ONNX')

    args = parser.parse_args()

    # Handle legacy arguments
    if args.videos:
        args.left, args.right = args.videos
    elif args.streams:
        args.left, args.right = args.streams

    if args.imgsz:
        args.size = args.imgsz

    if args.single:
        logger.error("Single video mode not supported in stereo detection (use --mono)")
        sys.exit(1)

    # Validate required arguments
    if args.mono:
        args.right = None
    else:
        if not args.right:
            logger.error("Both --left and --right sources are required (or pass --mono for single video)")
            sys.exit(1)

    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    logger.info(f"Model: {args.model}")
    logger.info(f"Left source: {args.left}")
    if args.right:
        logger.info(f"Right source: {args.right}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IoU threshold: {args.iou}")
    logger.info(f"Input size: {args.size}")

    # Initialize detection system
    try:
        counter = StereoOysterCounter(
                args.model,
                calibration_file=args.calibration,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                input_size=args.size
                )

        # Start HTTP server if requested
        if args.http:
            import threading
            from http.server import HTTPServer, BaseHTTPRequestHandler

            class StreamHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        # Serve simple HTML page (adatta la size per mono/stereo)
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        if args.mono:
                            img_tag = '<img src="/stream.mjpg" width="640" height="480">'
                            title = "Mono Oyster Detection"
                        else:
                            img_tag = '<img src="/stream.mjpg" width="1280" height="480">'
                            title = "Stereo Oyster Detection"
                        html = f'''
                        <!DOCTYPE html>
                        <html>
                        <head><title>{title}</title></head>
                        <body>
                            <h1>{title} - Live Stream</h1>
                            {img_tag}
                            <p>Press 'q' in terminal to quit</p>
                        </body>
                        </html>
                        '''
                        self.wfile.write(html.encode())

                    elif self.path == '/stream.mjpg':
                        # Serve MJPEG stream
                        self.send_response(200)
                        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                        self.end_headers()

                        try:
                            while True:
                                frame_data = counter.http_streamer.get_frame()
                                if frame_data:
                                    self.wfile.write(b'\r\n--frame\r\n')
                                    self.send_header('Content-type', 'image/jpeg')
                                    self.send_header('Content-length', len(frame_data))
                                    self.end_headers()
                                    self.wfile.write(frame_data)
                                time.sleep(0.033)  # ~30 FPS
                        except Exception:
                            pass
                    else:
                        self.send_error(404)

                def log_message(self, format, *args):
                    pass  # Suppress HTTP logs

            httpd = HTTPServer(('0.0.0.0', args.http_port), StreamHandler)
            http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            http_thread.start()
            logger.info(f"HTTP stream available at: http://localhost:{args.http_port}")

        # Run detection
        if args.mono:
            counter.run_mono(
                    args.left,
                    output_dir=args.output,
                    save_video=args.save_video,
                    headless=args.no_display
                    )
        else:
            counter.run(
                    args.left,
                    args.right,
                    output_dir=args.output,
                    save_video=args.save_video,
                    headless=args.no_display
                    )

        if args.http:
            httpd.shutdown()

    except Exception as e:
        logger.error(f"Failed to initialize detection system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
