import cv2
import numpy as np
from models.multi_angle_detector import MultiAngleDetector
from utils.angle_augmentation import AngleAugmentation
import os
import json

class ThermalTrackMultiAngle:
    def __init__(self):
        # Initialize multi-angle detector with different model paths
        model_paths = {
            'top': 'models/yolo_top_view.pt',
            'side': 'models/yolo_side_view.pt',
            'angled': 'models/yolo_angled_view.pt'
        }
        
        self.detector = MultiAngleDetector(model_paths)
        self.augmentor = AngleAugmentation()
        
    def process_video(self, video_path: str, output_path: str = None):
        """Process video with multi-angle detection"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance thermal contrast
            enhanced_frame = self.augmentor.enhance_thermal_contrast(frame)
            
            # Perform multi-angle detection
            results = self.detector.detect_with_angle_adaptation(enhanced_frame)
            
            # Visualize results
            annotated_frame = self._visualize_results(frame, results)
            
            if output_path:
                out.write(annotated_frame)
            
            cv2.imshow('Multi-Angle Thermal Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def _visualize_results(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Visualize detection results on frame"""
        annotated_frame = frame.copy()
        
        # Add angle type information
        angle_type = results.get('angle_type', 'unknown')
        cv2.putText(annotated_frame, f'Angle: {angle_type}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detections
        if 'detections' in results and results['detections']:
            for detection in results['detections']:
                if detection.boxes is not None:
                    boxes = detection.boxes.xyxy.cpu().numpy()
                    scores = detection.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence score
                        cv2.putText(annotated_frame, f'{score:.2f}', 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated_frame
    
    def process_video_with_results(self, video_path: str, output_path: str = None):
        """Process video with multi-angle detection and return results"""
        cap = cv2.VideoCapture(video_path)
        detection_results = []
        frame_count = 0
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance thermal contrast
            enhanced_frame = self.augmentor.enhance_thermal_contrast(frame)
            
            # Perform multi-angle detection
            results = self.detector.detect_with_angle_adaptation(enhanced_frame)
            
            # Store detection results
            frame_result = {
                'frame_number': frame_count,
                'angle_type': results.get('angle_type', 'unknown'),
                'detections': []
            }
            
            # Extract detection data
            if 'detections' in results and results['detections']:
                for detection in results['detections']:
                    if detection.boxes is not None:
                        boxes = detection.boxes.xyxy.cpu().numpy()
                        scores = detection.boxes.conf.cpu().numpy()
                        classes = detection.boxes.cls.cpu().numpy() if detection.boxes.cls is not None else []
                        
                        for i, (box, score) in enumerate(zip(boxes, scores)):
                            x1, y1, x2, y2 = map(float, box)
                            class_id = int(classes[i]) if i < len(classes) else 0
                            
                            frame_result['detections'].append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(score),
                                'class_id': class_id,
                                'thermal_signature': self._analyze_thermal_signature(enhanced_frame, box)
                            })
            
            detection_results.append(frame_result)
            
            # Visualize results
            annotated_frame = self._visualize_results_enhanced(frame, results)
            
            if output_path:
                out.write(annotated_frame)
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        
        return {
            'total_frames': frame_count,
            'total_detections': sum(len(f['detections']) for f in detection_results),
            'frames': detection_results[:100]  # Limit to first 100 frames for performance
        }
    
    def _analyze_thermal_signature(self, thermal_frame: np.ndarray, bbox: np.ndarray) -> dict:
        """Analyze thermal signature within bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract ROI
        roi = thermal_frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {'avg_temp': 0, 'max_temp': 0, 'min_temp': 0}
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Calculate temperature statistics (simulated)
        avg_intensity = np.mean(roi_gray)
        max_intensity = np.max(roi_gray)
        min_intensity = np.min(roi_gray)
        
        # Convert intensity to temperature (simplified mapping)
        temp_scale = 50  # Scale factor for temperature mapping
        avg_temp = (avg_intensity / 255.0) * temp_scale + 20  # 20-70°C range
        max_temp = (max_intensity / 255.0) * temp_scale + 20
        min_temp = (min_intensity / 255.0) * temp_scale + 20
        
        return {
            'avg_temp': round(avg_temp, 1),
            'max_temp': round(max_temp, 1),
            'min_temp': round(min_temp, 1),
            'thermal_intensity': round(avg_intensity, 1)
        }
    
    def _visualize_results_enhanced(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Enhanced visualization with thermal information"""
        annotated_frame = frame.copy()
        
        # Add angle type information
        angle_type = results.get('angle_type', 'unknown')
        cv2.putText(annotated_frame, f'Angle: {angle_type}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detections with enhanced info
        if 'detections' in results and results['detections']:
            detection_count = 0
            for detection in results['detections']:
                if detection.boxes is not None:
                    boxes = detection.boxes.xyxy.cpu().numpy()
                    scores = detection.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = map(int, box)
                        detection_count += 1
                        
                        # Draw bounding box with thermal colors
                        color = self._get_thermal_color(score)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add detection info
                        label = f'Detection {detection_count}: {score:.2f}'
                        cv2.putText(annotated_frame, label, 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Add thermal signature analysis
                        thermal_info = self._analyze_thermal_signature(frame, box)
                        temp_text = f"Temp: {thermal_info['avg_temp']}°C"
                        cv2.putText(annotated_frame, temp_text, 
                                   (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add detection summary
            cv2.putText(annotated_frame, f'Total Detections: {detection_count}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return annotated_frame
    
    def _get_thermal_color(self, confidence: float) -> tuple:
        """Get color based on thermal confidence"""
        if confidence > 0.8:
            return (0, 0, 255)  # Red for high confidence
        elif confidence > 0.6:
            return (0, 165, 255)  # Orange for medium confidence
        else:
            return (0, 255, 255)  # Yellow for low confidence

if __name__ == "__main__":
    tracker = ThermalTrackMultiAngle()
    
    # Process video file
    video_path = "input/thermal_video.mp4"
    output_path = "output/multi_angle_detection.avi"
    
    if os.path.exists(video_path):
        tracker.process_video(video_path, output_path)
    else:
        print(f"Video file not found: {video_path}")
