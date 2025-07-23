import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import math

class MultiAngleDetector:
    def __init__(self, model_paths: Dict[str, str]):
        """
        Initialize multi-angle detector with different YOLO models
        
        Args:
            model_paths: Dict with keys like 'top', 'side', 'angled' and model file paths
        """
        self.models = {}
        self.load_models(model_paths)
        self.angle_threshold = 30  # degrees
        
    def load_models(self, model_paths: Dict[str, str]):
        """Load different YOLO models for different viewing angles"""
        for angle_type, path in model_paths.items():
            try:
                self.models[angle_type] = YOLO(path)
                print(f"Loaded {angle_type} model from {path}")
            except Exception as e:
                print(f"Failed to load {angle_type} model: {e}")
    
    def estimate_viewing_angle(self, image: np.ndarray) -> str:
        """
        Estimate viewing angle based on image features
        
        Args:
            image: Input thermal image
            
        Returns:
            Estimated viewing angle type ('top', 'side', 'angled')
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect edges to analyze image geometry
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'top'  # Default to top view
            
        # Analyze aspect ratios and orientations
        aspect_ratios = []
        orientations = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                angle = rect[2]
                
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    aspect_ratios.append(aspect_ratio)
                    orientations.append(abs(angle))
        
        if not aspect_ratios:
            return 'top'
            
        avg_aspect_ratio = np.mean(aspect_ratios)
        avg_orientation = np.mean(orientations)
        
        # Decision logic based on features
        if avg_aspect_ratio > 2.0 and avg_orientation > 45:
            return 'side'
        elif avg_orientation > 20 and avg_aspect_ratio > 1.5:
            return 'angled'
        else:
            return 'top'
    
    def detect_with_angle_adaptation(self, image: np.ndarray) -> Dict:
        """
        Perform detection using the most appropriate model based on viewing angle
        
        Args:
            image: Input thermal image
            
        Returns:
            Detection results with confidence scores
        """
        angle_type = self.estimate_viewing_angle(image)
        
        # Use the appropriate model
        if angle_type in self.models:
            model = self.models[angle_type]
        else:
            # Fallback to first available model
            model = list(self.models.values())[0]
            angle_type = list(self.models.keys())[0]
        
        # Run detection
        results = model(image)
        
        return {
            'detections': results,
            'angle_type': angle_type,
            'model_used': angle_type
        }
    
    def ensemble_detect(self, image: np.ndarray) -> Dict:
        """
        Run all models and combine results using ensemble method
        
        Args:
            image: Input thermal image
            
        Returns:
            Combined detection results
        """
        all_results = {}
        
        # Run detection with all available models
        for angle_type, model in self.models.items():
            results = model(image)
            all_results[angle_type] = results
        
        # Combine results using weighted average based on confidence
        combined_detections = self._combine_detections(all_results, image)
        
        return combined_detections
    
    def _combine_detections(self, results_dict: Dict, image: np.ndarray) -> Dict:
        """Combine detections from multiple models"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        estimated_angle = self.estimate_viewing_angle(image)
        
        for angle_type, results in results_dict.items():
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # Weight scores based on estimated viewing angle
                    weight = self._get_angle_weight(angle_type, estimated_angle)
                    weighted_scores = scores * weight
                    
                    all_boxes.append(boxes)
                    all_scores.append(weighted_scores)
                    all_classes.append(classes)
        
        if not all_boxes:
            return {'detections': [], 'combined': True}
        
        # Concatenate all detections
        all_boxes = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)
        
        # Apply NMS to remove duplicates
        indices = cv2.dnn.NMSBoxes(
            all_boxes.tolist(),
            all_scores.tolist(),
            score_threshold=0.5,
            nms_threshold=0.4
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            final_boxes = all_boxes[indices]
            final_scores = all_scores[indices]
            final_classes = all_classes[indices]
        else:
            final_boxes = []
            final_scores = []
            final_classes = []
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'classes': final_classes,
            'combined': True,
            'estimated_angle': estimated_angle
        }
    
    def _get_angle_weight(self, model_angle: str, estimated_angle: str) -> float:
        """Get weight for model based on estimated viewing angle"""
        if model_angle == estimated_angle:
            return 1.0
        elif (model_angle == 'angled' and estimated_angle in ['top', 'side']) or \
             (estimated_angle == 'angled' and model_angle in ['top', 'side']):
            return 0.7
        else:
            return 0.3
