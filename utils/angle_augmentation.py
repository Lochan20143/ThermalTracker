import cv2
import numpy as np
from typing import Tuple, List
import random

class AngleAugmentation:
    def __init__(self):
        self.perspective_variants = 5
        
    def simulate_viewing_angles(self, image: np.ndarray, 
                              bboxes: List = None) -> List[Tuple[np.ndarray, List]]:
        """
        Generate multiple viewing angle variants of the same image
        
        Args:
            image: Original thermal image
            bboxes: Bounding boxes in format [x1, y1, x2, y2, class]
            
        Returns:
            List of (augmented_image, adjusted_bboxes) tuples
        """
        h, w = image.shape[:2]
        augmented_data = []
        
        # Original image
        augmented_data.append((image.copy(), bboxes.copy() if bboxes else []))
        
        # Generate perspective transforms for different viewing angles
        for i in range(self.perspective_variants):
            # Create perspective transformation
            transform_matrix = self._generate_perspective_transform(w, h, i)
            
            # Apply transformation to image
            transformed_image = cv2.warpPerspective(image, transform_matrix, (w, h))
            
            # Transform bounding boxes if provided
            transformed_bboxes = []
            if bboxes:
                transformed_bboxes = self._transform_bboxes(bboxes, transform_matrix)
            
            augmented_data.append((transformed_image, transformed_bboxes))
        
        return augmented_data
    
    def _generate_perspective_transform(self, width: int, height: int, 
                                      variant: int) -> np.ndarray:
        """Generate perspective transformation matrix for different viewing angles"""
        # Define source points (corners of original image)
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        # Define destination points based on variant
        if variant == 0:  # Slight top-down angle
            dst_points = np.float32([
                [width * 0.1, height * 0.1],
                [width * 0.9, height * 0.1],
                [width * 0.8, height * 0.9],
                [width * 0.2, height * 0.9]
            ])
        elif variant == 1:  # Side angle simulation
            dst_points = np.float32([
                [width * 0.2, 0],
                [width * 0.8, height * 0.2],
                [width * 0.8, height * 0.8],
                [width * 0.2, height]
            ])
        elif variant == 2:  # Angled view
            dst_points = np.float32([
                [width * 0.15, height * 0.05],
                [width * 0.85, height * 0.15],
                [width * 0.9, height * 0.85],
                [width * 0.1, height * 0.95]
            ])
        elif variant == 3:  # Rotated perspective
            dst_points = np.float32([
                [width * 0.05, height * 0.2],
                [width * 0.8, height * 0.05],
                [width * 0.95, height * 0.8],
                [width * 0.2, height * 0.95]
            ])
        else:  # Extreme angle
            dst_points = np.float32([
                [width * 0.3, 0],
                [width * 0.7, height * 0.3],
                [width * 0.7, height * 0.7],
                [width * 0.3, height]
            ])
        
        return cv2.getPerspectiveTransform(src_points, dst_points)
    
    def _transform_bboxes(self, bboxes: List, transform_matrix: np.ndarray) -> List:
        """Transform bounding boxes according to perspective transformation"""
        transformed_bboxes = []
        
        for bbox in bboxes:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                class_id = bbox[4] if len(bbox) > 4 else 0
                
                # Create points for all corners of the bounding box
                corners = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.float32)
                
                # Transform corners
                corners_homogeneous = np.column_stack([corners, np.ones(4)])
                transformed_corners = transform_matrix @ corners_homogeneous.T
                
                # Convert back to cartesian coordinates
                transformed_corners = transformed_corners[:2] / transformed_corners[2]
                transformed_corners = transformed_corners.T
                
                # Find new bounding box
                min_x = np.min(transformed_corners[:, 0])
                min_y = np.min(transformed_corners[:, 1])
                max_x = np.max(transformed_corners[:, 0])
                max_y = np.max(transformed_corners[:, 1])
                
                # Ensure coordinates are within image bounds
                min_x = max(0, min_x)
                min_y = max(0, min_y)
                
                transformed_bboxes.append([min_x, min_y, max_x, max_y, class_id])
        
        return transformed_bboxes
    
    def enhance_thermal_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance thermal image contrast for better angle detection"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
