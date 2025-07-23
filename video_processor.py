import cv2
import numpy as np
import os
import logging
from ultralytics import YOLO
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_FOLDER = 'processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    logger.error(f"YOLOv8 model loading failed: {e}")
    model = None

def estimate_temp(intensity):
    return round((intensity / 255) * 100, 1)

def process_video(video_path, output_path=None):
    if model is None:
        logger.error("YOLOv8 model not loaded.")
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        # If output_path is not provided, generate one in the PROCESSED_FOLDER
        if output_path is None:
            output_name = os.path.basename(video_path)
            output_name = secure_filename(output_name)
            output_path = os.path.join(PROCESSED_FOLDER, f"processed_{output_name}")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_boxes = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = (mag > 2.0).astype(np.uint8) * 255

            results = model(thermal, conf=0.5, verbose=False)[0]
            annotated = thermal.copy()
            new_boxes = []
            hidden_alert = False

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} ({conf:.2f})"
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(annotated.shape[1] - 1, x2), min(annotated.shape[0] - 1, y2)

                box_area = (x2 - x1) * (y2 - y1)
                motion_threshold = max(100, int(box_area * 0.05))
                roi_motion = motion_mask[y1:y2, x1:x2]
                motion_pixels = cv2.countNonZero(roi_motion)
                status = "Moving" if motion_pixels > motion_threshold else "Static"

                temp_pixel = gray[center[1], center[0]]
                temperature = estimate_temp(temp_pixel)

                color = (0, 0, 255) if status == "Moving" else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} ({status})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Temp: {temperature} °C", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                new_boxes.append((center, cls, conf))

            for old_center, old_cls, old_conf in prev_boxes:
                if all(np.linalg.norm(np.array(old_center) - np.array(new_center)) > 60
                       for new_center, _, _ in new_boxes):
                    hidden_alert = True

            if hidden_alert:
                cv2.putText(annotated, "⚠ Hidden Object Alert!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            prev_boxes = new_boxes
            prev_gray = gray
            out.write(annotated)

        cap.release()
        out.release()
        return output_path

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return None