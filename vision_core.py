# --- vision_core.py (Updated for YOLOv5x Default + Output Saving) ---
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image

# Load YOLOv8 model (optional)
yolo_v8_model = YOLO('yolov8n.pt')

# Load YOLOv5x model (default for UAV object detection)
yolo_v5_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
yolo_v5_model.conf = 0.3
yolo_v5_model.iou = 0.45

cap = cv2.VideoCapture(0)
back_sub = cv2.createBackgroundSubtractorMOG2()

IR_MODE = True
STABILISE = False
USE_YOLOV5 = True  # Enable YOLOv5x by default
CONFIDENCE_THRESHOLD = 0.5

# Load MobileNetV2 model
mobilenet = mobilenet_v2(pretrained=True)
mobilenet.eval()

# Load ImageNet labels
with open("imagenet_classes.txt", "r") as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def toggle_ir_mode():
    global IR_MODE
    IR_MODE = not IR_MODE

def toggle_stabilisation():
    global STABILISE
    STABILISE = not STABILISE

def toggle_model():
    global USE_YOLOV5
    USE_YOLOV5 = not USE_YOLOV5

def estimate_temp(intensity):
    return round((intensity / 255) * 100, 1)

def classify_image_pil(pil_image):
    image = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = mobilenet(image)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, 5)
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        if idx < len(imagenet_labels):
            return imagenet_labels[idx], round(top_probs[i].item() * 100, 2)
    return "Unknown object", round(top_probs[0].item() * 100, 2)

def generate_frames():
    global IR_MODE, STABILISE, USE_YOLOV5
    prev_gray = None

    # Save processed output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'static/uploads/processed_video_output.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        if IR_MODE:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        else:
            display = frame.copy()

        if STABILISE:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                valid_prev = prev_pts[status == 1]
                valid_curr = curr_pts[status == 1]
                m, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
                if m is not None:
                    display = cv2.warpAffine(display, m, (display.shape[1], display.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            prev_gray = curr_gray

        fg_mask = back_sub.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if USE_YOLOV5:
            results = yolo_v5_model(frame)
            boxes_data = results.xyxy[0].cpu().numpy()
            annotated = display.copy()
            for box in boxes_data:
                x1, y1, x2, y2, conf, cls = box[:6]
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                label = yolo_v5_model.names[int(cls)]
                crop = frame[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                alt_label, score = classify_image_pil(pil_crop)
                if score > 50:
                    label = f"{alt_label} (classified)"
                color = tuple(np.random.randint(128, 255, size=3).tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            results = yolo_v8_model(frame, verbose=False)[0]
            annotated = display.copy()
            for box in results.boxes:
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = yolo_v8_model.names[cls]
                crop = frame[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                alt_label, score = classify_image_pil(pil_crop)
                if score > 50:
                    label = f"{alt_label} (classified)"
                color = tuple(np.random.randint(128, 255, size=3).tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(annotated)
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()
