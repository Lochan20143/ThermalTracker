
# ThermalTrack: Heat Signature & Object Detection System

## Features
- Real-time RGB ↔ IR simulation
- Video stabilisation
- YOLOv8 object detection
- Image classification using MobileNetV2
- Video upload and frame-wise playback

## Setup Instructions

### 1. Clone and Install
```bash
git clone <repo_url>
cd ThermalTrack_HeatSignature
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Access the UI
Open your browser to `http://localhost:5000`

## Notes
- Make sure `imagenet_classes.txt` is present for image classification.
- YOLOv8 uses `yolov8n.pt` by default. You can replace it with `yolov8s.pt` or `yolov8m.pt` for better accuracy.
