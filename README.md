# ThermalTrack: Advanced Heat Signature Detection 🔥🎯

ThermalTrack is an AI-powered sensor video image classification and object detection system designed for real-time defence and surveillance use cases. Leveraging infrared (IR) thermal imaging, object detection (YOLOv8), background subtraction, and image classification (ResNet50/MobileNetV2), this tool detects and classifies both static and moving objects in live or uploaded video streams. It also provides RGB-to-IR conversion, video stabilisation, and estimated temperature heat signatures.

## 🚀 Features

- ✅ **Real-time Object Detection** (YOLOv8)
- ✅ **Image Classification** using ResNet50/MobileNetV2
- ✅ **Thermal Heat Signature Estimation**
- ✅ **Live Webcam Feed or Video Upload Support**
- ✅ **Infrared (IR) and RGB Mode Toggle**
- ✅ **Video Stabilisation for Moving Objects**
- ✅ **Static vs Moving Object Differentiation**
- ✅ **Responsive Futuristic UI with TailwindCSS**

## 🖼️ Sample Output

<img width="1920" height="1080" alt="Screenshot (16)" src="https://github.com/user-attachments/assets/02fc505f-fd06-43d2-8afe-a23953d0a281" />

<img width="1920" height="1080" alt="Screenshot (27)" src="https://github.com/user-attachments/assets/931ae754-3380-44bc-8bab-20e097b246a6" />



## 🛠️ Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- ResNet50 / MobileNetV2 (TorchVision)
- Flask
- Tailwind CSS + HTML
- PIL (Image processing)
- Torch & NumPy

## 📁 Folder Structure
ThermalTrack/
├── app.py # Flask backend
├── vision_core.py # Object detection, IR mode, classification
├── templates/
│ └── index.html # Frontend UI
├── static/
│ ├── uploads/ # User uploaded videos/images
│ └── processed/ # Output videos after processing
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## ⚙️ Setup Instructions

1. Clone the repository

git clone https://github.com/yourusername/ThermalTrack.git
cd ThermalTrack

2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies
pip install -r requirements.txt

4. Download YOLOv8 model (if not already included)
   # Visit https://github.com/ultralytics/ultralytics to download 'yolov8n.pt'

5. Run the Flask app
python app.py

6. Access via browser
 http://127.0.0.1:5000
  
📝 License
This project is licensed under the MIT License. See LICENSE for more details.
