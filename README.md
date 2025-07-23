ThermalTrack: Advanced Heat Signature Detection 🔥🎯
ThermalTrack is an AI-powered sensor video image classification and object detection system designed for real-time defence and surveillance use cases. Leveraging infrared (IR) thermal imaging, object detection (YOLOv8), background subtraction, and image classification (ResNet50/MobileNetV2), this tool detects and classifies both static and moving objects in live or uploaded video streams. It also provides RGB-to-IR conversion, video stabilisation, and estimated temperature heat signatures.

🚀 Features
✅ Real-time Object Detection (YOLOv8)
✅ Image Classification using ResNet50/MobileNetV2
✅ Thermal Heat Signature Estimation
✅ Live Webcam Feed or Video Upload Support
✅ Infrared (IR) and RGB Mode Toggle
✅ Video Stabilisation for Moving Objects
✅ Static vs Moving Object Differentiation
✅ Responsive Futuristic UI with TailwindCSS
🖼️ Sample Output
Thermal Detection Demo
<img width="1920" height="1080" alt="Screenshot (28)" src="https://github.com/user-attachments/assets/a6aa7c1d-701d-428a-9945-f036586d1caf" />

<img width="1920" height="1080" alt="Screenshot (16)" src="https://github.com/user-attachments/assets/70f905fe-15cc-4934-ae1b-ef0d2f6f7629" />


🛠️ Tech Stack
Python
OpenCV
YOLOv8 (Ultralytics)
ResNet50 / MobileNetV2 (TorchVision)
Flask
Tailwind CSS + HTML
PIL (Image processing)
Torch & NumPy
📁 Folder Structure
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

⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/yourusername/ThermalTrack.git cd ThermalTrack

Create virtual environment (optional but recommended) python -m venv venv source venv/bin/activate # or venv\Scripts\activate on Windows

Install dependencies pip install -r requirements.txt

Download YOLOv8 model (if not already included)

Visit https://github.com/ultralytics/ultralytics to download 'yolov8n.pt'
Run the Flask app python app.py

Access via browser http://127.0.0.1:5000

📝 License This project is licensed under the MIT License. See LICENSE for more details.
