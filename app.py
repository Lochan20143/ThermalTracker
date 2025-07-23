from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import os
import cv2  # Add this import
from PIL import Image  # Move this import to the top
import io  # Move this import to the top
from vision_core import generate_frames, toggle_ir_mode, toggle_stabilisation, classify_image_pil  # Add classify_image_pil here
from main_multi_angle import ThermalTrackMultiAngle  # Add this import

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize multi-angle thermal tracker
thermal_tracker = ThermalTrackMultiAngle()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_ir', methods=['POST'])
def toggle_ir():
    toggle_ir_mode()
    return jsonify({"status": "IR mode toggled"})

@app.route('/toggle_stabilisation', methods=['POST'])
def toggle_stabilisation_route():
    toggle_stabilisation()
    return jsonify({"status": "Stabilisation toggled"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files.get('file')
    # Defensive: check file and filename
    if not file or not hasattr(file, 'filename') or not file.filename:
        return jsonify({"status": "No file provided"}), 400
    # Defensive: check extension
    filename = file.filename.lower()
    if not (filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov')):
        return jsonify({"status": "Invalid file format"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, 'uploaded_video.mp4')
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"status": f"Failed to save video: {str(e)}"}), 500
    return jsonify({
        "status": "Video uploaded",
        "original_url": url_for('video_playback'),
        "processed_url": url_for('processed_video_playback')
    })

@app.route('/video_playback')
def video_playback():
    def generate_uploaded_video():
        cap = cv2.VideoCapture(os.path.join(UPLOAD_FOLDER, 'uploaded_video.mp4'))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    return Response(generate_uploaded_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_video_playback')
def processed_video_playback():
    """Stream processed video with IR detection"""
    def generate_processed_video():
        processed_path = os.path.join(UPLOAD_FOLDER, 'processed_video.mp4')
        if not os.path.exists(processed_path):
            return
            
        cap = cv2.VideoCapture(processed_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    
    return Response(generate_processed_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_uploaded_video', methods=['POST'])
def process_uploaded_video():
    """Process uploaded video with IR detection"""
    try:
        video_path = os.path.join(UPLOAD_FOLDER, 'uploaded_video.mp4')
        if not os.path.exists(video_path):
            return jsonify({"error": "No video uploaded"}), 400

        # Check if the uploaded file is actually a video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "No video file found or file is not a video"}), 400
        cap.release()
        
        processed_path = os.path.join(UPLOAD_FOLDER, 'processed_video.mp4')
        
        # Defensive: check if video file is not empty and can be opened
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Uploaded video cannot be opened or is corrupted"}), 400
        ret, _ = cap.read()
        cap.release()
        if not ret:
            return jsonify({"error": "Uploaded video is empty or unreadable"}), 400
        
        # Process with multi-angle IR detection
        detection_results = thermal_tracker.process_video_with_results(video_path, processed_path)
        
        # Defensive: ensure detection_results is valid and has expected keys
        if not detection_results or not isinstance(detection_results, dict):
            return jsonify({"error": "Processing failed, no results returned"}), 500
        if 'total_frames' in detection_results and detection_results['total_frames'] == 0:
            return jsonify({"error": "No frames detected in video"}), 400
        
        return jsonify({
            "status": "Video processed successfully",
            "detections": detection_results,
            "processed_url": url_for('processed_video_playback')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classify_image', methods=['POST'])  # Move this route before the if __name__ block
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        label, confidence = classify_image_pil(image)
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        print(f"Classification error: {str(e)}")  # For debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=True)
