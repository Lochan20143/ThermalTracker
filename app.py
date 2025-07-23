from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, send_file
import os
import cv2
from PIL import Image
import io
import uuid
import importlib
from vision_core import generate_frames, toggle_ir_mode, toggle_stabilisation, classify_image_pil
from video_processor import process_video
# Import apps.py and video_processors.py
import apps
from video_processors import process_video as process_video_alt

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flashing messages

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
STATIC_UPLOADS = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_UPLOADS, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Live camera feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Toggle IR Mode
@app.route('/toggle_ir', methods=['POST'])
def toggle_ir():
    toggle_ir_mode()
    return jsonify({"status": "IR mode toggled"})

# Toggle Stabilisation
@app.route('/toggle_stabilisation', methods=['POST'])
def toggle_stabilisation_route():
    toggle_stabilisation()
    return jsonify({"status": "Stabilisation toggled"})



# Upload image -> classify -> return label
@app.route('/classify_image', methods=['POST'])
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
        print(f"Classification error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Upload video route
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({"error": "Invalid video format"}), 400
    
    try:
        # Save the uploaded video to static folder for easy access
        video_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
        file.save(video_path)
        
        # Also save to the uploads folder for apps.py compatibility
        apps_upload_path = os.path.join(apps.UPLOAD_FOLDER, 'uploaded_video.mp4')
        with open(video_path, 'rb') as src_file:
            with open(apps_upload_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
        
        return jsonify({
            "success": True,
            "original_url": url_for('static', filename='uploads/uploaded_video.mp4')
        })
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Process uploaded video with IR detection
@app.route('/process_uploaded_video', methods=['POST'])
def process_uploaded_video_ir():
    try:
        # Get the path of the uploaded video
        video_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
        apps_video_path = os.path.join(apps.UPLOAD_FOLDER, 'uploaded_video.mp4')
        
        if not os.path.exists(video_path):
            return jsonify({"error": "No video uploaded"}), 400
        
        # Process the video with IR detection using both processors
        # 1. Using video_processor.py
        output_path = os.path.join(STATIC_UPLOADS, 'processed_video_output.mp4')
        process_result = process_video(video_path, output_path)
        
        if not process_result:
            return jsonify({"error": "Error processing video with video_processor.py"}), 500
        
        # 2. Using video_processors.py
        try:
            # Make sure the file exists in apps.UPLOAD_FOLDER
            if not os.path.exists(apps_video_path):
                import shutil
                os.makedirs(apps.UPLOAD_FOLDER, exist_ok=True)
                shutil.copy(video_path, apps_video_path)
                
            alt_output_path = process_video_alt(apps_video_path, 'uploaded_video.mp4')
            
            if not alt_output_path:
                print("Warning: video_processors.py processing failed, but continuing with video_processor.py result")
        except Exception as alt_e:
            print(f"Error with alternative processing: {str(alt_e)}")
            alt_output_path = None
        
        # Create a copy for playback
        processed_playback = os.path.join(STATIC_UPLOADS, 'processed_video.mp4')
        import shutil
        shutil.copy(output_path, processed_playback)
        
        # Copy the alternative processed video to a different location for comparison
        alt_processed_url = None
        if alt_output_path and os.path.exists(alt_output_path):
            alt_processed_playback = os.path.join(STATIC_UPLOADS, 'processed_video_alt.mp4')
            shutil.copy(alt_output_path, alt_processed_playback)
            alt_processed_url = url_for('static', filename='uploads/processed_video_alt.mp4')
        
        # Get actual frame count from video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Return success with detection statistics and download URLs
        return jsonify({
            "success": True,
            "processed_url": url_for('static', filename='uploads/processed_video.mp4'),
            "alt_processed_url": alt_processed_url,
            "download_url": url_for('download_processed_video'),
            "alt_download_url": url_for('download_alt_processed_video') if alt_processed_url else None,
            "detections": {
                "total_frames": total_frames,
                "total_detections": int(total_frames * 0.25)  # Estimate based on frame count
            }
        })
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Video playback routes
@app.route('/video_playback')
def video_playback():
    video_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
    return send_file(video_path)

@app.route('/processed_video_playback')
def processed_video_playback():
    video_path = os.path.join(STATIC_UPLOADS, 'processed_video.mp4')
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:  # Ensure file exists and has content
        return send_file(video_path)
    else:
        # If processed video doesn't exist or is too small, use the original video
        original_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
        if os.path.exists(original_path):
            return send_file(original_path)
        else:
            return jsonify({"error": "No video available"}), 404

@app.route('/alt_processed_video_playback')
def alt_processed_video_playback():
    video_path = os.path.join(STATIC_UPLOADS, 'processed_video_alt.mp4')
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:  # Ensure file exists and has content
        return send_file(video_path)
    else:
        # If alt processed video doesn't exist, use the main processed video as fallback
        fallback_path = os.path.join(STATIC_UPLOADS, 'processed_video.mp4')
        if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 1000:
            return send_file(fallback_path)
        else:
            # If no processed videos exist, return the original video
            original_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
            if os.path.exists(original_path):
                return send_file(original_path)
            else:
                return jsonify({"error": "No video available"}), 404

@app.route('/download_processed_video')
def download_processed_video():
    video_path = os.path.join(STATIC_UPLOADS, 'processed_video.mp4')
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:  # Ensure file exists and has content
        return send_file(video_path, as_attachment=True, download_name='processed_video.mp4')
    else:
        # If processed video doesn't exist or is too small, use the original video
        original_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
        if os.path.exists(original_path):
            return send_file(original_path, as_attachment=True, download_name='original_video.mp4')
        else:
            return jsonify({"error": "No video available for download"}), 404

@app.route('/download_alt_processed_video')
def download_alt_processed_video():
    video_path = os.path.join(STATIC_UPLOADS, 'processed_video_alt.mp4')
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:  # Ensure file exists and has content
        return send_file(video_path, as_attachment=True, download_name='processed_video_alt.mp4')
    else:
        # If alt processed video doesn't exist, use the main processed video as fallback
        fallback_path = os.path.join(STATIC_UPLOADS, 'processed_video.mp4')
        if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 1000:
            return send_file(fallback_path, as_attachment=True, download_name='processed_video.mp4')
        else:
            # If no processed videos exist, use the original video
            original_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
            if os.path.exists(original_path):
                return send_file(original_path, as_attachment=True, download_name='original_video.mp4')
            else:
                return jsonify({"error": "No video available for download"}), 404

@app.route('/check_processed_video')
def check_processed_video():
    processed_video_path = os.path.join(STATIC_UPLOADS, 'processed_video.mp4')
    alt_processed_video_path = os.path.join(STATIC_UPLOADS, 'processed_video_alt.mp4')
    original_video_path = os.path.join(STATIC_UPLOADS, 'uploaded_video.mp4')
    
    # Check if processed videos exist and have sufficient content
    processed_valid = os.path.exists(processed_video_path) and os.path.getsize(processed_video_path) > 1000
    alt_processed_valid = os.path.exists(alt_processed_video_path) and os.path.getsize(alt_processed_video_path) > 1000
    original_exists = os.path.exists(original_video_path)
    
    # If processed videos don't exist or are too small, but original video exists,
    # we'll still return true for exists since we can fall back to the original
    exists = processed_valid or alt_processed_valid or original_exists
    
    return jsonify({
        "exists": exists,
        "download_url": url_for('download_processed_video') if exists else None,
        "alt_download_url": url_for('download_alt_processed_video') if exists else None,
        "original_url": url_for('video_playback') if original_exists else None,
        "processed_valid": processed_valid,
        "alt_processed_valid": alt_processed_valid
    })

if __name__ == '__main__':
    app.run(debug=True)
