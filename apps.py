from flask import Flask, render_template, request, send_file, redirect, url_for, flash, jsonify
import os
import logging
import shutil

# Import both video processors for redundancy
try:
    from video_processor import process_video as process_video_main
except ImportError as e:
    logging.error(f"Error importing video_processor: {e}")
    process_video_main = None

try:
    from video_processors import process_video as process_video_alt
    from video_processors import load_model as load_alt_model
except ImportError as e:
    logging.error(f"Error importing video_processors: {e}")
    process_video_alt = None
    load_alt_model = None

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'video_file' not in request.files:
        flash('No file uploaded!')
        return redirect(url_for('index'))
    
    file = request.files['video_file']
    if file.filename == '':
        flash('No file selected!')
        return redirect(url_for('index'))
    
    filename = file.filename
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)
    
    # Try to process with both processors for redundancy
    output_path = None
    error_messages = []
    
    # Try main processor first
    if process_video_main is not None:
        try:
            output_path = process_video_main(upload_path)
            if output_path and os.path.exists(output_path):
                print(f"Successfully processed with main processor: {output_path}")
            else:
                error_messages.append("Main processor failed to produce output")
        except Exception as e:
            error_messages.append(f"Error with main processor: {str(e)}")
    else:
        error_messages.append("Main video processor not available")
    
    # If main processor failed, try alternative processor
    if output_path is None and process_video_alt is not None:
        try:
            # Try to load the model first
            if load_alt_model and load_alt_model():
                alt_output_path = process_video_alt(upload_path, filename)
                if alt_output_path and os.path.exists(alt_output_path):
                    output_path = alt_output_path
                    print(f"Successfully processed with alternative processor: {output_path}")
                else:
                    error_messages.append("Alternative processor failed to produce output")
            else:
                error_messages.append("Failed to load model for alternative processor")
        except Exception as e:
            error_messages.append(f"Error with alternative processor: {str(e)}")
    
    # Return the processed video if successful
    if output_path and os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    else:
        error_message = "\n".join(error_messages)
        print(f"Video processing failed: {error_message}")
        flash(f'Error processing video: {error_message}')
        return redirect(url_for('index'))

@app.route('/api/process', methods=['POST'])
def api_process():
    """API endpoint for processing videos"""
    if 'video_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['video_file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = file.filename
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)
    
    # Try both processors
    main_output = None
    alt_output = None
    errors = []
    
    # Try main processor
    if process_video_main is not None:
        try:
            main_output = process_video_main(upload_path)
        except Exception as e:
            errors.append(f"Main processor error: {str(e)}")
    
    # Try alternative processor
    if process_video_alt is not None:
        try:
            if load_alt_model and load_alt_model():
                alt_output = process_video_alt(upload_path, filename)
        except Exception as e:
            errors.append(f"Alternative processor error: {str(e)}")
    
    # Return results
    if main_output or alt_output:
        return jsonify({
            "success": True,
            "main_processor": {
                "success": main_output is not None,
                "output_path": main_output if main_output else None
            },
            "alt_processor": {
                "success": alt_output is not None,
                "output_path": alt_output if alt_output else None
            }
        })
    else:
        return jsonify({
            "success": False,
            "errors": errors
        }), 500

if __name__ == '__main__':
    app.run(debug=True)