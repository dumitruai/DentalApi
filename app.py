import os
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import uuid
import cv2
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'
MODEL_PATH = 'model/best.pt'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload and annotated directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Set up logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Load YOLOv8 model once at startup
try:
    model = YOLO(MODEL_PATH)
    app.logger.info(f"Loaded YOLOv8 model from {MODEL_PATH}")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    raise e


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File Too Large'}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/detect', methods=['POST'])
def detect():
    # Check if the post request has the file part
    if 'image' not in request.files:
        app.logger.warning("No image part in the request")
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    # If user does not select file, browser may submit an empty part without filename
    if file.filename == '':
        app.logger.warning("No selected image")
        return jsonify({'error': 'No selected image'}), 400

    if file and allowed_file(file.filename):
        # Generate a unique filename to prevent collisions
        filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded image
        try:
            file.save(filepath)
            app.logger.info(f"Saved uploaded image to {filepath}")
        except Exception as e:
            app.logger.error(f"Error saving uploaded image: {e}")
            return jsonify({'error': 'Failed to save uploaded image'}), 500

        # Perform inference
        try:
            results = model.predict(source=filepath, save=False, verbose=False)

            # Check if any detections were made
            if results and results[0].boxes:
                # Get the annotated image as a numpy array
                annotated_image = results[0].plot()

                # Convert RGB (from YOLO) to BGR (for OpenCV)
                annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

                # Save the annotated image to annotated folder
                annotated_filename = 'annotated_' + filename
                annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_filename)
                cv2.imwrite(annotated_path, annotated_image_bgr)
                app.logger.info(f"Saved annotated image to {annotated_path}")

                # Prepare the image to send back
                return send_file(
                    annotated_path,
                    mimetype='image/jpeg',
                    as_attachment=True,
                    download_name=annotated_filename
                )
            else:
                # No detections made, return the original image
                app.logger.info(f"No detections found in {filename}. Returning original image.")
                return send_file(
                    filepath,
                    mimetype='image/jpeg',
                    as_attachment=True,
                    download_name=file.filename
                )
        except Exception as e:
            app.logger.error(f"Error during inference: {e}")
            return jsonify({'error': 'Failed to perform inference'}), 500
    else:
        app.logger.warning("Invalid file type uploaded")
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/', methods=['GET'])
def home():
    return '''
    <!doctype html>
    <title>YOLOv8 Object Detection</title>
    <h1>Upload an Image for Object Detection</h1>
    <form method=post enctype=multipart/form-data action="/detect">
      <input type=file name=image>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    # Run the Flask app
    # For production, use a WSGI server like Gunicorn and disable debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)