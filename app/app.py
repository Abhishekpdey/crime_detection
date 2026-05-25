# app.py

import os
import cv2
import time
import logging
import requests
import streamlink
import numpy as np

from urllib.parse import urlparse
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_from_directory
)

# FIXED IMPORTS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------------------------------
# LOGGING
# ---------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# FLASK APP
# ---------------------------------------------------

app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)

# ---------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------

live_stream = None
live_stream_url = None
should_stop_live_stream = False
current_prediction = "No prediction yet"

# ---------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crime_detection_model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

# ---------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# ---------------------------------------------------

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ---------------------------------------------------
# LABELS
# ---------------------------------------------------

LABELS = ["crime", "non_crime"]

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    else:
        logger.error(f"❌ Model file not found: {MODEL_PATH}")

except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")

# ---------------------------------------------------
# STREAM URL EXTRACTION
# ---------------------------------------------------

def get_direct_stream_url(webpage_url):
    """
    Extract direct stream URL.
    """

    try:
        streams = streamlink.streams(webpage_url)

        if streams:
            logger.info("✅ Stream URL extracted using streamlink")
            return streams["best"].url

    except Exception as e:
        logger.warning(f"Streamlink failed: {str(e)}")

    try:
        parsed_url = urlparse(webpage_url)

        if 'earthcam.com' in webpage_url:

            cam_id = webpage_url.split('cam=')[1].split('&')[0]

            api_url = f"https://api.earthcam.com/cameras/public/{cam_id}/stream"

            response = requests.get(api_url)

            if response.status_code == 200:

                data = response.json()

                stream_url = data.get('url')

                if stream_url:
                    logger.info("✅ Stream URL extracted from EarthCam API")
                    return stream_url

    except Exception as e:
        logger.warning(f"EarthCam extraction failed: {str(e)}")

    return webpage_url

# ---------------------------------------------------
# LOAD VIDEO FRAMES
# ---------------------------------------------------

def load_video_frames(video_path, max_frames=30):

    frames = []

    try:
        cap = cv2.VideoCapture(video_path)

        count = 0

        while count < max_frames:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (64, 64))

            frame = img_to_array(frame)

            frame = frame.astype("float32") / 255.0

            frames.append(frame)

            count += 1

        cap.release()

        if len(frames) < max_frames:
            logger.warning(f"Only {len(frames)} frames found")

        return np.array(frames)

    except Exception as e:
        logger.error(f"Error loading frames: {str(e)}")
        return np.array([])

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

def predict_crime(frames):

    if model is None:
        return "Model not loaded."

    if len(frames) < 30:
        return "Insufficient frames for prediction."

    try:

        frames = np.expand_dims(frames, axis=0)

        predictions = model.predict(frames)

        average_prediction = np.mean(predictions)

        result = (
            "No crime detected."
            if average_prediction >= 0.30
            else "Crime detected!"
        )

        logger.info(f"Prediction: {result}")

        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Prediction failed."

# ---------------------------------------------------
# LIVE STREAM PROCESSING
# ---------------------------------------------------

def process_live_stream():

    global live_stream
    global should_stop_live_stream
    global current_prediction
    global live_stream_url

    frames_buffer = []

    reconnect_attempts = 0
    max_reconnect_attempts = 5

    while not should_stop_live_stream:

        try:

            if live_stream is None or not live_stream.isOpened():

                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error("❌ Max reconnect attempts reached")
                    break

                if live_stream_url:

                    logger.info("Reconnecting stream...")

                    live_stream = cv2.VideoCapture(live_stream_url)

                    reconnect_attempts += 1

                time.sleep(1)

                continue

            ret, frame = live_stream.read()

            if not ret:
                continue

            processed_frame = cv2.resize(frame, (64, 64))

            processed_frame = img_to_array(processed_frame)

            processed_frame = processed_frame.astype("float32") / 255.0

            frames_buffer.append(processed_frame)

            if len(frames_buffer) >= 30:

                prediction = predict_crime(np.array(frames_buffer))

                current_prediction = prediction

                frames_buffer = frames_buffer[15:]

            _, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame +
                b'\r\n'
            )

            reconnect_attempts = 0

        except Exception as e:

            logger.error(f"Live stream error: {str(e)}")

            time.sleep(1)

    if live_stream is not None and live_stream.isOpened():
        live_stream.release()

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.route('/')
def index():
    return render_template('frontend1.html')

# ---------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video = request.files['file']

    if video.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:

        video_path = os.path.join(
            UPLOAD_FOLDER,
            video.filename
        )

        video.save(video_path)

        frames = load_video_frames(video_path)

        if len(frames) == 0:
            return jsonify({'error': 'Frame extraction failed'}), 400

        result = predict_crime(frames)

        if os.path.exists(video_path):
            os.remove(video_path)

        return jsonify({'result': result})

    except Exception as e:

        logger.error(f"Video processing error: {str(e)}")

        return jsonify({'error': 'Video processing failed'}), 500

# ---------------------------------------------------

@app.route('/start_live_stream', methods=['POST'])
def start_live_stream():

    global live_stream
    global live_stream_url
    global should_stop_live_stream
    global current_prediction

    data = request.get_json()

    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:

        if live_stream is not None and live_stream.isOpened():
            live_stream.release()

        should_stop_live_stream = False

        direct_url = get_direct_stream_url(url)

        if not direct_url:
            return jsonify({'error': 'Could not extract stream'}), 400

        live_stream = cv2.VideoCapture(direct_url)

        live_stream_url = direct_url

        current_prediction = "No prediction yet"

        if not live_stream.isOpened():
            return jsonify({'error': 'Failed to connect'}), 400

        logger.info("✅ Stream started")

        return jsonify({'message': 'Stream started successfully'})

    except Exception as e:

        logger.error(f"Stream start error: {str(e)}")

        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------

@app.route('/stop_live_stream', methods=['POST'])
def stop_live_stream_route():

    global live_stream
    global should_stop_live_stream

    should_stop_live_stream = True

    if live_stream is not None and live_stream.isOpened():
        live_stream.release()

    live_stream = None

    logger.info("✅ Stream stopped")

    return jsonify({'message': 'Stream stopped successfully'})

# ---------------------------------------------------

@app.route('/video_feed')
def video_feed():

    return Response(
        process_live_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ---------------------------------------------------

@app.route('/get_current_prediction')
def get_current_prediction():

    global current_prediction

    return jsonify({
        'prediction': current_prediction
    })

# ---------------------------------------------------

@app.route('/static/<path:filename>')
def serve_static(filename):

    return send_from_directory(
        app.static_folder,
        filename
    )

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
