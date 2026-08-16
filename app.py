from flask import Flask, render_template, Response, jsonify
import cv2

from utils.predictor import (
    process_frame,
    reset,
    toggle_voice
)

app = Flask(__name__)

camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Check camera
if not camera.isOpened():
    print("⚠️ Camera could not be opened.")


# -----------------------------
# FRAME GENERATOR
# -----------------------------
def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            print("⚠️ Failed to read camera frame.")
            continue

        # AI processing
        # NOTE: process_frame() already flips the frame,
        # so we don't flip it here again.
        frame = process_frame(frame)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


# -----------------------------
# TEST ROUTE
# -----------------------------
@app.route('/test')
def test():
    return "Flask Working!"


# -----------------------------
# CLEAR BUTTON
# -----------------------------
@app.route('/clear')
def clear():
    reset()

    return jsonify({
        "status": "cleared",
        "message": "Word cleared successfully"
    })


# -----------------------------
# VIDEO STREAM
# -----------------------------
@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# -----------------------------
# REAL-TIME DATA API
# -----------------------------
@app.route('/get_data')
def get_data():

    from utils.predictor import (
        current_word,
        last_pred,
        current_confidence,
        total_predictions_count,
        get_session_duration,
        voice_enabled
    )

    return jsonify({
        "prediction": last_pred,
        "word": current_word,
        "confidence": round(
            float(current_confidence * 100), 1
        ),
        "stats": {
            "total": total_predictions_count,
            "duration": get_session_duration(),
            "voice": voice_enabled
        }
    })


# -----------------------------
# VOICE TOGGLE API
# -----------------------------
@app.route('/toggle_voice')
def toggle_voice_api():

    status = toggle_voice()

    return jsonify({
        "voice": status
    })


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    print("🚀 Sign Language Translator starting...")
    print("🌐 Open: http://127.0.0.1:8000")

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )