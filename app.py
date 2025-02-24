from flask import Flask, request, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
from hand_recognition import detect_hand  # Your existing hand detection module
from fitness_feedback import analyze_pose  # Import our new fitness feedback module
from waitress import serve

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

latest_frame = None  # Global variable to store the latest processed frame
frame_lock = threading.Lock()  # Thread lock for safe frame sharing

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame
    data = request.json.get('image', '')

    if not data:
        print("❌ No image data received")
        return "No image received", 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame = cv2.flip(frame, 1)  # Correct mirroring

        # Convert to RGB and process with MediaPipe Pose
        frame_proc = frame.copy()
        rgb_frame = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        feedback = ""
        if result.pose_landmarks:
            print("✅ Pose detected - Drawing landmarks")
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Get fitness feedback for the squat
            feedback = analyze_pose(result.pose_landmarks, frame.shape[1], frame.shape[0])
            # Removed overlaying the feedback text directly on the frame

        # Run hand detection
        hand_detected, frame = detect_hand(frame)
        if hand_detected:
            print("✅ Hand detected - Updating frame")

        # Update latest_frame safely
        with frame_lock:
            latest_frame = frame.copy()
            print("✅ latest_frame updated!")

        # Return both hand detection and fitness feedback in the JSON response
        return jsonify({"hand_detected": hand_detected, "feedback": feedback})

    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return "Error processing frame", 500

@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                # Encode the latest processed frame as JPEG
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, ssl_context=('cert.pem', 'key.pem'), debug=True)
    # Alternatively, use Waitress for production:
    # serve(app, host='0.0.0.0', port=5001)
