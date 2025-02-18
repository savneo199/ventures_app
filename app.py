from flask import Flask, request, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
from fitness_feedback import analyze_pose  # Import the fitness feedback module

app = Flask(__name__)

# Initialize MediaPipe Pose and its drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

latest_frame = None  # Global variable to store the latest processed frame
frame_lock = threading.Lock()  # Thread lock to safely update latest_frame

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
        # Decode the base64 image to a numpy array and then to a BGR image
        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Flip the frame horizontally (correct mirroring)
        frame = cv2.flip(frame, 1)

        # Convert the flipped frame to RGB for pose processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        feedback = ""
        if result.pose_landmarks:
            print("✅ Pose landmarks detected")
            # Draw the landmarks on the original (flipped) frame (which is still in BGR)
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Get fitness feedback (e.g., for squat form) using the detected landmarks
            height, width, _ = frame.shape
            feedback = analyze_pose(result.pose_landmarks, width, height)
            if feedback:
                cv2.putText(frame, feedback, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print("❌ No pose landmarks detected")

        # Safely update the global latest_frame
        with frame_lock:
            latest_frame = frame.copy()

        return jsonify({"feedback": feedback})
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
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run Flask over HTTPS using your certificate and key files
    app.run(host='0.0.0.0', port=5001, ssl_context=('cert.pem', 'key.pem'), debug=True)
