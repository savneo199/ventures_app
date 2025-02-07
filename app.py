from flask import Flask, request, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

latest_frame = None  # Store the latest processed frame
frame_lock = threading.Lock()  # Ensure thread safety

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
        # Convert base64 image to OpenCV format
        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        # If pose landmarks are detected, draw them
        if result.pose_landmarks:
            print("✅ Pose detected - Drawing landmarks")
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            print("⚠️ No pose landmarks detected")

        # Convert frame back to BGR for OpenCV display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Store the processed frame safely
        with frame_lock:
            latest_frame = frame.copy()

        return "OK"

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

                # Encode the frame as JPEG
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()

            print("🔄 Sending processed frame to video feed")

            # Send the latest frame immediately
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run Flask with SSL (HTTPS)
    app.run(host='0.0.0.0', port=5001, ssl_context=('cert.pem', 'key.pem'), debug=True)
