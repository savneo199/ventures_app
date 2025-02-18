import numpy as np
import math


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b given three points a, b, and c.
    Each point should be a tuple or list with (x, y) coordinates.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Create vectors BA and BC
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle using dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def analyze_pose(pose_landmarks, image_width, image_height):
    """
    Analyzes the pose landmarks to provide live fitness feedback for a squat.
    Expects pose_landmarks from MediaPipe Pose, along with the image dimensions.

    Returns:
        feedback (str): A string with feedback for proper squat form.
    """
    if not pose_landmarks:
        return ""

    try:
        # MediaPipe Pose landmark indices for squat:
        # Left hip: 23, Left knee: 25, Left ankle: 27
        # Right hip: 24, Right knee: 26, Right ankle: 28
        left_hip = pose_landmarks.landmark[23]
        left_knee = pose_landmarks.landmark[25]
        left_ankle = pose_landmarks.landmark[27]

        right_hip = pose_landmarks.landmark[24]
        right_knee = pose_landmarks.landmark[26]
        right_ankle = pose_landmarks.landmark[28]

        # Convert normalized coordinates to pixel values
        left_hip_coord = (left_hip.x * image_width, left_hip.y * image_height)
        left_knee_coord = (left_knee.x * image_width, left_knee.y * image_height)
        left_ankle_coord = (left_ankle.x * image_width, left_ankle.y * image_height)

        right_hip_coord = (right_hip.x * image_width, right_hip.y * image_height)
        right_knee_coord = (right_knee.x * image_width, right_knee.y * image_height)
        right_ankle_coord = (right_ankle.x * image_width, right_ankle.y * image_height)

        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip_coord, left_knee_coord, left_ankle_coord)
        right_knee_angle = calculate_angle(right_hip_coord, right_knee_coord, right_ankle_coord)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        # Provide simple feedback based on the average knee angle
        if avg_knee_angle > 160:
            feedback = "Stand tall. Begin your squat by bending your knees."
        elif 120 < avg_knee_angle <= 160:
            feedback = "Lower your hips further for a deeper squat."
        elif 90 < avg_knee_angle <= 120:
            feedback = "Good squat form!"
        elif avg_knee_angle <= 90:
            feedback = "Too low! Keep control to avoid injury."
        else:
            feedback = ""

        return feedback

    except Exception as e:
        print("Error in analyze_pose:", e)
        return ""
