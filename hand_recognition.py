import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def detect_hand(frame):
    """
    Detects if a hand is present in the given frame.
    Draws hand landmarks if detected.

    Returns:
        - (bool) `hand_detected`: True if a hand is detected, else False.
        - (numpy.ndarray) `frame`: Processed frame with drawn hand landmarks.
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame using MediaPipe Hands
    result = hands.process(rgb_frame)

    hand_detected = False
    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return hand_detected, frame  # Return detection status & updated frame