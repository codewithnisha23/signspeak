import cv2
import mediapipe as mp
import numpy as np

from .config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE


class HandTracker:
    """Detects one hand and returns a normalized 63-value landmark vector."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, result

        hand = result.multi_hand_landmarks[0]
        points = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
        points -= points[0]  # make the wrist the origin
        scale = np.max(np.abs(points))
        if scale > 0:
            points /= scale
        return points.flatten(), result

    def draw(self, frame, result):
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )

    def close(self):
        self.hands.close()
