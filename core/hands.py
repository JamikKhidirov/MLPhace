from mediapipe.tasks.python.vision import hand_landmarker as hl
from mediapipe.tasks.python.core import base_options as base_opts

import config


def classify_gesture(landmarks, handedness):
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    fingers = []

    for i in range(5):
        if i == 0:
            extended = (
                landmarks[tip_ids[0]].x < landmarks[pip_ids[0]].x
                if handedness == "Right"
                else landmarks[tip_ids[0]].x > landmarks[pip_ids[0]].x
            )
        else:
            extended = landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y
        fingers.append(extended)

    count = sum(fingers)
    if count == 0:
        return "Кулак"
    if count == 5:
        return "Ладонь"
    if fingers == [0, 1, 1, 0, 0]:
        return "Мир"
    if fingers == [0, 1, 0, 0, 0]:
        return "Тык"
    if fingers == [1, 0, 0, 0, 0]:
        return "Класс"
    if fingers == [1, 0, 0, 0, 1]:
        return "Позвони"
    if fingers == [0, 1, 1, 1, 0]:
        return "Три"
    if count >= 2:
        return str(count)
    return ""


GESTURE_EMOJI = {
    "Кулак": "✊",
    "Ладонь": "🖐",
    "Мир": "✌️",
    "Тык": "☝️",
    "Класс": "👍",
    "Позвони": "🤙",
    "Три": "🤟",
}

HAND_COLORS = {"Left": (255, 100, 100), "Right": (100, 100, 255)}


class HandDetector:
    def __init__(self):
        options = hl.HandLandmarkerOptions(
            base_options=base_opts.BaseOptions(
                model_asset_path=config.HAND_MODEL
            ),
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=0.5,
        )
        self._detector = hl.HandLandmarker.create_from_options(options)

    def detect(self, mp_image):
        return self._detector.detect(mp_image)

    def close(self):
        self._detector.close()
