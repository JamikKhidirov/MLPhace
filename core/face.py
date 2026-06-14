import numpy as np

from mediapipe.tasks.python.vision import face_landmarker as fl
from mediapipe.tasks.python.core import base_options as base_opts

import config


def estimate_head_pose(face_pts):
    nose = face_pts[1]
    x1 = min(p[0] for p in face_pts)
    y1 = min(p[1] for p in face_pts)
    x2 = max(p[0] for p in face_pts)
    y2 = max(p[1] for p in face_pts)

    face_w = x2 - x1
    face_h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    dx = (nose[0] - cx) / face_w if face_w > 0 else 0
    dy = (nose[1] - cy) / face_h if face_h > 0 else 0

    if abs(dx) < 0.08 and abs(dy) < 0.08:
        return "center"
    if abs(dx) > abs(dy):
        return "left" if dx < -0.08 else "right"
    return "up" if dy < -0.08 else "down"


def estimate_gaze(face_pts, face_blendshapes=None):
    if face_blendshapes:
        bs_map = {b.category_name: b.score for b in face_blendshapes}
        left = bs_map.get("eyeLookOutLeft", 0) + bs_map.get("eyeLookInRight", 0)
        right = bs_map.get("eyeLookInLeft", 0) + bs_map.get("eyeLookOutRight", 0)
        up = bs_map.get("eyeLookUpLeft", 0) + bs_map.get("eyeLookUpRight", 0)
        down = bs_map.get("eyeLookDownLeft", 0) + bs_map.get("eyeLookDownRight", 0)
        total = left + right + up + down
        if total > 0.5:
            if max(left, right, up, down) == left:
                return "влево"
            if max(left, right, up, down) == right:
                return "вправо"
            if max(left, right, up, down) == up:
                return "вверх"
            if max(left, right, up, down) == down:
                return "вниз"
    return "прямо"


def is_mouth_open(face_blendshapes, threshold=0.5):
    if not face_blendshapes:
        return False
    bs_map = {b.category_name: b.score for b in face_blendshapes}
    return bs_map.get("jawOpen", 0) > threshold


def is_smiling(face_blendshapes, threshold=0.4):
    if not face_blendshapes:
        return False
    bs_map = {b.category_name: b.score for b in face_blendshapes}
    left = bs_map.get("mouthSmileLeft", 0)
    right = bs_map.get("mouthSmileRight", 0)
    return (left + right) / 2 > threshold


def is_tongue_out(face_blendshapes, threshold=0.3):
    if not face_blendshapes:
        return False
    bs_map = {b.category_name: b.score for b in face_blendshapes}
    return bs_map.get("tongueOut", 0) > threshold


class FaceDetector:
    def __init__(self):
        options = fl.FaceLandmarkerOptions(
            base_options=base_opts.BaseOptions(
                model_asset_path=config.FACE_MODEL
            ),
            num_faces=config.NUM_FACES,
            output_face_blendshapes=True,
        )
        self._detector = fl.FaceLandmarker.create_from_options(options)
        self.connections = fl.FaceLandmarksConnections

    def detect(self, mp_image):
        return self._detector.detect(mp_image)

    def close(self):
        self._detector.close()
