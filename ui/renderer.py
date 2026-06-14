import cv2
import numpy as np

import config
from core.hands import classify_gesture, GESTURE_EMOJI, HAND_COLORS


def draw_pose(frame, landmarks, connections, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in connections.POSE_LANDMARKS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (255, 200, 80), 2)
    for pt in pts:
        cv2.circle(frame, pt, 3, (200, 150, 50), -1)


def draw_face_mesh(frame, landmarks, connections, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in connections.FACE_LANDMARKS_TESSELATION:
        cv2.line(frame, pts[conn.start], pts[conn.end], (40, 120, 40), 1)
    for conn in connections.FACE_LANDMARKS_CONTOURS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 1, (0, 200, 0), -1)
    x1 = min(p[0] for p in pts)
    y1 = min(p[1] for p in pts)
    x2 = max(p[0] for p in pts)
    y2 = max(p[1] for p in pts)
    return x1, y1, x2, y2, pts


def draw_hands(frame, tr, landmarks_list, handedness_list, w, h):
    for landmarks, handedness in zip(landmarks_list, handedness_list):
        h_name = handedness[0].category_name
        color = HAND_COLORS.get(h_name, (200, 200, 200))
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, color, -1)
        wrist = landmarks[0]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        gesture = classify_gesture(landmarks, h_name)
        emoji = GESTURE_EMOJI.get(gesture, "")
        prefix = "Л" if h_name == "Left" else "П"
        text = f"{prefix} {emoji} {gesture}" if gesture else prefix
        tr.put(text, (wx, wy - 20), color, 14)


def draw_emotion_info(tr, emotion_ru, age, gender_ru):
    y = 40
    if emotion_ru:
        emoji = config.EMOTION_EMOJI.get(emotion_ru, "")
        tr.put(f"Настроение: {emoji} {emotion_ru}", (20, y), (0, 255, 255), 22)
        y += 36
    if age is not None:
        tr.put(f"Возраст: ~{int(age)} лет", (20, y), (200, 255, 200), 18)
        y += 30
    if gender_ru:
        tr.put(f"Пол: {gender_ru}", (20, y), (200, 200, 255), 18)
        y += 30
    return y


def draw_blendshapes(tr, blendshapes, x1, y1):
    if not blendshapes:
        return
    high = [(b.category_name, b.score) for b in blendshapes if b.score > 0.5]
    if not high:
        return
    y_off = y1 - 20
    for name, score in high[:3]:
        ru = config.BLENDSHAPES_RU.get(name, name)
        tr.put(f"{ru}: {score:.2f}", (x1, y_off), (0, 255, 255), 11)
        y_off -= 18


def draw_head_info(tr, head_pose_ru, gaze_ru, x1, y1):
    tr.put(f"Голова: {head_pose_ru}", (x1, y1 - 6), (255, 200, 100), 12)
    tr.put(f"Взгляд: {gaze_ru}", (x1, y1 - 22), (255, 200, 100), 12)


def draw_extra_info(tr, mouth_open, smiling, tongue, x1, y2):
    labels = []
    if mouth_open:
        labels.append("рот открыт")
    if smiling:
        labels.append("улыбается")
    if tongue:
        labels.append("язык")
    if labels:
        tr.put(", ".join(labels), (x1, y2 + 18), (100, 255, 255), 11)


def draw_stats_panel(tr, num_faces, num_hands, num_poses):
    tr.put(f"Людей: {num_faces}  Рук: {num_hands}  Тел: {num_poses}",
           (920, 30), (200, 200, 200), 16)


def draw_fps(tr, fps):
    tr.put(f"FPS: {fps:.1f}", (20, 640), (100, 255, 100), 16)


def draw_status(tr, recording, w):
    if recording:
        tr.put("REC", (w - 80, 18), (0, 0, 255), 18)


def draw_help(tr, h):
    tr.put("S-скрин  R-запись  F-фильтры  C-запомнить  ESC-выход",
           (20, h - 20), (150, 150, 150), 13)


def draw_face_id(tr, fid, x1, y1, color):
    tr.put(f"#{fid}", (x1 - 30, y1 - 6), color, 16)


def draw_filter_name(tr, filter_name):
    tr.put(f"Фильтр: {filter_name}", (20, 640), (200, 150, 255), 14)


def draw_known_name(tr, name, sim, x1, y2):
    tr.put(f"{name} ({sim:.0%})", (x1, y2 + 40), (255, 255, 0), 14)
