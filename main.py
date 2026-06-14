    1   import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import threading

from mediapipe.tasks.python.vision.core import image as mp_image

from core.face import (
    FaceDetector,
    estimate_head_pose,
    estimate_gaze,
    is_mouth_open,
    is_smiling,
    is_tongue_out,
)
from core.hands import HandDetector
from core.pose import PoseDetector
from core.analyzer import FaceAnalyzer

from ui.text_renderer import TextRenderer
from ui.renderer import (
    draw_pose,
    draw_face_mesh,
    draw_hands,
    draw_emotion_info,
    draw_blendshapes,
    draw_head_info,
    draw_extra_info,
    draw_stats_panel,
    draw_fps,
    draw_status,
    draw_help,
    draw_face_id,
    draw_filter_name,
    draw_known_name,
)
from ui.effects import apply_filter, FILTER_NONE, FILTER_NAMES
from ui.mood_graph import draw_mood_graph

from utils.fps import FPSCounter
from utils.tracker import FaceTracker
from utils.recorder import Recorder
from utils.recognizer import FaceDB

import config

print("╔══════════════════════════════════════╗")
print("║        MLPhace Ultra v3.0            ║")
print("║   Лицо · Руки · Тело · Эмоции       ║")
print("╚══════════════════════════════════════╝")
print("  S - скриншот   R - запись видео")
print("  F - переключить фильтры")
print("  C - запомнить лицо")
print("  ESC - выход")

face_detector = FaceDetector()
hand_detector = HandDetector()
pose_detector = PoseDetector()
analyzer = FaceAnalyzer()
tracker = FaceTracker()
recorder = Recorder()
facedb = FaceDB()
fps_counter = FPSCounter()

current_filter = FILTER_NONE
filter_names_list = list(FILTER_NAMES.keys())
filter_idx = 0
recognized_names = {}

cap = cv2.VideoCapture(0)
frame_count = 0


def _check_known_face(crop, fid):
    global recognized_names
    emb = analyzer.get_face_embedding(crop)
    if emb is not None:
        name, sim = facedb.recognize(emb)
        if name:
            recognized_names[fid] = (name, sim)


def _save_known_face(crop, name):
    emb = analyzer.get_face_embedding(crop)
    if emb is not None:
        facedb.add(name, emb)
        print(f"Лицо '{name}' сохранено в базу")
    else:
        print("Не удалось получить эмбеддинг лица")


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)

    tr = TextRenderer(frame)

    # --- Body pose ---
    pose_result = pose_detector.detect(mp_img)
    num_poses = len(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks:
            draw_pose(frame, lm, pose_detector.connections, w, h)

    # --- Face mesh ---
    face_result = face_detector.detect(mp_img)
    face_boxes = {}
    face_ids_map = {}
    num_faces = len(face_result.face_landmarks) if face_result.face_landmarks else 0

    if face_result.face_landmarks:
        boxes = []
        for fi, landmarks in enumerate(face_result.face_landmarks):
            x1, y1, x2, y2, pts = draw_face_mesh(
                frame, landmarks, face_detector.connections, w, h
            )
            boxes.append((x1, y1, x2, y2))
            face_boxes[fi] = (x1, y1, x2, y2)

            bs = (
                face_result.face_blendshapes[fi]
                if face_result.face_blendshapes and fi < len(face_result.face_blendshapes)
                else None
            )

            draw_blendshapes(tr, bs, x1, y1)
            apply_filter(frame, pts, current_filter)

            head_pose = estimate_head_pose(pts)
            gaze = estimate_gaze(pts, bs)
            draw_head_info(
                tr,
                config.HEAD_POSE_RU.get(head_pose, head_pose),
                gaze, x1, y1,
            )

            mouth_open = is_mouth_open(bs) if bs else False
            smiling = is_smiling(bs) if bs else False
            tongue = is_tongue_out(bs) if bs else False
            draw_extra_info(tr, mouth_open, smiling, tongue, x1, y2)

        ids = tracker.update(boxes)
        for fi, fid in enumerate(ids):
            face_ids_map[fid] = fi
            bx1, by1, bx2, by2 = boxes[fi]
            color = tracker.get_id_color(fid)
            draw_face_id(tr, fid, bx1, by1, color)

            if fid in recognized_names:
                name, sim = recognized_names[fid]
                draw_known_name(tr, name, sim, bx1, by2)

            if frame_count % 30 == 0 and fid in face_ids_map:
                crop = frame[by1:by2, bx1:bx2]
                if crop.size > 10:
                    threading.Thread(
                        target=_check_known_face,
                        args=(crop.copy(), fid),
                        daemon=True,
                    ).start()

    # --- Hands ---
    hand_result = hand_detector.detect(mp_img)
    num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
    if hand_result.hand_landmarks:
        draw_hands(frame, tr, hand_result.hand_landmarks, hand_result.handedness, w, h)

    # --- Emotion + Age + Gender ---
    if frame_count % 30 == 0:
        analyzer.analyze_async(frame)

    emotion_ru, emotion_en, age, gender_ru, history, gender_en = analyzer.get_state()
    draw_emotion_info(tr, emotion_ru, age, gender_ru)
    draw_mood_graph(frame, history)

    # --- Overlays ---
    draw_stats_panel(tr, num_faces, num_hands, num_poses)
    draw_fps(tr, fps_counter.fps)
    draw_status(tr, recorder.is_recording, w)
    draw_help(tr, h)
    draw_filter_name(tr, FILTER_NAMES.get(current_filter, ""))

    # --- PIL text → frame ---
    tr.apply(frame)

    recorder.write_frame(frame)
    fps_counter.tick()
    cv2.imshow("MLPhace Ultra", frame)
    key = cv2.waitKey(5) & 0xFF

    if key == 27:
        break
    elif key in (ord("s"), ord("ы")):
        path = recorder.screenshot(frame)
        print(f"Скриншот: {path}")
    elif key in (ord("r"), ord("к")):
        if recorder.is_recording:
            recorder.stop()
            print("Запись остановлена")
        else:
            path = recorder.start(w, h)
            print(f"Запись: {path}")
    elif key in (ord("f"), ord("а")):
        filter_idx = (filter_idx + 1) % len(filter_names_list)
        current_filter = filter_names_list[filter_idx]
        print(f"Фильтр: {FILTER_NAMES[current_filter]}")
    elif key in (ord("c"), ord("с")):
        if face_result.face_landmarks:
            pts = face_result.face_landmarks[0]
            xs = [int(p.x * w) for p in pts]
            ys = [int(p.y * h) for p in pts]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 10:
                print("Введите имя для этого лица:")
                try:
                    name = input().strip()
                    if name:
                        threading.Thread(
                            target=_save_known_face,
                            args=(crop.copy(), name),
                            daemon=True,
                        ).start()
                except EOFError:
                    pass

cap.release()
cv2.destroyAllWindows()
face_detector.close()
hand_detector.close()
pose_detector.close()
recorder.stop()
