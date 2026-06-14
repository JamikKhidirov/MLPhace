import sys, os, threading
sys.path.insert(0, os.path.dirname(__file__))
os.environ["PYTHONIOENCODING"] = "utf-8"
import cv2
import numpy as np
from mediapipe.tasks.python.vision.core import image as mp_image
from core.face import FaceDetector
from core.analyzer import FaceAnalyzer
from ui.text_renderer import TextRenderer
from ui.effects import apply_filter, FILTER_NONE, FILTER_NAMES
from ui.mood_graph import draw_mood_graph
from utils.fps import FPSCounter
from utils.recorder import Recorder
from utils.recognizer import FaceDB
import config

print("MLPhace — люди зелёные квадраты")
print("  S - скриншот   R - запись   F - фильтр   C - запомнить лицо")

face_detector = FaceDetector()
analyzer = FaceAnalyzer()
recorder = Recorder()
facedb = FaceDB()
fps_counter = FPSCounter()

current_filter = FILTER_NONE
filter_names = list(FILTER_NAMES.keys())
filter_idx = 0
recognized_names = {}

# Haar Cascade (очень быстрый) — для детекции лиц в каждом кадре
haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
face_boxes = []  # [(x1,y1,x2,y2), ...]
filter_pts = []  # landmarks для фильтров (только когда фильтр включён)

def check_known(crop, fid):
    global recognized_names
    emb = analyzer.get_face_embedding(crop)
    if emb is not None:
        name, sim = facedb.recognize(emb)
        if name:
            recognized_names[fid] = (name, sim)

def save_known(crop, name):
    emb = analyzer.get_face_embedding(crop)
    if emb is not None:
        facedb.add(name, emb)
        print(f"'{name}' сохранён")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)

    # === Быстрая детекция лиц (Haar Cascade) — каждый кадр ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_boxes = haar.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))
    face_boxes = [(x, y, x + ex, y + ey) for x, y, ex, ey in raw_boxes]
    num_faces = len(face_boxes)

    # === Медленная детекция (MediaPipe) — только для фильтров ===
    if current_filter != FILTER_NONE and num_faces > 0 and frame_count % 2 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
        mp_result = face_detector.detect(mp_img)
        filter_pts = []
        if mp_result.face_landmarks:
            for landmarks in mp_result.face_landmarks:
                pts = [(p.x, p.y) for p in landmarks]
                filter_pts.append(pts)
    elif current_filter == FILTER_NONE:
        filter_pts = []

    # === Рисуем зелёные квадраты ===
    for fi, (x1, y1, x2, y2) in enumerate(face_boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Человек {fi + 1}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if current_filter != FILTER_NONE and fi < len(filter_pts):
            apply_filter(frame, filter_pts[fi], current_filter)

        # Распознавание
        if frame_count % (config.COMPARE_INTERVAL * 2) == 0:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 10:
                threading.Thread(target=check_known, args=(crop.copy(), fi), daemon=True).start()

        if fi in recognized_names:
            name, sim = recognized_names[fi]
            cv2.putText(frame, f"{name} ({sim:.0%})", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # === Эмоции (раз в 30 кадров) ===
    if frame_count % config.EMOTION_INTERVAL == 0:
        analyzer.analyze_async(frame)
    emotion_ru, emoji, age, gender_ru, history, _ = analyzer.get_state()
    tr = TextRenderer(frame)
    tr.put(f"{emoji} {emotion_ru} | {age} {gender_ru}", (10, 30), (255, 255, 255), 18)
    tr.apply(frame)

    draw_mood_graph(frame, history)

    # === Оверлеи ===
    cv2.putText(frame, f"Людей: {num_faces}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps_counter.fps:.0f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if recorder.is_recording:
        cv2.putText(frame, "ЗАПИСЬ", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if current_filter != FILTER_NONE:
        cv2.putText(frame, FILTER_NAMES[current_filter], (w - 140, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    recorder.write_frame(frame)
    fps_counter.tick()
    cv2.imshow("MLPhace", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord("s"),):
        recorder.screenshot(frame)
    elif key in (ord("r"),):
        if recorder.is_recording:
            recorder.stop()
        else:
            recorder.start(w, h)
    elif key in (ord("f"),):
        filter_idx = (filter_idx + 1) % len(filter_names)
        current_filter = filter_names[filter_idx]
        print(f"Фильтр: {FILTER_NAMES[current_filter]}")
    elif key in (ord("c"),) and face_boxes:
        x1, y1, x2, y2 = face_boxes[0]
        crop = frame[y1:y2, x1:x2]
        if crop.size > 10:
            print("Имя для лица:")
            name = input().strip()
            if name:
                threading.Thread(target=save_known, args=(crop.copy(), name), daemon=True).start()

cap.release()
cv2.destroyAllWindows()
face_detector.close()
recorder.stop()
