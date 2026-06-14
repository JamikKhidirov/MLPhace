import sys, os, threading, time
sys.path.insert(0, os.path.dirname(__file__))
import cv2
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

cap = cv2.VideoCapture(0)
frame_count = 0

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

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
    tr = TextRenderer(frame)

    # === Детекция лиц ===
    face_result = face_detector.detect(mp_img)
    landmarks_list = face_result.face_landmarks
    num_faces = len(landmarks_list) if landmarks_list else 0

    if landmarks_list:
        for fi, landmarks in enumerate(landmarks_list):
            xs = [int(p.x * w) for p in landmarks]
            ys = [int(p.y * h) for p in landmarks]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # Зелёный квадрат
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Человек {fi + 1}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # AR фильтр
            pts = [(p.x, p.y) for p in landmarks]
            apply_filter(frame, pts, current_filter)

            # Распознавание лиц
            if frame_count % config.COMPARE_INTERVAL == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 10:
                    threading.Thread(target=check_known, args=(crop.copy(), fi), daemon=True).start()

            if fi in recognized_names:
                name, sim = recognized_names[fi]
                cv2.putText(frame, f"{name} ({sim:.0%})", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === Эмоции + Возраст + Пол ===
    if frame_count % config.EMOTION_INTERVAL == 0:
        analyzer.analyze_async(frame)
    emotion_ru, emoji, age, gender_ru, history, _ = analyzer.get_state()
    tr.text(f"{emoji} {emotion_ru} | {age} {gender_ru}", 10, 30, (255, 255, 255), scale=1.0)

    # === График эмоций ===
    draw_mood_graph(frame, history)

    # === Счётчик людей ===
    cv2.putText(frame, f"Людей: {num_faces}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === FPS + Статус ===
    cv2.putText(frame, f"FPS: {fps_counter.fps:.0f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if recorder.is_recording:
        cv2.putText(frame, "⏺ ЗАПИСЬ", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if current_filter != FILTER_NONE:
        cv2.putText(frame, FILTER_NAMES[current_filter], (w - 150, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # === PIL текст ===
    tr.apply(frame)
    recorder.write_frame(frame)
    fps_counter.tick()
    cv2.imshow("MLPhace", frame)

    key = cv2.waitKey(5) & 0xFF
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
    elif key in (ord("c"),):
        if landmarks_list:
            xs = [int(p.x * w) for p in landmarks_list[0]]
            ys = [int(p.y * h) for p in landmarks_list[0]]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
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
