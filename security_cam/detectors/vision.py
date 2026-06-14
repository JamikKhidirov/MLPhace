import time
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from config import (
    FIRE_HSV_LOWER, FIRE_HSV_UPPER, SMOKE_HSV_LOWER, SMOKE_HSV_UPPER,
    FIRE_MIN_AREA, FIRE_MIN_FRAMES, FIRE_COOLDOWN,
    ABANDONED_STILL_FRAMES, ABANDONED_PERSON_RADIUS, ABANDONED_COOLDOWN,
    ENABLE_FIRE, ENABLE_ABANDONED
)


class FireDetector:
    def __init__(self, on_alert=None):
        self.on_alert = on_alert
        self.cooldowns = {}
        self.consecutive_fire = 0
        self.fire_lower = np.array(FIRE_HSV_LOWER)
        self.fire_upper = np.array(FIRE_HSV_UPPER)
        self.smoke_lower = np.array(SMOKE_HSV_LOWER)
        self.smoke_upper = np.array(SMOKE_HSV_UPPER)

    def process(self, frame):
        if not ENABLE_FIRE:
            return frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv, self.fire_lower, self.fire_upper)
        smoke_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)

        fire_mask = cv2.erode(fire_mask, None, iterations=1)
        fire_mask = cv2.dilate(fire_mask, None, iterations=2)

        fire_area = cv2.countNonZero(fire_mask)
        smoke_area = cv2.countNonZero(smoke_mask)

        if fire_area > FIRE_MIN_AREA:
            self.consecutive_fire += 1
            if self.consecutive_fire >= FIRE_MIN_FRAMES:
                if time.time() - self.cooldowns.get("fire", 0) > FIRE_COOLDOWN:
                    self.cooldowns["fire"] = time.time()
                    if self.on_alert:
                        self.on_alert("fire", "🔥 Пожар! Обнаружен огонь", fire_area / 10000)
                contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > FIRE_MIN_AREA:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(frame, "🔥 ОГОНЬ", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.consecutive_fire = max(0, self.consecutive_fire - 1)

        if smoke_area > FIRE_MIN_AREA * 2:
            if time.time() - self.cooldowns.get("smoke", 0) > FIRE_COOLDOWN:
                self.cooldowns["smoke"] = time.time()
                if self.on_alert:
                    self.on_alert("smoke", "💨 Дым! Возможное возгорание", smoke_area / 10000)
            contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > FIRE_MIN_AREA * 2:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                    cv2.putText(frame, "💨 ДЫМ", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        return frame


class AbandonedObjectDetector:
    def __init__(self, on_alert=None):
        self.on_alert = on_alert
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"⚠️ YOLO не загрузилась: {e}")
            self.model = None

        self.tracks = {}
        self.next_id = 0
        self.current_boxes = {}
        self.person_positions = []
        self.frame_count = 0
        self.cooldowns = {}

    def process(self, frame):
        if not ENABLE_ABANDONED or self.model is None:
            return frame

        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return frame

        results = self.model.track(frame, persist=True, verbose=False, classes=[0, 24, 26, 28, 31])
        if results is None or len(results) == 0:
            return frame

        result = results[0]
        if result.boxes is None:
            return frame

        person_boxes = []
        object_boxes = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else None

            if cls_id == 0:
                person_boxes.append((x1, y1, x2, y2, track_id))
            else:
                object_boxes.append((x1, y1, x2, y2, track_id, cls_id))

        self.person_positions = [( (x1 + x2) // 2, (y1 + y2) // 2 ) for x1, y1, x2, y2, _ in person_boxes]

        for ox1, oy1, ox2, oy2, tid, cls_id in object_boxes:
            if tid is None:
                continue
            if tid not in self.tracks:
                near_person = any(
                    abs((ox1 + ox2) // 2 - px) < ABANDONED_PERSON_RADIUS
                    and abs((oy1 + oy2) // 2 - py) < ABANDONED_PERSON_RADIUS
                    for px, py in self.person_positions
                )
                self.tracks[tid] = {
                    "first_seen": self.frame_count,
                    "appeared_with_person": near_person,
                    "last_moved": self.frame_count,
                    "last_pos": ((ox1 + ox2) // 2, (oy1 + oy2) // 2),
                    "label": result.names[cls_id],
                }
            else:
                track = self.tracks[tid]
                cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                dist = np.sqrt((cx - track["last_pos"][0])**2 + (cy - track["last_pos"][1])**2)
                if dist > 10:
                    track["last_moved"] = self.frame_count
                    track["last_pos"] = (cx, cy)

                person_nearby = any(
                    abs(cx - px) < ABANDONED_PERSON_RADIUS
                    and abs(cy - py) < ABANDONED_PERSON_RADIUS
                    for px, py in self.person_positions
                )

                if not person_nearby and track["appeared_with_person"]:
                    frames_still = self.frame_count - track["last_moved"]
                    if frames_still > ABANDONED_STILL_FRAMES:
                        key = f"abandoned_{tid}"
                        if time.time() - self.cooldowns.get(key, 0) > ABANDONED_COOLDOWN:
                            self.cooldowns[key] = time.time()
                            if self.on_alert:
                                self.on_alert(
                                    "abandoned",
                                    f"📦 Подозрительный предмет: {track['label']}",
                                    frames_still / ABANDONED_STILL_FRAMES
                                )
                        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 165, 255), 3)
                        cv2.putText(frame, f"! {track['label']}", (ox1, oy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return frame
