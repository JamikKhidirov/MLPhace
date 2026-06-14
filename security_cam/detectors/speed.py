import time
import cv2
import numpy as np
from ultralytics import YOLO
from config import (
    SPEED_DISTANCE_METERS, SPEED_LINE1_Y, SPEED_LINE2_Y,
    SPEED_MIN_CONFIDENCE, SPEED_COOLDOWN, ENABLE_SPEED
)


class SpeedDetector:
    def __init__(self, on_alert=None):
        self.on_alert = on_alert
        self.model = None
        self.loaded = False
        self.crossings = {}
        self.cooldowns = {}
        self.line1_y = 0
        self.line2_y = 0
        self.frame_h = 0

    def load_model(self):
        if not ENABLE_SPEED:
            return False
        try:
            self.model = YOLO("yolov8n.pt")
            self.loaded = True
            return True
        except Exception as e:
            print(f"⚠️ Speed: YOLO не загрузилась: {e}")
            return False

    def process(self, frame):
        if not ENABLE_SPEED or not self.loaded:
            return frame

        h, w = frame.shape[:2]
        if self.frame_h != h:
            self.frame_h = h
            self.line1_y = int(h * SPEED_LINE1_Y)
            self.line2_y = int(h * SPEED_LINE2_Y)

        cv2.line(frame, (0, self.line1_y), (w, self.line1_y), (255, 255, 0), 2)
        cv2.putText(frame, "ЗОНА 1", (10, self.line1_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.line(frame, (0, self.line2_y), (w, self.line2_y), (255, 255, 0), 2)
        cv2.putText(frame, "ЗОНА 2", (10, self.line2_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        results = self.model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
        if results is None or len(results) == 0:
            return frame

        result = results[0]
        if result.boxes is None:
            return frame

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < SPEED_MIN_CONFIDENCE:
                continue
            tid = int(box.id[0]) if box.id is not None else None
            if tid is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"#{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if tid not in self.crossings:
                self.crossings[tid] = {"status": 0, "time1": None, "time2": None}
            track = self.crossings[tid]

            if track["status"] == 0 and cy >= self.line1_y:
                track["status"] = 1
                track["time1"] = time.time()
            elif track["status"] == 1 and cy >= self.line2_y:
                track["status"] = 2
                track["time2"] = time.time()
                dt = track["time2"] - track["time1"]
                if dt > 0:
                    speed_ms = SPEED_DISTANCE_METERS / dt
                    speed_kmh = speed_ms * 3.6
                    if 5 < speed_kmh < 250:
                        if time.time() - self.cooldowns.get("speed", 0) > SPEED_COOLDOWN:
                            self.cooldowns["speed"] = time.time()
                            if self.on_alert:
                                self.on_alert(
                                    "speed",
                                    f"🚗 Скорость #{tid}: {speed_kmh:.0f} км/ч",
                                    speed_kmh / 100
                                )
                        cv2.putText(frame, f"{speed_kmh:.0f} км/ч", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif track["status"] == 2:
                pass

        return frame

    def reset(self):
        self.crossings.clear()
