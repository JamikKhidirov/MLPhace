import time
import cv2
import numpy as np
import easyocr
from config import (
    LPR_MIN_CONFIDENCE, LPR_MIN_PLATE_AREA, LPR_WHITELIST,
    LPR_COOLDOWN_SAME_PLATE, ENABLE_LPR
)


class PlateReader:
    def __init__(self, on_alert=None):
        self.on_alert = on_alert
        self.reader = None
        self.last_plates = {}
        self.cooldowns = {}
        self.loaded = False

    def load_model(self):
        if not ENABLE_LPR:
            return False
        try:
            print("🔢 Загрузка EasyOCR (первый раз может быть долго)...")
            self.reader = easyocr.Reader(["en", "ru"], gpu=False)
            self.loaded = True
            return True
        except Exception as e:
            print(f"⚠️ EasyOCR не загрузилась: {e}")
            return False

    def _find_plates_contour(self, gray):
        candidates = []
        edged = cv2.Canny(gray, 100, 200)
        edged = cv2.dilate(edged, None, iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < LPR_MIN_PLATE_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h
            if 1.5 < aspect < 6.0 and h > 20:
                candidates.append((x, y, w, h))
        return candidates

    def process(self, frame):
        if not ENABLE_LPR or not self.loaded:
            return frame, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self._find_plates_contour(gray)

        result_plates = []
        for x, y, w, h in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plate_roi = gray[y:y + h, x:x + w]
            if plate_roi.size < LPR_MIN_PLATE_AREA:
                continue
            plate_rgb = cv2.cvtColor(plate_roi, cv2.COLOR_GRAY2RGB)
            plate_rgb = cv2.resize(plate_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, plate_rgb = cv2.threshold(plate_rgb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                results = self.reader.readtext(plate_rgb)
                for bbox, text, conf in results:
                    if conf < LPR_MIN_CONFIDENCE:
                        continue
                    text = text.strip().upper().replace(" ", "")
                    if len(text) < 4:
                        continue
                    is_whitelisted = text in LPR_WHITELIST
                    key = f"plate_{text}"
                    if time.time() - self.cooldowns.get(key, 0) > LPR_COOLDOWN_SAME_PLATE:
                        self.cooldowns[key] = time.time()
                        status = "✅ СВОЙ" if is_whitelisted else "⚠️ ЧУЖОЙ"
                        label = f"{text} {status}"
                        if self.on_alert:
                            self.on_alert(
                                "plate",
                                f"🚗 Номер {status}: {text}",
                                conf
                            )
                        result_plates.append((text, conf, is_whitelisted))
                    color = (0, 255, 0) if is_whitelisted else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{text}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception:
                pass

        return frame, result_plates
