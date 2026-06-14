import os
import cv2
from datetime import datetime

import config


class Recorder:
    def __init__(self):
        self._writer = None
        self._recording = False
        self._w = None
        self._h = None

    @property
    def is_recording(self):
        return self._recording

    def start(self, w, h):
        if self._recording:
            return
        os.makedirs(config.RECORDINGS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.RECORDINGS_DIR, f"record_{ts}.avi")
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), config.RECORD_FPS, (w, h)
        )
        self._recording = True
        self._w, self._h = w, h
        return path

    def write_frame(self, frame):
        if self._recording and self._writer is not None:
            self._writer.write(frame)

    def stop(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording = False

    def screenshot(self, frame):
        os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.SCREENSHOTS_DIR, f"shot_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path
