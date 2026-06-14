import time
import threading
import requests
import cv2
import os
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED, ALERTS_DIR


class Notifier:
    def __init__(self):
        self.pending = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def set_frame(self, frame):
        with self.frame_lock:
            self.frame = frame.copy()

    def alert(self, category, text, confidence=1.0):
        with self.lock:
            self.pending.append((category, text, confidence, time.time()))

    def _send_telegram(self, text, photo_path=None):
        if not TELEGRAM_ENABLED:
            return
        try:
            if photo_path and os.path.exists(photo_path):
                with open(photo_path, "rb") as f:
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                        data={"chat_id": TELEGRAM_CHAT_ID, "caption": text},
                        files={"photo": f},
                        timeout=10
                    )
            else:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                    timeout=10
                )
        except Exception as e:
            print(f"⚠️ Telegram error: {e}")

    def _worker(self):
        while self.running:
            items = []
            with self.lock:
                items = self.pending.copy()
                self.pending.clear()

            for category, text, confidence, t in items:
                ts = time.strftime("%H:%M:%S", time.localtime(t))
                full_text = f"[{ts}] {text}"
                print(full_text)

                if TELEGRAM_ENABLED:
                    photo_path = None
                    with self.frame_lock:
                        if self.frame is not None:
                            filename = f"{category}_{int(t)}.jpg"
                            photo_path = str(ALERTS_DIR / filename)
                            cv2.imwrite(photo_path, self.frame)

                    if photo_path:
                        photo_path = photo_path
                    threading.Thread(
                        target=self._send_telegram,
                        args=(full_text, photo_path),
                        daemon=True
                    ).start()

            time.sleep(0.5)
