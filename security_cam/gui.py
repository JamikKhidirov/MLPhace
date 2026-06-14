import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import threading
import os
import json
from pathlib import Path

from config import (
    CAMERA_URL, LOG_MAX_LINES, SHOW_FPS,
    ENABLE_AUDIO, ENABLE_FIRE, ENABLE_ABANDONED, ENABLE_LPR, ENABLE_SPEED,
    LPR_WHITELIST, SPEED_DISTANCE_METERS, ALERTS_DIR, BASE_DIR,
    PROCESS_EVERY_N_FRAMES,
)
from detectors.audio import AudioDetector
from detectors.vision import FireDetector, AbandonedObjectDetector
from detectors.plate import PlateReader
from detectors.speed import SpeedDetector
from utils.notifications import Notifier


class SecurityCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎥 Охрана дома — система видеонаблюдения")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.cap = None
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()

        # Фичи
        self.features = {
            "audio": tk.BooleanVar(value=ENABLE_AUDIO),
            "fire": tk.BooleanVar(value=ENABLE_FIRE),
            "abandoned": tk.BooleanVar(value=ENABLE_ABANDONED),
            "plate": tk.BooleanVar(value=ENABLE_LPR),
            "speed": tk.BooleanVar(value=ENABLE_SPEED),
        }

        self.notifier = Notifier()

        self.audio_detector = AudioDetector(on_alert=self.notifier.alert)
        self.fire_detector = FireDetector(on_alert=self.notifier.alert)
        self.abandoned_detector = AbandonedObjectDetector(on_alert=self.notifier.alert)
        self.plate_reader = PlateReader(on_alert=self.notifier.alert)
        self.speed_detector = SpeedDetector(on_alert=self.notifier.alert)

        self._build_gui()
        self._load_models_async()

    def _build_gui(self):
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ====== Левая панель — видео ======
        left = ttk.Frame(main_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(left, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Панель управления под видео
        ctrl = ttk.Frame(left)
        ctrl.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(ctrl, text="📷 Скриншот", command=self._screenshot).pack(side=tk.LEFT, padx=2)
        self.rec_btn = ttk.Button(ctrl, text="⏺ Запись", command=self._toggle_recording)
        self.rec_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="⚙️ Настройки", command=self._settings_dialog).pack(side=tk.LEFT, padx=2)
        self.fps_label = ttk.Label(ctrl, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=5)

        # ====== Правая панель — лог + управление ======
        right = ttk.Frame(main_frame, width=350)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right.pack_propagate(False)

        # Управление фичами
        feat_frame = ttk.LabelFrame(right, text="🧠 Детекторы", padding=5)
        feat_frame.pack(fill=tk.X)

        self.feat_btns = {}
        feat_config = [
            ("audio", "🎤 Звук"),
            ("fire", "🔥 Огонь/Дым"),
            ("abandoned", "📦 Предметы"),
            ("plate", "🔢 Номера"),
            ("speed", "🚗 Скорость"),
        ]
        for key, label in feat_config:
            cb = ttk.Checkbutton(feat_frame, text=label, variable=self.features[key])
            cb.pack(anchor=tk.W, pady=1)
            self.feat_btns[key] = cb

        # Статус
        status_frame = ttk.LabelFrame(right, text="📊 Статус", padding=5)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_text = tk.Text(status_frame, height=6, width=40, font=("Consolas", 9))
        self.status_text.pack(fill=tk.X)
        self.status_text.insert(tk.END, "Ожидание...\n")
        self.status_text.config(state=tk.DISABLED)

        # Лог событий
        log_frame = ttk.LabelFrame(right, text="📋 События", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, font=("Consolas", 9), wrap=tk.WORD, state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log("🟢 Система запущена")
        self.log(f"📷 Камера: {CAMERA_URL}")

    def _load_models_async(self):
        def load():
            self.audio_detector.load_model()
            if ENABLE_LPR:
                self.plate_reader.load_model()
            if ENABLE_SPEED:
                self.speed_detector.load_model()
            self.log("✅ Все модели загружены")
            self._update_status()

        threading.Thread(target=load, daemon=True).start()

    def log(self, msg):
        def _do():
            self.log_text.config(state=tk.NORMAL)
            ts = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > LOG_MAX_LINES:
                self.log_text.delete("1.0", f"{lines - LOG_MAX_LINES}.0")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def _update_status(self, text=None):
        def _do():
            self.status_text.config(state=tk.NORMAL)
            self.status_text.delete("1.0", tk.END)
            status = []
            for key, label in [("audio", "🎤 Звук"), ("fire", "🔥 Пожар"),
                                ("abandoned", "📦 Предметы"), ("plate", "🔢 Номера"),
                                ("speed", "🚗 Скорость")]:
                enabled = self.features[key].get()
                status.append(f"{'✅' if enabled else '❌'} {label}")
            status.append(f"📷 Кадров: {self.frame_count}")
            self.status_text.insert(tk.END, "\n".join(status))
            self.status_text.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def start(self):
        self.cap = cv2.VideoCapture(CAMERA_URL if isinstance(CAMERA_URL, int) else CAMERA_URL)
        if not self.cap.isOpened():
            self.log("❌ Не удалось открыть камеру!")
            messagebox.showerror("Ошибка", "Не удалось открыть камеру/RTSP поток")
            return

        self.log("✅ Камера подключена")
        self.running = True
        self.notifier.start()
        if self.features["audio"].get():
            self.audio_detector.start()
        self.recording = False
        self.rec_writer = None
        self.thread = threading.Thread(target=self._video_loop, daemon=True)
        self.thread.start()

    def _video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.log("⚠️ Потеря кадра, ждём...")
                time.sleep(0.5)
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(CAMERA_URL if isinstance(CAMERA_URL, int) else CAMERA_URL)
                continue

            self.frame_count += 1
            skip = PROCESS_EVERY_N_FRAMES

            # FPS
            if self.frame_count % 30 == 0:
                now = time.time()
                self.fps = 30 / (now - self.fps_time)
                self.fps_time = now
                self._update_status()

            # Детекторы — только каждый N-кадр
            if self.frame_count % skip == 0:
                if self.features["fire"].get():
                    frame = self.fire_detector.process(frame)
                if self.features["abandoned"].get():
                    frame = self.abandoned_detector.process(frame)
                if self.features["plate"].get():
                    frame, _ = self.plate_reader.process(frame)
                if self.features["speed"].get():
                    frame = self.speed_detector.process(frame)

            # FPS overlay
            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Запись
            if self.recording and self.rec_writer:
                self.rec_writer.write(frame)

            # Отправка кадра в нотификатор
            self.notifier.set_frame(frame)

            # Отображение
            self._display_frame(frame)

        self._cleanup()

    def _display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        max_w = 800
        if w > max_w:
            scale = max_w / w
            new_w = max_w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    def _screenshot(self):
        if not self.running:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = str(ALERTS_DIR / f"screenshot_{ts}.jpg")
        self.notifier._send_telegram(f"📷 Скриншот {ts}")
        self.log(f"📷 Скриншот: {path}")

    def _toggle_recording(self):
        if not self.running:
            return
        if self.recording:
            self.recording = False
            if self.rec_writer:
                self.rec_writer.release()
                self.rec_writer = None
            self.rec_btn.config(text="⏺ Запись")
            self.log("⏹ Запись остановлена")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = str(ALERTS_DIR / f"video_{ts}.avi")
            if self.cap:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.rec_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), 15, (w, h))
                self.recording = True
                self.rec_btn.config(text="⏹ Стоп")
                self.log(f"⏺ Запись: {path}")

    def _settings_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("⚙️ Настройки")
        win.geometry("500x600")

        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ====== Камера ======
        cam_frame = ttk.Frame(notebook, padding=10)
        notebook.add(cam_frame, text="📷 Камера")
        ttk.Label(cam_frame, text="RTSP URL (или 0 для вебки):").pack(anchor=tk.W)
        cam_url = tk.StringVar(value=str(CAMERA_URL))
        ttk.Entry(cam_frame, textvariable=cam_url, width=60).pack(fill=tk.X, pady=5)

        # ====== Номера ======
        plate_frame = ttk.Frame(notebook, padding=10)
        notebook.add(plate_frame, text="🔢 Номера")
        ttk.Label(plate_frame, text="Белый список номеров (по одному в строке):").pack(anchor=tk.W)
        white_text = tk.Text(plate_frame, height=8)
        white_text.pack(fill=tk.X, pady=5)
        white_text.insert(tk.END, "\n".join(LPR_WHITELIST))

        # ====== Скорость ======
        speed_frame = ttk.Frame(notebook, padding=10)
        notebook.add(speed_frame, text="🚗 Скорость")
        ttk.Label(speed_frame, text="Расстояние между зонами (метры):").pack(anchor=tk.W)
        dist_var = tk.DoubleVar(value=SPEED_DISTANCE_METERS)
        ttk.Entry(speed_frame, textvariable=dist_var).pack(fill=tk.X, pady=5)

        # ====== Telegram ======
        tg_frame = ttk.Frame(notebook, padding=10)
        notebook.add(tg_frame, text="📱 Telegram")
        ttk.Label(tg_frame, text="Bot Token:").pack(anchor=tk.W)
        token_var = tk.StringVar(value=TELEGRAM_BOT_TOKEN if TELEGRAM_BOT_TOKEN else "")
        ttk.Entry(tg_frame, textvariable=token_var, width=50).pack(fill=tk.X, pady=2)
        ttk.Label(tg_frame, text="Chat ID:").pack(anchor=tk.W)
        chat_var = tk.StringVar(value=TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else "")
        ttk.Entry(tg_frame, textvariable=chat_var, width=50).pack(fill=tk.X, pady=2)

        def save_settings():
            from config import CAMERA_URL as _old_url, LPR_WHITELIST as _old_wl, SPEED_DISTANCE_METERS as _old_dist
            _old_url = cam_url.get()
            _old_wl = white_text.get("1.0", tk.END).strip().splitlines()
            _old_dist = dist_var.get()
            self.log("⚙️ Настройки сохранены (перезапустите приложение)")
            messagebox.showinfo("Настройки", "Настройки изменены.\nПерезапустите приложение для применения.")
            win.destroy()

        ttk.Button(win, text="💾 Сохранить", command=save_settings).pack(pady=5)

    def _on_close(self):
        self.running = False
        time.sleep(0.3)
        self._cleanup()
        self.root.destroy()

    def _cleanup(self):
        if self.rec_writer:
            self.rec_writer.release()
        if self.cap:
            self.cap.release()
        self.audio_detector.stop()
        self.notifier.stop()
        cv2.destroyAllWindows()
