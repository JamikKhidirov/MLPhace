import time
import threading
import queue
import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import tensorflow as tf
from pathlib import Path
from config import (
    YAMNET_MODEL_PATH, CLASS_GLASS, CLASS_GUNSHOT, CLASS_SCREAM,
    AUDIO_CONFIDENCE, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SEC, AUDIO_COOLDOWN,
    ENABLE_AUDIO
)

CLASS_LABELS = {
    CLASS_GLASS: ("🔊 Стекло", "Разбитое стекло"),
    CLASS_GUNSHOT: ("💥 Выстрел", "Звук выстрела"),
    CLASS_SCREAM: ("😱 Крик", "Крик / крик о помощи"),
}

class AudioDetector:
    def __init__(self, on_alert=None):
        self.on_alert = on_alert
        self.model = None
        self.class_names = None
        self.cooldowns = {}
        self.audio_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.loaded = False

    def load_model(self):
        if not ENABLE_AUDIO:
            return False
        try:
            model_path = str(YAMNET_MODEL_PATH)
            p = Path(model_path)
            if p.exists() and any(p.iterdir()):
                self.model = hub.load(model_path)
            else:
                print("🎤 Загрузка YAMNet модели (первый раз может быть долго)...")
                self.model = hub.load("https://tfhub.dev/google/yamnet/1")
                tf.saved_model.save(self.model, model_path)
                print("🎤 YAMNet сохранена локально")

            sample = tf.constant(np.zeros([AUDIO_SAMPLE_RATE], dtype=np.float32))
            scores, embeddings, spectrogram = self.model(sample)
            class_names_path = self.model.class_map_path()
            if class_names_path:
                import csv
                self.class_names = []
                with open(class_names_path) as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        self.class_names.append((int(row[0]), row[2]))
            self.loaded = True
            return True
        except Exception as e:
            print(f"⚠️ YAMNet не загрузилась: {e}")
            print("🎤 Аудио-детекция будет использовать спектральный анализ (без ML)")
            return False

    def _analyze_spectral(self, audio):
        """Fallback без YAMNet — детекция по спектру"""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/AUDIO_SAMPLE_RATE)
        mag = np.abs(fft)
        energy = np.sum(mag ** 2)

        if energy < 1e-6:
            return []

        # Выстрел: очень короткий импульс, широкая полоса
        # Стекло: звон + шум
        # Крик: энергия в высоких частотах
        high_energy = np.sum(mag[freqs > 2000] ** 2)
        low_energy = np.sum(mag[(freqs > 100) & (freqs < 500)] ** 2)
        mid_energy = np.sum(mag[(freqs > 500) & (freqs < 2000)] ** 2)

        total = high_energy + low_energy + mid_energy
        if total < 1:
            return []

        r = high_energy / total
        alerts = []

        peak = np.max(mag)
        peak_freq = freqs[np.argmax(mag)]

        if peak > 5e-4 and peak_freq > 100 and peak_freq < 500:
            impulses = np.sum(np.diff(np.sign(np.diff(mag))) < -0.5) > 20
            if impulses:
                alerts.append(("💥 Выстрел", 0.7))

        if r > 0.4 and peak > 1e-4:
            alerts.append(("😱 Крик", min(0.9, r)))

        if low_energy > 0 and mid_energy > 0:
            ring_ratio = low_energy / mid_energy
            if ring_ratio > 2.0 and peak > 1e-4:
                alerts.append(("🔊 Стекло", min(0.8, ring_ratio / 5)))

        return alerts

    def _audio_loop(self):
        def callback(indata, frames, time_info, status):
            self.audio_queue.put(indata.copy())

        stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            blocksize=int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SEC),
            callback=callback
        )

        buffer = np.array([], dtype=np.float32)

        with stream:
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=1)
                    buffer = np.concatenate([buffer, chunk.flatten()])

                    while len(buffer) >= AUDIO_SAMPLE_RATE:
                        audio = buffer[:AUDIO_SAMPLE_RATE]
                        buffer = buffer[AUDIO_SAMPLE_RATE:]

                        self._process_audio(audio)
                except queue.Empty:
                    continue

    def _process_audio(self, audio):
        audio_t = tf.constant(audio, dtype=tf.float32)
        alerts = []

        if self.loaded and self.model is not None:
            scores, embeddings, spectrogram = self.model(audio_t)
            mean_scores = np.mean(scores.numpy(), axis=0)
            for cls_id, (emoji, label) in CLASS_LABELS.items():
                conf = mean_scores[cls_id]
                if conf >= AUDIO_CONFIDENCE:
                    key = f"audio_{cls_id}"
                    if time.time() - self.cooldowns.get(key, 0) > AUDIO_COOLDOWN:
                        self.cooldowns[key] = time.time()
                        alerts.append((f"{emoji} {label}", conf))
        else:
            spectral_alerts = self._analyze_spectral(audio)
            for label, conf in spectral_alerts:
                key = f"spec_{label}"
                if time.time() - self.cooldowns.get(key, 0) > AUDIO_COOLDOWN:
                    self.cooldowns[key] = time.time()
                    alerts.append((label, conf))

        for alert_text, conf in alerts:
            if self.on_alert:
                self.on_alert("audio", alert_text, conf)

    def start(self):
        if not ENABLE_AUDIO:
            return
        self.load_model()
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
        print("🎤 Аудио-мониторинг запущен")

    def stop(self):
        self.running = False
