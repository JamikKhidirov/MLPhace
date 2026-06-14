import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
ALERTS_DIR = BASE_DIR / "alerts"
ALERTS_DIR.mkdir(exist_ok=True)

# ====== Источник видео ======
CAMERA_URL = 0  # 0 — вебка; или RTSP: "rtsp://admin:pass@192.168.1.100:554/stream1"

# ====== Включение/выключение фич ======
ENABLE_AUDIO = True
ENABLE_FIRE = True
ENABLE_ABANDONED = True
ENABLE_LPR = True
ENABLE_SPEED = True

# ====== YAMNet (аудио-детекция) ======
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_MODEL_PATH = MODELS_DIR / "yamnet"
# Индексы классов YAMNet для интересующих звуков
# Полный список: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
CLASS_GLASS = 422    # Glass
CLASS_GUNSHOT = 428  # Gunshot, gunfire
CLASS_SCREAM = 241   # Scream
AUDIO_CONFIDENCE = 0.6
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SEC = 1.0
AUDIO_COOLDOWN = 15  # сек между уведомлениями

# ====== Огонь и дым (цветовой HSV-метод) ======
FIRE_HSV_LOWER = (0, 120, 100)    # красный/оранжевый
FIRE_HSV_UPPER = (30, 255, 255)
SMOKE_HSV_LOWER = (0, 0, 150)     # серый/белый
SMOKE_HSV_UPPER = (180, 30, 255)
FIRE_MIN_AREA = 500
FIRE_MIN_FRAMES = 3               # подтверждение подряд
FIRE_COOLDOWN = 30

# ====== Оставленные предметы ======
ABANDONED_STILL_FRAMES = 60       # сколько кадров предмет стоит на месте
ABANDONED_PERSON_RADIUS = 100     # пикселей от человека
ABANDONED_COOLDOWN = 60

# ====== Номера (LPR) ======
LPR_MIN_CONFIDENCE = 0.4
LPR_MIN_PLATE_AREA = 1500
LPR_WHITELIST = [
    # "А123ВС77",
    # "О456ТТ77",
]
LPR_COOLDOWN_SAME_PLATE = 120     # повторное уведомление об одном номере
LPR_CUSTOM_DICT = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАВЕКМНОРСТУХ"
LPR_LANGUAGE = "ru"

# ====== Скорость ======
# Расстояние между линиями в метрах (замерь рулеткой у камеры)
SPEED_DISTANCE_METERS = 10
# Позиции линий (в процентах от высоты кадра)
SPEED_LINE1_Y = 0.3
SPEED_LINE2_Y = 0.7
SPEED_MIN_CONFIDENCE = 0.5       # уверенность что это машина
SPEED_COOLDOWN = 10

# ====== Telegram уведомления ======
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# ====== Производительность ======
PROCESS_EVERY_N_FRAMES = 5  # обработка детекторов каждый N-кадр (1 — все кадры)
                           # для слабых ноутбуков ставь 5-10

# ====== Общее ======
COOLDOWN_DEFAULT = 15
SHOW_FPS = True
LOG_MAX_LINES = 200
MODEL_DOWNLOAD_TIMEOUT = 120
