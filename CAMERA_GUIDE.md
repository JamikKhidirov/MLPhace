# ML для уличных камер видеонаблюдения

## 1. Что можно сделать

У тебя уже есть камеры на улице? Отлично. Подключаем к ним ML.

| Функция | Описание |
|---------|----------|
| **Детекция движения** | Камера видит движение → уведомление в Telegram |
| **Детекция человека** | Не просто тень, а именно человек |
| **Детекция машины** | Кто-то подъехал к дому / воротам |
| **Детекция лица** | Узнаёт своих, чужих → оповещение |
| **Распознавание номеров** | Какая машина проехала |
| **Охрана периметра** | Кто-то пересёк линию забора |
| **Запись по событию** | Пишет видео только когда есть движение |
| **Ночной режим** | Детекция в темноте (ИК-камеры) |

### Как это работает

```
Уличная камера → RTSP поток → Твой ПК / Сервер
                                    ↓
                          ML модель (YOLO / MediaPipe)
                                    ↓
                        Движение / Человек / Машина?
                                    ↓
                         Уведомление в Telegram
                                    ↓
                         Сохранение фото/видео
```

---

## 2. Как подключиться к уличной камере

### 2.1 Узнать IP адрес камеры

Уличные камеры работают через RTSP протокол.
Чтобы подключиться — нужен IP адрес и порт.

**Как узнать IP камеры:**
```
1. Зайди в роутер (обычно 192.168.1.1 или 192.168.0.1)
2. Найди список подключённых устройств (DHCP clients)
3. Найди камеру по названию (Hikvision, Dahua, etc.)
4. Запиши её IP адрес (например 192.168.1.100)
```

**Или через программу:**
```
Скачай ONVIF Device Manager (бесплатно)
→ Он найдёт все камеры в сети
→ Покажет IP, порт, RTSP URL
```

### 2.2 RTSP URL — как выглядит

```
rtsp://логин:пароль@IP_камеры:порт/путь
```

**Примеры для популярных брендов:**

```python
# Hikvision
rtsp://admin:pass123@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:pass123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# TP-Link
rtsp://admin:pass123@192.168.1.100:554/stream1

# HiWatch
rtsp://admin:pass123@192.168.1.100:554/h264

# RVi / EKF / любые ONVIF
rtsp://admin:pass123@192.168.1.100:554/onvif1

# Если камера через видеорегистратор (DVR/NVR)
rtsp://admin:pass123@192.168.1.1:554/Streaming/Channels/201
#    ↑ IP регистратора                          ↑ канал 2
```

**Где взять логин и пароль:**
```
По умолчанию у большинства камер:
  Логин: admin
  Пароль: admin / 12345 / пустой

Если меняли — смотри в настройках камеры.
Если забыли — сброс кнопкой на камере (держать 10 сек).
```

### 2.3 Проверка подключения

```bash
# Скачай VLC Media Player
# Открой: Media → Open Network Stream
# Вставь RTSP URL
# Если video идёт — всё ок

# Или через ffmpeg:
ffmpeg -i "rtsp://admin:pass@192.168.1.100:554/stream1" -frames 1 test.jpg
```

### 2.4 Если камер несколько (DVR/NVR)

Видеорегистратор собирает все камеры в один пул.
Подключаешься к регистратору, а он отдаёт любой канал.

```python
# Канал 1 (первая камера)
rtsp://admin:pass@192.168.1.1:554/Streaming/Channels/101

# Канал 2 (вторая камера)
rtsp://admin:pass@192.168.1.1:554/Streaming/Channels/201

# Канал 3
rtsp://admin:pass@192.168.1.1:554/Streaming/Channels/301
```

Подставь свой IP регистратора и перебирай каналы.

---

## 3. Запуск на ПК (самый простой вариант)

У тебя есть старый ПК / ноутбук? Поставь его рядом с роутером,
подключи к нему камеры через RTSP и получай уведомления.

### 3.1 Детекция движения на уличной камере

```python
import cv2
import time
import requests

BOT_TOKEN = "ваш_токен_бота"
CHAT_ID = "ваш_id_чата"

# RTSP твоей уличной камеры
CAMERA_URL = "rtsp://admin:pass@192.168.1.100:554/Streaming/Channels/101"

cap = cv2.VideoCapture(CAMERA_URL)

# Ждём первый кадр
ret, prev = cap.read()
if not ret:
    print("❌ Камера не отвечает. Проверь RTSP URL")
    print("   Возможные проблемы:")
    print("   - Неправильный IP/порт")
    print("   - Неверный логин/пароль")
    print("   - Камера выключена")
    exit()

prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (21, 21), 0)

MIN_AREA = 8000   # чувствительность (чем меньше, тем чувствительнее)
COOLDOWN = 30     # секунд между уведомлениями
last_notify = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Потеря связи с камерой, переподключаюсь...")
        cap.release()
        time.sleep(3)
        cap = cv2.VideoCapture(CAMERA_URL)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(prev, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    area = cv2.countNonZero(thresh)

    if area > MIN_AREA and time.time() - last_notify > COOLDOWN:
        last_notify = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S")

        cv2.imwrite(f"alert_{ts}.jpg", frame)
        print(f"🚨 Движение! {time.strftime('%H:%M:%S')}")

        # Отправка в Telegram
        with open(f"alert_{ts}.jpg", "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                data={"chat_id": CHAT_ID,
                      "caption": f"🚨 Движение на улице!\n{time.strftime('%H:%M:%S %d.%m.%Y')}"},
                files={"photo": f},
            )

    prev = gray
    time.sleep(0.05)  # небольшая пауза, чтобы не грузить процессор

cap.release()
```

### 3.2 Детекция человека (не просто движение)

```python
import cv2
import time
import requests
import numpy as np

BOT_TOKEN = "..."
CHAT_ID = "..."

# Модель YOLO для детекции людей и машин
# Скачай файлы:
# https://github.com/ultralytics/ultralytics
# yolov8n.pt или yolov8n.onnx
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # маленькая, быстрая

CAMERA_URL = "rtsp://admin:pass@192.168.1.100:554/stream1"
cap = cv2.VideoCapture(CAMERA_URL)
COOLDOWN = 30
last_notify = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        time.sleep(3)
        cap = cv2.VideoCapture(CAMERA_URL)
        continue

    # YOLO детекция (каждый 3-й кадр для скорости)
    results = model(frame, classes=[0, 2, 3, 5, 7])  # человек, машина, мото, автобус, грузовик

    person_count = 0
    car_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == 0:  # person
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Человек {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif cls_id in (2, 3, 5, 7):  # car, motorbike, bus, truck
                car_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Машина {conf:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if (person_count > 0 or car_count > 0) and time.time() - last_notify > COOLDOWN:
        last_notify = time.time()
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"alert_{ts}.jpg", frame)

        msg = ""
        if person_count > 0:
            msg += f"👤 Человек! ({person_count}) "
        if car_count > 0:
            msg += f"🚗 Машина! ({car_count})"
        msg += f"\n{time.strftime('%H:%M:%S %d.%m.%Y')}"

        with open(f"alert_{ts}.jpg", "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                data={"chat_id": CHAT_ID, "caption": msg},
                files={"photo": f},
            )

    cv2.imshow("Улица", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
```

---

## 4. Подключение нескольких уличных камер

### 4.1 Одновременный мониторинг всех камер

```python
import cv2
import time
import requests
import threading

BOT_TOKEN = "..."
CHAT_ID = "..."

# Твои уличные камеры
CAMERAS = {
    "Ворота":   "rtsp://admin:pass@192.168.1.101:554/stream1",
    "Двор":     "rtsp://admin:pass@192.168.1.102:554/stream1",
    "Гараж":    "rtsp://admin:pass@192.168.1.103:554/stream1",
    "Калитка":  "rtsp://admin:pass@192.168.1.104:554/stream1",
}

MIN_AREA = 8000

def monitor_camera(name, url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"❌ {name}: не подключилась")
        return

    ret, prev = cap.read()
    if not ret:
        print(f"❌ {name}: нет сигнала")
        return

    prev = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    last = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ {name}: потеря сигнала, переподключение...")
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(url)
            continue

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        diff = cv2.threshold(cv2.absdiff(prev, gray), 25, 255, cv2.THRESH_BINARY)[1]
        area = cv2.countNonZero(cv2.dilate(diff, None, iterations=2))

        if area > MIN_AREA and time.time() - last > 30:
            last = time.time()
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"alert_{name}_{ts}.jpg"
            cv2.imwrite(path, frame)

            with open(path, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                    data={"chat_id": CHAT_ID,
                          "caption": f"🚨 {name}: движение!\n{time.strftime('%H:%M:%S')}"},
                    files={"photo": f},
                )

        prev = gray
        time.sleep(0.05)

# Запускаем каждую камеру в отдельном потоке
threads = []
for name, url in CAMERAS.items():
    t = threading.Thread(target=monitor_camera, args=(name, url), daemon=True)
    t.start()
    threads.append(t)
    print(f"✅ {name}: запущена")
    time.sleep(1)  # пауза между подключениями

while True:
    time.sleep(1)
```

---

## 5. Запись видео по движению

```python
import cv2
import time

CAMERA_URL = "rtsp://admin:pass@192.168.1.100:554/stream1"
cap = cv2.VideoCapture(CAMERA_URL)

ret, prev = cap.read()
prev = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (21, 21), 0)

writer = None
recording = False
no_motion_since = 0
MIN_AREA = 8000

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        time.sleep(3)
        cap = cv2.VideoCapture(CAMERA_URL)
        continue

    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    motion = cv2.countNonZero(cv2.threshold(cv2.absdiff(prev, gray), 25, 255, cv2.THRESH_BINARY)[1])

    if motion > MIN_AREA:
        if not recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(f"record_{ts}.avi",
                                     cv2.VideoWriter_fourcc(*"XVID"), 15, (w, h))
            recording = True
            print(f"🎥 Запись: {ts}")
        writer.write(frame)
        no_motion_since = time.time()

    elif recording and time.time() - no_motion_since > 5:
        writer.release()
        writer = None
        recording = False
        print("⏹ Стоп. Движения нет 5 секунд")

    prev = gray
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break

if writer:
    writer.release()
cap.release()
```

---

## 6. Telegram Bot для управления

### 6.1 Создание бота

```
1. Telegram → @BotFather → /newbot
2. Введи название (например "МойДомОхрана")
3. Получишь TOKEN вида: 123456789:ABCdefGhIjKlMnOpQrStUvWxYz
4. @userinfobot → /start → получишь свой CHAT_ID
```

### 6.2 Бот с командами

```python
import cv2, asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

BOT_TOKEN = "ваш_токен"
CHAT_ID = "ваш_id"
CAMERAS = {
    "Ворота": "rtsp://admin:pass@192.168.1.101:554/stream1",
    "Двор":   "rtsp://admin:pass@192.168.1.102:554/stream1",
}

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

def take_snapshot(url):
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("snap.jpg", frame)
        return True
    return False

@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "🏠 Охрана дома\n"
        "/photo — фото с камер\n"
        "/status — статус системы\n"
        "/alarm_on — включить охрану\n"
        "/alarm_off — выключить"
    )

@dp.message(Command("photo"))
async def photo(msg: types.Message):
    for name, url in CAMERAS.items():
        if take_snapshot(url):
            with open("snap.jpg", "rb") as f:
                await msg.answer_photo(
                    types.BufferedInputFile(f.read(), f"{name}.jpg"),
                    caption=f"📷 {name}"
                )
        else:
            await msg.answer(f"❌ {name}: нет сигнала")

@dp.message(Command("status"))
async def status(msg: types.Message):
    text = "🏠 Система видеонаблюдения\n\n"
    for name, url in CAMERAS.items():
        cap = cv2.VideoCapture(url)
        ok = "✅" if cap.isOpened() and cap.read()[0] else "❌"
        cap.release()
        text += f"{ok} {name}\n"
    await msg.answer(text)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Запуск 24/7 (без монитора, без входа в систему)

### Windows — как служба

```bash
# 1. Скачай NSSM (Non-Sucking Service Manager)
#    https://nssm.cc/download

# 2. Установи скрипт как службу Windows
nssm install CameraGuard

# Заполни:
#   Path:     C:\Users\user\.venv\Scripts\python.exe
#   Arguments: C:\Users\user\camera_guard.py
#   Startup:   Automatic (запуск при старте Windows)

# 3. Запусти
nssm start CameraGuard

# 4. Управление
nssm stop CameraGuard
nssm restart CameraGuard
nssm remove CameraGuard
```

### Linux / Raspberry Pi — через systemd

```bash
# /etc/systemd/system/camera-guard.service
[Unit]
Description=Умная охрана дома
After=network.target

[Service]
ExecStart=/home/pi/venv/bin/python /home/pi/camera_guard.py
Restart=always
RestartSec=10
User=pi

[Install]
WantedBy=multi-user.target

# Активация
sudo systemctl daemon-reload
sudo systemctl enable camera-guard
sudo systemctl start camera-guard

# Проверка
sudo systemctl status camera-guard
sudo journalctl -u camera-guard -f  # логи в реальном времени
```

---

## 8. Что если камера не ONVIF / не RTSP?

### Вариант: видеорегистратор с HDMI выходом

Если камеры старые и идут только на видеорегистратор,
а у регистратора есть HDMI выход:

```
1. Купи USB захват видео (HDMI→USB) — $10-20
2. Подключи регистратор → захват → ПК
3. В коде используй cap = cv2.VideoCapture(0) (как вебкамера)
```

### Вариант: Wi-Fi камера с приложением

Некоторые камеры не дают RTSP, работают только через своё приложение.

```python
# Для таких камер есть 2 пути:
# 1. FFmpeg может захватить поток
# 2. Некоторые камеры отдают MJPEG по HTTP

# MJPEG поток (попробуй этот URL):
cap = cv2.VideoCapture("http://192.168.1.100:8080/video")
```

---

## 9. Что нужно для старта (минимально)

### Минимальный набор

```
1. Любой ПК / ноутбук (даже старый)
   → Можно оставить включённым 24/7

2. Кабель до роутера (WiFi может рвать поток)
   → Если WiFi — убедись, что сигнал стабильный

3. Установить Python + библиотеки:
```

```bash
pip install opencv-python requests numpy ultralytics aiogram
```

### Проверка за 5 минут

```bash
# 1. Узнай IP камеры (через роутер или ONVIF Device Manager)

# 2. Проверь RTSP через VLC:
#    Media → Open Network Stream → вставь RTSP URL

# 3. Запусти минимальный скрипт:
python -c "
import cv2
cap = cv2.VideoCapture('rtsp://admin:pass@192.168.1.100:554/stream1')
ret, frame = cap.read()
if ret:
    cv2.imwrite('test.jpg', frame)
    print('✅ Камера работает! Фото: test.jpg')
else:
    print('❌ Камера не отвечает')
cap.release()
"
```

---

## 10. Схема подключения

```
                      ┌──────────────┐
   Уличная камера ───→│   Роутер     │
   (RTSP поток)       │  192.168.1.1 │
                      └──────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
         │  ПК     │   │Raspberry│   │Сервер   │
         │  (Python)│   │Pi       │   │(VPS)    │
         └─────────┘   └─────────┘   └─────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                     ┌──────┴──────┐
                     │  Telegram   │
                     │  Уведомления│
                     └─────────────┘
```

---

## 11. Готовая программа (.exe) с 5 крутыми фичами

Готовая программа с графическим интерфейсом, которая делает ВСЁ:

```
┌──────────────────────────────────────────────────────┐
│  🎥 Охрана дома — система видеонаблюдения             │
├──────────────────────────┬───────────────────────────┤
│                          │  🧠 Детекторы              │
│     Видео с камеры       │  ☑ 🎤 Звук                │
│     (с разметкой)        │  ☑ 🔥 Огонь/Дым           │
│                          │  ☑ 📦 Предметы            │
│                          │  ☑ 🔢 Номера              │
│                          │  ☑ 🚗 Скорость            │
│                          │                           │
│                          │  📊 Статус                │
│                          │  ✅ 🎤 Звук               │
│                          │  ✅ 🔥 Пожар              │
│                          │  ...                      │
│                          │                           │
│                          │  📋 События               │
│                          │  [12:34] 🚗 Номер А123ВС  │
│                          │  [12:30] 💥 Выстрел!      │
│                          │  [12:29] 🔥 Обнаружен огонь│
│                          │  [12:25] 🚗 Скорость 68  │
├──────────────────────────┴───────────────────────────┤
│  [📷 Скриншот] [⏺ Запись] [⚙️ Настройки]           │
└──────────────────────────────────────────────────────┘
```

### 11.1 Как получить

```
1. Открой папку security_cam/
2. Запусти build_exe.ps1 (сборка .exe)
   → powershell -ExecutionPolicy Bypass -File build_exe.ps1

3. Готовый файл: security_cam/dist/SecurityGuard.exe
4. Просто запусти его — всё работает!
```

### 11.2 Что внутри

| Фича | Как работает |
|------|-------------|
| **🎤 Звук** | YAMNet (нейросеть от Google) слушает микрофон. Распознаёт: разбитое стекло, выстрел, крик. При обнаружении → уведомление в Telegram + звуковой сигнал |
| **🔥 Огонь/Дым** | Анализ цвета и текстуры в кадре. Ищет оранжево-красные области (огонь) и серо-белые (дым). Не реагирует на фонари/фары (проверка на мерцание) |
| **📦 Оставленные предметы** | YOLOv8 отслеживает все объекты. Если человек оставил сумку/коробку и ушёл, а предмет стоит > 2 секунд — тревога |
| **🔢 Номера (LPR)** | Находит номер на машине → EasyOCR читает символы → сверяет с белым списком. Свои номера — зелёная рамка, чужие — красная + Telegram |
| **🚗 Скорость** | Две виртуальные линии на видео. Засекает время проезда машины между ними. Зная расстояние (настраивается) — считает км/ч |

### 11.3 Настройка перед запуском

Открой `config.py` и пропиши:

```python
CAMERA_URL = "rtsp://admin:pass@192.168.1.100:554/stream1"  # своя камера

TELEGRAM_BOT_TOKEN = "123456789:ABCdefGhIjKlMnOpQrStUvWxYz"  # от @BotFather
TELEGRAM_CHAT_ID = "123456789"  # от @userinfobot

LPR_WHITELIST = ["А123ВС77", "О456ТТ77"]  # свои номера

SPEED_DISTANCE_METERS = 10  # расстояние между линиями (замерь рулеткой)
```

Или настрой через GUI: кнопка ⚙️ Настройки → вкладки Камера, Номера, Скорость, Telegram.

### 11.4 Запуск 24/7 как служба Windows

```bash
# После сборки .exe установи как службу:
nssm install SecurityGuard
# Path: C:\путь\к\SecurityGuard.exe
# Startup: Automatic
nssm start SecurityGuard
```

### 11.5 Что потребуется для сборки

```bash
# Установи Python 3.10+
# Установи зависимости:
pip install -r requirements.txt
pip install pyinstaller

# Собери:
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

Первый запуск собранного .exe скачает модели нейросетей (~50 MB всего).
Дальше работает полностью офлайн.

---

## 12. Полезные ссылки

- **ONVIF Device Manager**: https://sourceforge.net/projects/onvifdm
  (находит все камеры в сети, показывает RTSP URL)
- **RTSP URL для разных камер**: https://www.ispyconnect.com
- **NSSM (Windows Service)**: https://nssm.cc
- **Telegram BotFather**: https://t.me/BotFather
- **Aiogram**: https://docs.aiogram.dev
- **YOLOv8**: https://docs.ultralytics.com
- **YAMNet**: https://tfhub.dev/google/yamnet/1
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **PyInstaller**: https://pyinstaller.org
