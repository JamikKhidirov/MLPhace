import cv2
import numpy as np

import config

EMOTION_ORDER = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise",
]

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 100, 100),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "neutral": (200, 200, 200),
    "sad": (255, 100, 0),
    "surprise": (255, 255, 0),
}


def draw_mood_graph(frame, history, x=20, y=105, width=180, height=70):
    if len(history) < 2:
        return

    canvas = frame[y:y + height, x:x + width]
    overlay = np.full_like(canvas, (30, 30, 30), dtype=np.uint8)
    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)

    recent = history[-width:]
    for i in range(1, len(recent)):
        e0 = recent[i - 1]
        e1 = recent[i]
        c0 = EMOTION_COLORS.get(e0, (200, 200, 200))
        idx0 = EMOTION_ORDER.index(e0) if e0 in EMOTION_ORDER else 3
        idx1 = EMOTION_ORDER.index(e1) if e1 in EMOTION_ORDER else 3
        y0 = int(height - 10 - (idx0 / len(EMOTION_ORDER)) * (height - 20))
        y1 = int(height - 10 - (idx1 / len(EMOTION_ORDER)) * (height - 20))
        cv2.line(overlay, (i - 1, y0), (i, y1), c0, 2)

    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

    if not history:
        return

    last_en = history[-1]
    last_ru = config.EMOTION_RU.get(last_en, last_en)
    color = EMOTION_COLORS.get(last_en, (200, 200, 200))
    cv2.putText(
        frame, f"Сейчас: {last_ru}",
        (x + width + 10, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
    )

    counts = {e: history.count(e) for e in set(history)}
    top_en = max(counts, key=counts.get)
    top_ru = config.EMOTION_RU.get(top_en, top_en)
    pct = counts[top_en] / len(history) * 100
    cv2.putText(
        frame, f"Чаще: {top_ru} {pct:.0f}%",
        (x + width + 10, y + 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, EMOTION_COLORS.get(top_en, (200, 200, 200)), 1,
    )
