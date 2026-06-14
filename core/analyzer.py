import threading
import numpy as np
from deepface import DeepFace

import config


class FaceAnalyzer:
    def __init__(self):
        self.emotion_en = ""
        self.emotion_ru = ""
        self.age = None
        self.gender_en = ""
        self.gender_ru = ""
        self._lock = threading.Lock()
        self.emotion_history = []

    def analyze_async(self, frame):
        threading.Thread(
            target=self._analyze, args=(frame.copy(),), daemon=True
        ).start()

    def _analyze(self, frame):
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion", "age", "gender"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]

            with self._lock:
                en = result.get("dominant_emotion", "")
                self.emotion_en = en
                self.emotion_ru = config.EMOTION_RU.get(en, en)
                self.age = result.get("age")
                ge = result.get("dominant_gender", "")
                self.gender_en = ge
                self.gender_ru = config.GENDER_RU.get(ge, ge)

                if en:
                    self.emotion_history.append(en)
                    if len(self.emotion_history) > 60:
                        self.emotion_history.pop(0)
        except Exception:
            pass

    def get_state(self):
        with self._lock:
            return (
                self.emotion_ru,
                self.emotion_en,
                self.age,
                self.gender_ru,
                list(self.emotion_history),
                self.gender_en,
            )

    def get_face_embedding(self, face_img):
        try:
            emb = DeepFace.represent(
                face_img,
                model_name="Facenet",
                enforce_detection=False,
                silent=True,
            )
            if isinstance(emb, list) and len(emb) > 0:
                return np.array(emb[0]["embedding"])
        except Exception:
            pass
        return None
