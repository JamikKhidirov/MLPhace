import json
import os
import numpy as np

import config


class FaceDB:
    def __init__(self):
        self._faces = {}
        self._load()

    def _load(self):
        if os.path.exists(config.KNOWN_FACES_PATH):
            with open(config.KNOWN_FACES_PATH, "r") as f:
                data = json.load(f)
                for name, emb_list in data.items():
                    self._faces[name] = np.array(emb_list)

    def _save(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        data = {name: emb.tolist() for name, emb in self._faces.items()}
        with open(config.KNOWN_FACES_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, name, embedding):
        self._faces[name] = embedding
        self._save()

    def recognize(self, embedding, threshold=0.5):
        best_name = None
        best_sim = 0

        for name, ref_emb in self._faces.items():
            sim = np.dot(embedding, ref_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_emb)
            )
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name and best_sim > threshold:
            return best_name, best_sim
        return None, 0

    @property
    def names(self):
        return list(self._faces.keys())
