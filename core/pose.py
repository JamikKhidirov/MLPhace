from mediapipe.tasks.python.vision import pose_landmarker as pl
from mediapipe.tasks.python.core import base_options as base_opts

import config


class PoseDetector:
    def __init__(self):
        options = pl.PoseLandmarkerOptions(
            base_options=base_opts.BaseOptions(
                model_asset_path=config.POSE_MODEL
            ),
            num_poses=config.NUM_POSES,
        )
        self._detector = pl.PoseLandmarker.create_from_options(options)
        self.connections = pl.PoseLandmarksConnections

    def detect(self, mp_image):
        return self._detector.detect(mp_image)

    def close(self):
        self._detector.close()
