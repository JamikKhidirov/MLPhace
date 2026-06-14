import time


class FPSCounter:
    def __init__(self, avg_frames=30):
        self._times = []
        self._avg_frames = avg_frames

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) > self._avg_frames:
            self._times.pop(0)

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])
