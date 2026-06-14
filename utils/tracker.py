class FaceTracker:
    def __init__(self, iou_threshold=0.3):
        self._next_id = 1
        self._faces = {}
        self._iou_threshold = iou_threshold

    def update(self, boxes):
        new_faces = {}
        used = set()

        for bid, box in enumerate(boxes):
            best_id = None
            best_iou = 0
            for fid, prev_box in self._faces.items():
                iou = self._iou(box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = fid

            if best_id is not None and best_iou > self._iou_threshold:
                new_faces[best_id] = box
                used.add(best_id)
            else:
                new_id = self._next_id
                self._next_id += 1
                new_faces[new_id] = box

        self._faces = new_faces
        return list(self._faces.keys())

    def get_id_color(self, fid):
        colors = [
            (0, 255, 0),
            (255, 200, 0),
            (0, 200, 255),
            (200, 0, 255),
            (255, 0, 200),
            (0, 255, 200),
            (200, 200, 0),
            (200, 0, 200),
        ]
        return colors[(fid - 1) % len(colors)]

    @staticmethod
    def _iou(a, b):
        x1, y1, x2, y2 = a
        x3, y3, x4, y4 = b
        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (x4 - x3) * (y4 - y3)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0
