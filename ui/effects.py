import cv2
import numpy as np

FILTER_NONE = 0
FILTER_SUNGLASSES = 1
FILTER_HAT = 2
FILTER_CROWN = 3
FILTER_MASK = 4
FILTER_CLOWN = 5

FILTER_NAMES = {
    FILTER_NONE: "Нет",
    FILTER_SUNGLASSES: "Очки",
    FILTER_HAT: "Кепка",
    FILTER_CROWN: "Корона",
    FILTER_MASK: "Маска",
    FILTER_CLOWN: "Клоун",
}


def _eye_centers(face_pts):
    left_eye = [33, 133, 157, 158, 159, 160, 161, 173]
    right_eye = [362, 263, 387, 386, 385, 384, 398, 466]
    le = np.mean([face_pts[i] for i in left_eye], axis=0).astype(int)
    re = np.mean([face_pts[i] for i in right_eye], axis=0).astype(int)
    return le, re


def _nose_bridge(face_pts):
    return face_pts[168], face_pts[6]


def _mouth(face_pts):
    return face_pts[0], face_pts[17], face_pts[13], face_pts[14]


def _forehead(face_pts):
    pts = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]
    return np.mean([face_pts[i] for i in pts], axis=0).astype(int)


def apply_filter(frame, face_pts, filter_id):
    if filter_id == FILTER_NONE:
        return

    le, re = _eye_centers(face_pts)
    eye_dist = int(np.linalg.norm(le - re))
    fh = max(p[1] for p in face_pts) - min(p[1] for p in face_pts)
    fw = max(p[0] for p in face_pts) - min(p[0] for p in face_pts)

    if filter_id == FILTER_SUNGLASSES:
        _draw_sunglasses(frame, le, re, eye_dist)

    elif filter_id == FILTER_HAT:
        fh_pt = _forehead(face_pts)
        x1 = max(0, fh_pt[0] - fw // 2)
        x2 = min(frame.shape[1], fh_pt[0] + fw // 2)
        y1 = max(0, fh_pt[1] - fh // 3)
        y2 = fh_pt[1] + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 200), -1)
        cv2.ellipse(overlay, ((x1 + x2) // 2, y1), (fw // 3, 20), 0, 0, 180, (200, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    elif filter_id == FILTER_CROWN:
        fh_pt = _forehead(face_pts)
        cx, cy = fh_pt[0], fh_pt[1] - fh // 4
        pts = np.array([
            [cx - fw // 3, cy + 30], [cx - fw // 4, cy],
            [cx - fw // 6, cy + 20], [cx, cy - 30],
            [cx + fw // 6, cy + 20], [cx + fw // 4, cy],
            [cx + fw // 3, cy + 30],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (0, 215, 255))
        cv2.polylines(frame, [pts], True, (0, 150, 200), 2)

        for i in range(3):
            x = cx - fw // 4 + i * (fw // 4)
            cv2.circle(frame, (x, cy + 10), 4, (255, 255, 0), -1)

    elif filter_id == FILTER_MASK:
        m0, m17, m13, m14 = _mouth(face_pts)
        y_top = m13[1] - 5
        y_bot = min(frame.shape[0] - 1, m14[1] + 30)
        x_left = max(0, m0[0] - 20)
        x_right = min(frame.shape[1] - 1, m17[0] + 20)
        overlay = frame.copy()
        cv2.ellipse(
            overlay,
            ((x_left + x_right) // 2, (y_top + y_bot) // 2),
            ((x_right - x_left) // 2, (y_bot - y_top) // 2),
            0, 0, 360, (100, 100, 100), -1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    elif filter_id == FILTER_CLOWN:
        le, re = _eye_centers(face_pts)
        m0, m17, _, _ = _mouth(face_pts)
        m_cx, m_cy = (m0[0] + m17[0]) // 2, (m0[1] + m17[1]) // 2

        for (cx, cy) in [le, re]:
            cv2.circle(frame, (cx - 2, cy - 2), 18, (255, 0, 0), -1)
            cv2.circle(frame, (cx - 2, cy - 2), 10, (0, 0, 255), -1)
            cv2.circle(frame, (cx - 2, cy - 2), 4, (255, 255, 255), -1)

        cv2.circle(frame, (m_cx, m_cy), 25, (255, 0, 0), -1)
        cv2.circle(frame, (m_cx - 8, m_cy), 10, (255, 255, 255), -1)
        cv2.circle(frame, (m_cx + 8, m_cy), 10, (255, 255, 255), -1)


def _draw_sunglasses(frame, le, re, eye_dist):
    w = eye_dist * 2
    h = eye_dist
    cx, cy = (le[0] + re[0]) // 2, (le[1] + re[1]) // 2
    x1 = max(0, cx - w // 2)
    x2 = min(frame.shape[1] - 1, cx + w // 2)
    y1 = max(0, cy - h // 2)
    y2 = min(frame.shape[0] - 1, cy + h // 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.ellipse(overlay, ((x1 + x2) // 2, (y1 + y2) // 2),
                (w // 2, h // 2), 0, 0, 360, (20, 20, 20), -1)

    bridge_x = (le[0] + re[0]) // 2
    cv2.line(overlay, (bridge_x, y1), (bridge_x, y2), (20, 20, 20), max(2, eye_dist // 20))

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
