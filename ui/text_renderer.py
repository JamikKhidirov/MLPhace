from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"
_font_cache = {}


def _get_font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


class TextRenderer:
    def __init__(self, frame):
        self._pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._draw = ImageDraw.Draw(self._pil)

    def put(self, text, pos, color=(255, 255, 255), size=20):
        self._draw.text(pos, text, font=_get_font(size), fill=(color[2], color[1], color[0]))

    def apply(self, frame):
        frame[:] = cv2.cvtColor(np.array(self._pil), cv2.COLOR_RGB2BGR)
