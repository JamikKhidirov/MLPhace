import os

DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DIR, "models")
DATA_DIR = os.path.join(DIR, "data")
SCREENSHOTS_DIR = os.path.join(DATA_DIR, "screenshots")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
KNOWN_FACES_PATH = os.path.join(DATA_DIR, "known_faces.json")

FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

NUM_FACES = 10
NUM_HANDS = 6
NUM_POSES = 10
EMOTION_INTERVAL = 30
COMPARE_INTERVAL = 60
RECORD_FPS = 20.0

EMOTION_RU = {
    "happy": "счастье",
    "sad": "грусть",
    "angry": "злость",
    "fear": "страх",
    "surprise": "удивление",
    "disgust": "отвращение",
    "neutral": "нейтрально",
}

EMOTION_EMOJI = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐",
}

GENDER_RU = {"Man": "Мужчина", "Woman": "Женщина"}

BLENDSHAPES_RU = {
    "browDownLeft": "бровь вниз Л",
    "browDownRight": "бровь вниз П",
    "browInnerUp": "брови вверх",
    "browOuterUpLeft": "бровь вверх Л",
    "browOuterUpRight": "бровь вверх П",
    "cheekPuff": "надул щеки",
    "cheekSquintLeft": "щурится Л",
    "cheekSquintRight": "щурится П",
    "eyeBlinkLeft": "моргнул Л",
    "eyeBlinkRight": "моргнул П",
    "eyeLookDownLeft": "глаз вниз Л",
    "eyeLookDownRight": "глаз вниз П",
    "eyeLookInLeft": "глаз внутрь Л",
    "eyeLookInRight": "глаз внутрь П",
    "eyeLookOutLeft": "глаз наружу Л",
    "eyeLookOutRight": "глаз наружу П",
    "eyeLookUpLeft": "глаз вверх Л",
    "eyeLookUpRight": "глаз вверх П",
    "eyeSquintLeft": "прищур Л",
    "eyeSquintRight": "прищур П",
    "eyeWideLeft": "глаза широко Л",
    "eyeWideRight": "глаза широко П",
    "jawForward": "челюсть вперед",
    "jawLeft": "челюсть влево",
    "jawRight": "челюсть вправо",
    "jawOpen": "рот открыт",
    "mouthClose": "рот закрыт",
    "mouthDimpleLeft": "ямочка Л",
    "mouthDimpleRight": "ямочка П",
    "mouthFrownLeft": "хмурится Л",
    "mouthFrownRight": "хмурится П",
    "mouthFunnel": "трубочка",
    "mouthLeft": "рот влево",
    "mouthRight": "рот вправо",
    "mouthLowerDownLeft": "губа вниз Л",
    "mouthLowerDownRight": "губа вниз П",
    "mouthPressLeft": "губы сжаты Л",
    "mouthPressRight": "губы сжаты П",
    "mouthPucker": "губы бантиком",
    "mouthRollLower": "губа rolled",
    "mouthRollUpper": "губа rolled",
    "mouthShrugLower": "плечи губы",
    "mouthShrugUpper": "плечи губы",
    "mouthSmileLeft": "улыбка Л",
    "mouthSmileRight": "улыбка П",
    "mouthStretchLeft": "растяжка Л",
    "mouthStretchRight": "растяжка П",
    "mouthUpperUpLeft": "губа вверх Л",
    "mouthUpperUpRight": "губа вверх П",
    "noseSneerLeft": "нос морщит Л",
    "noseSneerRight": "нос морщит П",
    "tongueOut": "язык",
}

HEAD_POSE_RU = {
    "center": "прямо",
    "left": "влево",
    "right": "вправо",
    "up": "вверх",
    "down": "вниз",
}
