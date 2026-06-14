# ML модели — полный гайд

## 1. TensorFlow — что это?

**TensorFlow** — библиотека от Google для создания и обучения нейросетей.

### Как устроен TensorFlow

```
Данные → Tensor (многомерный массив) → Граф операций → Результат
```

**Tensor** — это просто многомерный массив (как numpy array), но который может
считаться на GPU / TPU.

```python
import tensorflow as tf

# Создаём тензор
a = tf.constant([[1, 2], [3, 4]])  # shape (2, 2)
b = tf.constant([[5, 6], [7, 8]])

# Операции над тензорами
c = tf.matmul(a, b)  # умножение матриц
```

### Что внутри TensorFlow?

| Компонент | Что делает |
|-----------|------------|
| `tf.keras` | Высокоуровневое API для создания сеток |
| `tf.data` | Загрузка и подготовка данных |
| `tf.train` | Оптимизаторы (Adam, SGD и т.д.) |
| `tf.nn` | Нейросетевые операции (conv2d, relu, softmax) |
| `tf.lite` | Конвертация для мобилок / edge |
| `tf.js` | Запуск моделей в браузере |

### Пример простой сети на TensorFlow

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 класса
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

---

## 2. Как модель определяет "рот открыт" или "улыбка"?

### Что такое blendshapes (блендшейпы)?

**Blendshape** — это числовой коэффициент (от 0 до 1), который показывает,
насколько сильно выражено определённое движение лица.

```
jawOpen = 0.92  ← рот широко открыт
jawOpen = 0.05  ← рот закрыт
mouthSmileLeft = 0.87 ← левый угол рта в улыбке
```

### Как MediaPipe это делает?

```
Кадр с камеры → Face Landmarker → 468 точек лица → Blendshapes
```

**468 точек лица** — это 468 ключевых точек (landmarks) на лице:

```
Точка 0: левый угол рта
Точка 13: центр верхней губы
Точка 14: центр нижней губы
Точка 17: правый угол рта
Точка 1: кончик носа
Точка 33: левый глаз
Точка 263: правый глаз
...
```

**Blendshapes** вычисляются из положения этих 468 точек.

Как работает конкретно `jawOpen` (рот открыт):

```
1. Берём точки вокруг рта (landmarks 0, 13, 14, 17 и т.д.)
2. Вычисляем расстояние между верхней губой (13) и нижней губой (14)
3. Нормализуем (делим на ширину лица)
4. Получаем коэффициент: 0.0 = закрыт, 1.0 = открыт максимум
```

Аналогично для **улыбки** (`mouthSmileLeft`):

```
1. Берём точку угла рта (0) и точку щеки
2. Вычисляем, насколько угол рта смещён в сторону и вверх
3. Чем больше смещение → тем выше score
```

То есть модель **не думает** "это улыбка". Она просто измеряет
геометрическое положение точек и сопоставляет с эталоном.

### Как это обучали?

```
Шаг 1: Взяли 10000 фото лиц
Шаг 2: Разметили на каждом 468 точек лица вручную
Шаг 3: Обучили нейросеть предсказывать эти точки по картинке
         Вход: картинка 256×256
         Выход: 468 × 2 координаты = 936 чисел
Шаг 4: Для блендшейпов взяли ещё 10000 фото
         На каждом фото эксперт оценил emotion от 0 до 1
         (jawOpen: 0.8, mouthSmile: 0.9 и т.д.)
Шаг 5: Обучили маленькую сетку предсказывать emotion из 468 точек
```

Итог: **две сети, работающие последовательно**:
```
Картинка → Detector (468 точек) → Regressor (52 blendshape)
```

---

## 3. Как обучить свою ML модель с нуля

### Полный пайплайн

```
1. Понять задачу
2. Собрать данные
3. Разметить данные
4. Выбрать архитектуру
5. Написать код обучения
6. Обучить
7. Валидировать
8. Экспортировать
9. Использовать
```

### Шаг 1. Понять задачу

| Вопрос | Пример |
|--------|--------|
| Что предсказываем? | Кошки vs собаки |
| Это классификация? | Да, 2 класса |
| Сколько данных нужно? | Минимум 100 на класс |
| Какая точность нужна? | 95%+ |
| Где будет работать? | На сервере / телефоне |

### Шаг 2. Собрать данные

**Где брать данные:**

- Kaggle: https://kaggle.com/datasets
- Roboflow: https://universe.roboflow.com
- Hugging Face: https://huggingface.co/datasets
- Google Dataset Search
- Свои фото / видео / скрапинг

**Форматы данных:**

```
Классификация:
  dataset/
    cats/      ← папка = класс
      cat1.jpg
      cat2.jpg
    dogs/
      dog1.jpg
      dog2.jpg

Detection:
  dataset/
    images/
      img1.jpg
    labels/
      img1.txt  ← YOLO формат: класс x y w h

Landmarks:
  dataset/
    images/
      face1.jpg
    labels.csv  ← имя_файла, x1,y1, x2,y2, ..., x468,y468
```

### Шаг 3. Разметить данные

**Инструменты:**

| Инструмент | Для чего |
|------------|----------|
| LabelImg | Bounding boxes |
| CVAT | Bboxes + polygons + keypoints |
| LabelMe | Polygons (сегментация) |
| Roboflow | Всё + аугментация |
| Makesense.ai | Бесплатно, онлайн |

**Сколько нужно данных:**

| Задача | Минимум | Хорошо | Идеально |
|--------|---------|--------|----------|
| Классификация (2 класса) | 100 шт | 1000+ | 10000+ |
| Классификация (10 классов) | 500 шт | 5000+ | 50000+ |
| Detection | 200 шт | 2000+ | 20000+ |
| Landmarks | 500 шт | 5000+ | 50000+ |
| Segmentation | 100 шт | 1000+ | 10000+ |

### Шаг 4. Выбрать архитектуру

| Задача | Модель для старта | Почему |
|--------|-------------------|--------|
| Классификация фото | MobileNetV2 | Быстрая, маленькая |
| Классификация фото | ResNet50 | Точнее, но тяжелее |
| Detection | YOLOv8 | Лучшая по скорости/точности |
| Detection | SSD | Проще для старта |
| Landmarks | MediaPipe | Готовая, не надо учить |
| Landmarks | MobileNet + regression | Своя реализация |
| Segmentation | U-Net | Стандарт для сегментации |
| Emotion | Mini-Xception | Маленькая, точная |

Для **самой первой модели** возьми **MobileNetV2** — она предобучена
на ImageNet (1.2 млн фото), маленькая (14 MB), быстрая.

### Шаг 5. Написать код обучения

#### Классификация с нуля (на своих данных):

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Загружаем данные
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    subset='training',
)

val_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    subset='validation',
)

# 2. Создаём модель
base = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base.trainable = False  # заморозка — не трогаем веса

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),  # 2 класса
])

# 3. Компилируем
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# 4. Обучаем
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
)

# 5. Сохраняем
model.save('my_model.h5')
```

#### Если данных мало — Transfer Learning (дообучение):

```python
# Разморозить верхние слои предобученной модели
base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False  # первые 100 слоёв заморожены

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
)
model.fit(train_gen, validation_data=val_gen, epochs=10)
```

### Шаг 6. Обучить (где считать)

| Вариант | Цена | GPU |
|---------|------|-----|
| Свой PC | Бесплатно | Любая NVIDIA |
| Google Colab | Бесплатно | T4 / V100 |
| Kaggle | Бесплатно | P100 |
| Paperspace | $8/мес | RTX 4000 |
| RunPod | $0.2/час | A100 |
| Lambda Lab | $0.5/час | H100 |

**Google Colab — лучший для старта:**
```
https://colab.research.google.com

Runtime → Change runtime type → GPU
```

### Шаг 7. Валидировать (как понять, что работает)

```python
# Метрики
val_loss, val_acc = model.evaluate(val_gen)
print(f"Точность: {val_acc:.2%}")

# Матрица ошибок
from sklearn.metrics import confusion_matrix
y_pred = model.predict(val_gen)
y_true = val_gen.classes
cm = confusion_matrix(y_true, y_pred.argmax(axis=1))

# Проблемы:
# - Overfitting: loss на train падает, на val растёт
#   → добавить Dropout, уменьшить сеть, больше данных
# - Underfitting: loss везде высокий
#   → усложнить сеть, убрать dropout, больше эпох
```

### Шаг 8. Экспортировать

```python
# Сохранить модель
model.save('my_model.h5')

# Конвертировать в TFLite (для мобилок / MediaPipe)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Конвертировать в ONNX (универсальный формат)
# pip install tf2onnx
import tf2onnx
onnx_model = tf2onnx.convert.from_keras(model)
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

### Шаг 9. Использовать

```python
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    class_id = pred.argmax()
    confidence = pred[class_id]

    label = f"Класс {class_id}: {confidence:.2%}"
    cv2.putText(frame, label, (20, 40), ...)
    cv2.imshow('Webcam', frame)
```

---

## 4. Как работает детекция лица в нашем проекте

```
Шаг 1:
  Кадр с вебки → MediaPipe FaceLandmarker
  └→ Нейросеть находит лицо и 468 точек

Шаг 2:
  Из 468 точек → Blendshapes (52 коэффициента)
  └→ Маленькая нейросеть считает emotion по точкам

Шаг 3:
  JawOpen > 0.5  → "рот открыт"
  MouthSmile > 0.4 → "улыбается"
  TongueOut > 0.3 → "язык"
  BrowDownLeft > 0.5 → "нахмурился"
```

---

## 5. Как запустить ML модель на телефоне

### 5.1 TensorFlow Lite (TFLite) — Android + iOS

Самый популярный способ. Конвертируешь модель в `.tflite` и запускаешь на телефоне.

**Шаг 1: Конвертировать модель в TFLite**

```bash
# Из Keras
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('my_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # сжатие
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

**Шаг 2: Добавить в Android проект**

```kotlin
// build.gradle (app)
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

```kotlin
// Inference на Android
class Classifier(context: Context) {
    private val interpreter: Interpreter

    init {
        val model = FileUtil.loadMappedFile(context, "model.tflite")
        interpreter = Interpreter(model)
    }

    fun predict(bitmap: Bitmap): FloatArray {
        val input = preprocess(bitmap)  // 224x224, normalize
        val output = Array(1) { FloatArray(2) }
        interpreter.run(input, output)
        return output[0]  // [0.1, 0.9] — вероятности классов
    }
}
```

**Шаг 3: Для iOS (Swift)**

```swift
import TensorFlowLite

class Classifier {
    private var interpreter: Interpreter

    init() {
        let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite")!
        interpreter = try! Interpreter(modelPath: modelPath)
        try! interpreter.allocateTensors()
    }

    func predict(image: UIImage) -> [Float] {
        let input = preprocess(image)
        try! interpreter.copy(input, toInputAt: 0)
        try! interpreter.invoke()
        let output = try! interpreter.output(at: 0)
        return [Float](unsafeData: output.data) ?? []
    }
}
```

### 5.2 MediaPipe Tasks — готовые решения для мобилок

MediaPipe Tasks — это готовые предобученные модели + API для Android/iOS/Web.

**Что есть из коробки:**

| Задача | Android | iOS | Модель |
|--------|---------|-----|--------|
| Face Detection | ✅ | ✅ | `face_detector.task` |
| Face Landmarker | ✅ | ✅ | `face_landmarker.task` |
| Hand Landmarker | ✅ | ✅ | `hand_landmarker.task` |
| Pose Landmarker | ✅ | ✅ | `pose_landmarker.task` |
| Image Classification | ✅ | ✅ | `efficientnet.tflite` |
| Object Detection | ✅ | ✅ | `efficientdet.tflite` |
| Text Classification | ✅ | ✅ | BERT-based |

**Пример на Android (Kotlin) — Face Landmarker:**

```kotlin
// build.gradle
dependencies {
    implementation 'com.google.mediapipe:tasks-vision:0.10.0'
}

// В Activity
val options = FaceLandmarkerOptions.builder()
    .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
    .setNumFaces(5)
    .setOutputFaceBlendshapes(true)
    .build()

val faceLandmarker = FaceLandmarker.createFromOptions(context, options)

// На каждом кадре:
val result = faceLandmarker.detect(image)

for (landmarks in result.faceLandmarks) {
    for (lm in landmarks) {
        // lm.x, lm.y — координаты точек лица
    }
}

// Blendshapes (эмоции):
for (blendshape in result.faceBlendshapes) {
    for (bs in blendshape) {
        Log.d("ML", "${bs.categoryName}: ${bs.score}")
        // "jawOpen: 0.92", "mouthSmileLeft: 0.87"
    }
}
```

**Пример на iOS (Swift):**

```swift
import MediaPipeTasksVision

let options = FaceLandmarkerOptions(
    baseOptions: BaseOptions(modelAssetPath: "face_landmarker.task"),
    numFaces: 5,
    outputFaceBlendshapes: true
)
let faceLandmarker = try! FaceLandmarker(options: options)

let result = try! faceLandmarker.detect(image: mpImage)
for blendshape in result.faceBlendshapes {
    for bs in blendshape {
        print("\(bs.categoryName): \(bs.score)")
    }
}
```

### 5.3 Сравнение инструментов

| Инструмент | Платформы | Скорость | Размер | Сложность |
|------------|-----------|----------|--------|-----------|
| **TFLite** | Android, iOS | ⚡️⚡️⚡️ | Маленький | 🟢 Легко |
| **MediaPipe Tasks** | Android, iOS, Web | ⚡️⚡️⚡️ | Средний | 🟢 Легко |
| **ML Kit (Google)** | Android, iOS | ⚡️⚡️⚡️ | Облако/локально | 🟢 Очень легко |
| **Core ML (Apple)** | iOS только | ⚡️⚡️⚡️⚡️ | Средний | 🟡 Средне |
| **PyTorch Mobile** | Android, iOS | ⚡️⚡️ | Большой | 🔴 Сложно |
| **ONNX Runtime** | Android, iOS, Web | ⚡️⚡️⚡️ | Средний | 🟡 Средне |
| **NCNN (Tencent)** | Android, iOS | ⚡️⚡️⚡️⚡️ | Маленький | 🔴 Сложно |
| **MNN (Alibaba)** | Android, iOS | ⚡️⚡️⚡️⚡️ | Маленький | 🔴 Сложно |

### 5.4 ML Kit — самый простой способ (без кода модели)

Google ML Kit — это готовые API, работающие в облаке или на устройстве.
Никаких моделей — просто вызываешь функцию.

```kotlin
// Android — распознавание лиц за 3 строки:
val detector = FaceDetection.getClient(FaceDetectorOptions.Builder()
    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
    .build())

detector.process(image)
    .addOnSuccessListener { faces ->
        for (face in faces) {
            val bounds = face.boundingBox
            face.smilingProbability    // 0.87 → улыбается
            face.leftEyeOpenProbability
            face.rightEyeOpenProbability
        }
    }
```

**Что умеет ML Kit:**
- Face Detection (лица + ключевые точки)
- Text Recognition (OCR)
- Barcode Scanning
- Image Labeling
- Object Detection
- Pose Detection
- Selfie Segmentation

### 5.5 Как портировать нашу программу на телефон

Наша `main.py` использует **MediaPipe Tasks** — это уже работает на мобилках.

**Надо сделать:**
1. Взять те же `.task` файлы (face_landmarker, hand_landmarker, pose_landmarker)
2. Написать Android/iOS приложение на Kotlin/Swift
3. Использовать MediaPipe Tasks API (код выше)
4. Emotion (DeepFace) заменить на ML Kit или MediaPipe blendshapes

DeepFace не работает на мобилках — слишком тяжёлый.
Вместо него используй blendshapes от FaceLandmarker:
```
mouthSmileLeft > 0.5 → 😊
browDownLeft > 0.5 → 😠
jawOpen > 0.5 → 😲
```

### 5.6 Flutter — один код для Android + iOS

```yaml
# pubspec.yaml
dependencies:
  google_mlkit_face_detection: ^0.12.0
  camera: ^0.10.0
```

```dart
// main.dart
import 'package:google_mlkit_face_detection/face_detection.dart';

final detector = FaceDetector(
  options: FaceDetectorOptions(
    enableClassification: true,  // эмоции
    enableLandmarks: true,       // точки лица
  ),
);

final inputImage = InputImage.fromFile(file);
final faces = await detector.processImage(inputImage);

for (final face in faces) {
  print('Улыбка: ${face.smilingProbability}');
  print('Правый глаз открыт: ${face.rightEyeOpenProbability}');
}
```

### 5.7 Быстрый старт — что скачать и попробовать

```
Лучший путь для первой мобильной ML:

1. Скачай MediaPipe Tasks примеры:
   https://github.com/google-ai-edge/mediapipe-samples

2. Открой examples/face_landmarker/android/

3. Вставь свою модель .task в assets/

4. Запусти на телефоне — готово!

Альтернатива — ML Kit без моделей:
   https://developers.google.com/ml-kit/vision/face-detection
```

---

## 6. Полезные ссылки

- **TensorFlow Lite**: https://tensorflow.org/lite
- **MediaPipe Tasks**: https://ai.google.dev/edge/mediapipe/solutions/guide
- **MediaPipe Samples (GitHub)**: https://github.com/google-ai-edge/mediapipe-samples
- **ML Kit**: https://developers.google.com/ml-kit
- **PyTorch Mobile**: https://pytorch.org/mobile
- **ONNX Runtime Mobile**: https://onnxruntime.ai
- **Kaggle datasets**: https://kaggle.com/datasets
- **Roboflow**: https://roboflow.com
- **Google Colab**: https://colab.research.google.com
- **Papers with Code**: https://paperswithcode.com

---

## 7. План обучения "с нуля до своей модели"

```
Неделя 1: Python + numpy + OpenCV
  - Читать картинки, видео
  - Изменять размер, цвета, фильтры
  - Рисовать на кадре

Неделя 2: TensorFlow/Keras basics
  - Linear Regression на tf
  - Простая сеть для MNIST (цифры)
  - Понимать: tensor, layer, loss, optimizer

Неделя 3: Свёрточные сети (CNN)
  - Conv2D, MaxPooling, Flatten, Dense
  - Классификация кошек/собак
  - Data augmentation

Неделя 4: Transfer Learning
  - MobileNetV2, ResNet
  - Fine-tuning
  - Сохранение и загрузка

Неделя 5: Object Detection
  - YOLO / SSD
  - Разметка в LabelImg
  - Обучение детектора

Неделя 6: Свой проект
  - Сбор датасета (свои фото)
  - Разметка
  - Обучение
  - Запуск на вебку
```
