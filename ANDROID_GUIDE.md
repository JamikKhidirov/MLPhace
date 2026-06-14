# ML на Android — полное руководство

## 0. Форматы ML моделей

### Какие бывают расширения

| Формат | Расширение | Для чего | Работает на мобилке? |
|--------|------------|----------|----------------------|
| **TensorFlow SavedModel** | `.pb` / `saved_model/` | TF модели | ❌ Нужна конвертация |
| **Keras** | `.h5` / `.keras` | Keras модели | ❌ Нужна конвертация |
| **TensorFlow Lite** | `.tflite` | Оптимизировано для мобил | ✅ Да |
| **MediaPipe Task** | `.task` | MediaPipe (лицо, руки, тело) | ✅ Да |
| **PyTorch** | `.pt` / `.pth` | PyTorch модели | ❌ Нужна конвертация |
| **PyTorch Mobile** | `.ptl` | Оптимизировано для мобил | ✅ Да |
| **ONNX** | `.onnx` | Универсальный формат | ⚠️ Через ONNX Runtime |
| **Core ML** | `.mlmodel` / `.mlpackage` | Только iOS | ⚠️ Только iOS |
| **OpenVINO** | `.xml` + `.bin` | Intel оптимизация | ❌ |
| **Scikit-learn** | `.pkl` / `.joblib` | Классический ML | ❌ |
| **OpenCV Cascade** | `.xml` | Каскады Хаара (лица) | ✅ Да |
| **ML Kit** | API (без файлов) | Готовые облачные решения | ✅ Да |

### Что работает на Android

```
✅ TFLite (.tflite)       — основной формат, все фреймворки
✅ MediaPipe (.task)       — Face, Hand, Pose
✅ ML Kit (API)            — без файлов, облачные API
✅ OpenCV Cascade (.xml)   — для детекции лиц
✅ PyTorch Mobile (.ptl)   — если очень нужно
✅ ONNX Runtime (.onnx)    — универсальный вариант
❌ TensorFlow (.pb/.h5)    — надо конвертировать
❌ PyTorch (.pt/.pth)      — надо конвертировать
❌ Scikit-learn (.pkl)     — не работает
```

### Как конвертировать в TFLite

```bash
# Keras → TFLite
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
open('model.tflite', 'wb').write(tflite)
"

# PyTorch → ONNX → TFLite
python -c "
import torch
import torch.onnx
model = torch.load('model.pt', map_location='cpu')
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'model.onnx')

import onnx
from onnx_tf.backend import prepare
tf_rep = prepare(onnx.load('model.onnx'))
tf_rep.export_graph('model.pb')

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('model.pb')
tflite = converter.convert()
open('model.tflite', 'wb').write(tflite)
"

# PyTorch → PyTorch Mobile (.ptl)
python -c "
import torch
model = torch.load('model.pt', map_location='cpu')
model.eval()
scripted = torch.jit.script(model)
scripted.save('model.ptl')
"
```

### Сравнение размеров

| Модель | Формат | Размер |
|--------|--------|--------|
| MobileNetV2 | `.h5` | 14 MB |
| MobileNetV2 | `.tflite` (float32) | 14 MB |
| MobileNetV2 | `.tflite` (float16) | 7 MB |
| MobileNetV2 | `.tflite` (int8) | 3.5 MB |
| YOLOv8n | `.pt` | 6 MB |
| YOLOv8n | `.tflite` | 4 MB |
| Face Landmarker | `.task` | 3.7 MB |
| Hand Landmarker | `.task` | 7.8 MB |
| Pose Landmarker | `.task` | 5.7 MB |

**Для мобилки: TFLite с INT8 квантованием — идеал (маленький, быстрый).**

---

---

## 2. Инструменты

| Инструмент | Для чего | Ссылка |
|------------|----------|--------|
| Android Studio | Писать код | https://developer.android.com/studio |
| MediaPipe Tasks | Face / Hand / Pose | `com.google.mediapipe:tasks-vision` |
| TFLite | Свои .tflite модели | `org.tensorflow:tensorflow-lite` |
| ML Kit | Готовые API от Google | `com.google.mlkit:*` |

---

## 3. Использование MediaPipe Tasks (наш проект)

Это самый простой путь — берёшь `.task` файлы и используешь готовый API.

### 2.1 Подключение к проекту

**build.gradle (app):**
```groovy
dependencies {
    // MediaPipe Tasks Vision — для Face / Hand / Pose
    implementation 'com.google.mediapipe:tasks-vision:0.10.0'
    
    // Камера
    implementation 'androidx.camera:camera-core:1.3.0'
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
}
```

### 2.2 Добавляем модели в проект

```
app/
  src/
    main/
      assets/           ← сюда кладёшь .task файлы
        face_landmarker.task
        hand_landmarker.task
        pose_landmarker_lite.task
```

Модели те же самые, что в нашем Python проекте — скачай с Google Storage:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

### 2.3 Face Landmarker — полный код (Kotlin)

```kotlin
// FaceAnalyzer.kt

import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.core.RunningMode

class FaceAnalyzer(private val context: Context) {

    private val faceLandmarker: FaceLandmarker

    init {
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath("face_landmarker.task")
                    .build()
            )
            .setNumFaces(5)
            .setOutputFaceBlendshapes(true)  // ← эмоции!
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultCallback { result, _ ->
                onFaceResult(result)
            }
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(context, options)
    }

    private fun onFaceResult(result: FaceLandmarkerResult) {
        for ((i, landmarks) in result.faceLandmarks().withIndex()) {
            // 468 точек лица
            for (lm in landmarks) {
                val x = lm.x()  // 0..1
                val y = lm.y()  // 0..1
                // рисуем точку на canvas
            }

            // Blendshapes — эмоции!
            if (i < result.faceBlendshapes().size) {
                val blendshapes = result.faceBlendshapes()[i]
                for (bs in blendshapes) {
                    val name = bs.categoryName()
                    val score = bs.score()
                    when {
                        name == "jawOpen" && score > 0.5 -> "рот открыт"
                        name == "mouthSmileLeft" && score > 0.4 -> "улыбается"
                        name == "browDownLeft" && score > 0.5 -> "хмурится"
                        name == "eyeBlinkLeft" && score > 0.5 -> "моргнул"
                        name == "tongueOut" && score > 0.3 -> "язык"
                    }
                }
            }
        }
    }

    fun detect(imageProxy: ImageProxy) {
        val mpImage = MediaPipeImage.fromImageProxy(imageProxy)
        faceLandmarker.detectAsync(mpImage, SystemClock.uptimeMillis())
    }

    fun close() {
        faceLandmarker.close()
    }
}
```

### 2.4 Камера + Face Landmarker — Activity

```kotlin
// MainActivity.kt

class MainActivity : AppCompatActivity() {
    private lateinit var faceAnalyzer: FaceAnalyzer
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView  // кастомный View для рисования

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        
        faceAnalyzer = FaceAnalyzer(this)
        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { image ->
                        faceAnalyzer.detect(image)
                        image.close()
                    }
                }
            
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_FRONT_CAMERA,
                preview, imageAnalyzer
            )
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        faceAnalyzer.close()
    }
}
```

### 2.5 Отрисовка точек поверх камеры (OverlayView)

```kotlin
// OverlayView.kt

class OverlayView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private var landmarks: List<List<NormalizedLandmark>> = listOf()
    private var blendshapes: List<List<Category>> = listOf()

    fun update(landmarks: List<List<NormalizedLandmark>>, blendshapes: List<List<Category>>) {
        this.landmarks = landmarks
        this.blendshapes = blendshapes
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val paint = Paint().apply {
            color = Color.GREEN
            strokeWidth = 2f
            style = Paint.Style.FILL
        }

        val textPaint = Paint().apply {
            color = Color.YELLOW
            textSize = 30f
        }

        for ((i, faceLms) in landmarks.withIndex()) {
            for (lm in faceLms) {
                val x = lm.x() * width
                val y = lm.y() * height
                canvas.drawCircle(x, y, 3f, paint)
            }

            // Рисуем текст с эмоциями
            if (i < blendshapes.size) {
                val bs = blendshapes[i]
                for (b in bs) {
                    if (b.score() > 0.5) {
                        canvas.drawText(
                            "${b.categoryName()}: ${b.score()}",
                            20f, 200f + i * 200 + i * 30f,
                            textPaint
                        )
                    }
                }
            }
        }
    }
}
```

---

## 4. Использование своей обученной модели (TFLite)

### 4.1 Конвертация Python → TFLite

```bash
# У тебя есть обученная модель my_model.h5
# Конвертируешь её в .tflite

python -c "
import tensorflow as tf

# Загружаем Keras модель
model = tf.keras.models.load_model('my_model.h5')

# Конвертируем в TFLite с оптимизацией
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # сжатие в 4 раза
converter.target_spec.supported_types = [tf.float16]  # половинная точность
tflite_model = converter.convert()

# Сохраняем
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f'Размер: {len(tflite_model) / 1024:.1f} KB')
"
```

**Важно:** модель должна быть маленькой!
- До 10 MB — отлично
- 10-50 MB — нормально
- 50+ MB — слишком много для мобилки

### 4.2 TFLite на Android — полный код

```kotlin
// Classifier.kt

import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Classifier(context: Context, modelName: String = "model.tflite") {
    
    private val interpreter: Interpreter
    private val inputSize: Int
    private val labels: List<String>

    init {
        // Загружаем модель из assets
        val modelBuffer = FileUtil.loadMappedFile(context, modelName)
        interpreter = Interpreter(modelBuffer)
        
        // Определяем размер входа (по модели)
        val inputShape = interpreter.getInputTensor(0).shape()
        inputSize = inputShape[1]  // например 224 для MobileNet
        
        // Читаем лейблы
        labels = context.assets.open("labels.txt").bufferedReader().readLines()
    }

    fun predict(bitmap: Bitmap): Prediction {
        // 1. Изменяем размер до inputSize
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        // 2. Преобразуем в ByteBuffer
        val input = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        input.order(ByteOrder.nativeOrder())
        input.rewind()
        
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (pixel in pixels) {
            input.putFloat(((pixel shr 16) and 0xFF) / 255.0f)  // R
            input.putFloat(((pixel shr 8) and 0xFF) / 255.0f)   // G
            input.putFloat((pixel and 0xFF) / 255.0f)            // B
        }
        
        // 3. Запускаем инференс
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)
        
        // 4. Находим лучший класс
        val probs = output[0]
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidence = probs[maxIdx]
        val label = labels[maxIdx]
        
        return Prediction(label, confidence, probs)
    }

    fun close() {
        interpreter.close()
    }
}

data class Prediction(
    val label: String,
    val confidence: Float,
    val allProbabilities: FloatArray
)
```

### 4.3 Использование Classifier в Activity

```kotlin
// MainActivity.kt

class MainActivity : AppCompatActivity() {
    private lateinit var classifier: Classifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        classifier = Classifier(this, "model.tflite")
        
        // Берём кадр с камеры и передаём
        imageAnalyzer.setAnalyzer { imageProxy ->
            val bitmap = imageProxyToBitmap(imageProxy)
            val result = classifier.predict(bitmap)
            
            runOnUiThread {
                textView.text = "${result.label}: ${result.confidence}"
            }
            
            imageProxy.close()
        }
    }
}
```

### 4.4 ImageProxy → Bitmap (конвертация кадра)

```kotlin
fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val buffer = image.planes[0].buffer
    val pixels = ByteArray(buffer.remaining())
    buffer.get(pixels)
    
    val bitmap = Bitmap.createBitmap(
        image.width, image.height, Bitmap.Config.ARGB_8888
    )
    bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels))
    
    return bitmap
}
```

---

## 5. ML Kit — без моделей, готовые API

### 5.1 Face Detection с эмоциями

```kotlin
// build.gradle
dependencies {
    implementation 'com.google.mlkit:face-detection:16.1.6'
}
```

```kotlin
// FaceDetectionActivity.kt

val detector = FaceDetection.getClient(
    FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)  // эмоции!
        .build()
)

// На каждом кадре:
val image = InputImage.fromBitmap(bitmap, 0)
detector.process(image)
    .addOnSuccessListener { faces ->
        for (face in faces) {
            val bounds = face.boundingBox
            val smileProb = face.smilingProbability       // 0.87 — улыбается
            val leftEyeOpen = face.leftEyeOpenProbability  // 0.95 — открыт
            val rightEyeOpen = face.rightEyeOpenProbability
            val eulerY = face.headEulerAngleY              // поворот головы
            val eulerZ = face.headEulerAngleZ
            
            // Точки лица
            val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)
            val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)
            val noseBase = face.getLandmark(FaceLandmark.NOSE_BASE)
            val mouthBottom = face.getLandmark(FaceLandmark.MOUTH_BOTTOM)
        }
    }
```

### 5.2 Другие ML Kit API

| API | Что делает | Ссылка |
|-----|------------|--------|
| Face Detection | Лица + эмоции + точки | `com.google.mlkit:face-detection` |
| Text Recognition | Распознаёт текст | `com.google.mlkit:text-recognition` |
| Barcode Scanning | QR/штрихкоды | `com.google.mlkit:barcode-scanning` |
| Object Detection | Любые объекты | `com.google.mlkit:object-detection` |
| Pose Detection | Скелет человека | `com.google.mlkit:pose-detection` |
| Selfie Segmentation | Вырезает человека | `com.google.mlkit:selfie-segmentation` |
| Image Labeling | Описание картинки | `com.google.mlkit:image-labeling` |

---

## 6. Как портировать наш Python проект на Android

### Что есть в Python → что будет на Android:

| Python | Android |
|--------|---------|
| OpenCV Haar Cascade | ML Kit Face Detection |
| MediaPipe FaceLandmarker | MediaPipe Tasks FaceLandmarker |
| MediaPipe HandLandmarker | MediaPipe Tasks HandLandmarker |
| MediaPipe PoseLandmarker | MediaPipe Tasks PoseLandmarker |
| DeepFace emotion | Blendshapes от FaceLandmarker |
| DeepFace age/gender | ML Kit Face Detection |

DeepFace НЕ работает на мобилках — слишком тяжёлый.
Вместо него используй blendshapes.

### Соответствие эмоций:

```kotlin
// Вместо DeepFace на Android — blendshapes от FaceLandmarker

fun getEmotion(blendshapes: List<Category>): String {
    val map = blendshapes.associate { it.categoryName() to it.score() }
    
    return when {
        map["mouthSmileLeft"]?.let { it > 0.4 } == true &&
        map["mouthSmileRight"]?.let { it > 0.4 } == true -> "😊"
        map["browDownLeft"]?.let { it > 0.5 } == true &&
        map["browDownRight"]?.let { it > 0.5 } == true -> "😠"
        map["jawOpen"]?.let { it > 0.5 } == true -> "😲"
        map["browInnerUp"]?.let { it > 0.5 } == true -> "😨"
        map["mouthFrownLeft"]?.let { it > 0.3 } == true -> "😢"
        else -> "😐"
    }
}
```

---

## 7. Полный проект Android — минимальный шаблон

### Структура файлов:

```
MyMLApp/
├── app/
│   ├── src/main/
│   │   ├── assets/
│   │   │   ├── face_landmarker.task
│   │   │   └── hand_landmarker.task
│   │   ├── java/com/example/mymlapp/
│   │   │   ├── MainActivity.kt
│   │   │   ├── FaceAnalyzer.kt
│   │   │   ├── HandAnalyzer.kt
│   │   │   └── OverlayView.kt
│   │   ├── res/layout/
│   │   │   └── activity_main.xml
│   │   └── AndroidManifest.xml
│   └── build.gradle
├── build.gradle
└── settings.gradle
```

### activity_main.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <androidx.camera.view.PreviewView
        android:id="@+id/previewView"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />

    <com.example.mymlapp.OverlayView
        android:id="@+id/overlayView"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />

    <TextView
        android:id="@+id/emotionText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:textSize="24sp"
        android:textColor="@android:color/white"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        android:layout_margin="20dp"/>

</androidx.constraintlayout.widget.ConstraintLayout>
```

### AndroidManifest.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-feature android:name="android.hardware.camera" android:required="true" />
    <uses-permission android:name="android.permission.CAMERA" />

    <application
        android:allowBackup="true"
        android:label="MLPhace"
        android:theme="@style/Theme.AppCompat.NoActionBar">

        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

---

## 8. Быстрый старт за 10 минут

```
1. Открой Android Studio → New Project → Empty Activity

2. Добавь в build.gradle зависимости (раздел 2.1)

3. Скачай face_landmarker.task и положи в app/src/main/assets/

4. Скопируй код FaceAnalyzer.kt (раздел 2.3) и OverlayView.kt (раздел 2.5)

5. Скопируй код MainActivity.kt (раздел 2.4)

6. Запусти на телефоне → увидишь сетку лица и эмоции
```

---

## 9. Ресурсы

- **MediaPipe Tasks для Android**: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/android
- **ML Kit для Android**: https://developers.google.com/ml-kit/vision/face-detection/android
- **TensorFlow Lite для Android**: https://tensorflow.org/lite/android
- **Пример проекта MediaPipe на GitHub**: https://github.com/google-ai-edge/mediapipe-samples
- **Конвертер в TFLite**: https://tensorflow.org/lite/convert
