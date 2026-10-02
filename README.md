# 😊 Emotion Detection Using Deep Learning

A web-based **Facial Emotion Detection** application built with **Python, TensorFlow/Keras, OpenCV, and Flask**.

The application detects a face from an uploaded image, preprocesses the detected face into a **48×48 grayscale image**, and predicts the dominant facial emotion using a trained deep learning CNN model.

---

## 🚀 Features

* Facial emotion recognition using a trained CNN model
* Face detection using **OpenCV DNN SSD**
* Image upload-based emotion prediction
* Emotion confidence score
* Bounding box around detected faces
* Results displayed through a Flask web interface
* Webcam-based real-time emotion detection implementation
* Responsive HTML/CSS user interface
* Supports seven emotion classes

---

## 🎭 Supported Emotions

The model predicts the following seven emotions:

| Emotion  |
| -------- |
| Angry    |
| Disgust  |
| Fear     |
| Happy    |
| Sad      |
| Surprise |
| Neutral  |

---

## 🧠 Technology Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Network (CNN)

### Computer Vision

* OpenCV
* OpenCV DNN Face Detector
* SSD-based face detection

### Web Framework

* Flask

### Data Processing

* NumPy

### Frontend

* HTML5
* CSS3
* Jinja2 Templates

---

## 🏗️ Project Architecture

```text
Input Image
     │
     ▼
OpenCV DNN Face Detection
     │
     ▼
Face Region Extraction
     │
     ▼
Convert to Grayscale
     │
     ▼
Resize to 48 × 48
     │
     ▼
Normalize Pixel Values
     │
     ▼
CNN Emotion Classification
     │
     ▼
Predicted Emotion + Confidence
     │
     ▼
Flask Web Interface
```

---

## 📂 Project Structure

```text
emotion_detection/
│
├── app.py
├── fer.py
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── models/
│   ├── deploy.prototxt.txt
│   └── res10_300x300_ssd_iter_140000_fp16.caffemodel
│
├── templates/
│   ├── upload.html
│   ├── analysis.html
│   ├── result.html
│   └── blabla.html
│
└── model.keras
```

> **Note:** `model.keras` is required by the Flask application but is not currently visible in the repository tree. Make sure the trained model file is available in the project root before running the application.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/mohit26061999/emotion_detection.git
cd emotion_detection
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

The project currently specifies:

```text
flask
opencv-python
tensorflow
numpy
```

You can also install them manually:

```bash
pip install flask opencv-python tensorflow numpy
```

---

# 🤖 Model

The emotion classifier is a CNN trained on facial-expression images using **48×48 grayscale input images**.

The model architecture contains:

* Convolutional layers
* Batch Normalization
* ReLU activation
* Max Pooling
* Dropout
* Fully Connected layers
* Softmax output layer

### Input

```text
48 × 48 × 1
```

### Output

```text
7 emotion classes
```

The training script uses:

* Adam optimizer
* Categorical Cross-Entropy loss
* Accuracy as a training metric
* Early Stopping
* Model Checkpoint
* Reduce Learning Rate on Plateau

---

# 🔍 Face Detection

The application uses an **OpenCV DNN SSD face detector**.

The following model files are included inside the `models/` directory:

```text
deploy.prototxt.txt
res10_300x300_ssd_iter_140000_fp16.caffemodel
```

A detection confidence threshold of approximately **60%** is used before processing a detected face.

---

# 🖼️ Image Emotion Detection

The primary Flask application is implemented in:

```text
app.py
```

### Run the application

```bash
python app.py
```

The application will start on the Flask development server.

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

### Workflow

1. Open the application.
2. Upload an image containing a face.
3. The application detects the face using OpenCV DNN.
4. The face is converted to grayscale.
5. The face is resized to `48 × 48`.
6. Pixel values are normalized between `0` and `1`.
7. The CNN predicts the emotion.
8. The result is displayed with:

   * Detected emotion
   * Confidence percentage
   * Face bounding box
   * Processed image

---

# 📷 Webcam Emotion Detection

The file:

```text
nn.py
```

contains an implementation for **real-time webcam emotion detection**.

It uses:

* OpenCV webcam capture
* Haar Cascade face detection
* The trained Keras model
* Real-time emotion labeling

The webcam stream is exposed through a Flask endpoint:

```text
/video
```

The implementation is separate from the main image-upload workflow.

---

# 🧪 Model Training

The model training and evaluation workflow is contained in:

```text
fer.py
```

The script includes:

### Data Loading

Images are loaded using Keras `ImageDataGenerator` and `flow_from_directory()`.

Expected structure:

```text
images/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
│
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

Images are processed as:

```text
48 × 48
Grayscale
Categorical classes
```

### Training Configuration

The training code uses:

```text
Batch Size: 128
Input Size: 48 × 48 × 1
Optimizer: Adam
Loss: Categorical Cross-Entropy
Output Classes: 7
```

Callbacks included:

```text
ModelCheckpoint
EarlyStopping
ReduceLROnPlateau
```

---

# 📊 Model Evaluation

The training script also calculates:

* Accuracy
* Weighted F1 Score
* Confusion Matrix
* Classification Report

Example evaluation code:

```python
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score
)
```

The confusion matrix is visualized using Seaborn.

---

# 🌐 Flask Endpoints

The main Flask application contains the following routes:

| Route                 | Method | Purpose                                       |
| --------------------- | ------ | --------------------------------------------- |
| `/`                   | GET    | Displays upload page                          |
| `/predict`            | POST   | Processes uploaded image and predicts emotion |
| `/uploads/<filename>` | GET    | Serves uploaded/processed images              |

The webcam implementation in `nn.py` additionally provides:

| Route    | Method | Purpose                       |
| -------- | ------ | ----------------------------- |
| `/`      | GET    | Webcam interface              |
| `/video` | GET    | Real-time webcam video stream |

---

# 🔬 Prediction Process

For an uploaded image, the application performs the following preprocessing:

```python
gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

gray_face = cv2.resize(
    gray_face,
    (48, 48),
    interpolation=cv2.INTER_AREA
)

gray_face = gray_face.astype("float") / 255.0
```

The processed image is then passed to the trained model:

```python
prediction = model.predict(gray_face, verbose=0)[0]
```

The predicted class is selected using:

```python
emotion_index = np.argmax(prediction)
```

The corresponding emotion is returned along with its confidence.

---

# ⚠️ Limitations

Emotion recognition from facial expressions can be affected by:

* Poor lighting
* Face orientation
* Occlusion
* Low-resolution images
* Small faces
* Multiple faces
* Unusual facial expressions
* Dataset limitations
* Differences between training images and real-world images

The application currently returns the **dominant predicted emotion**, rather than a complete multi-emotion interpretation.

---

# 🔐 Privacy

Images uploaded to the application are processed locally by the Flask application and saved in the configured upload directory:

```text
static/uploads/
```

For production deployments, consider adding:

* Automatic deletion of uploaded images
* File type validation
* File size limits
* Secure filename handling
* Authentication
* HTTPS

---

# 🛠️ Future Improvements

Possible improvements include:

* Improve model accuracy through better training and augmentation
* Support multiple faces in a single image
* Display probability for all seven emotions
* Improve real-time webcam detection
* Add REST API endpoints
* Add model performance dashboards
* Deploy the application using Docker
* Add GPU inference support
* Add model versioning
* Add automated testing
* Improve security and upload validation

---

# 📸 Demo

You can add screenshots or GIFs of the application here:

```markdown
![Upload Page](screenshots/upload.png)

![Emotion Detection Result](screenshots/result.png)

![Webcam Detection](screenshots/webcam.png)
```

---

# 💻 Example

After starting the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

Upload a facial image and the application will return a result similar to:

```text
Detected emotion: Happy (87.4%)
```

---

# 👨‍💻 Author

**Mohit Kumar**

AI/ML Engineer | Generative AI | Computer Vision | Machine Learning

GitHub:
https://github.com/mohit26061999

---

# ⭐ Acknowledgements

This project uses open-source technologies including:

* TensorFlow / Keras
* OpenCV
* Flask
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

# 📄 License

This project does not currently include a license file.

Consider adding an open-source license such as the **MIT License** if you plan to allow others to use, modify, and distribute the project.
