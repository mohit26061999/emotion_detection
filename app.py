from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load emotion detection model (.h5 or .keras format)
model = load_model("model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    'models/deploy.prototxt.txt',
    'models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('analysis.html', error="No file uploaded", input_image=None, output_image=None)

    file = request.files['image']
    if file.filename == '':
        return render_template('analysis.html', error="No file selected", input_image=None, output_image=None)

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    image = cv2.imread(input_path)
    (h, w) = image.shape[:2]

    # DNN face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    label = "No Face Detected"
    found = False
    confidence_score = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Ensure box coordinates are within image boundaries
            x = max(0, x)
            y = max(0, y)
            x1 = min(w, x1)
            y1 = min(h, y1)

            # Extract face region
            face = image[y:y1, x:x1]
            if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                continue

            # Convert to grayscale
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Resize face to 48x48
            gray_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)

            # Normalize and expand dimensions to fit model input
            gray_face = gray_face.astype("float") / 255.0
            gray_face = np.expand_dims(gray_face, axis=-1)
            gray_face = np.expand_dims(gray_face, axis=0)

            # Predict emotion
            prediction = model.predict(gray_face, verbose=0)[0]
            emotion_index = np.argmax(prediction)
            label = emotion_labels[emotion_index]
            confidence_score = prediction[emotion_index] * 100

            # Draw bounding box and label
            cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
            text = f"{label}: {confidence_score:.1f}%"
            y_pos = max(y - 10, 20)
            cv2.putText(image, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            found = True
            break

    processed_filename = "processed_" + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(output_path, image)

    result_message = f"Detected emotion: {label} ({confidence_score:.1f}%)" if found else "No face detected"

    return render_template('analysis.html',
                           input_image=filename,
                           output_image=processed_filename,
                           message=result_message,
                           error=None if found else "No face detected")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
