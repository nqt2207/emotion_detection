
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from joblib import load
from skimage.feature import hog

app = Flask(__name__)

# Load models
facemodel = YOLO('model/YOLO/yolov8m-face.pt')
svm_clf_loaded = load('model/svm_model.joblib')

def extract_hog_features(image):
    features, _ = hog(image, orientations=8, pixels_per_cell=(8, 8),
                      cells_per_block=(1, 1), visualize=True, multichannel=False)
    return features

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    in_memory = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    faces_48x48 = []
    results = facemodel(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            faces_48x48.append(resized)

    if not faces_48x48:
        return jsonify({'result': 'No face detected'})

    # For demo: predict the first face only
    features = extract_hog_features(faces_48x48[0]).reshape(1, -1)
    pred = svm_clf_loaded.predict(features)[0]
    return jsonify({'result': str(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# Trich xuat HOG features
def extract_hog_features(image):
    features = hog(image,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True)
    return features

features_list = [extract_hog_features(f) for f in faces_48x48]
X_test = np.array(features_list)

if len(X_test) > 0:
    y_pred = svm_clf_loaded.predict(X_test)

    print(f"Detected {len(y_pred)} faces:")

    results = facemodel(img, stream=True)
    i = 0
    for r in results:
        for box in r.boxes:
            if i >= len(y_pred):
                break
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = str(y_pred[i])

            # Vẽ khung và text
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            i += 1

    # Hiển thị ảnh kết quả
    display(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

else:
    print("No face found.")
