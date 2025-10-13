import ultralytics
import cvzone
import cv2
import requests
import urllib
import numpy as np

from ultralytics import YOLO
from PIL import Image
from IPython.display import display
from joblib import load
from tensorflow import keras
from skimage.feature import hog

facemodel = YOLO('model/YOLO/yolov8m-face.pt')

# File model
svm_clf_loaded = load('model/svm_model.joblib')  # đường dẫn tới file mới upload

image_path = "test.jpg"

# Read the image
img = cv2.imread(image_path)

if img is None:
    raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {image_path}")

faces_48x48 = []

results = facemodel(img, stream=True)  # dùng stream=True để duyệt qua từng frame/ảnh

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Cắt khuôn mặt
        face = img[y1:y2, x1:x2]

        # Kiểm tra vùng ảnh hợp lệ
        if face.size == 0:
            continue

        # Chuyển grayscale + resize 48x48
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))

        # Thêm vào list
        faces_48x48.append(resized)

        # Vẽ khung để xem
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

display(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

print(f"Phát hiện được {len(faces_48x48)} khuôn mặt.")
if len(faces_48x48):
    print("Kích thước ảnh:", faces_48x48[0].shape)

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
