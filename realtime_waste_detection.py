"""
realtime_waste_detection.py — Real-time waste classification using pretrained model
-----------------------------------------------------------------------------------
Usage:
    python realtime_waste_detection.py
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import time

# ==== CONFIG ====
MODEL_PATH = r"C:\Users\Dell\Downloads\archive\garbage-dataset\runs\mobilenetv2_waste_model_finetuned.h5"
IMG_SIZE = 224

# ==== LOAD MODEL ====
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (must match your training dataset order)
class_names = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]
print("[INFO] Classes:", class_names)

# ==== CAMERA SETUP ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting real-time classification... Press 'q' to quit.")
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds)
    class_name = class_names[class_idx]
    confidence = preds[0][class_idx] * 100

    # Display results
    label = f"{class_name} ({confidence:.1f}%)"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.imshow("Waste Classifier (Press 'q' to exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Real-time session ended.")
