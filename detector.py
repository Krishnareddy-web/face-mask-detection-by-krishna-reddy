import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mask_model = load_model('model/mask_detector.h5')

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def detect_and_predict_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    preds = []
    boxes = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        processed = preprocess_face(face_img)
        pred = mask_model.predict(processed)[0]
        preds.append(pred)
        boxes.append((x, y, w, h))
    return boxes, preds
