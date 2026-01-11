import cv2
import numpy as np
import os
from collections import deque, Counter

# Paths to model files
AGE_PROTO = os.path.join('Models', 'age_deploy (1).prototxt')
AGE_MODEL = os.path.join('Models', 'age_net.caffemodel')
FACE_PROTO = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Age ranges as per model, with custom label for 20-25
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '20-25', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
face_cascade = cv2.CascadeClassifier(FACE_PROTO)

# Start webcam
cap = cv2.VideoCapture(0)
age_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        pred_idx = age_preds[0].argmax()
        age = AGE_LIST[pred_idx]
        # If model predicts '(15-20)' or '(25-32)', show '20-25' if confidence is close
        if age in ['(15-20)', '(25-32)']:
            conf_15_20 = age_preds[0][4] if len(age_preds[0]) > 4 else 0
            conf_25_32 = age_preds[0][5] if len(age_preds[0]) > 5 else 0
            if abs(conf_15_20 - conf_25_32) < 0.15:
                age = '20-25'
        age_buffer.append(age)
        # Use the most frequent age in the buffer for stability
        stable_age = Counter(age_buffer).most_common(1)[0][0]
        label = f"Age: {stable_age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Age Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()