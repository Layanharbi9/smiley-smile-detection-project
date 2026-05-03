import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "smile_model.h5"
IMG_SIZE   = 224
THRESHOLD  = 0.4 

print("[1] Loading the model...")
model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("[2] Camera started successfully. Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size > 0:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            prediction = model.predict(face_input, verbose=0)[0][0]
            
            if prediction >= THRESHOLD:
                label = f"Smiling: {prediction*100:.1f}%"
                color = (0, 255, 0) 
            else:
                label = f"Not Smiling: {(1-prediction)*100:.1f}%"
                color = (0, 0, 255) 

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Smiley Project - Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()