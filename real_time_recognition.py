import os
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("face_recognition_model.keras")

# Load class labels
with open("classnames.txt", "r") as f:
    CLASSES = f.read().splitlines()

IMG_SIZE = 100  # Image size used for training

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.repeat(face, 3, axis=-1)  # Convert grayscale to 3-channel
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize & reshape

        # Predict person
        predictions = model.predict(face)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]

        # Determine test result
        if confidence > 0.60:  # Threshold for a match
            person_name = CLASSES[class_index]
            test_result = "Matched"
            color = (0, 255, 0)  # Green
        else:
            person_name = "Unknown"
            test_result = "Not Matched"
            color = (0, 0, 255)  # Red

        label = f"{person_name} ({confidence*100:.2f}%) - {test_result}"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
