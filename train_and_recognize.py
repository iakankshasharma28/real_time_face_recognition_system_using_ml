import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
DATA_DIR = "dataset"
CLASSES = sorted(os.listdir(DATA_DIR))  # Ensure consistent ordering
IMG_SIZE = 100

X, y = [], []
for label, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for image_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip invalid images
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

# Convert lists to numpy arrays
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
y = to_categorical(np.array(y), num_classes=len(CLASSES))  # One-hot encode labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert grayscale to 3-channel for MobileNetV2
X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.2, height_shift_range=0.2,
                             horizontal_flip=True, zoom_range=0.2)
datagen.fit(X_train)

# Load MobileNetV2 (pretrained on ImageNet)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze pretrained layers

# Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(len(CLASSES), activation="softmax")(x)

# Compile Model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
model.fit(datagen.flow(X_train, y_train, batch_size=16), validation_data=(X_test, y_test), epochs=10)

# Save Model & Labels
model.save("face_recognition_model.keras")
with open("classnames.txt", "w") as f:
    f.write("\n".join(CLASSES))

print("✅ Model trained and saved successfully!")
