#transfer learning model(Xception)
import os
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import wave
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from pydub import AudioSegment
from pydub.playback import play

# Define dataset path
data_path = "C:\\Users\\tsnte\\Downloads\\archive\\data"

# Load and extract MFCC features
def extract_features(file_path, max_pad_len=100):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    if np.max(np.abs(audio)) < 0.015:  # Silence detection threshold
        return None
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# Prepare dataset
X, y = [], []
labels = {'male': 0, 'female': 1}

for label in labels.keys():
    folder_path = os.path.join(data_path, label)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(labels[label])

X = np.array(X)
y = np.array(y)

# Convert grayscale (40,100,1) to (40,100,3)
X = np.repeat(X[..., np.newaxis], 3, axis=-1)

# Resize to (71,100,3) for Xception
X_resized = np.array([tf.image.resize(x, (71, 100)).numpy() for x in X])

# Split dataset (Stratified to balance male & female)
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, stratify=y, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Compute class weights to handle dataset imbalance
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Load pre-trained Xception model (without the top layers)
base_model = Xception(weights="imagenet", include_top=False, input_shape=(71, 100, 3))

# Freeze base layers to retain learned features
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(2, activation='softmax')(x)

# Define final model
model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with class weights
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), class_weight=class_weights)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Save the model
model.save("voice_gender_model_xception.h5")