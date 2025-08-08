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

data_path = "C:\DL_Project\Gender_Detection_Dataset\data"

def extract_features(file_path, max_pad_len=100):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    if np.max(np.abs(audio)) < 0.015: 
        return None
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

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

X = np.repeat(X[..., np.newaxis], 3, axis=-1)

X_resized = np.array([tf.image.resize(x, (71, 100)).numpy() for x in X])

X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, stratify=y, random_state=42)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

base_model = Xception(weights="imagenet", include_top=False, input_shape=(71, 100, 3))

for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(2, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), class_weight=class_weights)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

model.save("voice_gender_model_xception.h5")
