import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

#  Load Dataset
data_path = "C:\\Users\\jagad\\Downloads\\DL\\data"

def extract_features(file_path, max_pad_len=100):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    if np.max(np.abs(audio)) < 0.015: return None
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.pad(mfccs, ((0, 0), (0, max(0, max_pad_len - mfccs.shape[1]))), mode='constant')[:, :max_pad_len]
    return np.repeat(mfccs[..., np.newaxis], 3, axis=-1)

X, y = [], []
labels = {'male': 0, 'female': 1}

for label, idx in labels.items():
    for file in os.listdir(os.path.join(data_path, label)):
        if file.endswith(".wav"):
            feat = extract_features(os.path.join(data_path, label, file))
            if feat is not None: X.append(feat); y.append(idx)

X, y = np.array(X), np.array(y)
X_resized = np.array([tf.image.resize(x, (71, 100)).numpy() for x in X])
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, stratify=y, random_state=42)
y_train, y_test = to_categorical(y_train, 2), to_categorical(y_test, 2)
class_weights = {i: w for i, w in enumerate(compute_class_weight("balanced", classes=np.unique(y), y=y))}

base_model = Xception(weights="imagenet", include_top=False, input_shape=(71, 100, 3))
for layer in base_model.layers: layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), class_weight=class_weights)

model.save("voice_gender_model_xception.h5")

def plot_training_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axs[0].set_title('Model Accuracy'); axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Accuracy'); axs[0].legend()
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Val Loss')
    axs[1].set_title('Model Loss'); axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Loss'); axs[1].legend()
    plt.show()

plot_training_history(history)

y_pred_probs = model.predict(X_test)
y_pred, y_true = np.argmax(y_pred_probs, axis=1), np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.show()

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Male", "Female"]))
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")