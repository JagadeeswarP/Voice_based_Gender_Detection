import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import pyttsx3  # Text-to-Speech
from tensorflow.keras.models import load_model

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed

# Load the trained model
model = load_model("voice_gender_model_xception.h5")  # Ensure correct file path

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=100):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    # Silence detection threshold
    if np.max(np.abs(audio)) < 0.015:
        return None

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Pad or truncate to max_pad_len
    if mfccs.shape[1] < max_pad_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_pad_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    # Convert to (40, 100, 3) to match model input
    mfccs = np.repeat(mfccs[..., np.newaxis], 3, axis=-1)

    # Resize to (71, 100, 3) for Xception model
    mfccs_resized = tf.image.resize(mfccs, (71, 100)).numpy()

    return mfccs_resized

# Function to predict gender and provide voice feedback
def predict_gender(audio_path):
    features = extract_features(audio_path)
    if features is None:
        print("Silence detected! Please record again.")
        engine.say("Silence detected! Please record again.")
        engine.runAndWait()
        return

    # Reshape to match model input (1, 71, 100, 3)
    features = np.expand_dims(features, axis=0)

    predictions = model.predict(features)[0]
    male_prob, female_prob = predictions

    result = "Male" if male_prob > female_prob else "Female"

    print(f"Male Probability: {male_prob * 100:.2f}%")
    print(f"Female Probability: {female_prob * 100:.2f}%")
    print(f"Predicted Gender: {result}")

    # Voice feedback
    engine.say(f"The predicted gender is {result}")
    engine.runAndWait()

# Function to record audio
def record_audio(filename, duration=3, samplerate=22050):
    print("Recording...")
    engine.say("Recording started")
    engine.runAndWait()

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, recording, samplerate)

    print("Recording complete.")
    engine.say("Recording complete")
    engine.runAndWait()

# Main execution
if __name__ == "_main_":
    recorded_file = "recorded_audio.wav"
    record_audio(recorded_file)
    predict_gender(recorded_file)