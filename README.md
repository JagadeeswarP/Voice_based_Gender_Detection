DataSets Link:
Common Voice : https://commonvoice.mozilla.org/en/datasets
Kaggle : https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal/data

Summary:
This project focuses on developing an intelligent system to classify a person's gender (male or female) based on their voice using deep learning techniques. The approach combines audio signal processing with a Convolutional Neural Network (CNN) trained on MFCC (Mel-Frequency Cepstral Coefficients) features extracted from speech samples.

MFCC (Mel-Frequency Cepstral Coefficients) – What It Does and Why It’s Used
1. What is MFCC?
MFCC (Mel-Frequency Cepstral Coefficients) is a feature extraction technique used in speech and audio processing. It helps convert a raw audio signal into a set of numerical values that represent the most important characteristics of the sound.

2. Why is MFCC Used in Voice-Based Gender Detection?
Human voices differ based on pitch, tone, and frequency content.

MFCC captures these differences by analyzing the sound frequencies most relevant to human perception.

It helps in distinguishing male and female voices, as male voices usually have lower frequency components than female voices.


For training, Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from the recordings to capture essential speech features. The pre-trained Xception model is fine-tuned using these extracted features for binary classification (male/female).


