import librosa
import numpy as np
import joblib

# Feature extraction function
def extract_feature(file_name):
    audio, sample_rate = librosa.load(file_name, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Load trained model
model = joblib.load("emotion_model.pkl")

# Give path of any audio file from dataset
audio_file = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

# Extract features
feature = extract_feature(audio_file)

# Predict emotion
prediction = model.predict([feature])

print("Predicted Emotion:", prediction[0])