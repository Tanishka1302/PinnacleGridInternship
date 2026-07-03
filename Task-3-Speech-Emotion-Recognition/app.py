import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# Load model
model = joblib.load("emotion_model.pkl")

# Feature extraction
def extract_feature(file_name):
    audio, sample_rate = librosa.load(file_name, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Emojis
emoji = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "fearful": "😨",
    "calm": "😌",
    "neutral": "😐",
    "disgust": "🤢",
    "surprised": "😲"
}

st.title("🎤 Speech Emotion Recognition System")

st.markdown("""
Upload a **WAV** audio file and the trained Machine Learning model will predict the speaker's emotion.
""")

uploaded_file = st.file_uploader("📂 Upload Audio File", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    feature = extract_feature(temp_path)

    prediction = model.predict([feature])[0]

    st.success(
        f"{emoji.get(prediction,'🎭')} Predicted Emotion: **{prediction.capitalize()}**"
    )

st.markdown("---")
st.caption("Developed using Python, Librosa, Scikit-Learn & Streamlit")