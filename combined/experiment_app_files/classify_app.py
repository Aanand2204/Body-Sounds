import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM as OriginalLSTM
from tensorflow.keras.layers import GRU as OriginalGRU


# -------------------------
# Custom Layers (time_major fix)
# -------------------------
class CustomLSTM(OriginalLSTM):
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs:
            del kwargs['time_major']
        super().__init__(*args, **kwargs)


class CustomGRU(OriginalGRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)


# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_heart_model():
    model = tf.keras.models.load_model(
        "lstm_model.h5",
        custom_objects={'LSTM': CustomLSTM}
    )
    return model


@st.cache_resource
def load_lung_model():
    model = tf.keras.models.load_model(
        "lung_model.h5",
        custom_objects={"GRU": CustomGRU}
    )
    return model


# -------------------------
# Constants
# -------------------------
LUNG_CLASSES = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]
HEART_CLASSES = ["Normal", "Murmur"]  # Adjust based on your model


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Body Sound Classification")
st.title("🩺 Body Sound Classification (Heart & Lung)")

# Sidebar for model selection
sound_type = st.sidebar.selectbox(
    "Select Sound Type",
    ["Heart Sound", "Lung Sound"]
)

uploaded_file = st.file_uploader(f"Upload {sound_type.lower()} (.wav)", type=["wav", "mp3"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=22050)

    # Display waveform
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"{sound_type} Waveform")
    st.pyplot(fig)

    # -------------------------
    # Heart Sound Classification
    # -------------------------
    if sound_type == "Heart Sound":
        model = load_heart_model()

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        # Reshape for LSTM (samples, timesteps, features)
        X_input = np.expand_dims(mfcc_scaled, axis=0)
        X_input = np.expand_dims(X_input, axis=2)

        # Prediction
        prediction = model.predict(X_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        st.subheader("🔮 Heart Sound Prediction")
        st.success(f"Predicted: **{HEART_CLASSES[predicted_class] if predicted_class < len(HEART_CLASSES) else predicted_class}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        st.subheader("Class Probabilities")
        if len(prediction[0]) == len(HEART_CLASSES):
            st.bar_chart({HEART_CLASSES[i]: prediction[0][i] for i in range(len(HEART_CLASSES))})
        else:
            st.bar_chart({f"Class {i}": prediction[0][i] for i in range(len(prediction[0]))})

    # -------------------------
    # Lung Sound Classification
    # -------------------------
    elif sound_type == "Lung Sound":
        model = load_lung_model()

        # Extract MFCC features with augmentation
        y_aug = librosa.effects.time_stretch(y, rate=1.2)
        mfcc = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=52)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Reshape for GRU
        X = mfcc_mean.reshape(1, 1, 52)

        # Prediction
        preds = model.predict(X)
        preds = np.squeeze(preds)

        cls = np.argmax(preds)
        confidence = preds[cls]

        st.subheader("🔮 Lung Sound Prediction")
        st.success(f"Disease: **{LUNG_CLASSES[cls]}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

        st.subheader("Class Probabilities")
        st.bar_chart({LUNG_CLASSES[i]: preds[i] for i in range(len(LUNG_CLASSES))})
        