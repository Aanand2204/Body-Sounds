import streamlit as st
import librosa
import librosa.display
import os
import tempfile
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt

from classification import load_heart_model, load_lung_model, HeartSoundClassifier, LungSoundClassifier
from signal_processing import HeartbeatAnalyzer, LungSoundAnalyzer
from utils import pretty_print_analysis, export_json

# ---------------------------
# Suppress TF / warnings
# ---------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ---------------------------
# CLASS LABELS
# ---------------------------
HEART_CLASS_MAP = {
    0: "Artifact",
    1: "Murmur",
    2: "Normal"
}

LUNG_CLASSES = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]

# ===========================
# HEART SOUND FUNCTIONS
# ===========================
def run_heart_classification(y, sr):
    """
    Run LSTM classification on heart sound.
    """
    st.subheader("🔎 Heart Sound Classification (Deep Learning)")
    model = load_heart_model()
    classifier = HeartSoundClassifier(model)

    pred_class, scores = classifier.predict(y, sr)
    class_name = HEART_CLASS_MAP.get(pred_class, "Unknown")

    st.write(f"**Predicted Class:** {class_name} ({pred_class})")
    st.write("**Raw Scores:**", scores.tolist())
    
    st.subheader("Waveform of Input Sound")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Heart Sound Waveform")
    st.pyplot(fig)


def run_heart_signal_processing(uploaded_file):
    """
    Run full signal processing analysis for heart sound.
    """
    st.subheader("📊 Heart Sound Signal Processing Analysis")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_wav_path = tmp.name

    analyzer = HeartbeatAnalyzer(temp_wav_path)
    results = analyzer.analyze()

    pretty_print_analysis(results)

    st.json({k: v for k, v in results.items() if k != "_data"})

    os.makedirs("reports", exist_ok=True)
    export_json(results, "reports/heartbeat_report.json")

    analyzer.plot_all()

    os.remove(temp_wav_path)

# ===========================
# LUNG SOUND FUNCTIONS
# ===========================
def run_lung_classification(y, sr):
    """
    Run GRU-based lung disease classification.
    """
    st.subheader("🔎 Lung Sound Classification (Deep Learning)")

    model = load_lung_model()
    classifier = LungSoundClassifier(model)

    label, confidence, scores = classifier.predict(y, sr)

    st.write(f"**Predicted Disease:** {label}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
    st.write("**Raw Scores:**", scores.tolist())

    st.subheader("Waveform of Input Sound")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Lung Sound Waveform")
    st.pyplot(fig)


def run_lung_signal_processing(uploaded_file):
    """
    Run lung signal processing analysis.
    """
    st.subheader("📊 Lung Sound Signal Processing Analysis")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_wav_path = tmp.name

    analyzer = LungSoundAnalyzer(temp_wav_path)
    results = analyzer.analyze()

    pretty_print_analysis(results)

    st.json({k: v for k, v in results.items() if k != "_data"})

    os.makedirs("reports", exist_ok=True)
    export_json(results, "reports/lung_report.json")

    analyzer.plot_all()

    os.remove(temp_wav_path)

# ===========================
# MAIN APP
# ===========================
def main():
    st.set_page_config(page_title="Body Sound Analysis App", layout="wide")

    st.title("🩺 Body Sound Analysis App")
    st.write("Analyze heart and lung sounds using Deep Learning and Signal Processing.")

    # Sidebar for sound type selection
    sound_type = st.sidebar.selectbox(
        "Select Sound Type",
        ["Heart Sound", "Lung Sound"]
    )

    st.header(f"{'❤️' if sound_type == 'Heart Sound' else '🫁'} {sound_type} Analysis")

    uploaded_file = st.file_uploader(
        f"Upload {sound_type.lower()} audio (.wav)", 
        type=["wav"],
        key=f"{sound_type}_uploader"
    )

    if uploaded_file is not None:
        # Load audio with librosa
        y, sr = librosa.load(uploaded_file, sr=22050)

        # Tabs: Classification vs Signal Processing
        tab1, tab2 = st.tabs(["Classification", "Signal Processing"])

        if sound_type == "Heart Sound":
            with tab1:
                run_heart_classification(y, sr)

            with tab2:
                run_heart_signal_processing(uploaded_file)

        elif sound_type == "Lung Sound":
            with tab1:
                run_lung_classification(y, sr)

            with tab2:
                run_lung_signal_processing(uploaded_file)

    else:
        st.info(f"Please upload a {sound_type.lower()} file to begin analysis.")


# ===========================
# ENTRY POINT
# ===========================
if __name__ == "__main__":
    main()