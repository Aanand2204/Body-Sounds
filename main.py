import streamlit as st
import librosa
import librosa.display
import os
import tempfile
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Import utilities
from heart_murmur_analysis.utils import pretty_print_analysis, export_json

# Import classification modules
from heart_murmur_analysis.classification import load_heart_model, load_lung_model, load_bowel_model, HeartSoundClassifier, LungSoundClassifier, BowelSoundClassifier

# Import signal processing modules
from heart_murmur_analysis.signal_processing import HeartbeatAnalyzer, LungSoundAnalyzer, BowelSoundAnalyzer

# Import report generators
from heart_murmur_analysis.report_generator.report_generator import (
    generate_heart_report, 
    generate_lung_report,
    generate_hospital_report
)

# Agent imports
try:
    from heart_murmur_analysis.agent.agent import build_heartbeat_agent, build_bowel_agent, build_lung_agent
except ImportError:
    # Agent modules may not exist yet
    build_heartbeat_agent = None
    build_bowel_agent = None
    build_lung_agent = None

# -----------------------------
# Environment & warnings
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# -----------------------------
# Constants
# -----------------------------
BOWEL_CLASS_MAP = {
    0: "Noise",
    1: "Bowel Sound"
}

HEART_CLASS_MAP = {
    0: "Artifact",
    1: "Murmur",
    2: "Normal"
}

BOWEL_REPORT_PATH = "reports/bowel_report.json"
HEART_REPORT_PATH = "reports/heartbeat_report.json"
LUNG_REPORT_PATH = "reports/lung_report.json"


@st.cache_data
def load_audio(uploaded_file, sr=None):
    """Cache audio loading to avoid redundant processing."""
    return librosa.load(uploaded_file, sr=sr)


@st.cache_data
def cached_bowel_analysis(_file_buffer, file_name):
    """Cached wrapper for bowel sound analysis."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(_file_buffer)
        path = tmp.name
    try:
        analyzer = BowelSoundAnalyzer(path)
        results = analyzer.analyze()
        return results
    finally:
        if os.path.exists(path):
            os.remove(path)

@st.cache_data
def cached_heart_analysis(_file_buffer, file_name):
    """Cached wrapper for heart sound analysis."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(_file_buffer)
        path = tmp.name
    try:
        analyzer = HeartbeatAnalyzer(path)
        results = analyzer.analyze()
        return results
    finally:
        if os.path.exists(path):
            os.remove(path)

@st.cache_data
def cached_lung_analysis(_file_buffer, file_name):
    """Cached wrapper for lung sound analysis."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(_file_buffer)
        path = tmp.name
    try:
        analyzer = LungSoundAnalyzer(path)
        results = analyzer.analyze()
        return results
    finally:
        if os.path.exists(path):
            os.remove(path)


# ===========================
# BOWEL SOUND FUNCTIONS
# ===========================

def run_bowel_classification(y, sr, results_dict):
    """Run deep learning classification for bowel sounds."""
    st.subheader("🔎 Bowel Sound Classification (Deep Learning)")
    
    model = load_bowel_model()
    classifier = BowelSoundClassifier(model)

    pred_class, prob = classifier.predict(y, sr)

    if pred_class is None:
        st.error("Audio too short or invalid.")
        return results_dict

    class_name = BOWEL_CLASS_MAP.get(pred_class, "Unknown")

    st.write(f"**Predicted Class:** {class_name}")
    st.write(f"**Confidence:** {prob:.2f}")

    # Add classification result
    results_dict["classification"] = class_name

    # Display waveform
    st.subheader("Waveform of Input Sound")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Bowel Sound Waveform")
    st.pyplot(fig)
    plt.close()

    return results_dict


def run_bowel_signal_processing(uploaded_file, results_dict):
    """Run signal processing analysis for bowel sounds."""
    st.subheader("📊 Bowel Signal Processing Analysis")

    # Use cached analysis
    sp_results = cached_bowel_analysis(uploaded_file.getbuffer(), uploaded_file.name)

    # Merge classification + signal processing results
    results_dict.update(sp_results)

    # Save results
    os.makedirs("reports", exist_ok=True)
    export_json(results_dict, BOWEL_REPORT_PATH)

    # Display results
    st.json({k: v for k, v in results_dict.items() if k != "_data"})

    # Show plots using cached results
    data = sp_results["_data"]
    from heart_murmur_analysis.signal_processing.visualizer import plot_results
    plot_results(
        time=data["time"],
        raw=data["raw"],
        filtered=data["filtered"],
        env=data["env"],
        fs=sp_results["fs"],
        peaks=data["events"],
        intervals_s=data["intervals_s"],
        sound_type='bowel'
    )


def run_bowel_agent_chat(patient_info: dict):
    """Run AI agent chat for bowel sound analysis."""
    st.subheader("💬 Chat with Bowel Sound Agent")
    
    if build_bowel_agent is None:
        st.info("ℹ️ Bowel sound agent is not available.")
        return
    
    if not os.path.exists(BOWEL_REPORT_PATH):
        st.warning("⚠️ Please run Signal Processing first to generate a report.")
        return

    # Initialize chat history
    if "bowel_chat_history" not in st.session_state:
        st.session_state.bowel_chat_history = []

    # Build agent if not exists or if patient_info/report changed
    current_agent_key = f"bowel_agent_{hash(json.dumps(patient_info))}"
    if st.session_state.get("bowel_agent_key") != current_agent_key:
        st.session_state.bowel_agent = build_bowel_agent(BOWEL_REPORT_PATH, patient_info=patient_info)
        st.session_state.bowel_agent_key = current_agent_key
        st.session_state.bowel_chat_history = [] # Reset history on context change

    # Chat input
    user_input = st.chat_input("Ask me about your bowel sound analysis...")
    if user_input:
        st.session_state.bowel_chat_history.append(("user", user_input))

        async def get_response():
            response = await st.session_state.bowel_agent.ainvoke({"messages": [("user", user_input)]})
            return response["messages"][-1].content if response["messages"] else "No response."

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        reply = loop.run_until_complete(get_response())
        st.session_state.bowel_chat_history.append(("bot", reply))

    # Display chat
    for role, msg in st.session_state.bowel_chat_history:
        with st.chat_message(role):
            st.write(msg)


# ===========================
# HEART SOUND FUNCTIONS
# ===========================

def run_heart_classification(y, sr, results_dict):
    """Run deep learning classification for heart sounds."""
    st.subheader("🔎 Heart Sound Classification (Deep Learning)")
    
    model = load_heart_model()
    classifier = HeartSoundClassifier(model)

    pred_class, scores = classifier.predict(y, sr)
    class_name = HEART_CLASS_MAP.get(pred_class, "Unknown")

    st.write(f"**Predicted Class:** {class_name} ({pred_class})")
    st.write("**Raw Scores:**", scores.tolist())

    # Add classification result
    results_dict["classification"] = class_name

    # Display waveform
    st.subheader("Waveform of Input Sound")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Heart Sound Waveform")
    st.pyplot(fig)
    plt.close()

    return results_dict


def run_heart_signal_processing(uploaded_file, results_dict):
    """Run signal processing analysis for heart sounds."""
    st.subheader("📊 Heart Signal Processing Analysis")

    # Use cached analysis
    sp_results = cached_heart_analysis(uploaded_file.getbuffer(), uploaded_file.name)

    # Merge classification + signal processing results
    results_dict.update(sp_results)

    # Save results
    os.makedirs("reports", exist_ok=True)
    export_json(results_dict, HEART_REPORT_PATH)

    # Display results
    st.json({k: v for k, v in results_dict.items() if k != "_data"})

    # Show plots
    data = sp_results["_data"]
    from heart_murmur_analysis.signal_processing.visualizer import plot_results
    plot_results(
        time=data["time"],
        raw=data["raw"],
        filtered=data["filtered"],
        env=data["env"],
        fs=sp_results["fs"],
        peaks=data["peaks"],
        intervals_s=data["intervals_s"]
    )


def run_heart_agent_chat(patient_info: dict):
    """Run AI agent chat for heart sound analysis."""
    st.subheader("💬 Chat with Heart Sound Agent")
    
    if build_heartbeat_agent is None:
        st.info("ℹ️ Heartbeat agent is not available.")
        return

    if not os.path.exists(HEART_REPORT_PATH):
        st.warning("⚠️ Please run Signal Processing first to generate a report.")
        return

    # Initialize chat history
    if "heart_chat_history" not in st.session_state:
        st.session_state.heart_chat_history = []

    # Build agent if not exists or if patient_info/report changed
    current_agent_key = f"heart_agent_{hash(json.dumps(patient_info))}"
    if st.session_state.get("heart_agent_key") != current_agent_key:
        st.session_state.heart_agent = build_heartbeat_agent(HEART_REPORT_PATH, patient_info=patient_info)
        st.session_state.heart_agent_key = current_agent_key
        st.session_state.heart_chat_history = [] # Reset history on context change

    # Chat input
    user_input = st.chat_input("Ask me about your heartbeat analysis...")
    if user_input:
        st.session_state.heart_chat_history.append(("user", user_input))

        async def get_response():
            response = await st.session_state.heart_agent.ainvoke({"messages": [("user", user_input)]})
            return response["messages"][-1].content if response["messages"] else "No response."

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        reply = loop.run_until_complete(get_response())
        st.session_state.heart_chat_history.append(("bot", reply))

    # Display chat
    for role, msg in st.session_state.heart_chat_history:
        with st.chat_message(role):
            st.write(msg)


# ===========================
# LUNG SOUND FUNCTIONS
# ===========================

def run_lung_classification(y, sr, results_dict):
    """Run deep learning classification for lung sounds."""
    st.subheader("🔎 Lung Sound Classification (Deep Learning)")

    model = load_lung_model()
    classifier = LungSoundClassifier(model)

    predicted_label, confidence, probs = classifier.predict(y, sr)

    st.write(f"**Predicted Disease:** {predicted_label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # Display waveform
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Lung Sound Waveform")
    st.pyplot(fig)
    plt.close()

    # Save classification result for report
    results_dict["classification"] = predicted_label
    results_dict["confidence"] = float(confidence)

    return results_dict


def run_lung_signal_processing(uploaded_file, results_dict):
    """Run signal processing analysis for lung sounds."""
    st.subheader("📊 Lung Signal Processing Analysis")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        wav_path = tmp.name

    # Analyze
    analyzer = LungSoundAnalyzer(wav_path)
    sp_results = analyzer.analyze()

    # Merge results (DO NOT overwrite classification)
    results_dict.update(sp_results)

    # Save JSON safely
    os.makedirs("reports", exist_ok=True)
    export_json(results_dict, LUNG_REPORT_PATH)

    # Show numeric results
    st.json({k: v for k, v in results_dict.items() if k != "_data"})

    # Plots
    analyzer.plot_all()

    # Cleanup
    os.remove(wav_path)


def run_lung_agent_chat(patient_info: dict):
    """Run AI agent chat for lung sound analysis."""
    st.subheader("💬 Chat with Lung Sound Agent")
    
    if build_lung_agent is None:
        st.info("ℹ️ Lung sound agent is not available.")
        return
    
    if not os.path.exists(LUNG_REPORT_PATH):
        st.warning("⚠️ Please run Signal Processing first to generate a report.")
        return

    # Initialize chat history
    if "lung_chat_history" not in st.session_state:
        st.session_state.lung_chat_history = []

    # Build agent if not exists or if patient_info/report changed
    current_agent_key = f"lung_agent_{hash(json.dumps(patient_info))}"
    if st.session_state.get("lung_agent_key") != current_agent_key:
        st.session_state.lung_agent = build_lung_agent(LUNG_REPORT_PATH, patient_info=patient_info)
        st.session_state.lung_agent_key = current_agent_key
        st.session_state.lung_chat_history = [] # Reset history on context change

    # Chat input
    user_input = st.chat_input("Ask me about your lung sound analysis...")
    if user_input:
        st.session_state.lung_chat_history.append(("user", user_input))

        async def get_response():
            response = await st.session_state.lung_agent.ainvoke({"messages": [("user", user_input)]})
            return response["messages"][-1].content if response["messages"] else "No response."

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        reply = loop.run_until_complete(get_response())
        st.session_state.lung_chat_history.append(("bot", reply))

    # Display chat
    for role, msg in st.session_state.lung_chat_history:
        with st.chat_message(role):
            st.write(msg)


# ===========================
# MAIN APPLICATION
# ===========================

def main():
    st.set_page_config(
        page_title="🫀 Body Sound Detection App", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar for sound type selection
    with st.sidebar:
        st.title("🎵 Sound Type Selection")
        sound_type = st.radio(
            "Select the type of sound to analyze:",
            options=["🩺 Bowel Sound", "❤️ Heart Sound", "🫁 Lung Sound"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Patient Information")
        patient_name = st.text_input("Name", value="Demo Patient")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        patient_info = {
            "name": patient_name,
            "age": patient_age,
            "gender": patient_gender
        }

    # Main content
    if sound_type == "🩺 Bowel Sound":
        st.title("🩺 Bowel Sound Analysis App")
        st.write(
            "Upload a bowel sound `.wav` file to analyze using Deep Learning, "
            "Signal Processing, and chat with the AI agent."
        )
        
        uploaded_file = st.file_uploader(
            "Upload bowel sound audio (.wav)", 
            type=["wav"],
            key="bowel_upload"
        )

        if uploaded_file is not None:
            y, sr = load_audio(uploaded_file, sr=44100)
            
            # Initialize or reset results_dict in session_state
            if "bowel_results_dict" not in st.session_state or st.session_state.get("bowel_file_name") != uploaded_file.name:
                st.session_state.bowel_results_dict = {}
                st.session_state.bowel_file_name = uploaded_file.name
            
            results_dict = st.session_state.bowel_results_dict

            tab1, tab2, tab3, tab4 = st.tabs([
                "Classification", 
                "Signal Processing", 
                "💬 Chat with Agent", 
                "Final Report"
            ])

            with tab1:
                results_dict = run_bowel_classification(y, sr, results_dict)
                st.session_state.bowel_results_dict = results_dict

            with tab2:
                run_bowel_signal_processing(uploaded_file, results_dict)
                st.session_state.bowel_results_dict = results_dict

            with tab3:
                run_bowel_agent_chat(patient_info)

            with tab4:
                st.header("📄 Bowel Sound Report")
                if os.path.exists(BOWEL_REPORT_PATH):
                    generate_hospital_report(BOWEL_REPORT_PATH, patient_info)
                else:
                    st.warning("⚠️ Please complete Signal Processing to generate the report.")

    elif sound_type == "❤️ Heart Sound":
        st.title("❤️ Heart Sound Analysis App")
        st.write(
            "Upload a heartbeat `.wav` file to analyze using Deep Learning, "
            "Signal Processing, and chat with the AI agent."
        )
        
        uploaded_file = st.file_uploader(
            "Upload heartbeat audio (.wav)", 
            type=["wav"],
            key="heart_upload"
        )

        if uploaded_file is not None:
            y, sr = load_audio(uploaded_file, sr=22050)
            
            # Initialize or reset results_dict in session_state
            if "heart_results_dict" not in st.session_state or st.session_state.get("heart_file_name") != uploaded_file.name:
                st.session_state.heart_results_dict = {}
                st.session_state.heart_file_name = uploaded_file.name
            
            results_dict = st.session_state.heart_results_dict

            tab1, tab2, tab3, tab4 = st.tabs([
                "Classification", 
                "Signal Processing", 
                "💬 Chat with Agent", 
                "Final Report"
            ])

            with tab1:
                results_dict = run_heart_classification(y, sr, results_dict)
                st.session_state.heart_results_dict = results_dict

            with tab2:
                run_heart_signal_processing(uploaded_file, results_dict)
                st.session_state.heart_results_dict = results_dict

            with tab3:
                run_heart_agent_chat(patient_info)

            with tab4:
                st.header("📄 Heart Sound Report")
                if os.path.exists(HEART_REPORT_PATH):
                    generate_heart_report(HEART_REPORT_PATH, patient_info)
                else:
                    st.warning("⚠️ Please complete Signal Processing to generate the report.")

    else:  # Lung Sound
        st.title("🫁 Lung Sound Analysis App")
        st.write(
            "Upload a lung sound `.wav` file to analyze respiratory conditions using "
            "Deep Learning and Signal Processing."
        )
        
        uploaded_file = st.file_uploader(
            "Upload lung sound (.wav)", 
            type=["wav"],
            key="lung_upload"
        )

        if uploaded_file is not None:
            y, sr = load_audio(uploaded_file, sr=22050)
            
            # Initialize or reset results_dict in session_state
            if "lung_results_dict" not in st.session_state or st.session_state.get("lung_file_name") != uploaded_file.name:
                st.session_state.lung_results_dict = {}
                st.session_state.lung_file_name = uploaded_file.name
            
            results_dict = st.session_state.lung_results_dict

            tab1, tab2, tab3, tab4 = st.tabs([
                "Classification", 
                "Signal Processing", 
                "💬 Chat with Agent", 
                "Final Report"
            ])

            with tab1:
                results_dict = run_lung_classification(y, sr, results_dict)
                st.session_state.lung_results_dict = results_dict

            with tab2:
                run_lung_signal_processing(uploaded_file, results_dict)
                st.session_state.lung_results_dict = results_dict

            with tab3:
                run_lung_agent_chat(patient_info)

            with tab4:
                st.header("📄 Lung Sound Report")
                if os.path.exists(LUNG_REPORT_PATH):
                    generate_lung_report(LUNG_REPORT_PATH, patient_info)
                else:
                    st.warning("⚠️ Please complete Signal Processing to generate the report.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()