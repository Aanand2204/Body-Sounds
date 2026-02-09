# 🩺 AI Body Sound Analyzer

An end-to-end medical diagnostics platform using **Deep Learning**, **Digital Signal Processing (DSP)**, and **Agentic AI** to analyze biological sounds (Heart, Lung, and Bowel).

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Key Features

### 1. ❤️ Heart Sound Analysis
- **Deep Learning**: Uses a trained **LSTM (Long Short-Term Memory)** model to classify heart sounds into *Normal*, *Murmur*, or *Artifact*.
- **Signal Processing**: Extracts S1/S2 heartbeats using energy envelope detection and 20-500Hz bandpass filtering.
- **Heart Rate Variability (HRV)**: Detailed timing analysis of heart rate patterns.

### 2. 🫁 Lung Sound Analysis
- **Classification**: Powered by a **GRU (Gated Recurrent Unit)** model to detect *Normal*, *Wheeze*, or *Crackle* sounds.
- **Breath Cycle Detection**: Sophisticated prominence-based detection of inspiration and expiration phases.

### 3. 🩺 Bowel Sound Analysis
- **Event Detection**: High-resolution detection of bowel sounds using **CRNN (Convolutional Recurrent Neural Networks)**.
- **Metric Extraction**: Analysis of event frequency and rhythmic patterns.

### 4. 💬 Specialized AI Agents
- Integrated with **LangGraph** and **Groq (Llama-3)** for rapid medical insight.
- **Context-Aware**: Agents receive patient information (age, gender, name) to provide personalized explanations.
- **RAG Powered**: Uses semantic search over analysis reports to answer technical questions about detected anomalies.

---

## 📦 Installation

Install the package directly from PyPI or locally in editable mode:

### From PyPI
```powershell
pip install body-sound-detection
```

### Locally (for development)
```powershell
pip install -e .
```

> [!IMPORTANT]
> Ensure you have **FFmpeg** installed on your system for audio processing (required by `librosa`).

---

## 🏃 Usage

### Programmatic Usage
You can use the package as a library in your own Python projects:
```python
from body_sound_detection import HeartbeatAnalyzer, load_model

# Run Signal Processing
analyzer = HeartbeatAnalyzer("heartbeat_sample.wav")
results = analyzer.analyze()
print(f"Detected Heart Rate: {results['heart_rate_bpm']} BPM")
```

---

## ⚙️ Configuration

The AI Agents require the following environment variables (or a `.env` file):
- `GROQ_API_KEY`: For LLM generation.
- `HF_TOKEN`: For downloading models from HuggingFace.

---

## 📂 Project Structure

```text
├── body_sound_detection/ 
│   ├── agent/             # LangGraph AI logic
│   ├── classification/    # Deep Learning (H5 Models)
│   ├── signal_processing/ # Audio DSP routines
│   ├── report_generator/  # PDF/JSON report logic
│   └── cli.py             # App entry point
├── main.py                # Main Streamlit dashboard
├── pyproject.toml         # Packaging metadata
└── README.md              # You are here!
```

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for more information.
