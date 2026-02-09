import os

# -----------------------------
# Hugging Face Model Config
# -----------------------------
# Heart model
HF_HEART_REPO_ID = "vaidehibh/heart_model"  # Update with actual repo
HF_HEART_MODEL_FILENAME = "lstm_model.h5"

# Lung model
HF_LUNG_REPO_ID = "vaidehibh/lung_model"
HF_LUNG_MODEL_FILENAME = "lung_model.h5"

# Bowel model
HF_BOWEL_REPO_ID = "vaidehibh/crnn"
HF_BOWEL_MODEL_FILENAME = "model.weights.h5"

# -----------------------------
# Audio Config
# -----------------------------
SAMPLE_RATE = 22050
N_MFCC = 52

# -----------------------------
# Lung Disease Classes
# -----------------------------
CLASSES = [
    "COPD",
    "Bronchiolitis",
    "Pneumonia",
    "URTI",
    "Healthy"
]

# -----------------------------
# TensorFlow Optimization
# -----------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
