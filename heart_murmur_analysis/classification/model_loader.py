import tensorflow as tf
from tensorflow.keras.layers import LSTM as OriginalLSTM
from tensorflow.keras.layers import GRU as OriginalGRU
from tensorflow.keras import layers, models
from huggingface_hub import hf_hub_download
from heart_murmur_analysis.config import (
    HF_HEART_REPO_ID, HF_HEART_MODEL_FILENAME,
    HF_LUNG_REPO_ID, HF_LUNG_MODEL_FILENAME,
    HF_BOWEL_REPO_ID, HF_BOWEL_MODEL_FILENAME
)
import streamlit as st


# Custom LSTM to handle 'time_major' argument in saved model
class CustomLSTM(OriginalLSTM):
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs:
            del kwargs['time_major']
        super().__init__(*args, **kwargs)


# Custom GRU to handle 'time_major' argument
class CustomGRU(OriginalGRU):
    def __init__(self, *args, **kwargs):
        if "time_major" in kwargs:
            del kwargs["time_major"]
        super().__init__(*args, **kwargs)


@st.cache_resource
def load_heart_model(use_huggingface: bool = True, local_path: str = "models/lstm_model.h5"):
    """
    Loads the trained LSTM model for heart sound classification.
    
    Args:
        use_huggingface: If True, download from Hugging Face Hub
        local_path: Local path to model if use_huggingface is False
    """
    try:
        if use_huggingface:
            st.info("📥 Downloading heart model from Hugging Face...")
            model_path = hf_hub_download(
                repo_id="vaidehibh/heart_model",
                filename="lstm_model.h5",
                repo_type="model"
            )
        else:
            model_path = local_path
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'LSTM': CustomLSTM}
        )
        st.success("✅ Heart model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load heart model: {str(e)}")
        raise e


@st.cache_resource
def load_lung_model():
    """
    Load lung GRU model from Hugging Face with custom GRU fix.
    """
    try:
        st.info("📥 Downloading lung model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="vaidehibh/lung_model",
            filename="lung_model.h5",
            repo_type="model"
        )

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"GRU": CustomGRU}
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        st.success("✅ Lung model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load lung model: {str(e)}")
        raise e


def build_crnn_model(seq_len=9, n_freq_bins=16):
    """
    Build CRNN model architecture for bowel sound classification.
    
    Architecture:
    - Conv2D (30 filters, 3x3) + MaxPool
    - Conv2D (60 filters, 4x2) + MaxPool
    - Bidirectional GRU (80 units)
    - Dense (1 unit, sigmoid)
    
    Args:
        seq_len: Sequence length (default 9 frames)
        n_freq_bins: Number of frequency bins (default 16)
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(seq_len, n_freq_bins, 1), name='input')
    
    # Convolutional block 1
    x = layers.Conv2D(30, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(1, 2), name='pool1')(x)
    
    # Convolutional block 2
    x = layers.Conv2D(60, (4, 2), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D(pool_size=(1, 2), name='pool2')(x)
    
    # Reshape: collapse frequency and channel dimensions, keep time
    x = layers.Reshape((seq_len, -1), name='reshape')(x)
    
    # Dropout
    x = layers.Dropout(0.4, name='dropout1')(x)
    
    # Bidirectional GRU
    x = layers.Bidirectional(
        layers.GRU(80, activation='relu', return_sequences=False),
        name='bi_gru'
    )(x)
    
    # Dropout
    x = layers.Dropout(0.4, name='dropout2')(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = models.Model(inputs, outputs, name='CRNN')
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


@st.cache_resource
def load_bowel_model():
    """
    Load trained bowel sound CRNN model from Hugging Face.
    Downloads weights file and loads into model architecture.
    """
    try:
        st.info("📥 Downloading bowel model from Hugging Face...")
        
        weights_path = hf_hub_download(
            repo_id="vaidehibh/crnn",
            filename="model.weights.h5",
            repo_type="model"
        )

        # Build model with architecture
        # These parameters match the training configuration
        model = build_crnn_model(seq_len=9, n_freq_bins=16)

        # Load only the weights into the model
        model.load_weights(weights_path)

        st.success("✅ Bowel model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load bowel model: {str(e)}")
        raise e