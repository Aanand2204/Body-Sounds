import numpy as np
import librosa
from body_sound_detection.config import SAMPLE_RATE, N_MFCC, CLASSES


class HeartSoundClassifier:
    def __init__(self, model):
        """
        Initialize classifier with a pre-trained model.
        """
        self.model = model

    def extract_features(self, y, sr, n_mfcc=40):
        """
        Extract MFCC features from the audio.
        """
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled

    def prepare_input(self, mfcc_scaled):
        """
        Reshape features for LSTM input.
        Expected shape: (samples, timesteps, features)
        """
        X_input = np.expand_dims(mfcc_scaled, axis=0)  # Add batch dimension
        X_input = np.expand_dims(X_input, axis=2)      # Add feature dimension
        return X_input

    def predict(self, y, sr):
        """
        Full pipeline: extract features, prepare input, and predict class.
        Returns predicted_class, raw_scores
        """
        features = self.extract_features(y, sr)
        X_input = self.prepare_input(features)
        prediction = self.model.predict(X_input)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        return predicted_class, prediction


class LungSoundClassifier:
    def __init__(self, model):
        """
        Initialize lung sound classifier with pre-trained GRU model.
        """
        self.model = model

    def extract_features(self, y, sr):
        """
        Extract MFCC features from lung sound.
        Uses mean pooling across time.
        """
        # Optional augmentation (same as training)
        y = librosa.effects.time_stretch(y, rate=1.2)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled

    def prepare_input(self, mfcc_scaled):
        """
        Reshape features for GRU input.
        Expected shape: (batch, timesteps, features)
        """
        X_input = np.expand_dims(mfcc_scaled, axis=0)  # (1, 52)
        X_input = np.expand_dims(X_input, axis=1)      # (1, 1, 52)
        return X_input

    def predict(self, y, sr):
        """
        Full pipeline:
        - Extract MFCC
        - Prepare GRU input
        - Predict lung disease

        Returns:
        predicted_label, confidence, raw_scores
        """
        features = self.extract_features(y, sr)
        X_input = self.prepare_input(features)

        preds = self.model.predict(X_input)
        preds = np.squeeze(preds)  # (5,)

        class_index = int(np.argmax(preds))
        confidence = float(preds[class_index])
        label = CLASSES[class_index]

        return label, confidence, preds


class BowelSoundClassifier:
    def __init__(self, model, sr=44100):
        """
        Initialize classifier with a pre-trained model.
        """
        self.model = model
        self.sr = sr

    def extract_features(self, y, sr):
        """
        Extract log-mel spectrogram features.
        """
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=16,  # Changed from 64 to 16 to match model input shape
            n_fft=1024,
            hop_length=256
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Transpose to (time, features)
        return log_mel.T

    def build_sequences(self, X_frames, seq_len=9):
        """
        Build sequences for CRNN input.
        """
        if len(X_frames) < seq_len:
            return None

        sequences = []
        for i in range(len(X_frames) - seq_len + 1):
            seq = X_frames[i:i + seq_len]
            sequences.append(seq)

        X_seq = np.array(sequences)

        # Add channel dimension
        X_seq = X_seq[..., np.newaxis]
        return X_seq

    def predict(self, y, sr):
        """
        Full pipeline: feature extraction → sequence building → prediction.
        Returns predicted_class, probability
        """
        X_frames = self.extract_features(y, sr)
        X_seq = self.build_sequences(X_frames)

        if X_seq is None:
            return None, None

        preds = self.model.predict(X_seq)

        # Average predictions
        prob = float(np.mean(preds))

        predicted_class = 1 if prob > 0.5 else 0

        return predicted_class, prob