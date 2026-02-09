import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile
import json
import os
import scipy.io.wavfile as wav
import scipy.signal as signal
from typing import Dict, Any, Tuple

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
def load_heart_model(model_path="lstm_model.h5"):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'LSTM': CustomLSTM}
    )


@st.cache_resource
def load_lung_model(model_path="lung_model.h5"):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"GRU": CustomGRU}
    )


# -------------------------
# Constants
# -------------------------
LUNG_CLASSES = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]
HEART_CLASSES = ["Normal", "Murmur"]  # Adjust based on your model


# -------------------------
# Signal Processing Functions (Heart Sound)
# -------------------------
def load_wav(path: str, target_fs: int = None) -> Tuple[int, np.ndarray]:
    rate, data = wav.read(path)
    if data.dtype.kind == "i":
        maxv = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / maxv
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if target_fs is not None and target_fs != rate:
        n_samples = round(len(data) * float(target_fs) / rate)
        data = signal.resample(data, n_samples)
        rate = target_fs
    return rate, data


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(x: np.ndarray, fs: int, lowcut=20.0, highcut=500.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y


def envelope(x: np.ndarray, fs: int, win_ms: float = 50.0):
    win = int(fs * win_ms / 1000.0)
    win = max(1, win)
    sq = x ** 2
    kernel = np.ones(win) / win
    env = np.sqrt(np.convolve(sq, kernel, mode="same"))
    return env


def detect_peaks_from_envelope(env: np.ndarray, fs: int,
                               min_bpm: float = 30, max_bpm: float = 220,
                               prominence_factor: float = 0.6):
    min_interval_s = 60.0 / max_bpm
    min_distance = int(min_interval_s * fs * 0.8)
    height = np.percentile(env, 50) + (np.max(env) - np.percentile(env, 50)) * 0.3
    peaks, props = signal.find_peaks(
        env,
        distance=min_distance,
        height=height,
        prominence=prominence_factor * np.std(env)
    )
    return peaks, props


def compute_intervals(peaks: np.ndarray, fs: int) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs


def compute_hrv_metrics(intervals_s: np.ndarray) -> Dict[str, Any]:
    out = {}
    if intervals_s.size == 0:
        return out
    ibi_ms = intervals_s * 1000.0
    out["mean_IBI_ms"] = float(np.mean(ibi_ms))
    out["median_IBI_ms"] = float(np.median(ibi_ms))
    out["bpm_mean"] = float(60000.0 / np.mean(ibi_ms))
    out["SDNN_ms"] = float(np.std(ibi_ms, ddof=1))
    diff_ms = np.diff(ibi_ms)
    out["RMSSD_ms"] = float(np.sqrt(np.mean(diff_ms ** 2))) if diff_ms.size > 0 else None
    out["pNN50_pct"] = float(np.sum(np.abs(diff_ms) > 50.0) / diff_ms.size * 100.0) if diff_ms.size > 0 else None
    out["CV"] = float(out["SDNN_ms"] / out["mean_IBI_ms"]) if out["mean_IBI_ms"] > 0 else None
    if diff_ms.size > 0:
        sd1 = np.sqrt(0.5 * np.var(diff_ms, ddof=1))
        sd2 = np.sqrt(2 * np.var(ibi_ms, ddof=1) - 0.5 * np.var(diff_ms, ddof=1))
        out["SD1_ms"] = float(sd1)
        out["SD2_ms"] = float(sd2)
    return out


def estimate_snr(signal_data: np.ndarray, peaks: np.ndarray, fs: int, window_ms: int = 100) -> float:
    w = max(1, int(fs * window_ms / 1000.0))
    sig_powers = []
    mask = np.ones_like(signal_data, dtype=bool)
    for p in peaks:
        start = max(0, p - w)
        end = min(len(signal_data), p + w)
        mask[start:end] = False
        sig_powers.append(np.mean(signal_data[start:end] ** 2))
    if not sig_powers:
        return None
    noise_power = np.mean(signal_data[mask] ** 2) if np.any(mask) else 1e-12
    signal_power = np.mean(sig_powers)
    return float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


def energy_distribution(signal_data: np.ndarray, fs: int, cutoff_hz: float = 200.0):
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(4096, len(signal_data)))
    total_energy = np.trapz(Pxx, f)
    below_ix = f <= cutoff_hz
    energy_below = np.trapz(Pxx[below_ix], f[below_ix])
    pct_below = float(100.0 * energy_below / (total_energy + 1e-12))
    centroid = float(np.sum(f * Pxx) / (np.sum(Pxx) + 1e-12))
    return {"pct_energy_below_{}Hz".format(int(cutoff_hz)): pct_below,
            "spectral_centroid_hz": centroid}


def s1_s2_amplitude_ratio(env: np.ndarray, peaks: np.ndarray):
    if len(peaks) < 2:
        return None
    amps = env[peaks]
    s1 = amps[::2]
    s2 = amps[1::2]
    if np.mean(s2) == 0:
        return None
    return {"S1_mean": float(np.mean(s1)),
            "S2_mean": float(np.mean(s2)),
            "S1_to_S2_ratio": float(np.mean(s1) / np.mean(s2))}


def detect_extra_peaks_per_cycle(peaks: np.ndarray, fs: int, intervals_s: np.ndarray) -> Dict[str, Any]:
    res = {"cycles_checked": 0, "cycles_with_extra_peaks": 0}
    if len(peaks) < 2 or intervals_s.size == 0:
        return res
    mean_cycle = np.mean(intervals_s)
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end_time = (peaks[i] / fs) + mean_cycle
        end = int(round(end_time * fs))
        in_cycle = peaks[(peaks >= start) & (peaks <= end)]
        if len(in_cycle) > 2:
            res["cycles_with_extra_peaks"] += 1
        res["cycles_checked"] += 1
    return res


def irregular_spacing_stats(intervals_s: np.ndarray) -> Dict[str, Any]:
    out = {}
    if intervals_s.size == 0:
        return out
    out["intervals_mean_s"] = float(np.mean(intervals_s))
    out["intervals_std_s"] = float(np.std(intervals_s, ddof=1))
    out["intervals_cv"] = out["intervals_std_s"] / out["intervals_mean_s"]
    q1, q3 = np.percentile(intervals_s, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = intervals_s[(intervals_s < lower) | (intervals_s > upper)]
    out["n_outliers"] = int(len(outliers))
    return out


def frequency_band_energy(signal_data: np.ndarray, fs: int, band=(150, 500)):
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(4096, len(signal_data)))
    band_ix = (f >= band[0]) & (f <= band[1])
    band_energy = np.trapz(Pxx[band_ix], f[band_ix]) if np.any(band_ix) else 0.0
    total_energy = np.trapz(Pxx, f)
    return {"band_hz": band,
            "band_energy_pct": float(100.0 * band_energy / (total_energy + 1e-12))}


def plot_heart_results(time, raw, filtered, env, fs, peaks, intervals_s):
    figs = []

    fig1 = plt.figure(figsize=(14, 4))
    plt.plot(time, raw, label="raw", alpha=0.4)
    plt.plot(time, filtered, label="filtered", linewidth=0.8)
    if np.max(env) > 0:
        plt.plot(time, env / np.max(env) * 0.8 * np.max(filtered), label="envelope (scaled)", linewidth=1)
    plt.scatter(peaks / fs, env[peaks], c="red", marker="x", label="detected peaks")
    plt.xlabel("Time (s)")
    plt.title("Heart Sound Waveform with detected peaks")
    plt.legend()
    plt.tight_layout()
    figs.append(fig1)

    fig2 = plt.figure(figsize=(14, 4))
    f, t, Sxx = signal.spectrogram(filtered, fs=fs, nperseg=1024, noverlap=512)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    plt.ylim(0, 800)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("Spectrogram (dB)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    figs.append(fig2)

    if intervals_s.size > 0:
        fig3 = plt.figure(figsize=(8, 3))
        plt.hist(intervals_s * 1000.0, bins=20)
        plt.xlabel("IBI (ms)")
        plt.title("Inter-beat interval histogram")
        plt.tight_layout()
        figs.append(fig3)

    if intervals_s.size > 1:
        fig4 = plt.figure(figsize=(5, 5))
        x, y = intervals_s[:-1] * 1000.0, intervals_s[1:] * 1000.0
        plt.scatter(x, y, s=10)
        plt.xlabel("IBI_n (ms)")
        plt.ylabel("IBI_{n+1} (ms)")
        plt.title("Poincaré Plot")
        plt.tight_layout()
        figs.append(fig4)

    return figs


def analyze_heartbeat(wav_path: str, resample_fs: int = 2000) -> Dict[str, Any]:
    fs, raw = load_wav(wav_path, target_fs=resample_fs)
    raw = raw - np.mean(raw)
    filtered = bandpass_filter(raw, fs, lowcut=20.0, highcut=500.0, order=4)
    env = envelope(filtered, fs, win_ms=40.0)
    peaks, _ = detect_peaks_from_envelope(env, fs)
    intervals_s = compute_intervals(peaks, fs)

    hrv = compute_hrv_metrics(intervals_s)
    snr_db = estimate_snr(filtered, peaks, fs)
    energy = energy_distribution(filtered, fs)
    s1s2 = s1_s2_amplitude_ratio(env, peaks)
    extra_peaks = detect_extra_peaks_per_cycle(peaks, fs, intervals_s)
    irregular = irregular_spacing_stats(intervals_s)
    band = frequency_band_energy(filtered, fs)

    report = {
        "file": wav_path,
        "fs": fs,
        "duration_s": len(raw) / fs,
        "beats_detected": len(peaks),
        "bpm": hrv.get("bpm_mean"),
        "hrv": hrv,
        "SNR_dB": snr_db,
        "energy": energy,
        "S1S2": s1s2,
        "extra_peaks": extra_peaks,
        "irregular_spacing": irregular,
        "150_500Hz_band": band,
        "_data": {"time": np.arange(len(raw)) / fs,
                  "raw": raw,
                  "filtered": filtered,
                  "env": env,
                  "peaks": peaks,
                  "intervals_s": intervals_s}
    }
    return report


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Body Sound Analyzer (Experimental)", layout="wide")
st.title("🩺 Body Sound Analyzer – Classification + Signal Processing")

# Sidebar for sound type selection
sound_type = st.sidebar.selectbox(
    "Select Sound Type",
    ["Heart Sound", "Lung Sound"]
)

uploaded_file = st.file_uploader(f"Upload {sound_type.lower()} (.wav)", type=["wav", "mp3"])

tab1, tab2 = st.tabs(["Classification", "Signal Processing"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ============================================================
    # HEART SOUND PROCESSING
    # ============================================================
    if sound_type == "Heart Sound":
        # ---------------- CLASSIFICATION ----------------
        with tab1:
            st.header("🎵 Heart Sound Classification (LSTM)")
            try:
                model = load_heart_model()
                st.success("Heart model loaded.")
            except Exception as e:
                st.error(f"Could not load heart model: {e}")
                model = None

            if model:
                y, sr = librosa.load(tmp_path, sr=22050)
                
                st.subheader("Waveform")
                fig, ax = plt.subplots(figsize=(10, 2))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                st.pyplot(fig)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                mfcc_scaled = np.mean(mfcc.T, axis=0)
                X_input = np.expand_dims(mfcc_scaled, axis=0)
                X_input = np.expand_dims(X_input, axis=2)

                prediction = model.predict(X_input)
                predicted_class = int(np.argmax(prediction, axis=1)[0])
                confidence = prediction[0][predicted_class]

                st.subheader("🔮 Prediction Result")
                st.success(f"Predicted Class: **{HEART_CLASSES[predicted_class] if predicted_class < len(HEART_CLASSES) else predicted_class}**")
                st.info(f"Confidence: **{confidence*100:.2f}%**")

                if len(prediction[0]) == len(HEART_CLASSES):
                    st.bar_chart({HEART_CLASSES[i]: prediction[0][i] for i in range(len(HEART_CLASSES))})

        # ---------------- SIGNAL PROCESSING ----------------
        with tab2:
            st.header("🔬 Heart Sound Signal Processing")
            with st.spinner("Running signal processing..."):
                try:
                    report = analyze_heartbeat(tmp_path, resample_fs=2000)
                except Exception as e:
                    st.error(f"Signal processing failed: {e}")
                    report = None

            if report:
                st.markdown("### Analysis Report")
                st.write(f"**Duration (s):** {report['duration_s']:.2f}")
                st.write(f"**Beats detected:** {report['beats_detected']}")
                st.write(f"**Estimated BPM:** {report['bpm']:.2f}" if report['bpm'] else "Estimated BPM: N/A")

                st.markdown("**HRV metrics**")
                st.json(report["hrv"])

                st.write(f"**SNR (dB):** {report['SNR_dB']:.2f}" if report['SNR_dB'] else "SNR: N/A")
                st.json(report["energy"])

                if report["S1S2"]:
                    st.json(report["S1S2"])

                st.json(report["extra_peaks"])
                st.json(report["irregular_spacing"])
                st.json(report["150_500Hz_band"])

                dat = report["_data"]
                figs = plot_heart_results(dat["time"], dat["raw"], dat["filtered"], 
                                         dat["env"], report["fs"], dat["peaks"], dat["intervals_s"])
                for fig in figs:
                    st.pyplot(fig)

                rep_copy = dict(report)
                rep_copy.pop("_data", None)
                st.download_button(
                    "Download Report (JSON)",
                    data=json.dumps(rep_copy, indent=2),
                    file_name="heart_report.json",
                    mime="application/json"
                )

    # ============================================================
    # LUNG SOUND PROCESSING
    # ============================================================
    elif sound_type == "Lung Sound":
        # ---------------- CLASSIFICATION ----------------
        with tab1:
            st.header("🫁 Lung Sound Classification (GRU)")
            try:
                model = load_lung_model()
                st.success("Lung model loaded.")
            except Exception as e:
                st.error(f"Could not load lung model: {e}")
                model = None

            if model:
                y, sr = librosa.load(tmp_path, sr=22050)
                
                st.subheader("Waveform")
                fig, ax = plt.subplots(figsize=(10, 2))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                st.pyplot(fig)

                y = librosa.effects.time_stretch(y, rate=1.2)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=52)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                X = mfcc_mean.reshape(1, 1, 52)

                preds = np.squeeze(model.predict(X))
                cls = np.argmax(preds)
                confidence = preds[cls]

                st.subheader("🔮 Prediction Result")
                st.success(f"Predicted Disease: **{LUNG_CLASSES[cls]}**")
                st.info(f"Confidence: **{confidence*100:.2f}%**")
                st.bar_chart({LUNG_CLASSES[i]: preds[i] for i in range(len(LUNG_CLASSES))})

        # ---------------- SIGNAL PROCESSING ----------------
        with tab2:
            st.header("📊 Lung Sound Signal Processing")
            
            try:
                from signal_processing.analyzer import LungSoundAnalyzer
                analyzer = LungSoundAnalyzer(tmp_path)
                results = analyzer.analyze()

                st.json({k: v for k, v in results.items() if k != "_data"})
                analyzer.plot_all()

                st.download_button(
                    "Download Report (JSON)",
                    data=json.dumps({k: v for k, v in results.items() if k != "_data"}, indent=2),
                    file_name="lung_report.json",
                    mime="application/json"
                )
            except ImportError:
                st.warning("LungSoundAnalyzer not available. Skipping signal processing for lung sounds.")
            except Exception as e:
                st.error(f"Lung signal processing failed: {e}")

    os.remove(tmp_path)

else:
    st.info(f"Upload a {sound_type.lower()} file to begin analysis.")