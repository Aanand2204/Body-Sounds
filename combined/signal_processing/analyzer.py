import numpy as np
import librosa
from typing import Dict, Any

# Heart sound imports
from .loader import load_wav
from .preprocessing import bandpass_filter, envelope
from .features import (
    # Heart features
    detect_peaks_from_envelope,
    compute_intervals,
    compute_hrv_metrics,
    estimate_snr,
    energy_distribution,
    s1_s2_amplitude_ratio,
    detect_extra_peaks_per_cycle,
    irregular_spacing_stats,
    frequency_band_energy,
    # Lung features
    detect_breath_cycles,
    compute_breathing_rate,
    band_energy,
    spectral_centroid,
    wheeze_index,
    crackle_index,
    # Bowel features
    detect_bowel_events,
    compute_event_intervals,
    compute_event_metrics,
)
from .visualizer import plot_results


# ===========================
# BOWEL SOUND ANALYZER
# ===========================
class BowelSoundAnalyzer:
    """
    Analyzer for bowel sound signals with event detection and spectral analysis.
    """
    def __init__(self, wav_path: str, resample_fs: int = 4000):
        self.wav_path = wav_path
        self.resample_fs = resample_fs
        self._last_results = None

    def analyze(self) -> Dict[str, Any]:
        """
        Run full bowel sound analysis pipeline.
        
        Returns:
            dict: Analysis results including event metrics, SNR, spectral data.
        """
        # Load and preprocess
        fs, raw = load_wav(self.wav_path, target_fs=self.resample_fs)
        raw = raw - np.mean(raw)  # remove DC offset

        # Bowel sounds mostly 100–1000 Hz
        filtered = bandpass_filter(raw, fs, lowcut=100.0, highcut=1000.0, order=4)

        # Envelope extraction
        env = envelope(filtered, fs, win_ms=50.0)

        # Event detection
        events, _ = detect_bowel_events(env, fs)
        intervals_s = compute_event_intervals(events, fs)

        # Compute features
        metrics = compute_event_metrics(intervals_s)
        snr_db = estimate_snr(filtered, events, fs)
        energy = energy_distribution(filtered, fs, cutoff_hz=300.0)
        band = frequency_band_energy(filtered, fs, band=(150, 1000))

        results = {
            "file": self.wav_path,
            "fs": fs,
            "duration_s": len(raw) / fs,
            "events_detected": len(events),
            "event_rate_per_min": metrics.get("rate_per_min"),
            "event_metrics": metrics,
            "SNR_dB": snr_db,
            "energy": energy,
            "150_1000Hz_band": band,
            "_data": {
                "time": np.arange(len(raw)) / fs,
                "raw": raw,
                "filtered": filtered,
                "env": env,
                "events": events,
                "intervals_s": intervals_s
            }
        }

        self._last_results = results
        return results

    def plot_all(self):
        """
        Generate all bowel sound visualization plots.
        """
        if self._last_results is None:
            raise RuntimeError("Run analyze() before calling plot_all().")

        data = self._last_results["_data"]
        plot_results(
            time=data["time"],
            raw=data["raw"],
            filtered=data["filtered"],
            env=data["env"],
            fs=self._last_results["fs"],
            peaks=data["events"],
            intervals_s=data["intervals_s"],
            sound_type='bowel'
        )


# ===========================
# HEART SOUND ANALYZER
# ===========================
class HeartbeatAnalyzer:
    """
    Analyzer for heart sound signals with HRV, valve sound, and spectral analysis.
    """
    def __init__(self, wav_path: str, resample_fs: int = 2000):
        self.wav_path = wav_path
        self.resample_fs = resample_fs
        self._last_results = None

    def analyze(self) -> Dict[str, Any]:
        """
        Run full heart sound analysis pipeline.
        
        Returns:
            dict: Analysis results including HRV metrics, SNR, spectral data, etc.
        """
        # Load and preprocess
        fs, raw = load_wav(self.wav_path, target_fs=self.resample_fs)
        raw = raw - np.mean(raw)  # remove DC offset
        
        # Filtering and envelope
        filtered = bandpass_filter(raw, fs, lowcut=20.0, highcut=500.0, order=4)
        env = envelope(filtered, fs, win_ms=40.0)
        
        # Peak detection
        peaks, _ = detect_peaks_from_envelope(env, fs)
        intervals_s = compute_intervals(peaks, fs)

        # Compute features
        hrv = compute_hrv_metrics(intervals_s)
        snr_db = estimate_snr(filtered, peaks, fs)
        energy = energy_distribution(filtered, fs)
        s1s2 = s1_s2_amplitude_ratio(env, peaks)
        extra_peaks = detect_extra_peaks_per_cycle(peaks, fs, intervals_s)
        irregular = irregular_spacing_stats(intervals_s)
        band = frequency_band_energy(filtered, fs)

        results = {
            "file": self.wav_path,
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
            "_data": {
                "time": np.arange(len(raw)) / fs,
                "raw": raw,
                "filtered": filtered,
                "env": env,
                "peaks": peaks,
                "intervals_s": intervals_s
            }
        }

        self._last_results = results
        return results

    def plot_all(self):
        """
        Generate all heart sound visualization plots.
        """
        if self._last_results is None:
            raise RuntimeError("Run analyze() before calling plot_all().")

        data = self._last_results["_data"]
        plot_results(
            time=data["time"],
            raw=data["raw"],
            filtered=data["filtered"],
            env=data["env"],
            fs=self._last_results["fs"],
            peaks=data["peaks"],
            intervals_s=data["intervals_s"]
        )


# ===========================
# LUNG SOUND ANALYZER
# ===========================
class LungSoundAnalyzer:
    """
    Analyzer for lung sound signals with breathing rate, spectral, and adventitious sound detection.
    """
    def __init__(self, wav_path: str, target_fs: int = 4000):
        self.wav_path = wav_path
        self.target_fs = target_fs
        self._last_results = None

    def analyze(self) -> Dict[str, Any]:
        """
        Run full lung sound analysis pipeline.
        
        Returns:
            dict: Analysis results including breathing rate, spectral data, wheeze/crackle indices.
        """
        # Load audio
        y, fs = librosa.load(self.wav_path, sr=self.target_fs)
        y = y - np.mean(y)  # remove DC offset

        # Signal processing
        filtered = bandpass_filter(y, fs)
        env = envelope(filtered, fs)

        # Breath detection
        peaks = detect_breath_cycles(env, fs)
        breathing_rate = compute_breathing_rate(peaks, fs)

        # Compute metrics
        results = {
            "file": self.wav_path,
            "fs": fs,
            "duration_s": len(y) / fs,
            "breaths_detected": int(len(peaks)),
            "breathing_rate": breathing_rate,
            "SNR_dB": estimate_snr(filtered),
            "spectral": {
                "low_band_energy_%": band_energy(filtered, fs, (50, 400)),
                "mid_band_energy_%": band_energy(filtered, fs, (400, 1600)),
                "high_band_energy_%": band_energy(filtered, fs, (1600, 2500)),
                "spectral_centroid_hz": spectral_centroid(filtered, fs),
            },
            "adventitious": {
                "wheeze_index_%": wheeze_index(filtered, fs),
                "crackle_index_%": crackle_index(filtered, fs),
            },
            "_data": {
                "raw": y,
                "filtered": filtered,
                "envelope": env,
                "peaks": peaks,
            },
        }

        self._last_results = results
        return results

    def plot_all(self):
        """
        Generate all lung sound visualization plots.
        """
        if self._last_results is None:
            raise RuntimeError("Run analyze() before plot_all().")

        data = self._last_results["_data"]
        plot_results(data=data, fs=self._last_results["fs"], sound_type='lung')