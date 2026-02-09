import numpy as np
import scipy.signal as signal
from typing import Dict, Any, Tuple


# ===========================
# BOWEL SOUND FEATURES
# ===========================

def detect_bowel_events(env: np.ndarray, fs: int,
                        min_interval_s: float = 0.1,
                        prominence_factor: float = 0.5):
    """
    Detect bowel sound events from envelope signal.
    
    Args:
        env: Envelope signal
        fs: Sampling frequency
        min_interval_s: Minimum time between events (seconds)
        prominence_factor: Peak prominence threshold factor
        
    Returns:
        Tuple of (peaks, properties)
    """
    min_distance = int(min_interval_s * fs)

    height = np.percentile(env, 60)

    peaks, props = signal.find_peaks(
        env,
        distance=min_distance,
        height=height,
        prominence=prominence_factor * np.std(env)
    )
    return peaks, props


def compute_event_intervals(events: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute inter-event intervals in seconds.
    
    Args:
        events: Event peak indices
        fs: Sampling frequency
        
    Returns:
        Array of intervals in seconds
    """
    if len(events) < 2:
        return np.array([])
    return np.diff(events) / fs


def compute_event_metrics(intervals_s: np.ndarray) -> Dict[str, Any]:
    """
    Compute bowel event metrics from intervals.
    
    Args:
        intervals_s: Inter-event intervals in seconds
        
    Returns:
        Dictionary with mean, std, and rate metrics
    """
    out = {}
    if intervals_s.size == 0:
        return out

    out["mean_interval_s"] = float(np.mean(intervals_s))
    out["std_interval_s"] = float(np.std(intervals_s))
    out["rate_per_min"] = float(60.0 / np.mean(intervals_s))

    return out


# ===========================
# HEART SOUND FEATURES
# ===========================

# ---------- Peak Detection ----------
def detect_peaks_from_envelope(env: np.ndarray, fs: int,
                            min_bpm: float = 30, max_bpm: float = 220,
                            prominence_factor: float = 0.6) -> Tuple[np.ndarray, Dict]:
    """
    Detect heartbeat peaks from envelope signal.
    
    Args:
        env: Envelope signal
        fs: Sampling frequency
        min_bpm: Minimum expected heart rate
        max_bpm: Maximum expected heart rate
        prominence_factor: Peak prominence threshold factor
        
    Returns:
        Tuple of (peaks, properties)
    """
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


# ---------- Interval / HRV ----------
def compute_intervals(peaks: np.ndarray, fs: int) -> np.ndarray:
    """Compute inter-beat intervals in seconds."""
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs


def compute_hrv_metrics(intervals_s: np.ndarray) -> Dict[str, Any]:
    """
    Compute Heart Rate Variability metrics.
    
    Args:
        intervals_s: Inter-beat intervals in seconds
        
    Returns:
        Dictionary containing HRV metrics (SDNN, RMSSD, pNN50, etc.)
    """
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


# ---------- Signal Quality (Heart) ----------
def estimate_snr_heart(signal_data: np.ndarray, peaks: np.ndarray, fs: int, window_ms: int = 100) -> float:
    """
    Estimate Signal-to-Noise Ratio for heart sounds.
    
    Args:
        signal_data: Filtered heart sound signal
        peaks: Detected peak indices
        fs: Sampling frequency
        window_ms: Window size around peaks in milliseconds
        
    Returns:
        SNR in dB
    """
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


def energy_distribution(signal_data: np.ndarray, fs: int, cutoff_hz: float = 200.0) -> Dict[str, float]:
    """
    Compute energy distribution below cutoff frequency.
    
    Args:
        signal_data: Heart sound signal
        fs: Sampling frequency
        cutoff_hz: Frequency cutoff
        
    Returns:
        Dictionary with percentage energy below cutoff and spectral centroid
    """
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(4096, len(signal_data)))
    total_energy = np.trapz(Pxx, f)
    below_ix = f <= cutoff_hz
    energy_below = np.trapz(Pxx[below_ix], f[below_ix])
    pct_below = float(100.0 * energy_below / (total_energy + 1e-12))
    centroid = float(np.sum(f * Pxx) / (np.sum(Pxx) + 1e-12))
    
    return {
        f"pct_energy_below_{int(cutoff_hz)}Hz": pct_below,
        "spectral_centroid_hz": centroid
    }


def s1_s2_amplitude_ratio(env: np.ndarray, peaks: np.ndarray) -> Dict[str, float]:
    """
    Compute S1/S2 valve sound amplitude ratio.
    
    Args:
        env: Envelope signal
        peaks: Detected peak indices
        
    Returns:
        Dictionary with S1 mean, S2 mean, and their ratio
    """
    if len(peaks) < 2:
        return None
    
    amps = env[peaks]
    s1 = amps[::2]
    s2 = amps[1::2]
    
    if np.mean(s2) == 0:
        return None
    
    return {
        "S1_mean": float(np.mean(s1)),
        "S2_mean": float(np.mean(s2)),
        "S1_to_S2_ratio": float(np.mean(s1) / np.mean(s2))
    }


# ---------- Abnormalities ----------
def detect_extra_peaks_per_cycle(peaks: np.ndarray, fs: int, intervals_s: np.ndarray) -> Dict[str, int]:
    """
    Detect extra peaks within cardiac cycles (potential murmurs/irregularities).
    
    Args:
        peaks: Detected peak indices
        fs: Sampling frequency
        intervals_s: Inter-beat intervals
        
    Returns:
        Dictionary with cycle counts
    """
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
    """
    Compute statistics on interval irregularity.
    
    Args:
        intervals_s: Inter-beat intervals in seconds
        
    Returns:
        Dictionary with mean, std, CV, and outlier count
    """
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


def frequency_band_energy(signal_data: np.ndarray, fs: int, band: Tuple[int, int] = (150, 500)) -> Dict[str, Any]:
    """
    Compute energy in specific frequency band.
    
    Args:
        signal_data: Heart sound signal
        fs: Sampling frequency
        band: Frequency band tuple (low, high)
        
    Returns:
        Dictionary with band info and energy percentage
    """
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(4096, len(signal_data)))
    band_ix = (f >= band[0]) & (f <= band[1])
    band_energy = np.trapz(Pxx[band_ix], f[band_ix]) if np.any(band_ix) else 0.0
    total_energy = np.trapz(Pxx, f)
    
    return {
        "band_hz": band,
        "band_energy_pct": float(100.0 * band_energy / (total_energy + 1e-12))
    }


# ===========================
# LUNG SOUND FEATURES
# ===========================

# ---------- Breath Detection ----------
def detect_breath_cycles(env: np.ndarray, fs: int) -> np.ndarray:
    """
    Detect breath cycle peaks from envelope signal.
    
    Args:
        env: Envelope signal
        fs: Sampling frequency
        
    Returns:
        Array of peak indices
    """
    peaks, _ = signal.find_peaks(
        env,
        distance=int(0.8 * fs),
        prominence=np.std(env)
    )
    return peaks


def compute_breathing_rate(peaks: np.ndarray, fs: int) -> float:
    """
    Compute breathing rate in breaths per minute.
    
    Args:
        peaks: Detected breath peak indices
        fs: Sampling frequency
        
    Returns:
        Breathing rate or None if insufficient peaks
    """
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / fs
    return 60.0 / np.mean(intervals)


# ---------- Spectral Features ----------
def band_energy(x: np.ndarray, fs: int, band: Tuple[int, int]) -> float:
    """
    Compute energy percentage in frequency band.
    
    Args:
        x: Signal
        fs: Sampling frequency
        band: Frequency band (low, high)
        
    Returns:
        Percentage of total energy in band
    """
    f, Pxx = signal.welch(x, fs, nperseg=2048)
    idx = (f >= band[0]) & (f <= band[1])
    total = np.trapz(Pxx, f)
    band_e = np.trapz(Pxx[idx], f[idx])
    return 100 * band_e / (total + 1e-12)


def spectral_centroid(x: np.ndarray, fs: int) -> float:
    """
    Compute spectral centroid.
    
    Args:
        x: Signal
        fs: Sampling frequency
        
    Returns:
        Spectral centroid in Hz
    """
    f, Pxx = signal.welch(x, fs)
    return float(np.sum(f * Pxx) / (np.sum(Pxx) + 1e-12))


# ---------- Adventitious Sounds ----------
def wheeze_index(x: np.ndarray, fs: int) -> float:
    """
    Compute wheeze index (mid-frequency energy).
    
    Args:
        x: Lung sound signal
        fs: Sampling frequency
        
    Returns:
        Wheeze index percentage
    """
    return band_energy(x, fs, (400, 1600))


def crackle_index(x: np.ndarray, fs: int) -> float:
    """
    Compute crackle index (low-frequency energy).
    
    Args:
        x: Lung sound signal
        fs: Sampling frequency
        
    Returns:
        Crackle index percentage
    """
    return band_energy(x, fs, (100, 400))


# ---------- Signal Quality (Lung) ----------
def estimate_snr_lung(x: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio for lung sounds.
    
    Args:
        x: Lung sound signal
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(x**2)
    noise_power = np.var(x - signal.medfilt(x, 11))
    return 10 * np.log10(signal_power / (noise_power + 1e-12))


# ===========================
# UNIFIED SNR FUNCTION
# ===========================
def estimate_snr(signal_data: np.ndarray, peaks: np.ndarray = None, fs: int = None, 
                 window_ms: int = 100) -> float:
    """
    Unified SNR estimation function that handles both heart and lung sounds.
    
    Args:
        signal_data: Audio signal
        peaks: Peak indices (optional, for heart sound analysis)
        fs: Sampling frequency (optional, for heart sound analysis)
        window_ms: Window size in ms (for heart sound analysis)
        
    Returns:
        SNR in dB
    """
    if peaks is not None and fs is not None:
        # Heart sound SNR
        return estimate_snr_heart(signal_data, peaks, fs, window_ms)
    else:
        # Lung sound SNR
        return estimate_snr_lung(signal_data)