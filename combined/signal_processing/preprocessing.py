import numpy as np
import scipy.signal as signal
from typing import Dict, Tuple


# ===========================
# BANDPASS FILTERING
# ===========================

def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(x: np.ndarray, fs: int, lowcut: float = 20.0, highcut: float = 500.0, order: int = 4) -> np.ndarray:
    """
    Apply a safe bandpass filter to audio signal.
    
    This function automatically adjusts cutoff frequencies to prevent
    invalid digital filter errors and works for both heart and lung sounds.
    
    Args:
        x: Input signal
        fs: Sampling frequency (Hz)
        lowcut: Low cutoff frequency (Hz). Default 20 Hz for heart, use 50+ for lung
        highcut: High cutoff frequency (Hz). Default 500 Hz for heart, use 2500+ for lung
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal
        
    Examples:
        >>> # Heart sound filtering
        >>> filtered_heart = bandpass_filter(signal, fs=2000, lowcut=20, highcut=500)
        
        >>> # Lung sound filtering
        >>> filtered_lung = bandpass_filter(signal, fs=4000, lowcut=50, highcut=2500)
    """
    nyq = 0.5 * fs

    # Safety checks to prevent invalid filter parameters
    lowcut = max(lowcut, 1.0)
    highcut = min(highcut, nyq - 1.0)

    low = lowcut / nyq
    high = highcut / nyq

    # Return original signal if filter parameters are invalid
    if low <= 0 or high >= 1 or low >= high:
        return x

    b, a = signal.butter(order, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, x)


# ===========================
# ENVELOPE EXTRACTION
# ===========================

def envelope(x: np.ndarray, fs: int, win_ms: float = 50.0) -> np.ndarray:
    """
    Compute smoothed RMS envelope of signal.
    
    Used for:
    - Heart sound: Detecting S1/S2 peaks
    - Lung sound: Detecting breathing cycles and energy analysis
    
    Args:
        x: Input signal
        fs: Sampling frequency (Hz)
        win_ms: Window size in milliseconds (default: 50ms)
        
    Returns:
        Envelope signal
        
    Examples:
        >>> # For heart sounds (shorter window)
        >>> env_heart = envelope(filtered, fs=2000, win_ms=40)
        
        >>> # For lung sounds (longer window)
        >>> env_lung = envelope(filtered, fs=4000, win_ms=50)
    """
    win = int(fs * win_ms / 1000.0)
    win = max(1, win)

    squared = x ** 2
    kernel = np.ones(win) / win

    env = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return env


# ===========================
# ENERGY METRICS
# ===========================

def signal_energy(x: np.ndarray) -> float:
    """
    Compute total signal energy.
    
    Args:
        x: Input signal
        
    Returns:
        Total energy (sum of squared samples)
    """
    return float(np.sum(x ** 2))


def zero_crossing_rate(x: np.ndarray) -> float:
    """
    Compute zero-crossing rate.
    
    Useful for:
    - Wheeze detection in lung sounds (higher ZCR)
    - Voice/noise discrimination
    
    Args:
        x: Input signal
        
    Returns:
        Zero-crossing rate (normalized)
    """
    return float(np.mean(np.abs(np.diff(np.sign(x)))))


# ===========================
# SPECTRAL FEATURES
# ===========================

def spectral_features(x: np.ndarray, fs: int) -> Dict[str, float]:
    """
    Compute spectral centroid and bandwidth.
    
    Args:
        x: Input signal
        fs: Sampling frequency (Hz)
        
    Returns:
        Dictionary containing:
        - spectral_centroid_hz: Center of mass of spectrum
        - spectral_bandwidth_hz: Spread of spectrum around centroid
    """
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(2048, len(x)))

    centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    bandwidth = np.sqrt(
        np.sum(((freqs - centroid) ** 2) * psd) / (np.sum(psd) + 1e-10)
    )

    return {
        "spectral_centroid_hz": float(centroid),
        "spectral_bandwidth_hz": float(bandwidth)
    }


# ===========================
# UTILITY FUNCTIONS
# ===========================

def normalize_signal(x: np.ndarray) -> np.ndarray:
    """
    Normalize signal to [-1, 1] range.
    
    Args:
        x: Input signal
        
    Returns:
        Normalized signal
    """
    max_val = np.max(np.abs(x))
    if max_val > 0:
        return x / max_val
    return x


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    """
    Remove DC offset (mean) from signal.
    
    Args:
        x: Input signal
        
    Returns:
        Zero-mean signal
    """
    return x - np.mean(x)


def preprocess_audio(x: np.ndarray, fs: int, sound_type: str = 'heart') -> np.ndarray:
    """
    Complete preprocessing pipeline for bowel, heart, or lung sounds.
    
    Args:
        x: Input signal
        fs: Sampling frequency
        sound_type: Either 'bowel', 'heart', or 'lung'
        
    Returns:
        Preprocessed signal
    """
    # Remove DC offset
    x = remove_dc_offset(x)
    
    # Apply appropriate bandpass filter
    if sound_type == 'bowel':
        x = bandpass_filter(x, fs, lowcut=100.0, highcut=1000.0, order=4)
    elif sound_type == 'heart':
        x = bandpass_filter(x, fs, lowcut=20.0, highcut=500.0, order=4)
    elif sound_type == 'lung':
        x = bandpass_filter(x, fs, lowcut=50.0, highcut=2500.0, order=4)
    else:
        raise ValueError(f"Unknown sound_type: {sound_type}. Use 'bowel', 'heart', or 'lung'.")
    
    return x