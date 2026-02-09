import numpy as np
import librosa
import scipy.io.wavfile as wav
import scipy.signal as signal
from typing import Tuple, Optional


def load_wav(path: str, target_fs: Optional[int] = None, method: str = 'auto') -> Tuple[int, np.ndarray]:
    """
    Load a WAV file, convert to mono float32, and optionally resample.
    
    This function supports two loading methods:
    - 'librosa': Uses librosa (better for audio ML, automatic resampling)
    - 'scipy': Uses scipy (lower-level, more control)
    - 'auto': Chooses librosa if target_fs is specified, else scipy
    
    Args:
        path: Path to WAV file
        target_fs: Target sampling frequency (Hz). If None, uses original sample rate.
        method: Loading method ('librosa', 'scipy', or 'auto')
        
    Returns:
        Tuple of (sample_rate, audio_data) where audio_data is float32 in [-1, 1]
        
    Examples:
        >>> fs, audio = load_wav('heart.wav', target_fs=2000)
        >>> fs, audio = load_wav('lung.wav', target_fs=4000, method='librosa')
    """
    if method == 'auto':
        # Use librosa if target_fs is specified (better resampling)
        method = 'librosa' if target_fs is not None else 'scipy'
    
    if method == 'librosa':
        return _load_wav_librosa(path, target_fs)
    elif method == 'scipy':
        return _load_wav_scipy(path, target_fs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'librosa', 'scipy', or 'auto'.")


def _load_wav_librosa(path: str, target_fs: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """
    Load WAV using librosa (high-quality resampling, automatic preprocessing).
    
    Args:
        path: Path to WAV file
        target_fs: Target sampling frequency
        
    Returns:
        Tuple of (sample_rate, audio_data)
    """
    if target_fs is None:
        # Load at native sample rate
        y, sr = librosa.load(path, sr=None, mono=True)
    else:
        # Load and resample
        y, sr = librosa.load(path, sr=target_fs, mono=True)
    
    y = y.astype(np.float32)
    return sr, y


def _load_wav_scipy(path: str, target_fs: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """
    Load WAV using scipy (lower-level, more control over data types).
    
    Args:
        path: Path to WAV file
        target_fs: Target sampling frequency for resampling
        
    Returns:
        Tuple of (sample_rate, audio_data)
    """
    rate, data = wav.read(path)

    # Convert to float32 in [-1, 1]
    if data.dtype.kind == "i":  # Integer type
        maxv = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / maxv
    else:
        data = data.astype(np.float32)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if target_fs is not None and target_fs != rate:
        n_samples = round(len(data) * float(target_fs) / rate)
        data = signal.resample(data, n_samples)
        rate = target_fs

    return rate, data


def load_wav_librosa(path: str, target_fs: int = 22050) -> Tuple[int, np.ndarray]:
    """
    Legacy function for librosa-based loading.
    Kept for backward compatibility.
    
    Args:
        path: Path to WAV file
        target_fs: Target sampling frequency (default: 22050 Hz)
        
    Returns:
        Tuple of (sample_rate, audio_data)
    """
    return _load_wav_librosa(path, target_fs)


def load_wav_scipy(path: str, target_fs: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """
    Legacy function for scipy-based loading.
    Kept for backward compatibility.
    
    Args:
        path: Path to WAV file
        target_fs: Target sampling frequency (optional)
        
    Returns:
        Tuple of (sample_rate, audio_data)
    """
    return _load_wav_scipy(path, target_fs)