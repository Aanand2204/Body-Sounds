import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import streamlit as st
from typing import Dict, Optional


# ===========================
# BOWEL SOUND VISUALIZATION
# ===========================

def plot_bowel_results(time: np.ndarray, raw: np.ndarray, filtered: np.ndarray,
                       env: np.ndarray, fs: int, peaks: np.ndarray, 
                       intervals_s: np.ndarray):
    """
    Generate comprehensive plots for bowel sound analysis (Streamlit-friendly).
    
    Args:
        time: Time array
        raw: Raw signal
        filtered: Filtered signal
        env: Envelope signal
        fs: Sampling frequency
        peaks: Detected event indices
        intervals_s: Inter-event intervals in seconds
    """
    # 1. Waveform + Events
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, raw, label="Raw", alpha=0.4, color='lightgray')
    ax.plot(time, filtered, label="Filtered", linewidth=0.8, color='blue')
    ax.plot(time, env / np.max(env) * 0.8 * np.max(filtered),
            label="Envelope (scaled)", linewidth=1, color='green')
    ax.scatter(peaks / fs, env[peaks], c="red", marker="x", s=100,
               label="Detected Events", zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Bowel Sound Waveform with Detected Events")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # 2. Spectrogram
    fig, ax = plt.subplots(figsize=(14, 4))
    f, t, Sxx = signal.spectrogram(filtered, fs=fs, nperseg=1024, noverlap=512)
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    ax.set_ylim(0, 1500)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Bowel Sound Spectrogram (dB)")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    st.pyplot(fig)
    plt.close()


# ===========================
# HEART SOUND VISUALIZATION
# ===========================

def plot_heart_results(time: np.ndarray, raw: np.ndarray, filtered: np.ndarray, 
                       env: np.ndarray, fs: int, peaks: np.ndarray, 
                       intervals_s: np.ndarray):
    """
    Generate comprehensive plots for heartbeat analysis (Streamlit-friendly).
    
    Args:
        time: Time array
        raw: Raw signal
        filtered: Filtered signal
        env: Envelope signal
        fs: Sampling frequency
        peaks: Detected peak indices
        intervals_s: Inter-beat intervals in seconds
    """
    # 1. Waveform + Peaks
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, raw, label="Raw", alpha=0.4, color='lightgray')
    ax.plot(time, filtered, label="Filtered", linewidth=0.8, color='blue')
    ax.plot(time, env / np.max(env) * 0.8 * np.max(filtered),
            label="Envelope (scaled)", linewidth=1, color='green')
    ax.scatter(peaks / fs, env[peaks], c="red", marker="x", s=100, 
               label="Detected Beats", zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Heart Sound Waveform with Detected Beats")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # 2. Spectrogram
    fig, ax = plt.subplots(figsize=(14, 4))
    f, t, Sxx = signal.spectrogram(filtered, fs=fs, nperseg=1024, noverlap=512)
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    ax.set_ylim(0, 800)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Heart Sound Spectrogram (dB)")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    st.pyplot(fig)
    plt.close()

    # 3. IBI Histogram
    if intervals_s.size > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ibi_ms = intervals_s * 1000.0
        ax.hist(ibi_ms, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(ibi_ms), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(ibi_ms):.1f} ms')
        ax.set_xlabel("Inter-Beat Interval (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Inter-Beat Interval Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # 4. Poincaré Plot (HRV visualization)
    if intervals_s.size > 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        x = intervals_s[:-1] * 1000.0
        y = intervals_s[1:] * 1000.0
        ax.scatter(x, y, s=30, alpha=0.6, color='purple')
        
        # Add diagonal line for reference
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'k--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel("IBI(n) (ms)")
        ax.set_ylabel("IBI(n+1) (ms)")
        ax.set_title("Poincaré Plot - Heart Rate Variability")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        st.pyplot(fig)
        plt.close()


# ===========================
# LUNG SOUND VISUALIZATION
# ===========================

def plot_lung_results(data: Dict[str, np.ndarray], fs: int):
    """
    Generate plots for lung sound analysis (Streamlit-friendly).
    
    Args:
        data: Dictionary containing 'raw', 'filtered', 'envelope', and 'peaks'
        fs: Sampling frequency
    """
    t = np.arange(len(data["raw"])) / fs

    # 1. Waveform + Breath Cycles
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, data["raw"], alpha=0.3, label="Raw", color='lightgray')
    ax.plot(t, data["filtered"], label="Filtered", linewidth=0.8, color='blue')
    
    # Scale envelope to match filtered signal amplitude
    env_scaled = data["envelope"] / np.max(data["envelope"]) * np.max(data["filtered"])
    ax.plot(t, env_scaled, label="Envelope (scaled)", linewidth=1, color='green')
    
    # Mark detected breaths
    ax.scatter(
        data["peaks"] / fs,
        data["envelope"][data["peaks"]],
        color="red",
        marker="x",
        s=100,
        label=f'Breaths Detected ({len(data["peaks"])})',
        zorder=5
    )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Lung Sound with Detected Breath Cycles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # 2. Spectrogram (optional, useful for wheeze/crackle visualization)
    fig, ax = plt.subplots(figsize=(14, 4))
    f, t_spec, Sxx = signal.spectrogram(data["filtered"], fs=fs, nperseg=2048, noverlap=1024)
    pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    ax.set_ylim(0, 2500)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Lung Sound Spectrogram (dB)")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    st.pyplot(fig)
    plt.close()

    # 3. Breathing Rate Timeline
    if len(data["peaks"]) > 1:
        peak_times = data["peaks"] / fs
        intervals = np.diff(peak_times)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(peak_times[1:], 60.0 / intervals, marker='o', linestyle='-', 
                color='steelblue', linewidth=2, markersize=6)
        ax.axhline(60.0 / np.mean(intervals), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {60.0/np.mean(intervals):.1f} breaths/min')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Breathing Rate (breaths/min)")
        ax.set_title("Instantaneous Breathing Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()


# ===========================
# UNIFIED PLOT FUNCTION
# ===========================

def plot_results(time: Optional[np.ndarray] = None, 
                 raw: Optional[np.ndarray] = None,
                 filtered: Optional[np.ndarray] = None,
                 env: Optional[np.ndarray] = None,
                 fs: Optional[int] = None,
                 peaks: Optional[np.ndarray] = None,
                 intervals_s: Optional[np.ndarray] = None,
                 data: Optional[Dict[str, np.ndarray]] = None,
                 sound_type: str = 'auto'):
    """
    Unified plotting function that handles bowel, heart, and lung sound visualization.
    
    Args:
        time: Time array (for heart/bowel sounds)
        raw: Raw signal
        filtered: Filtered signal
        env: Envelope signal
        fs: Sampling frequency
        peaks: Detected peak indices
        intervals_s: Inter-beat intervals (for heart/bowel sounds)
        data: Dictionary with signal data (for lung sounds)
        sound_type: Type of sound ('bowel', 'heart', 'lung', or 'auto')
        
    Examples:
        >>> # Bowel sound plotting
        >>> plot_results(time=t, raw=raw, filtered=filt, env=env, 
        ...              fs=4000, peaks=peaks, intervals_s=intervals, sound_type='bowel')
        
        >>> # Heart sound plotting
        >>> plot_results(time=t, raw=raw, filtered=filt, env=env, 
        ...              fs=2000, peaks=peaks, intervals_s=intervals)
        
        >>> # Lung sound plotting
        >>> plot_results(data={"raw": raw, "filtered": filt, 
        ...                    "envelope": env, "peaks": peaks}, fs=4000)
    """
    # Auto-detect sound type
    if sound_type == 'auto':
        if data is not None:
            sound_type = 'lung'
        elif intervals_s is not None:
            sound_type = 'heart'
        else:
            sound_type = 'lung'  # default
    
    if sound_type == 'bowel':
        if time is None or raw is None or filtered is None or env is None or fs is None or peaks is None:
            raise ValueError("Bowel sound plotting requires: time, raw, filtered, env, fs, peaks")
        if intervals_s is None:
            intervals_s = np.array([])
        plot_bowel_results(time, raw, filtered, env, fs, peaks, intervals_s)
    
    elif sound_type == 'heart':
        if time is None or raw is None or filtered is None or env is None or fs is None or peaks is None:
            raise ValueError("Heart sound plotting requires: time, raw, filtered, env, fs, peaks")
        if intervals_s is None:
            intervals_s = np.array([])
        plot_heart_results(time, raw, filtered, env, fs, peaks, intervals_s)
    
    elif sound_type == 'lung':
        if data is None:
            # Build data dict from individual arrays
            if raw is None or filtered is None or env is None or peaks is None:
                raise ValueError("Lung sound plotting requires data dict or individual arrays")
            data = {
                "raw": raw,
                "filtered": filtered,
                "envelope": env,
                "peaks": peaks
            }
        if fs is None:
            raise ValueError("Lung sound plotting requires fs")
        plot_lung_results(data, fs)
    
    else:
        raise ValueError(f"Unknown sound_type: {sound_type}. Use 'bowel', 'heart', 'lung', or 'auto'.")