import json
import numpy as np
from typing import Dict, Any


# ===========================
# BOWEL SOUND PRINTING
# ===========================

def pretty_print_bowel_analysis(result: Dict[str, Any]):
    """
    Nicely formatted console output for bowel sound analysis.
    
    Args:
        result: Dictionary containing bowel sound analysis results
    """
    print("=" * 60)
    print("BOWEL SOUND ANALYSIS SUMMARY")
    print("=" * 60)
    print(f" File: {result['file']}")
    print(f" Duration: {result['duration_s']:.2f} s")
    print(f" Events Detected: {result['events_detected']}")
    
    event_rate = result.get('event_rate_per_min')
    if event_rate is not None:
        print(f" Event Rate (per min): {event_rate:.2f}")
    else:
        print(" Event Rate (per min): N/A")
    
    print("=" * 60)

    # --- Event Metrics ---
    print("\n[Event Metrics]")
    metrics = result.get("event_metrics", {})
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:25s} : {v:.2f}")

    # --- Signal Quality ---
    print("\n[Signal Quality]")
    snr = result.get('SNR_dB')
    if snr is not None:
        print(f"  SNR (dB)                  : {snr:.2f}")
    else:
        print("  SNR (dB)                  : N/A")

    energy = result.get("energy", {})
    for k, v in energy.items():
        print(f"  {k:25s} : {v:.2f}")

    band = result.get("150_1000Hz_band", {})
    if band:
        print(f"  Band Energy ({band.get('band_hz', 'N/A')} Hz): {band.get('band_energy_pct', 0):.2f}%")

    print("=" * 60)


# ===========================
# HEART SOUND PRINTING
# ===========================

def pretty_print_heart_analysis(result: Dict[str, Any]):
    """
    Nicely formatted console output for heart sound analysis.
    
    Args:
        result: Dictionary containing heart sound analysis results
    """
    print("=" * 60)
    print("HEART SOUND ANALYSIS SUMMARY")
    print("=" * 60)
    print(f" File: {result['file']}")
    print(f" Duration: {result['duration_s']:.2f} s")
    print(f" Beats Detected: {result['beats_detected']}")
    
    bpm = result.get('bpm')
    if bpm is not None:
        print(f" Estimated BPM: {bpm:.2f}")
    else:
        print(" Estimated BPM: N/A")
    
    print("=" * 60)

    # --- HRV ---
    print("\n[Heart Rate Variability (HRV)]")
    for k, v in result.get("hrv", {}).items():
        if v is not None:
            print(f"  {k:20s} : {v:.2f}")

    # --- Signal Quality ---
    print("\n[Signal Quality]")
    snr = result.get('SNR_dB')
    if snr is not None:
        print(f"  SNR (dB)            : {snr:.2f}")
    else:
        print("  SNR (dB)            : N/A")
    
    for k, v in result.get("energy", {}).items():
        print(f"  {k:20s} : {v:.2f}")
    
    if result.get("S1S2"):
        for k, v in result["S1S2"].items():
            print(f"  {k:20s} : {v:.2f}")

    # --- Abnormality Detection ---
    print("\n[Abnormality Checks]")
    print(f"  Extra Peaks per Cycle: {result.get('extra_peaks', {})}")
    print(f"  Irregular Spacing     : {result.get('irregular_spacing', {})}")
    print(f"  150-500Hz Band Energy : {result.get('150_500Hz_band', {})}")

    print("=" * 60)


# ===========================
# LUNG SOUND PRINTING
# ===========================

def pretty_print_lung_analysis(result: Dict[str, Any]):
    """
    Console-friendly summary of lung sound analysis.
    
    Args:
        result: Dictionary containing lung sound analysis results
    """
    print("=" * 60)
    print("LUNG SOUND ANALYSIS SUMMARY")
    print("=" * 60)
    print(f" File: {result.get('file', 'N/A')}")
    print(f" Duration: {result.get('duration_s', 0):.2f} s")
    print(f" Breaths Detected: {result.get('breaths_detected', 0)}")
    
    br = result.get('breathing_rate')
    if br is not None:
        print(f" Breathing Rate: {br:.1f} breaths/min")
    else:
        print(" Breathing Rate: N/A")
    
    # Prediction (if available)
    if "classification" in result:
        print(f"\n Predicted Disease : {result['classification']}")
    if "confidence" in result:
        print(f" Confidence        : {result['confidence']*100:.2f}%")
    
    print("=" * 60)

    # --- Signal Quality ---
    print("\n[Signal Quality]")
    snr = result.get('SNR_dB')
    if snr is not None:
        print(f"  SNR (dB): {snr:.2f}")
    
    # --- Spectral Features ---
    print("\n[Spectral Features]")
    for k, v in result.get("spectral", {}).items():
        print(f"  {k:25s}: {v:.2f}")
    
    # --- Adventitious Sounds ---
    print("\n[Adventitious Sound Indices]")
    for k, v in result.get("adventitious", {}).items():
        print(f"  {k:25s}: {v:.2f}")
    
    # --- MFCC Statistics (if available) ---
    if "mfcc_stats" in result:
        print("\n[MFCC Statistics]")
        for k, v in result["mfcc_stats"].items():
            print(f"  {k:25s}: {v:.4f}")

    print("=" * 60)


# ===========================
# UNIFIED PRINTING
# ===========================

def pretty_print_analysis(result: Dict[str, Any], sound_type: str = 'auto'):
    """
    Unified function to print analysis results for bowel, heart, and lung sounds.
    
    Args:
        result: Dictionary containing analysis results
        sound_type: Type of sound ('bowel', 'heart', 'lung', or 'auto')
    """
    # Auto-detect sound type
    if sound_type == 'auto':
        if 'events_detected' in result and 'event_metrics' in result and 'hrv' not in result:
            sound_type = 'bowel'
        elif 'beats_detected' in result and 'hrv' in result:
            sound_type = 'heart'
        elif 'breaths_detected' in result or 'spectral' in result:
            sound_type = 'lung'
        else:
            sound_type = 'heart'  # default
    
    if sound_type == 'bowel':
        pretty_print_bowel_analysis(result)
    elif sound_type == 'heart':
        pretty_print_heart_analysis(result)
    elif sound_type == 'lung':
        pretty_print_lung_analysis(result)
    else:
        raise ValueError(f"Unknown sound_type: {sound_type}. Use 'bowel', 'heart', 'lung', or 'auto'.")


# ===========================
# JSON UTILITIES
# ===========================

def make_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy objects to JSON-serializable types.
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(v) for v in obj)
    else:
        return obj


def export_json(result: Dict[str, Any], path: str):
    """
    Save analysis result to JSON file safely.
    
    Handles numpy arrays and removes raw data arrays before export.
    
    Args:
        result: Dictionary containing analysis results
        path: Output JSON file path
    """
    # Make a copy to avoid modifying original
    safe_result = make_json_safe(result)
    
    # Remove raw data arrays if present
    safe_result.pop("_data", None)
    
    # Write to file
    with open(path, "w") as f:
        json.dump(safe_result, f, indent=4)
    
    print(f"[INFO] Exported results to {path}")


def export_bowel_json(result: Dict[str, Any], path: str):
    """
    Export bowel sound analysis results to JSON.
    
    Args:
        result: Bowel sound analysis results
        path: Output JSON file path
    """
    export_json(result, path)


def export_heart_json(result: Dict[str, Any], path: str):
    """
    Export heart sound analysis results to JSON.
    
    Args:
        result: Heart sound analysis results
        path: Output JSON file path
    """
    export_json(result, path)


def export_lung_json(result: Dict[str, Any], path: str):
    """
    Export lung sound analysis results to JSON.
    
    Args:
        result: Lung sound analysis results
        path: Output JSON file path
    """
    export_json(result, path)


# ===========================
# SUMMARY STATISTICS
# ===========================

def print_comparison_summary(heart_result: Dict[str, Any] = None, 
                            lung_result: Dict[str, Any] = None,
                            bowel_result: Dict[str, Any] = None):
    """
    Print a side-by-side comparison summary of bowel, heart, and lung analysis.
    
    Args:
        heart_result: Heart sound analysis results
        lung_result: Lung sound analysis results
        bowel_result: Bowel sound analysis results
    """
    print("\n" + "=" * 100)
    print("BODY SOUND ANALYSIS COMPARISON".center(100))
    print("=" * 100)
    
    print(f"{'METRIC':<40} {'BOWEL':<20} {'HEART':<20} {'LUNG':<20}")
    print("-" * 100)
    
    # Duration
    if bowel_result:
        print(f"{'Duration (s)':<40} {bowel_result.get('duration_s', 0):<20.2f}", end="")
    else:
        print(f"{'Duration (s)':<40} {'N/A':<20}", end="")
    
    if heart_result:
        print(f"{heart_result.get('duration_s', 0):<20.2f}", end="")
    else:
        print(f"{'N/A':<20}", end="")
    
    if lung_result:
        print(f"{lung_result.get('duration_s', 0):<20.2f}")
    else:
        print(f"{'N/A':<20}")
    
    # Events Detected
    if bowel_result:
        print(f"{'Events Detected':<40} {bowel_result.get('events_detected', 0):<20}", end="")
    else:
        print(f"{'Events Detected':<40} {'N/A':<20}", end="")
    
    if heart_result:
        print(f"{heart_result.get('beats_detected', 0):<20}", end="")
    else:
        print(f"{'N/A':<20}", end="")
    
    if lung_result:
        print(f"{lung_result.get('breaths_detected', 0):<20}")
    else:
        print(f"{'N/A':<20}")
    
    # Rate
    if bowel_result:
        event_rate = bowel_result.get('event_rate_per_min')
        rate_str = f"{event_rate:.1f} ev/min" if event_rate else "N/A"
        print(f"{'Rate':<40} {rate_str:<20}", end="")
    else:
        print(f"{'Rate':<40} {'N/A':<20}", end="")
    
    if heart_result:
        bpm = heart_result.get('bpm')
        bpm_str = f"{bpm:.1f} bpm" if bpm else "N/A"
        print(f"{bpm_str:<20}", end="")
    else:
        print(f"{'N/A':<20}", end="")
    
    if lung_result:
        br = lung_result.get('breathing_rate')
        br_str = f"{br:.1f} br/min" if br else "N/A"
        print(f"{br_str:<20}")
    else:
        print(f"{'N/A':<20}")
    
    # SNR
    if bowel_result:
        snr = bowel_result.get('SNR_dB')
        snr_str = f"{snr:.2f} dB" if snr else "N/A"
        print(f"{'SNR':<40} {snr_str:<20}", end="")
    else:
        print(f"{'SNR':<40} {'N/A':<20}", end="")
    
    if heart_result:
        snr = heart_result.get('SNR_dB')
        snr_str = f"{snr:.2f} dB" if snr else "N/A"
        print(f"{snr_str:<20}", end="")
    else:
        print(f"{'N/A':<20}", end="")
    
    if lung_result:
        snr = lung_result.get('SNR_dB')
        snr_str = f"{snr:.2f} dB" if snr else "N/A"
        print(f"{snr_str:<20}")
    else:
        print(f"{'N/A':<20}")
    
    print("=" * 100 + "\n")