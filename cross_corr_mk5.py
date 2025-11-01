# cross_corr_mk3.py

# Standard Library Imports
import argparse
import math
from pathlib import Path
from typing import Any, Dict, Literal, Tuple
import os, sys
import traceback
from time import perf_counter

from dataclasses import dataclass
from typing import List, Optional

# Third Party Imports
import numpy as np
import matplotlib.pyplot as plt


# Local Imports 
from _io_laeyr import get_io_rec
from cli import build_cli, resolve_params
from common import (convert_raw_buffer_to_complex, gen_sign, ncc, plot_cross_corr, plot_signal_structure, print_corruption_info, trace_prefix)
# Color Codes and Logging Tags
from common import (
    GREEN, BRIGHT_GREEN, RED, BRIGHT_RED, YELLOW, BRIGHT_YELLOW,
    GRAY, BRIGHT_GRAY, CYAN, BRIGHT_CYAN, BLUE, BRIGHT_BLUE, MAGENTA,
    BRIGHT_MAGENTA, WHITE, BRIGHT_WHITE, BLACK, BRIGHT_BLACK, 
    RESET,
    WARN, ERR, INFO, DBG, OK
)
from dmr import (DIBIT_TO_DEV_HZ, FSK_LOOKUP, SYNC_PATTERNS, analyze_peak_distances, estimate_sync_count_in_chunk, generate_iq_samples_C, generate_iq_samples_G, tst_sync_template, verify_template_vs_standard)

# =====================================================
# Identity
_MODULE_MARKER = Path(__file__).stem

FIN = f"{BRIGHT_MAGENTA}=== Finis sententiae ==={RESET}" # "Ita est"

BPS = 4  # Byte  per [complex] sample

# ==============================================================================

def build_cross_corr(signal_chunk: np.ndarray, srch_seq: np.ndarray, corr: np.ndarray) -> None:
    """
    In-place cross-correlation into preallocated `corr`.
    Contract:
      - mode='valid'
      - len(corr) == len(signal_chunk) - len(srch_seq) + 1
      - no normalization or windowing
    """
    assert signal_chunk.ndim == 1 and srch_seq.ndim == 1, "Inputs must be 1-D."
    expected_len = signal_chunk.size - srch_seq.size + 1
    assert expected_len > 0, "Search sequence longer than signal chunk."
    assert corr.shape == (expected_len,), "`corr` has wrong length."
    corr[:] = np.correlate(signal_chunk, np.conj(srch_seq), mode='valid')


def detect_positions(corr: np.ndarray, chunk_start_idx: int, min_distance: int, z_thr: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      peaks_abs: np.ndarray[int]  # absolute indices in the original signal (start of sync)
      vals:      np.ndarray[float]# correlation values at detected peaks
    Deterministic, thresholded, NMS with radius = min_distance.
    """
    assert min_distance >= 1, "min_distance must be >= 1"
    assert np.isfinite(corr).all(), "NaN/Inf in `corr`."
    
    x = corr.astype(np.float64, copy=False)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else 1e-12
    thr = med + z_thr * sigma

    work = x.copy()
    peaks_abs: list[int] = []
    vals: list[float] = []
    neg_inf = -np.inf
    n = work.size

    while True:
        i = int(np.argmax(work))
        v = work[i]
        if not np.isfinite(v) or v < thr:
            break
        abs_i = chunk_start_idx + i
        peaks_abs.append(abs_i)
        vals.append(x[i])

        L = max(0, i - (min_distance - 1))
        R = min(n, i + min_distance)
        work[L:R] = neg_inf

    return np.asarray(peaks_abs, dtype=int), np.asarray(vals, dtype=float)

rng = np.random.default_rng(42)
def corrupt(x, mode, level):
    x = np.asarray(x)
    if mode == "amp_scale":                # глобальне масштабування
        return x * level
    if mode == "awgn_snr_db":              # адитивний шум із заданим SNR дБ
        p_sig = np.mean(np.abs(x)**2)
        snr_lin = 10**(level/10.0)
        p_noise = p_sig / (snr_lin + 1e-12)
        if np.iscomplexobj(x):
            noise = (rng.normal(size=x.size) + 1j * rng.normal(size=x.size)) / np.sqrt(2)
        else:
            noise = rng.normal(size=x.size)
        noise = noise * np.sqrt(p_noise)
        return x + noise
    if mode == "contig_zero_frac":         # занулити суцільний відрізок частки length=level∈(0,1]
        assert 0 < level <= 1, "level must be in (0,1]"
        L = max(1, int(round(level * x.size)))
        s = rng.integers(0, x.size - L + 1)
        y = x.copy()
        y[s:s+L] = 0
        return y
    if mode == "sample_replace_frac":      # випадкові позиції замінити шумом (частка level)
        assert 0 < level <= 1, "level must be in (0,1]"
        k = max(1, int(round(level * x.size)))
        idx = rng.choice(x.size, size=k, replace=False)
        y = x.copy()

        if np.iscomplexobj(x):
            # енергетично узгоджений масштаб для комплексного сигналу
            scale = np.sqrt(np.mean(np.abs(x - np.mean(x))**2))
            rep = (rng.normal(size=k) + 1j*rng.normal(size=k)) / np.sqrt(2) * scale
        else:
            # реальний випадок
            scale = np.sqrt(np.mean((x - np.mean(x))**2))
            rep = rng.normal(size=k) * scale

        y[idx] = rep  # тип уже збігається, каст не потрібен → без ComplexWarning
        return y

    if mode == "global_phase_deg":         # глобальний фазовий поворот
        phi = np.deg2rad(level)
        y = x * np.exp(1j * phi)
        if not np.iscomplexobj(x):   # якщо сигнал дійсний
            y = np.real(y)
        return x * np.exp(1j*phi)
    if mode == "per_sample_phase_jitter_deg":  # фазовий шум N(0, level^2)
        phi = rng.normal(loc=0.0, scale=np.deg2rad(level), size=x.size)
        y = x * np.exp(1j * phi)
        if not np.iscomplexobj(x):
            y = np.real(y)
        return x * np.exp(1j*phi)
    if mode == "time_shift":               # циклічний зсув на level відліків (int)
        s = int(level)
        return np.roll(x, s)
    raise ValueError("unknown mode")


def compare_signals(a:np.ndarray, b:np.ndarray)->Dict[str,Any]:
    """
    Порівняння двох комплексних сигналів через кореляційні метрики
    
    Повертає метрики які працюють для:
    - Зміщених сигналів
    - Масштабованих сигналів  
    - Схожих, але не ідентичних сигналів
    """
    
    # Normalized correlation coefficient (0 до 1, де 1 = ідентичні)
    dot_prod = np.vdot(b, a)
    corr_coef = np.abs(dot_prod) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Максимум нормалізованої крос-кореляції (знаходить схожість навіть при зсуві)
    xcorr = np.correlate(a, np.conj(b), mode='full')
    max_xcorr = np.max(np.abs(xcorr))
    auto_c = np.vdot(a, a).real
    auto_g = np.vdot(b, b).real
    norm_xcorr = max_xcorr / np.sqrt(auto_c * auto_g)
    
    # Позиція піку (для визначення зсуву)
    peak_idx = np.argmax(np.abs(xcorr))
    time_shift = peak_idx - (len(a) - 1)
 
    return {
        'corr_coef': corr_coef,
        'xcorr_peak': norm_xcorr,
        'time_shift': time_shift
    }

def compare_signals_fast(a:np.ndarray, b:np.ndarray)->float:
    """
    Швидке порівняння комплексних сигналів
    Використовує лише correlation coefficient
    """
    # Normalized correlation coefficient - O(N) операцій
    dot_prod = np.vdot(b, a)  # одне проходження
    norm_c = np.linalg.norm(a)      # одне проходження
    norm_g = np.linalg.norm(b)      # одне проходження
    corr_coef = np.abs(dot_prod) / (norm_c * norm_g)
    
    return corr_coef


def compare_iq_samples(sync_c: np.ndarray, sync_g: np.ndarray, title: str = "IQ Samples Comparison"):
    """
    Порівняння двох масивів IQ-семплів на двох графіках один під одним
    
    Args:
        sync_c: Claude generated IQ samples
        sync_g: GPT generated IQ samples  
        title: Заголовок графіку
    """
    # Обмежуємо кількість точок для читабельності
    n_samples = min(1000, len(sync_c), len(sync_g))
    time_axis = np.arange(n_samples)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    def _on_key(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    plt.gcf().canvas.mpl_connect('key_press_event', _on_key)
    # Верхній subplot - Claude версія
    ax1.plot(time_axis, np.real(sync_c[:n_samples]), 'b-', label='I (Real)', alpha=0.7, linewidth=0.8)
    ax1.plot(time_axis, np.imag(sync_c[:n_samples]), 'r-', label='Q (Imag)', alpha=0.7, linewidth=0.8)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Claude version', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, n_samples])
    
    # Нижній subplot - GPT версія
    ax2.plot(time_axis, np.real(sync_g[:n_samples]), 'b-', label='I (Real)', alpha=0.7, linewidth=0.8)
    ax2.plot(time_axis, np.imag(sync_g[:n_samples]), 'g-', label='Q (Imag)', alpha=0.7, linewidth=0.8)
    ax2.set_xlabel('Sample index', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('GPT version', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, n_samples])
    

    
    # Додаткова статистика
    diff = sync_c[:n_samples] - sync_g[:n_samples]
    print(f"\nStatistics for first {n_samples} samples:")
    print(f"  Max absolute diff: {np.max(np.abs(diff)):.6f}")
    print(f"  Mean absolute diff: {np.mean(np.abs(diff)):.6f}")
    print(f"  RMS diff: {np.sqrt(np.mean(np.abs(diff)**2)):.6f}")
    
    plt.tight_layout()
    plt.show()


def test_dmr_sync_gen():
    Fs = 48e6
    if 0:
        for dibit in [(0,1), (0,0), (1,0), (1,1)]:
            fsk = FSK_LOOKUP[dibit]
            direct = DIBIT_TO_DEV_HZ[dibit]
            print(f"{dibit}: FSK={fsk.deviation_hz:_.1f} Hz, DIRECT={direct:_.1f} Hz")

    for id, sync in SYNC_PATTERNS.items():
        t0 = perf_counter()
        sync_c = generate_iq_samples_C(sync, Fs)
        sync_g = generate_iq_samples_G(sync, Fs)
        gen_t = perf_counter() - t0
        
        if 0:
            t0 = perf_counter()
            comp_res = compare_signals(sync_c, sync_g)
            comput_t = perf_counter() - t0
            
            print(f"{YELLOW}{id:18}{RESET}:"
                f"| dur {gen_t=}  {comput_t=}"
                f"|  corr_coef:  {CYAN}{comp_res["corr_coef"]:_.2f}{RESET}"
                f"|  xcorr_peak: {CYAN}{comp_res["xcorr_peak"]:_.2f}{RESET}"
                f"|  time_shift: {CYAN}{comp_res["time_shift"]:_.2f}{RESET}"
                )
        t0 = perf_counter()
        comp_res = compare_signals_fast(sync_c, sync_g)
        comput_t = perf_counter() - t0
        print(f"{YELLOW}{id:18}{RESET}:"
                f"| dur {gen_t=:.3f}  {comput_t=:.3f}"
                f"|  corr_coef:  {CYAN}{comp_res:_.2f}{RESET}"
        )
        # compare_iq_samples(sync_c, sync_g, title=f"{id} - IQ Samples Comparison")
    sys.exit(0)


def plot_sync_correlation(detections: dict, sync_ids: list = None, info: str = None, block: bool = False):
    """
    Візуалізує масиви кореляції для обраних SYNC типів.
    
    Args:
        detections: результат від search_sync_in_chunk
        sync_ids: список SYNC для відображення (None = всі)
    """
    if sync_ids is None:
        sync_ids = list(detections.keys())
    
    n_plots = len(sync_ids)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots), sharex=True)
    def _on_key(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    if n_plots == 1:
        axes = [axes]
    
    for ax, sync_id in zip(axes, sync_ids):
        det = detections[sync_id]
        corr = det['corr_array']
        peaks = det['peaks']
        threshold = det['threshold']
        
        # Визначаємо що малювати: z-scores або correlation
        if 'z_scores' in det:
            plot_array = det['z_scores']
            ylabel = 'Z-score'
            title_suffix = ' (z-score normalized)'
        else:
            plot_array = corr
            ylabel = 'Correlation'
            title_suffix = ''
        
        # Масив (z-score або correlation)
        ax.plot(plot_array, color='gray', lw=0.5, alpha=0.7, label=ylabel)
        
        # Threshold лінія
        ax.axhline(threshold, color='red', ls='--', lw=1, label=f'Threshold={threshold:.2f}')
        
        # Піки
        if len(peaks) > 0:
            peak_vals = plot_array[peaks]
            ax.scatter(peaks, peak_vals, color='red', s=30, zorder=5, label=f'Peaks ({len(peaks)})')
        
        # Статистика
        stats_text = f"max={np.max(plot_array):.2f} mean={np.mean(plot_array):.2f} std={np.std(plot_array):.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        title = f"{sync_id} correlation{title_suffix}"
        if info:
            title += f"\n{info}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Sample index')
    plt.tight_layout()
    plt.show(block=block)


def search_sync_in_iq_chunk(samples: np.ndarray, sync_templates: dict, threshold: float, 
                         sync_ids: list | None = None, use_zscore: bool = True,
                         verbose: bool = True) -> dict:
    """
    Search for SYNC patterns in a chunk of IQ samples.
    
    Args:
        samples: IQ samples to search in
        sync_templates: dict of all available SYNC templates
        threshold: if use_zscore=True: z-score threshold (e.g. 3.0), else correlation threshold [0..1]
        sync_ids: list of SYNC IDs to search (None = search all)
        use_zscore: use z-score normalization instead of absolute threshold
        verbose: print detection results
    
    Returns:
        dict: {sync_id: {'position': int, 'correlation': float, 'corr_array': np.ndarray, 
                         'peaks': np.ndarray, 'threshold': float, 'z_scores': np.ndarray (if use_zscore)}}
    """
    if sync_ids is None:
        sync_ids = list(sync_templates.keys())
    
    results = {}
    
    for sync_id in sync_ids:
        template = sync_templates[sync_id]
        
        # Cross-correlation
        corr = np.correlate(samples, np.conj(template), mode='valid')
        
        # Normalize correlation
        template_power = np.sum(np.abs(template)**2)
        for i in range(len(corr)):
            sig_window = samples[i:i+len(template)]
            sig_power = np.sum(np.abs(sig_window)**2)
            corr[i] /= np.sqrt(template_power * sig_power + 1e-12)
        
        corr_abs = np.abs(corr)
        
        # Z-score normalization (if enabled)
        if use_zscore:
            mean_corr = np.mean(corr_abs)
            std_corr = np.std(corr_abs)
            z_scores = (corr_abs - mean_corr) / (std_corr + 1e-12)
            detection_array = z_scores
        else:
            z_scores = None
            detection_array = corr_abs
        
        # Find ALL peaks above threshold
        peaks = np.where(detection_array > threshold)[0]
        
        # Find max peak
        peak_idx = np.argmax(detection_array)
        peak_val = detection_array[peak_idx]
        corr_val = corr_abs[peak_idx]  # original correlation value
        
        # Store result
        results[sync_id] = {
            'position': int(peak_idx),
            'correlation': float(corr_val),
            'corr_array': corr_abs,
            'peaks': peaks,
            'threshold': threshold,
        }
        
        if use_zscore:
            results[sync_id]['z_scores'] = z_scores
            results[sync_id]['peak_zscore'] = float(peak_val)
        
        # Report
        if verbose:
            if len(peaks) > 0:
                if use_zscore:
                    print(f"{OK} {sync_id:18} @ {peak_idx:10d}  corr={CYAN}{corr_val:.3f}{RESET}  "
                          f"z={CYAN}{peak_val:.2f}{RESET}  total_peaks={len(peaks)}")
                else:
                    print(f"{OK} {sync_id:18} @ {peak_idx:10d}  corr={CYAN}{peak_val:.3f}{RESET}  total_peaks={len(peaks)}")
            else:
                max_val = np.max(detection_array)
                print(f"{WARN} {sync_id:18} no peaks above thr (max={'z=' if use_zscore else ''}{max_val:.2f})")
    
    return results


def search_sync_in_phase_chunk(phase_diffs: np.ndarray, sync_phase_templates: dict, threshold: float, 
                               min_distance: int,
                                sync_ids: list | None = None, use_zscore: bool = False,
                                verbose: bool = True) -> dict:
    """
    Search for SYNC patterns using phase difference correlation.
    
    Args:
        phase_diffs: phase differences of IQ samples (np.diff(np.unwrap(np.angle(samples))))
        sync_phase_templates: dict of phase difference templates
        threshold: correlation threshold
        sync_ids: list of SYNC IDs to search (None = search all)
        use_zscore: use z-score normalization
        verbose: print detection results
    
    Returns:
        dict: {sync_id: detection_info}
    """
    if sync_ids is None:
        sync_ids = list(sync_phase_templates.keys())
    
    results = {}
    
    for sync_id in sync_ids:
        template = sync_phase_templates[sync_id]
        
        # Cross-correlation on phase differences
        corr = np.correlate(phase_diffs, template, mode='valid')
        
        # Normalize correlation
        template_power = np.sum(template**2)
        corr_normalized = np.empty(len(corr))
        for i in range(len(corr)):
            sig_window = phase_diffs[i:i+len(template)]
            sig_power = np.sum(sig_window**2)
            corr_normalized[i] = corr[i] / np.sqrt(template_power * sig_power + 1e-12)
        
        # Z-score normalization (if enabled)
        if use_zscore:
            mean_corr = np.mean(corr_normalized)
            std_corr = np.std(corr_normalized)
            z_scores = (corr_normalized - mean_corr) / (std_corr + 1e-12)
            detection_array = z_scores
        else:
            z_scores = None
            detection_array = corr_normalized
        
        # Find ALL peaks above threshold
        peaks_raw = np.where(detection_array > threshold)[0]
        # Apply greedy NMS with min_distance
        if len(peaks_raw) > 0:
            # Sort by correlation value (highest first)
            sorted_indices = np.argsort(-detection_array[peaks_raw])
            peaks_sorted = peaks_raw[sorted_indices]
            
            peaks = []
            for peak in peaks_sorted:
                # Check if too close to already selected peaks
                if not any(abs(peak - p) < min_distance for p in peaks):
                    peaks.append(peak)
            
            peaks = np.array(peaks, dtype=int)
        else:
            peaks = peaks_raw
        # Find max peak
        peak_idx = np.argmax(detection_array)
        peak_val = detection_array[peak_idx]
        corr_val = corr_normalized[peak_idx]
        
        # Store result
        results[sync_id] = {
            'position': int(peak_idx),
            'correlation': float(corr_val),
            'corr_array': corr_normalized,
            'peaks': peaks,
            'threshold': threshold,
        }

        if use_zscore:
            results[sync_id]['z_scores'] = z_scores
            results[sync_id]['peak_zscore'] = float(peak_val)
        
        # Report
        if verbose:
            if len(peaks) > 0:  # ← ТУТ! Було peaks_raw, має бути peaks
                if use_zscore:
                    print(f"{OK} {sync_id:18} @ {peak_idx:10d}  corr={CYAN}{corr_val:.3f}{RESET}  "
                          f"z={CYAN}{peak_val:.2f}{RESET}  total_peaks={len(peaks)}")  
                else:
                    print(f"{OK} {sync_id:18} @ {peak_idx:10d}  corr={CYAN}{peak_val:.3f}{RESET}  total_peaks={len(peaks)}") 
            else:
                max_val = np.max(detection_array)
                print(f"{WARN} {sync_id:18} no peaks above thr (max={'z=' if use_zscore else ''}{max_val:.2f})")
    
    return results


def look_for_sync_in_file():
    args = build_cli(_MODULE_MARKER).parse_args()
    file, header_sz, samp_count, Fs, dtype, bps, verbose= resolve_params(args, True)
    fsz = file.stat().st_size
    # Pre-Allocations
    _,  _, _CHUNK_BYTE_SZ_OPT =  get_io_rec()
    n_bytes = _CHUNK_BYTE_SZ_OPT
    chunk_sz = n_bytes
    chunk_ln = n_bytes // bps
    chunk_dur = chunk_ln / Fs
    chunk_dur_ms = (chunk_ln / Fs) * 1000
    chunks_count = (fsz-header_sz) // chunk_sz
    raw_byte_buffer = np.empty((n_bytes,), dtype=np.uint8)
    complex_buffer = np.empty((chunk_ln,), dtype=np.complex64)
    # Build all sync`s paterns on samp rate Fs
    sync_templates = {}
    sync_phase_templates = {}    
    for id, sync in SYNC_PATTERNS.items():
        iq_template = generate_iq_samples_C(sync, Fs, apply_rrc=True)
        sync_templates[id] = iq_template
        
        # Phase difference template
        phases = np.unwrap(np.angle(iq_template))
        phase_diffs = np.diff(phases)
        sync_phase_templates[id] = phase_diffs

    print(f"\n{INFO} Expected SYNC counts in chunk of {GREEN}{chunk_ln:_}{RESET} samples ({CYAN}{chunk_dur_ms:.2f}{RESET} ms):")
    for sync_type in ['MS_Voice', 'MS_Data']:
        expected = estimate_sync_count_in_chunk(sync_type, chunk_ln, Fs)
        print(f"  {YELLOW}{sync_type:18}{RESET} \t ~{GREEN}{expected}{RESET} SYNC patterns")
    
    template_len = len(next(iter(sync_templates.values())))
    phase_template_len = len(next(iter(sync_phase_templates.values())))
    MIN_DISTANCE = phase_template_len  // 2
    THRESHOLD = 0.59  # normalized correlation coefficient [0..1]: 0=uncorrelated, 1=perfect match
    
    print(f"{INFO} Generated {len(sync_templates)} templates:")
    print(f"        IQ templates: len={template_len}")
    print(f"        Phase diff templates: len={phase_template_len}")
    print(f"        min_dist={MIN_DISTANCE}, thr={CYAN}{THRESHOLD}{RESET}")  # Threshold normalized correlation coefficient [0..1]: 0=uncorrelated, 1=perfect match
   
    fd = open(file,"rb")  # file existance checked previously
    fd.seek(header_sz)
    try:
        # tst_sync_template()
        # verify_template_vs_standard()
        chunk_idx = 0
        t0 = perf_counter()
        while True:
            n_bytes_read = fd.readinto(raw_byte_buffer)
            if n_bytes_read < n_bytes:
                break

            samples = convert_raw_buffer_to_complex(raw_byte_buffer, n_bytes_read, complex_buffer=complex_buffer, dtype=dtype)
            dc_offset = np.mean(samples)
            samples -= dc_offset
            
            if 0:
                search_sync_in_iq_chunk(samples, sync_templates, THRESHOLD, 
                                            sync_ids=['MS_Voice', 'MS_Data'], 
                                            use_zscore=False,
                                            verbose=verbose)
            # Compute phase differences for chunk
            phases_chunk = np.unwrap(np.angle(samples))
            phase_diffs_chunk = np.diff(phases_chunk)
            
            detections = search_sync_in_phase_chunk(phase_diffs_chunk, sync_phase_templates, 
                                        THRESHOLD, MIN_DISTANCE,
                                        sync_ids=['MS_Voice', 'MS_Data'],
                                        verbose=verbose)
            

            analyze_peak_distances(detections, 'MS_Voice', Fs)
            # analyze_peak_distances(detections, 'MS_Data', Fs)
            # Малюємо тільки MS_Voice та MS_Data
            chunk_idx += 1
            info = f"{chunk_idx=}/{chunks_count}"
            plot_sync_correlation(detections, sync_ids=['MS_Voice', 'MS_Data'], info=info, block=True)
            
            # break  # PoC: exit after first chunk

        elapsed = perf_counter()-t0
    finally:
        fd.close()
    sys.exit(0)


if __name__ == "__main__":
    os.system('')  # Enables ANSI escape characters in terminal (Windows)
    print(f"\n{INFO} Starting cross-correlation sync detection mk3...")
    try:
        # test_dmr_sync_gen()
        look_for_sync_in_file()
        # -------------------------------
        # ПАРАМЕТРИ ГЕНЕРАЦІЇ СИГНАЛУ
        # -------------------------------
        SYNC_LN: int = 24           # довжина синхропослідовності (в символах)
        M: int = 120                # кількість символів зліва і справа від синхри
        SUPER_FRAME_LN: int = 6     # кількість бурстів у суперкадрі
        SF_COUNT: int = 48           # кількість суперкадрів
        GUARD: float = 0.2          # частка охоронного інтервалу (від довжини бурста)
        SPS: int = 10               # відліків на символ
        SIGN_STD: float = 50.0      # амплітудний розкид (U(-A, +A))
        SEED_SYNC: int = 42         # початкове зерно для синхри
        
        BURST_LEN = (2*M + SYNC_LN) * SPS
        GUARD_LEN = int(round(GUARD * BURST_LEN))
        burst_step = BURST_LEN + GUARD_LEN
        expected_positions = [
            (sf * SUPER_FRAME_LN + 0) * burst_step + M * SPS
            for sf in range(SF_COUNT)
        ]
        sync_corrupt_prob=0.25
        sync_corrupt_smb_part_max=0.3
        sync_corrupt_samp_part_max=0.3
        # sync_corrupt_samp_part_max=None
        signal, sync, corrupt_info = gen_sign(SYNC_LN=SYNC_LN,M=M,SUPER_FRAME_LN=SUPER_FRAME_LN,SF_COUNT=SF_COUNT,
                                GUARD=GUARD,SPS=SPS,SIGN_STD=SIGN_STD,
                                sync_corrupt_prob=sync_corrupt_prob,
                                sync_corrupt_smb_part_max=sync_corrupt_smb_part_max,
                                sync_corrupt_samp_part_max=sync_corrupt_samp_part_max,
                                seed_sync=SEED_SYNC)
        print_corruption_info(corruption_stats=corrupt_info, mode="brief")

        SYNC_LN_SAM = len(sync)           # довжина синхросеквенції у семплах
        SIGN_LN_SAM = len(signal)             # довжина повного сигналу у семплах

        MIN_DISTANCE = SYNC_LN_SAM + 2 * M * SPS  # відстань між піками в семплах
        # len_corr = len(sign) - len(sync) + 1.
        corr = np.zeros(SIGN_LN_SAM - SYNC_LN_SAM + 1)  # буфер для кореляції (mode='valid')

        TARGET_SNR_DB: float =1.1
        signal_power_mean = np.mean(signal**2)

        noise_power = signal_power_mean / (10**(TARGET_SNR_DB / 10))
        noise_std = np.sqrt(noise_power)

        np.random.seed(0)
        noise = np.random.normal(0, noise_std, size=signal.shape)
        signal_noisy = signal + noise
        signal_noised_power_mean = np.mean(signal_noisy**2)
        snr_actual = 10 * np.log10(signal_power_mean / np.mean(noise**2))
        snr_est_td = 10 * np.log10(np.var(signal) / np.var(noise))  # time-domain SNR estimate
        snr_est_fd = 10 * np.log10(np.sum(np.abs(np.fft.fft(signal))**2) / np.sum(np.abs(np.fft.fft(noise))**2)) # frequency-domain SNR estimate

        print(f"{INFO} Generated signal length: {GREEN}{SIGN_LN_SAM}{RESET}. Sync length: {GREEN}{SYNC_LN}{RESET}."
              f"\n       Signal power: {CYAN}{signal_power_mean:.2f}{RESET}. SING_STD: {CYAN}{SIGN_STD}{RESET}."
              f"\n       Signal noised power: {CYAN}{signal_noised_power_mean:.2f}{RESET}."
              f"\n       Target SNR: {CYAN}{TARGET_SNR_DB:.2f}{RESET} dB, actual {CYAN}{snr_actual:.2f}{RESET} dB.  Noise power: {CYAN}{noise_power:.2f}{RESET}. Noise std: {CYAN}{noise_std:.2f}{RESET}"
              f"\n       Time-domain SNR estimate: {CYAN}{snr_est_td:.2f}{RESET} dB. Frequency-domain SNR estimate: {CYAN}{snr_est_fd:.2f}{RESET} dB."
        )

        if 0:
            plot_signal_structure(signal=signal, SYNC_LN=SYNC_LN, M=M, SUPER_FRAME_LN=SUPER_FRAME_LN, SF_COUNT=SF_COUNT, SPS=SPS, GUARD=GUARD, sync_positions_only=True, 
                                  title_suffix=f"[clean signal, power={signal_power_mean:.1f} dB]", block=False)
            plot_signal_structure(signal=signal_noisy, SYNC_LN=SYNC_LN, M=M, SUPER_FRAME_LN=SUPER_FRAME_LN, SF_COUNT=SF_COUNT, SPS=SPS, GUARD=GUARD, sync_positions_only=True, 
                                  title_suffix=f"[noisy signal, power={signal_noised_power_mean:.1f} dB] ", block=False)
            input("Press Enter to continue...")


        if 1:
            # len_corr = len(sync) - len(sync_reduced) + 1
            sign_power = np.abs(sync)**2
            print(f"{INFO} Corr with reduced itself")
            for reduce in [1,1e2, 1e3, 1e4, 1e5, 1e6]:
                sync_reduced = sync / reduce
                sign_power_reduced = np.abs(sync_reduced)**2
                corr_vect = np.correlate(sync, np.conj(sync_reduced), mode='valid')
                corr_val = corr_vect[0]  # cause both seq has same len
                norm_coef = np.sqrt(np.sum(sign_power) * np.sum(sign_power_reduced))
                corr_val_norm = corr_val / norm_coef
                print(f"{int(reduce):>12_}  corr:  norm={CYAN}{corr_val_norm:_.2f}{RESET} raw={CYAN}{corr_val:_.2f}{RESET}  \t\t{norm_coef=:_} ")
            print()

            print(f"{INFO} Corr with corrupted itself (|NCC|, phase deg)")

            modes = [
                ("amp_scale",             [1, 1e-1, 1e-2, 1e-3]),
                ("awgn_snr_db",           [30, 20, 10, 0, -5]),
                ("contig_zero_frac",      [0.1, 0.3, 0.5]),
                ("sample_replace_frac",   [0.05, 0.1, 0.2]),
                ("global_phase_deg",      [0, 30, 90, 180]),
                ("per_sample_phase_jitter_deg", [5, 15, 30, 60]),
                ("time_shift",            [0, 1, 5, 20]),
            ]
            for mode, levels in modes:
                print(f"\n{YELLOW}{mode}{RESET}")
                for lv in levels:
                    y = corrupt(sync, mode, lv)
                    rho = ncc(sync, y)
                    if np.iscomplexobj(rho):
                        mag = np.abs(rho)
                        ph  = np.degrees(np.angle(rho))
                        rho_str = f"|rho|={CYAN}{mag:0.3f}{RESET}  phase={CYAN}{ph:6.1f}{RESET}"
                    else:
                        rho_str = f"rho={CYAN}{rho:0.3f}{RESET}"
                    print(f"  level={lv:>8}  {rho_str}")
            sys.exit(0)
            
        print(f"{INFO} Sync ampl reduction tests:")
        chunk_start = 0
        for r, reduce in enumerate([1e5], 1):
            sync_reduced = sync / reduce
            
            build_cross_corr(signal_chunk=signal_noisy, srch_seq=sync_reduced, corr=corr)

            peaks, vals = detect_positions(corr=corr, chunk_start_idx=chunk_start, min_distance=MIN_DISTANCE)
            
            is_findied_all_expected = all([p in peaks  for p in expected_positions])
            
            finded_redundant = [p for p in peaks if p not in expected_positions]
            is_redundant = len(finded_redundant) > 0
            msg_color = BRIGHT_GREEN if is_findied_all_expected else BRIGHT_RED
            msg = f"{msg_color}found  all{RESET}" if is_findied_all_expected else f"{msg_color}some missing/wrong{RESET}"
            if is_redundant:
                extra_count = len(finded_redundant)
                msg += f", redundant detections {BRIGHT_YELLOW}extra_count{RESET}"
            print(f"  {r:2} samples sync reduction {CYAN}{reduce}{RESET} - {msg}")
            if not is_findied_all_expected:
                print(f"    Detected peaks positions:")
                for i, (p, v) in enumerate(zip(peaks, vals), 1):
                    peak_color = BRIGHT_GREEN if p in expected_positions else BRIGHT_RED
                    print(f"      {i:2}: idx={peak_color}{int(p):>8_d}{RESET}, val={v:_.2f}")
                plot_cross_corr(peaks, vals, corr, title=f"Cross-correlation with noisy [snr={TARGET_SNR_DB} dB] signal. Sync sample ampl reducted by {reduce}.", block=True)


        print(f"{INFO} Finished.\n")
    except Exception:
        print(f"{ERR} An error occurred:{RESET}")
        traceback.print_exc()
        sys.exit(1)