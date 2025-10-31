
# ANSI color codes for terminal output
GREEN = "\033[92m"
BRIGHT_GREEN = "\033[1;92m"
RED = "\033[91m"
BRIGHT_RED = "\033[1;91m"
YELLOW = "\033[93m"
BRIGHT_YELLOW = "\033[1;93m"
GRAY = "\033[90m"
BRIGHT_GRAY = "\033[1;90m"
CYAN = "\033[96m"
BRIGHT_CYAN = "\033[1;96m"
BLUE = "\033[94m"
BRIGHT_BLUE = "\033[1;94m"
MAGENTA = "\033[95m"
BRIGHT_MAGENTA = "\033[1;95m"
WHITE = "\033[97m"
BRIGHT_WHITE = "\033[1;97m"
BLACK = "\033[30m"
BRIGHT_BLACK = "\033[1;30m"
RESET = "\033[0m"

WARN = BLUE + "[WARN]" + RESET
ERR = RED + "[CRIT]" + RESET
INFO = YELLOW + "[INFO]" + RESET
DBG = GRAY + "[DeBG]" + RESET
OK = GREEN + "[ OK ]" + RESET

import inspect
from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def trace_prefix():
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)
    return f"{GRAY}[{info.filename}:{info.lineno} ({info.function})]{RESET} "


# Helper Functions
def init_rand_subseq(arr: np.ndarray, l: int, r: int, _min: float, _max: float, seed: int = None) -> None:
    """
    Inplace Initialize a random subsequence of an pre allcated array between indices l and r.
    """
    if seed is not None:
        np.random.seed(seed)
    if l < 0 or r > len(arr) or l >= r:
        raise ValueError("Invalid indices for subsequence.")
    
    arr[l:r] = np.random.uniform(_min, _max, r - l)


def noise_signal(arr: np.ndarray, l: int, r: int, noise_min: float, noise_max: float, seed: int = None) -> None:
    """
    Inplace Add uniform random noise to the array[l:r].
    """
    if seed is not None:
        np.random.seed(seed)
    if l < 0 or r > len(arr) or l >= r:
        raise ValueError(f"{trace_prefix()} Invalid indices for subsequence.")
    noise = np.random.uniform(noise_min, noise_max, r - l)
    arr[l:r] += noise


def lab_detect(template: np.ndarray, clean_signal: np.ndarray, noisy_signal: np.ndarray) -> dict:
    """
    Навчальна функція без опцій.
    Вхід:
      template      — 1D шаблон (довжина K)
      clean_signal  — 1D сигнал з вбудованим шаблоном без шуму (довжина N)
      noisy_signal  — 1D сигнал = clean_signal + шум

    Повертає dict з ключами:
      corr_clean            — масив нормалізованої кореляції (N-K+1) для clean_signal
      corr_noisy            — масив нормалізованої кореляції (N-K+1) для noisy_signal
      peaks_clean_idx       — індекси топ-піків у corr_clean
      peaks_noisy_idx       — індекси топ-піків у corr_noisy
      conf_clean            — довіра піків corr_clean: peak / σ_фон
      conf_noisy            — довіра піків corr_noisy: peak / σ_фон
      snr_time_exact_db     — точний SNR у часі з використанням clean_signal
      snr_freq_exact_db     — точний SNR у частоті (Парсеваль)
      snr_time_blind_db     — «сліпа» оцінка SNR у часі за MAD високочастотного залишку noisy_signal
      snr_corr_blind_db     — «сліпа» метрика: 20*log10(peak/σ_фон) для corr_noisy
      meta                  — допоміжні параметри: N, K, top_k, min_distance, confidence_radius
    Вимоги:
      1D масиви, len(template)=K ≤ N=len(signal). std(template) > 0.
    """
    # --- перевірки
    if template.ndim != 1 or clean_signal.ndim != 1 or noisy_signal.ndim != 1:
        raise ValueError("Усі вектори мають бути 1D.")
    if clean_signal.shape != noisy_signal.shape:
        raise ValueError("clean_signal і noisy_signal мають однакову довжину.")
    K = int(template.size); N = int(clean_signal.size)
    if K < 1 or K > N:
        raise ValueError("Некоректна довжина шаблону відносно сигналу.")
    if not (np.isfinite(template).all() and np.isfinite(clean_signal).all() and np.isfinite(noisy_signal).all()):
        raise ValueError("Вхід містить NaN/Inf.")
    if np.std(template) == 0:
        raise ValueError("std(template) == 0.")

    # --- типи
    t = template.astype(np.float64, copy=False)
    x_clean = clean_signal.astype(np.float64, copy=False)
    x_noisy = noisy_signal.astype(np.float64, copy=False)
    eps = 1e-12

    # --- NCC (valid)
    def _ncc(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        s = np.convolve(x, t[::-1], mode='valid')                # сумарний добуток
        # ковзні статистики для x
        cx = np.concatenate(([0.0], np.cumsum(x)))
        cx2 = np.concatenate(([0.0], np.cumsum(x*x)))
        win_sum = cx[K:] - cx[:-K]
        win_sum2 = cx2[K:] - cx2[:-K]
        mux = win_sum / K
        ex2 = win_sum2 / K
        varx = np.maximum(ex2 - mux*mux, eps)
        sigx = np.sqrt(varx)
        # шаблон
        mut = float(np.mean(t))
        sigt = float(np.std(t) + eps)
        return (s - K*mux*mut) / (sigx*sigt)

    corr_clean = _ncc(x_clean, t)
    corr_noisy = _ncc(x_noisy, t)

    # --- піки: greedy NMS з фіксованими параметрами
    top_k = 5
    min_distance = max(1, K//2)
    def _top_peaks(arr: np.ndarray, k: int, d: int) -> np.ndarray:
        work = arr.copy()
        peaks = []
        neg_inf = -np.inf
        for _ in range(k):
            idx = int(np.argmax(work))
            if not np.isfinite(work[idx]):
                break
            peaks.append(idx)
            L = max(0, idx-(d-1)); R = min(work.size, idx+d)
            work[L:R] = neg_inf
        return np.array(peaks, dtype=int)

    peaks_clean_idx = _top_peaks(corr_clean, top_k, min_distance)
    peaks_noisy_idx = _top_peaks(corr_noisy, top_k, min_distance)

    # --- довіра: peak / σ_фон
    confidence_radius = 2*K
    def _conf(corr_arr: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        if peaks.size == 0:
            return np.empty(0)
        mask = np.ones_like(corr_arr, dtype=bool)
        for p in peaks:
            L = max(0, p - confidence_radius)
            R = min(corr_arr.size, p + confidence_radius + 1)
            mask[L:R] = False
        bg = corr_arr[mask]
        sigma_bg = float(max(np.std(bg) if bg.size >= 8 else np.std(corr_arr), eps))
        return np.asarray([float(corr_arr[p]) / sigma_bg for p in peaks])

    conf_clean = _conf(corr_clean, peaks_clean_idx)
    conf_noisy = _conf(corr_noisy, peaks_noisy_idx)

    # --- точний SNR (clean відомий)
    n = x_noisy - x_clean
    ps = float(np.mean(x_clean*x_clean))
    pn = float(np.mean(n*n))
    snr_time_exact_db = 10.0*np.log10(max(ps, eps)/max(pn, eps))
    Xc = np.fft.fft(x_clean); Xn = np.fft.fft(n)
    P_sig = (1.0/N)*np.sum(np.abs(Xc)**2)
    P_noise = (1.0/N)*np.sum(np.abs(Xn)**2)
    snr_freq_exact_db = 10.0*np.log10(max(P_sig, eps)/max(P_noise, eps))

    # --- «сліпі» оцінки шуму
    # 1) MAD високочастотного залишку
    w = max(3, (K | 1))  # непарне вікно ~ K
    k = w//2
    pad = np.pad(x_noisy, (k, k), mode='edge')
    ma = np.convolve(pad, np.ones(w)/w, mode='valid')
    r = x_noisy - ma
    med = float(np.median(r))
    sigma_mad = 1.4826*float(np.median(np.abs(r - med)))
    snr_time_blind_db = 10.0*np.log10(
        max(np.mean(x_noisy*x_noisy) - sigma_mad**2, eps) / max(sigma_mad**2, eps)
    )

    # 2) кореляційна «SNR-подібна» метрика
    # беремо найбільший пік corr_noisy та σ фону corr_noisy
    if peaks_noisy_idx.size:
        # σ фону вже оцінювали в _conf; перерахуємо один раз глобально
        mask = np.ones_like(corr_noisy, dtype=bool)
        for p in peaks_noisy_idx:
            L = max(0, p - confidence_radius)
            R = min(corr_noisy.size, p + confidence_radius + 1)
            mask[L:R] = False
        bg = corr_noisy[mask]
        sigma_bg = float(max(np.std(bg) if bg.size >= 8 else np.std(corr_noisy), eps))
        peak_val = float(corr_noisy[peaks_noisy_idx[0]])
        snr_corr_blind_db = 20.0*np.log10(max(peak_val, eps)/sigma_bg)
    else:
        snr_corr_blind_db = -np.inf

    return {
        "corr_clean": corr_clean,
        "corr_noisy": corr_noisy,
        "peaks_clean_idx": peaks_clean_idx,
        "peaks_noisy_idx": peaks_noisy_idx,
        "conf_clean": conf_clean,
        "conf_noisy": conf_noisy,
        "snr_time_exact_db": snr_time_exact_db,
        "snr_freq_exact_db": snr_freq_exact_db,
        "snr_time_blind_db": snr_time_blind_db,
        "snr_corr_blind_db": snr_corr_blind_db,
        "meta": {
            "N": N, "K": K,
            "top_k": top_k,
            "min_distance": min_distance,
            "confidence_radius": confidence_radius,
        }
    }


def plot_lab_detect_detailed(result: dict):
    """
    Візуалізує результати lab_detect() для навчального аналізу.
    Три сабплоти:
        1. Кореляція чистого сигналу.
        2. Кореляція зашумленого сигналу.
        3. Порівняння (різниця або відношення) між ними.
    Виводить також коротке резюме SNR та довіри.
    """
    corr_clean = result["corr_clean"]
    corr_noisy = result["corr_noisy"]
    peaks_clean = result["peaks_clean_idx"]
    peaks_noisy = result["peaks_noisy_idx"]
    conf_clean = np.array(result["conf_clean"])
    conf_noisy = np.array(result["conf_noisy"])

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --- 1. Clean correlation ---
    axs[0].plot(corr_clean, color='green', lw=1.0, label='Clean correlation')
    axs[0].scatter(peaks_clean, corr_clean[peaks_clean], color='red', marker='o', label='Detected peaks')
    axs[0].set_title("Normalized cross-correlation — CLEAN signal")
    axs[0].set_ylabel("Correlation")
    axs[0].grid(True, lw=0.3)
    axs[0].legend(loc='upper right', fontsize=8)

    # --- 2. Noisy correlation ---
    axs[1].plot(corr_noisy, color='gray', lw=1.0, label='Noisy correlation')
    axs[1].scatter(peaks_noisy, corr_noisy[peaks_noisy], color='red', marker='o', label='Detected peaks')
    axs[1].set_title("Normalized cross-correlation — NOISY signal")
    axs[1].set_ylabel("Correlation")
    axs[1].grid(True, lw=0.3)
    axs[1].legend(loc='upper right', fontsize=8)

    # --- 3. Difference/ratio plot ---
    diff = corr_noisy - corr_clean
    ratio = np.divide(corr_noisy, corr_clean, out=np.zeros_like(corr_noisy), where=corr_clean != 0)
    axs[2].plot(diff, color='blue', lw=0.8, label='Difference (noisy - clean)')
    axs[2].plot(ratio, color='orange', lw=0.8, label='Ratio (noisy / clean)', alpha=0.7)
    axs[2].set_title("Effect of noise: difference and ratio")
    axs[2].set_xlabel("Shift index")
    axs[2].set_ylabel("Δ / ratio")
    axs[2].grid(True, lw=0.3)
    axs[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

    # --- Summary metrics ---
    print("=== SUMMARY ===")
    print(f"SNR exact (time): {result['snr_time_exact_db']:.2f} dB")
    print(f"SNR exact (freq): {result['snr_freq_exact_db']:.2f} dB")
    print(f"SNR blind (time): {result['snr_time_blind_db']:.2f} dB")
    print(f"SNR blind (corr): {result['snr_corr_blind_db']:.2f} dB")
    print(f"Mean confidence clean: {np.mean(conf_clean):.3f}")
    print(f"Mean confidence noisy: {np.mean(conf_noisy):.3f}")
    print(f"Confidence drop: {np.mean(conf_clean) - np.mean(conf_noisy):.3f}")


from typing import Tuple
import numpy as np

def gen_sign(
    SYNC_LN: int = 24,
    M: int = 120,                 # сумарно M ліворуч і M праворуч від sync
    SUPER_FRAME_LN: int = 6,      # бурстів у суперкадрі; лише перший містить sync
    SF_COUNT: int = 4,            # кількість суперкадрів
    GUARD: float = 0.2,           # частка від довжини бурста у відліках
    SPS: int = 10,                # відліків на символ
    SIGN_STD: float = 1.0,        # амплітуда сигналу
    seed_sync: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерує сигнал із суперкадрами та бурстами однакової довжини.
    BURST_LEN_sym = 2*M + SYNC_LN  (у символах) — для всіх бурстів.
    Лише перший бурст кожного суперкадра містить SYNC, решта мають випадковий блок такої самої довжини.

    Структура бурста:
        1-й у суперкадрі: [MSG_1 (M)][SYNC (SYNC_LN)][MSG_2 (M)] + GUARD
        інші:              [MSG_1 (M)][RANDOM_BLOCK (SYNC_LN)][MSG_2 (M)] + GUARD

    Повертає:
        signal : np.ndarray (float32) — часовий ряд усіх суперкадрів
        sync   : np.ndarray (float32) — еталонний SYNC у відліках (довжина SYNC_LN*SPS)
    """
   # --- фіксована синхра ---
    np.random.seed(seed_sync)
    sync = np.random.uniform(-SIGN_STD, SIGN_STD, SYNC_LN * SPS).astype(np.float32)

    # --- рандомний seed для решти сигналу ---
    np.random.seed(None)

    BURST_LEN = (2*M + SYNC_LN) * SPS
    GUARD_LEN = int(round(GUARD * BURST_LEN))
    TOTAL_BURSTS = SUPER_FRAME_LN * SF_COUNT

    N = TOTAL_BURSTS * (BURST_LEN + GUARD_LEN)
    signal = np.random.uniform(-SIGN_STD, SIGN_STD, N).astype(np.float32)

    burst_step = BURST_LEN + GUARD_LEN
    for sf in range(SF_COUNT):
        for b in range(SUPER_FRAME_LN):
            base = (sf * SUPER_FRAME_LN + b) * burst_step
            mid_start = base + M * SPS
            mid_end = mid_start + SYNC_LN * SPS

            if b == 0:
                signal[mid_start:mid_end] = sync

            # guard зона
            guard_start = base + BURST_LEN
            guard_end = guard_start + GUARD_LEN
            signal[guard_start:guard_end] = 0.0

    return signal, sync


def plot_signal_structure(signal: np.ndarray, SYNC_LN: int, M: int, SUPER_FRAME_LN: int,
                          SF_COUNT: int, SPS: int, GUARD: float,
                          title_suffix: str = "",
                          sync_positions_only: bool = True, block: bool = True) -> None:
    """
    Візуалізує часову структуру сигналу, згенерованого gen_sign().

    Аргументи:
        signal — повний сигнал
        SYNC_LN, M, SUPER_FRAME_LN, SF_COUNT, SPS, GUARD — параметри генерації
        sync_positions_only — якщо True, показує лише перші бурсти з SYNC у кожному суперкадрі
    """
    BURST_LEN = (2*M + SYNC_LN) * SPS
    GUARD_LEN = int(round(GUARD * BURST_LEN))
    burst_step = BURST_LEN + GUARD_LEN

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, lw=0.6, color='gray')
    title = "Signal structure overview. " + title_suffix
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")

    for sf in range(SF_COUNT):
        for b in range(SUPER_FRAME_LN):
            base = (sf * SUPER_FRAME_LN + b) * burst_step
            mid_start = base + M * SPS
            mid_end = mid_start + SYNC_LN * SPS

            color = 'red' if b == 0 else 'blue'
            alpha = 0.5 if b == 0 else 0.15
            if not sync_positions_only or b == 0:
                ax.axvspan(mid_start, mid_end, color=color, alpha=alpha)

    ax.grid(True, lw=0.3)
    plt.tight_layout()


    def _on_key(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    plt.show(block=block)


def detect_sync_positions(signal_in: np.ndarray, sync: np.ndarray, min_distance: int, top: int=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        signal_in (np.ndarray): _description_
        sync (np.ndarray): _description_
        min_distance (int): 
        top (int, optional): Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            peaks, vals, corr = detect_sync_positions(...)
    """
    corr = np.correlate(signal_in, sync, mode='valid')
    peaks = []
    vals = []
    work = corr.copy()
    neg_inf = -np.inf
    # select top peaks with non-maximum suppression
    for _ in range(top): 
        i = int(np.argmax(work))
        if not np.isfinite(work[i]):
            break
        peaks.append(i)
        vals.append(corr[i])
        L = max(0, i - min_distance)
        R = min(work.size, i + min_distance)
        work[L:R] = neg_inf
    return np.array(peaks), np.array(vals), corr


def plot_cross_corr(peaks:np.ndarray, vals:np.ndarray, corr:np.ndarray, title: str = "Cross-correlation with sync pattern", block:bool = False) -> None:
    plt.figure(figsize=(12,4))
    plt.plot(corr, lw=0.8, color='gray', label="Correlation")
    plt.scatter(peaks, vals, color='red', s=20, label='Detected peaks')
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Correlation value")
    plt.legend()
    plt.grid(True, lw=0.3)

    def _on_key(event):
        if event.key == 'escape':
            plt.close(event.canvas.figure)
    plt.gcf().canvas.mpl_connect('key_press_event', _on_key)
    plt.show(block=block)
        
@dataclass
class CorrDetectResult:
    peaks_idx: np.ndarray          # (M,) індекси піків у кореляції
    peaks_score: np.ndarray        # (M,) значення NCC у піках ∈ [-1, 1]
    confidence: np.ndarray         # (M,) peak_value / σ_фону
    decision: np.ndarray           # (M,) True/False за порогом tau
    corr: Optional[np.ndarray]     # (N-K+1,), опціонально: весь масив NCC
    snr_time_db: Optional[float]   # SNR часу, якщо передано clean
    snr_freq_db: Optional[float]   # SNR частоти, якщо передано clean
    meta: dict                     # допоміжні поля: N, K, tau, radius, ...


