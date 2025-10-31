# cross_corr_mk1.py

# Standard Library Imports
import os, sys
import traceback

# Third Party Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Imports 
from common import trace_prefix, init_rand_subseq, noise_signal
# Color Codes and Logging Tags
from common import (
    GREEN, BRIGHT_GREEN, RED, BRIGHT_RED, YELLOW, BRIGHT_YELLOW,
    GRAY, BRIGHT_GRAY, CYAN, BRIGHT_CYAN, BLUE, BRIGHT_BLUE, MAGENTA,
    BRIGHT_MAGENTA, WHITE, BRIGHT_WHITE, BLACK, BRIGHT_BLACK, 
    RESET,
    WARN, ERR, INFO, DBG, OK
)


def ncc(x: np.ndarray, t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalized cross-correlation (valid mode).
    NCC[k] = sum_i (x[k+i]-mux_k)*(t[i]-mut) / (sigx_k*sigt)
    Returns array of length len(x)-len(t)+1
    """
    K = len(t)
    if K < 1 or len(x) < K:
        raise ValueError("Template longer than signal or empty.")

    # raw correlation
    s = np.convolve(x, t[::-1], mode='valid')  # sum x*rev(t)

    # running stats for x
    onesK = np.ones(K, dtype=x.dtype)
    mux = np.convolve(x, onesK, mode='valid') / K
    x2 = x * x
    ex2 = np.convolve(x2, onesK, mode='valid') / K
    varx = np.maximum(ex2 - mux * mux, eps)
    sigx = np.sqrt(varx)

    # template stats
    mut = float(np.mean(t))
    sigt = float(np.std(t) + eps)

    # center correlation and normalize
    num = s - K * mux * mut
    den = sigx * sigt
    return num / den


def top_peaks(corr: np.ndarray, top_k: int = 5, min_distance: int = 1) -> np.ndarray:
    """
    Greedy non-maximum suppression to pick top_k peak indices in 'corr'
    with minimum index spacing 'min_distance'.
    """
    if min_distance < 1:
        min_distance = 1
    corr_work = corr.copy()
    peaks = []
    neg_inf = -np.inf

    for _ in range(top_k):
        idx = int(np.argmax(corr_work))
        if not np.isfinite(corr_work[idx]):
            break
        peaks.append(idx)
        # suppress neighborhood
        left = max(0, idx - (min_distance - 1))
        right = min(len(corr_work), idx + min_distance)
        corr_work[left:right] = neg_inf

    return np.array(peaks, dtype=int)


# =========================
# Main
# =========================
if __name__ == "__main__":
    os.system('')  # Enables ANSI escape characters in terminal (Windows)
    try:
        N: int = 1024
        K: int = 64
        signal: np.ndarray = np.zeros(N, dtype=np.float32)
        sample_seq: np.ndarray = np.zeros(K, dtype=np.float32)

        SIG_MEAN: float = 0.0
        SIG_STD: float = 50.0
        NOISE_STD: float = 10.0

        # init clean background and template
        init_rand_subseq(signal, 0, signal.size, SIG_MEAN - SIG_STD, SIG_MEAN + SIG_STD, seed=24)
        init_rand_subseq(sample_seq, 0, K, SIG_MEAN - SIG_STD, SIG_MEAN + SIG_STD, seed=42)

        # insert template periodically
        insert_positions = [64, 256, 512]
        for i in insert_positions:
            signal[i:i + K] = sample_seq

        # keep a clean copy before noise
        clean = signal.copy()

        # ---------- Correlation on clean ----------
        corr_clean = ncc(clean.astype(np.float64), sample_seq.astype(np.float64))
        peaks_clean = top_peaks(corr_clean, top_k=5, min_distance=K // 2)
        print(f"{INFO} Cross-correlation peak indices {YELLOW}CLEAN{RESET}:\n{peaks_clean}")

        # ---------- Add noise ----------
        noise_signal(signal, 0, signal.size, -NOISE_STD, NOISE_STD, seed=24)
        noisy = signal
        n = noisy - clean

        # ---------- SNR (time domain) ----------
        snr_time = 10.0 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(n.astype(np.float64) ** 2))
        print(f"{INFO} Estimated SNR after adding noise (time): {BRIGHT_CYAN}{snr_time:.2f} dB{RESET}")

        # ---------- SNR (frequency domain consistent with Parseval) ----------
        Xc = np.fft.fft(clean.astype(np.float64))
        Xn = np.fft.fft(n.astype(np.float64))
        Nlen = len(clean)
        P_sig = (1.0 / Nlen) * np.sum(np.abs(Xc) ** 2)
        P_noise = (1.0 / Nlen) * np.sum(np.abs(Xn) ** 2)
        snr_freq = 10.0 * np.log10(P_sig / P_noise)
        print(f"{INFO} Estimated SNR (frequency): {BRIGHT_CYAN}{snr_freq:.2f} dB{RESET}")

        # ---------- Correlation on noisy ----------
        corr_noisy = ncc(noisy.astype(np.float64), sample_seq.astype(np.float64))
        peaks_noisy = top_peaks(corr_noisy, top_k=5, min_distance=K // 2)
        print(f"{INFO} Cross-correlation peak indices {YELLOW}NOISED{RESET}:\n{peaks_noisy}")

        # optional: show top-3 with scores
        order = np.argsort(corr_noisy[peaks_noisy])[-3:][::-1]
        top3 = peaks_noisy[order]
        scores = corr_noisy[top3]
        print(f"{INFO} Top-3 peaks (index:score): {BRIGHT_GREEN}{list(zip(top3.tolist(), np.round(scores, 4).tolist()))}{RESET}")

        # expected vs detected simple check
        expected = np.array(insert_positions, dtype=int)
        # for each expected, find nearest detected
        nearest = [int(peaks_noisy[np.argmin(np.abs(peaks_noisy - e))]) if len(peaks_noisy) else None for e in expected]
        deltas = [abs(n - e) if n is not None else None for n, e in zip(nearest, expected)]
        print(f"{INFO} Expected positions: {expected}")
        print(f"{INFO} Nearest detected:  {nearest}")
        print(f"{INFO} |Δ| indices:       {deltas}")


        plt.figure(figsize=(10, 4))
        plt.plot(corr_noisy, color='gray', lw=1.0, label='Cross-correlation')
        plt.scatter(peaks_noisy, corr_noisy[peaks_noisy], color='red', marker='o', label='Detected peaks')
        for e in insert_positions:
            plt.axvline(e, color='blue', ls='--', lw=0.8, label='Expected' if e == insert_positions[0] else "")
        plt.title("Normalized cross-correlation with detected peaks")
        plt.xlabel("Signal index (shift)")
        plt.ylabel("Correlation coefficient")
        plt.legend()
        plt.grid(True, lw=0.3)
        plt.tight_layout()
        plt.show()

    except Exception:
        print(f"{ERR} An error occurred:{RESET}")
        traceback.print_exc()
        sys.exit(1)
