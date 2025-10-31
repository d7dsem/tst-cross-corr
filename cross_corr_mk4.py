# cross_corr_mk3.py

# Standard Library Imports
from typing import Tuple
import os, sys
import traceback

from dataclasses import dataclass
from typing import List, Optional

# Third Party Imports
import numpy as np
import matplotlib.pyplot as plt


# Local Imports 
from common import (gen_sign, plot_cross_corr, plot_signal_structure, print_corruption_info, trace_prefix)
# Color Codes and Logging Tags
from common import (
    GREEN, BRIGHT_GREEN, RED, BRIGHT_RED, YELLOW, BRIGHT_YELLOW,
    GRAY, BRIGHT_GRAY, CYAN, BRIGHT_CYAN, BLUE, BRIGHT_BLUE, MAGENTA,
    BRIGHT_MAGENTA, WHITE, BRIGHT_WHITE, BLACK, BRIGHT_BLACK, 
    RESET,
    WARN, ERR, INFO, DBG, OK
)


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
    out = np.correlate(signal_chunk, srch_seq, mode='valid')
    # explicit copy to the provided buffer
    corr[:] = out


def detect_positions(corr: np.ndarray, chunk_start_idx: int, min_distance: int, z_thr: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      peaks_abs: np.ndarray[int]  # absolute indices in the original signal (start of sync)
      vals:      np.ndarray[float]# correlation values at detected peaks
    Deterministic, thresholded, NMS with radius = min_distance.
    """
    assert np.isfinite(corr).all(), "NaN/Inf in `corr`."
    med = np.median(corr)
    mad = np.median(np.abs(corr - med))
    sigma = 1.4826 * mad if mad > 0 else 1e-12
    thr = med + z_thr * sigma

    work = corr.copy()
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
        vals.append(corr[i])
        L = max(0, i - min_distance)
        R = min(n, i + min_distance)
        work[L:R] = neg_inf

    return np.asarray(peaks_abs, dtype=int), np.asarray(vals, dtype=float)


if __name__ == "__main__":
    os.system('')  # Enables ANSI escape characters in terminal (Windows)
    print(f"\n{INFO} Starting cross-correlation sync detection mk3...")
    try:
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
        SIGN_LN = len(signal)             # довжина повного сигналу у семплах

        MIN_DISTANCE = SYNC_LN_SAM + 2 * M * SPS  # відстань між піками в семплах
        corr = np.zeros(SIGN_LN - SYNC_LN_SAM + 1)  # буфер для кореляції (mode='valid')

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

        print(f"{INFO} Generated signal length: {GREEN}{SIGN_LN}{RESET}. Sync length: {GREEN}{SYNC_LN}{RESET}."
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