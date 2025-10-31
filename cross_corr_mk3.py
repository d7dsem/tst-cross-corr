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
from common import (detect_sync_positions, gen_sign, lab_detect, noise_signal, plot_cross_corr, plot_lab_detect_detailed, plot_signal_structure, trace_prefix, init_rand_subseq)
# Color Codes and Logging Tags
from common import (
    GREEN, BRIGHT_GREEN, RED, BRIGHT_RED, YELLOW, BRIGHT_YELLOW,
    GRAY, BRIGHT_GRAY, CYAN, BRIGHT_CYAN, BLUE, BRIGHT_BLUE, MAGENTA,
    BRIGHT_MAGENTA, WHITE, BRIGHT_WHITE, BLACK, BRIGHT_BLACK, 
    RESET,
    WARN, ERR, INFO, DBG, OK
)



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
        SF_COUNT: int = 4           # кількість суперкадрів
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

        signal, sync, _ = gen_sign(SYNC_LN=SYNC_LN,M=M,SUPER_FRAME_LN=SUPER_FRAME_LN,SF_COUNT=SF_COUNT,GUARD=GUARD,SPS=SPS,SIGN_STD=SIGN_STD,seed_sync=SEED_SYNC)
        if 0: plot_signal_structure(signal=signal,SYNC_LN=SYNC_LN,M=M,SUPER_FRAME_LN=SUPER_FRAME_LN,SF_COUNT=SF_COUNT,SPS=SPS,GUARD=GUARD,sync_positions_only=True, title_suffix="[clean signal]", block=False)


        TARGET_SNR_DB: float =0.1
        TARGET_SNR_DB: float =10.0
        TARGET_SNR_DB: float =.01

        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(TARGET_SNR_DB / 10))
        noise_std = np.sqrt(noise_power)
        
        np.random.seed(0)
        noise = np.random.normal(0, noise_std, size=signal.shape)
        signal_noisy = signal + noise
        snr_actual = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
        
        print(f"{INFO} Sign len: {GREEN}{len(signal_noisy)}{RESET}. Signal power: {CYAN}{signal_power:.2f}{RESET}. SING_STD: {CYAN}{SIGN_STD}{RESET}.")
        print(f"      Target SNR: {CYAN}{TARGET_SNR_DB:.2f}{RESET} dB, actual {CYAN}{snr_actual:.2f}{RESET} dB.  Noise power: {CYAN}{noise_power:.2f}{RESET}. Noise std: {CYAN}{noise_std:.2f}{RESET}")
       
        # Additional SNR levels for analysis
        if 0:
            for desired_snr_db in [0.1, 10.0, 20.0]:
                _noise_power = signal_power / (10**(desired_snr_db / 10))
                _noise_std = np.sqrt(_noise_power)
                print(f"  Desired SNR: {CYAN}{desired_snr_db:>5}{RESET} dB, noise std: {CYAN}{_noise_std:>8.3f}{RESET}  signal/noise power ratio: {CYAN}{signal_power:.1f}{RESET}/{CYAN}{_noise_power:.1f}{RESET} = {CYAN}{signal_power/_noise_power:>8.3f}{RESET}")
                plot_signal_structure(signal=signal_noisy,SYNC_LN=SYNC_LN,M=M,SUPER_FRAME_LN=SUPER_FRAME_LN,SF_COUNT=SF_COUNT,SPS=SPS,GUARD=GUARD,sync_positions_only=True, title_suffix=f"[noised {desired_snr_db} dB]", block=False)

        if 0: plot_signal_structure(signal=signal_noisy,SYNC_LN=SYNC_LN,M=M,SUPER_FRAME_LN=SUPER_FRAME_LN,SF_COUNT=SF_COUNT,SPS=SPS,GUARD=GUARD,sync_positions_only=True, title_suffix=f"[noised {TARGET_SNR_DB} dB]", block=False)

        top = 5
        if 1:
            print(f"{INFO} Sync ampl reduction tests:")
            for r, reduce in enumerate([10, 25, 50, 100, 1000],1):
                sync_reduced = sync / reduce
                peaks, vals, corr = detect_sync_positions(signal_in=signal_noisy, sync=sync_reduced, min_distance=SPS*(SYNC_LN+2*M), top=top)
                
                all_findings_correct = all([p in peaks  for p in expected_positions])
                msg_color = BRIGHT_GREEN if all_findings_correct else BRIGHT_RED
                msg = f"Found {msg_color}all{RESET}" if all_findings_correct else f"{msg_color}Some missing/wrong{RESET}"
                print(f"  {r:2}  reduction {CYAN}{reduce}{RESET}. {msg}.")
                if not all_findings_correct:
                    print(f"    Detected peaks positions:")
                    for i, (p, v) in enumerate(zip(peaks, vals), 1):
                        peak_color = BRIGHT_GREEN if p in expected_positions else BRIGHT_RED
                        print(f"      { i:2}: idx={peak_color}{p:>8_d}{RESET}, val={v:_.2f} ")
                    plot_cross_corr(peaks, vals, corr, title=f"Cross-correlation with noisy [snr={TARGET_SNR_DB} dB] signal. Sync reducted by {reduce}.", block=True)


        if 0:
            reduce = 1
            sync_reduced = sync / reduce
            peaks, vals, corr = detect_sync_positions(signal_in=signal_noisy, sync=sync_reduced, min_distance=SPS*(SYNC_LN+2*M), top=top)
            plot_cross_corr(peaks, vals, corr, title=f"Cross-correlation with noisy [snr={TARGET_SNR_DB} dB] signal", block=False)
            if 0: plot_signal_structure(signal=signal_noisy,SYNC_LN=SYNC_LN,M=M,SUPER_FRAME_LN=SUPER_FRAME_LN,SF_COUNT=SF_COUNT,SPS=SPS,GUARD=GUARD,sync_positions_only=True, title_suffix=f"[noised {TARGET_SNR_DB} dB, sync reduced x{reduce}]", block=False)


            print(f"{INFO} Corr length: {GREEN}{len(corr)}{RESET}. Peaks found: {GREEN}{len(peaks)}{RESET} {GRAY}{top=}{RESET}.")
            print(f"{INFO} Sync reduction {CYAN}{reduce}{RESET} (sync amplitude multiplied by this factor)")
            if 0:
                print(f"{INFO} Expected sync positions:")
                for i, pos in enumerate(expected_positions, 1):
                    print(f"  {i:2}  pos={GREEN}{pos:>8_d}{RESET} ")

            if 0:
                print(f"{INFO} Detected peaks positions:")
                for i, (p, v) in enumerate(zip(peaks, vals), 1):
                    peak_color = BRIGHT_GREEN if p in expected_positions else BRIGHT_RED
                    print(f"{ i:2}: idx={peak_color}{p:>8_d}{RESET}, val={v:_.2f} ")


        print(f"{INFO} Finished.\n")
    except Exception:
        print(f"{ERR} An error occurred:{RESET}")
        traceback.print_exc()
        sys.exit(1)