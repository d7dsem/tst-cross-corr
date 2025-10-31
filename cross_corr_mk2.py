# cross_corr_mk2.py

# Standard Library Imports
from ast import Tuple
import os, sys
import traceback

from dataclasses import dataclass
from typing import List, Optional

# Third Party Imports
import numpy as np
import matplotlib.pyplot as plt


# Local Imports 
from common import (gen_sign, lab_detect, noise_signal, plot_lab_detect_detailed, trace_prefix, init_rand_subseq)
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
    try:
        
        sign, sync = gen_sign()
        # ----- Параметри експерименту
        N = 1024
        K = 64
        SIG_MEAN = 0.0
        SIG_STD = 50.0
        INSERT_POS = [64, 256, 512]
        # ----- Підготовка даних
        clean_signal = np.zeros(N, dtype=np.float64)
        template = np.zeros(K, dtype=np.float64)
        # 1) випадковий фон і шаблон
        init_rand_subseq(clean_signal, 0, N, SIG_MEAN - SIG_STD, SIG_MEAN + SIG_STD, seed=24)
        init_rand_subseq(template, 0, K, SIG_MEAN - SIG_STD, SIG_MEAN + SIG_STD, seed=42)
        # 2) вставки шаблону в чистий сигнал
        for i in INSERT_POS:
            clean_signal[i:i+K] = template

        # розрахунок шуму за бажаним рівнем SNR
        TARGET_SNR_DB = 2  
        signal_power = np.mean(clean_signal**2)
        noise_power = signal_power / (10 ** (TARGET_SNR_DB / 10))
        noise_amp = np.sqrt(3 * noise_power)  # амплітуда рівномірного шуму
        NOISE_STD = noise_amp  # використовується для генерації шуму


        # 3) формування зашумленого сигналу
        noisy_signal = clean_signal.copy()
        noise_signal(noisy_signal, 0, N, -NOISE_STD, NOISE_STD, seed=24)

        # ----- Детекція і метрики
        res = lab_detect(template, clean_signal, noisy_signal)

        # ----- Вивід ключових результатів
        print(f"{INFO} Параметри: N={N}, K={K}, NOISE_STD={NOISE_STD}{RESET}")
        print(f"{INFO} Очікувані позиції вставок: {BRIGHT_YELLOW}{INSERT_POS}{RESET}")

        peaks_clean = res["peaks_clean_idx"].tolist()
        peaks_noisy = res["peaks_noisy_idx"].tolist()
        print(f"{INFO} Піки кореляції CLEAN: {BRIGHT_GREEN}{peaks_clean}{RESET}")
        print(f"{INFO} Піки кореляції NOISY: {BRIGHT_CYAN}{peaks_noisy}{RESET}")

        # Похибка по найближчому піку для кожної очікуваної позиції
        def nearest_deltas(expected, detected):
            if len(detected) == 0:
                return [None for _ in expected]
            d = []
            det = np.array(detected)
            for e in expected:
                d.append(int(np.min(np.abs(det - e))))
            return d

        deltas_clean = nearest_deltas(INSERT_POS, peaks_clean)
        deltas_noisy = nearest_deltas(INSERT_POS, peaks_noisy)
        print(f"{INFO} |Δ| до найближчих піків CLEAN: {BRIGHT_GREEN}{deltas_clean}{RESET}")
        print(f"{INFO} |Δ| до найближчих піків NOISY: {BRIGHT_CYAN}{deltas_noisy}{RESET}")

        # SNR-зведення
        print(f"{INFO} SNR exact (time): {BRIGHT_CYAN}{res['snr_time_exact_db']:.2f} dB{RESET}")
        print(f"{INFO} SNR exact (freq): {BRIGHT_CYAN}{res['snr_freq_exact_db']:.2f} dB{RESET}")
        print(f"{INFO} SNR blind (time, MAD): {BRIGHT_YELLOW}{res['snr_time_blind_db']:.2f} dB{RESET}")
        print(f"{INFO} SNR blind (corr-метрика): {BRIGHT_YELLOW}{res['snr_corr_blind_db']:.2f} dB{RESET}")

        # ----- Візуалізація
        plot_lab_detect_detailed(res)

    except Exception:
        print(f"{ERR} An error occurred:{RESET}")
        traceback.print_exc()
        sys.exit(1)

