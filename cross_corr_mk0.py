# cross_corr_mk0.py

# Standard Library Imports
import os, sys, time
import traceback

# Third Party Imports
import numpy as np

# Local Imports 
from common import (noise_signal, trace_prefix, init_rand_subseq)
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
        N: int = 1024
        K: int = 64
        signal: np.ndarray = np.zeros(N, dtype=np.float32)
        sample_seq :np.ndarray = np.zeros(K, dtype=np.float32)

        SIG_MEAN: float = 0.0
        SIG_STD: float = 50.0
        NOISE_STD: float = 10.0

        init_rand_subseq(signal, 0, signal.size, SIG_MEAN-SIG_STD, SIG_MEAN+SIG_STD, seed=24)
        init_rand_subseq(sample_seq, 0, K, SIG_MEAN-SIG_STD, SIG_MEAN+SIG_STD, seed=42)

        # Insert sample sequence at multiple locations periodically
        for i in [64, 256, 512]:
            signal[i:i+K] = sample_seq

        # Find cross-correlation peaks on the signal
        corr = np.correlate(signal, sample_seq, mode='valid')   
        peak_indices = np.argsort(corr)[-5:][::-1]  # Top 5 peaks
        print(f"{INFO} Cross-correlation peak indices {YELLOW}CLEAR{RESET}:\n{peak_indices}")

        # Add noise to the signal
        noise_signal(signal, 0, signal.size, -NOISE_STD, NOISE_STD, seed=24)
        # estimate snr in time domain
        signal_power = np.mean(signal ** 2)
        noise_power = NOISE_STD**2 / 3  # Variance of uniform distribution U(-a, a) is a^2/3
        snr = 10 * np.log10(signal_power / noise_power)
        print(f"{INFO} Estimated SNR after adding noise: {BRIGHT_CYAN}{snr:.2f} dB{RESET}")
        
        # estimate snr in frequency domain
        signal_fft = np.fft.fft(signal) 
        freq_bins = np.fft.fftfreq(len(signal))
        signal_power_freq = np.mean(np.abs(signal_fft) ** 2)
        noise_power_freq = noise_power * len(signal)  # Total noise power in frequency domain
        snr_freq = 10 * np.log10(signal_power_freq / noise_power_freq)
        print(f"{INFO} Estimated SNR in frequency domain: {BRIGHT_CYAN}{snr_freq:.2f} dB{RESET}")

        # Find cross-correlation peaks on the signal+noise
        corr = np.correlate(signal, sample_seq, mode='valid')
        peak_indices = np.argsort(corr)[-5:][::-1]  # Top 5 peaks
        print(f"{INFO} Cross-correlation peak indices {YELLOW}NOISED{RESET}:\n{peak_indices}")


    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        sys.exit(1) 