"""
ETSI TS 102 361-1 V2.6.1 (2023-05)
10.2.2 4FSK generation (p 112)
+-------+-------+--------+-----------------+
| Bit 1 | Bit 0 | Symbol | 4FSK deviation  |
+-------+-------+--------+-----------------+
|   0   |   1   |   +3   |   +1.944 kHz    |
|   0   |   0   |   +1   |   +0.648 kHz    |
|   1   |   0   |   -1   |   -0.648 kHz    |
|   1   |   1   |   -3   |   -1.944 kHz    |
+-------+-------+--------+-----------------+


ETSI TS 102 361-1 V2.6.1 (2023-05)

Table 9.2: SYNC patterns (p91)

BS sourced voice:
Hex: 7 5 5 F D 7 D F 7 5 F 7
Bin: 011101010101111111010111110111110111010111110111

BS sourced data:
Hex: D F F 5 7 D 7 5 D F 5 D
Bin: 110111111111010101110111010101101111110101101101

MS sourced voice:
Hex: 7 F 7 D 5 D D 5 7 D F D
Bin: 011111110111110101011011010101111101111111110101

MS sourced data:
Hex: D 5 D 7 F 7 7 F D 7 5 7
Bin: 110101011101011111110111011111110111010101110111

RC sync:
Hex: 7 7 D 5 5 F 7 D F D 7 7
Bin: 011101111101010101111101111110110111110101110111

TDMA direct mode slot 1 voice:
Hex: 5 D 5 7 7 F 7 7 5 7 F F
Bin: 010111010101011101111111011101110101011111111111

"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FSKSymbol:
    bit1: int
    bit0: int
    symbol: int
    deviation_hz: float


FSK_MAPPING = [
    FSKSymbol(bit1=0, bit0=1, symbol=+3, deviation_hz=+1.944e3),
    FSKSymbol(bit1=0, bit0=0, symbol=+1, deviation_hz=+0.648e3),
    FSKSymbol(bit1=1, bit0=0, symbol=-1, deviation_hz=-0.648e3),
    FSKSymbol(bit1=1, bit0=1, symbol=-3, deviation_hz=-1.944e3),
]

# Для швидкого доступу за бітовою парою:
FSK_LOOKUP = {
    (s.bit1, s.bit0): s for s in FSK_MAPPING
}

def hex_to_bin(hex_str: str) -> str:
    """Convert hex string to binary string"""
    hex_clean = hex_str.replace(" ", "")
    return bin(int(hex_clean, 16))[2:].zfill(len(hex_clean) * 4)

@dataclass
class SYNC:
    full_name: str
    id: str
    bit_seq: str
    hex_seq: str

SYNC_PATTERNS = {
    "BS_Voice": SYNC(
        full_name="Base Station sourced voice",
        id="BS_Voice",
        bit_seq=hex_to_bin("755FD7DF75F7"),
        hex_seq="755FD7DF75F7"
    ),
    "BS_Data": SYNC(
        full_name="Base Station sourced data",
        id="BS_Data",
        bit_seq=hex_to_bin("DFF57D75DF5D"),
        hex_seq="DFF57D75DF5D"
    ),
    "MS_Voice": SYNC(
        full_name="Mobile Station sourced voice",
        id="MS_Voice",
        bit_seq=hex_to_bin("7F7D5DD57DFD"),
        hex_seq="7F7D5DD57DFD"
    ),
    "MS_Data": SYNC(
        full_name="Mobile Station sourced data",
        id="MS_Data",
        bit_seq=hex_to_bin("D5D7F77FD757"),
        hex_seq="D5D7F77FD757"
    ),
    "RC_Sync": SYNC(
        full_name="Repeater Control synchronization",
        id="RC_Sync",
        bit_seq=hex_to_bin("77D55F7DFD77"),
        hex_seq="77D55F7DFD77"
    ),
    "DMO_Slot1_Voice": SYNC(
        full_name="TDMA direct mode slot 1 voice",
        id="DMO_Slot1_Voice",
        bit_seq=hex_to_bin("5D577F7757FF"),
        hex_seq="5D577F7757FF"
    )
}
SYMBOL_RATE = 4800.0  # symbols per second
TS = 1.0 / SYMBOL_RATE
# Dibit -> deviation in Hz: (Bit1, Bit0) per ETSI Table 10.3
DIBIT_TO_DEV_HZ = {
    (0, 1): +1944.0,
    (0, 0): +648.0,
    (1, 0): -648.0,
    (1, 1): -1944.0,
}

def _bits_to_symbols(bit_seq: str):
    """Convert '0101…' (length multiple of 2) to list of per-symbol deviations (Hz)."""
    if len(bit_seq) % 2 != 0:
        raise ValueError("SYNC.bit_seq length must be even (dibits).")
    devs = []
    for i in range(0, len(bit_seq), 2):
        b1 = int(bit_seq[i])
        b0 = int(bit_seq[i+1])
        try:
            devs.append(DIBIT_TO_DEV_HZ[(b1, b0)])
        except KeyError:
            raise ValueError(f"Invalid dibit {(b1, b0)} in bit_seq at pos {i}.")
    return np.asarray(devs, dtype=np.float64)


def generate_iq_samples_G(sync, Fs: float) -> np.ndarray:
    """
    Generate complex baseband IQ samples for a SYNC 4FSK sequence at sample rate Fs.
    GPT version
    Assumptions:
      - DMR 4FSK (4.8 kSym/s), piecewise-constant frequency per symbol (no SRRC shaping).
      - sync.bit_seq: '0'/'1' string, MSB-first, length multiple of 2 (48 для SYNC).
      - Output amplitude = 1.0 (unit envelope), continuous phase.

    Parameters
    ----------
    sync : SYNC
        Object with .bit_seq (str of '0'/'1').
    Fs : float
        Sample rate in samples/second.

    Returns
    -------
    np.ndarray (complex64)
        Complex baseband samples e^{j*phi[n]}.
    """
    if Fs <= 0:
        raise ValueError("Fs must be positive.")
    devs_hz = _bits_to_symbols(sync.bit_seq)  # per-symbol frequency offsets in Hz
    n_symbols = devs_hz.size
    total_time = n_symbols * TS
    total_samples = int(round(total_time * Fs))

    # Рівномірно розподілити кількість семплів по символах (уникаємо накопичення похибки)
    edges = np.round(np.linspace(0, total_samples, n_symbols + 1)).astype(int)
    counts = np.diff(edges)  # samples per symbol, sum == total_samples

    # Генерація з безперервною фазою
    out = np.empty(total_samples, dtype=np.complex64)
    phase = 0.0
    idx = 0
    two_pi_over_Fs = 2.0 * np.pi / Fs
    for sym_dev_hz, nSamp in zip(devs_hz, counts):
        if nSamp == 0:
            continue
        # миттєвий крок фази
        dphi = two_pi_over_Fs * sym_dev_hz
        # лінійний наріст фази по семплах цього символу
        phis = phase + dphi * np.arange(nSamp, dtype=np.float64)
        out[idx:idx+nSamp] = np.exp(1j * phis).astype(np.complex64)
        phase = float(phis[-1] + dphi)  # фаза на наступний семпл (безперервність)
        idx += nSamp

    return out


def generate_iq_samples_C_mk0(sync: SYNC, Fs: float) -> np.ndarray:
    """
    Generate complex IQ samples array at specified sample rate with FSK mapping
    Claude version
    Args:
        sync: SYNC pattern object containing bit sequence
        Fs: Sampling rate in Hz
    
    Returns:
        Complex IQ samples as numpy array
    """
    symbol_rate = 4800.0  # symbols/sec (DMR standard)
    
    # Розбиваємо на дібіти
    bit_seq = sync.bit_seq
    dibits = [(int(bit_seq[i]), int(bit_seq[i+1])) 
              for i in range(0, len(bit_seq), 2)]
    
    n_symbols = len(dibits)
    symbol_duration = 1.0 / symbol_rate
    total_duration = n_symbols * symbol_duration
    total_samples = int(round(total_duration * Fs))
    
    # Рівномірний розподіл семплів (як у GPT версії)
    edges = np.round(np.linspace(0, total_samples, n_symbols + 1)).astype(int)
    samples_per_symbol = np.diff(edges)
    
    # Генерація з безперервною фазою
    iq_samples = np.empty(total_samples, dtype=np.complex64)
    phase = 0.0
    idx = 0
    two_pi_over_Fs = 2.0 * np.pi / Fs
    
    for dibit, n_samples in zip(dibits, samples_per_symbol):
        if n_samples == 0:
            continue
            
        # Отримуємо девіацію
        fsk_symbol = FSK_LOOKUP[dibit]
        freq_deviation = fsk_symbol.deviation_hz
        
        # Крок фази на семпл
        phase_step = two_pi_over_Fs * freq_deviation
        
        # Генеруємо фази для всього символу
        phases = phase + phase_step * np.arange(n_samples, dtype=np.float64)
        iq_samples[idx:idx+n_samples] = np.exp(1j * phases).astype(np.complex64)
        
        # Оновлюємо фазу для безперервності
        phase = float(phases[-1] + phase_step)
        idx += n_samples
    
    return iq_samples


def generate_iq_samples_C(sync: SYNC, Fs: float, apply_rrc: bool = False, 
                          rrc_alpha: float = 0.2, rrc_span: int = 8) -> np.ndarray:
    """
    Generate complex IQ samples array at specified sample rate with FSK mapping
    
    Args:
        sync: SYNC pattern object containing bit sequence
        Fs: Sampling rate in Hz
        apply_rrc: Apply Root Raised Cosine pulse shaping filter
        rrc_alpha: RRC rolloff factor (0.2 typical for DMR)
        rrc_span: Filter span in symbols
    
    Returns:
        Complex IQ samples as numpy array
    """
    symbol_rate = 4800.0  # symbols/sec (DMR standard)
    
    # Розбиваємо на дібіти
    bit_seq = sync.bit_seq
    dibits = [(int(bit_seq[i]), int(bit_seq[i+1])) 
              for i in range(0, len(bit_seq), 2)]
    
    n_symbols = len(dibits)
    symbol_duration = 1.0 / symbol_rate
    total_duration = n_symbols * symbol_duration
    total_samples = int(round(total_duration * Fs))
    
    # Рівномірний розподіл семплів
    edges = np.round(np.linspace(0, total_samples, n_symbols + 1)).astype(int)
    samples_per_symbol = np.diff(edges)
    
    # Генерація з безперервною фазою
    iq_samples = np.empty(total_samples, dtype=np.complex64)
    phase = 0.0
    idx = 0
    two_pi_over_Fs = 2.0 * np.pi / Fs
    
    for dibit, n_samples in zip(dibits, samples_per_symbol):
        if n_samples == 0:
            continue
            
        # Отримуємо девіацію
        fsk_symbol = FSK_LOOKUP[dibit]
        freq_deviation = fsk_symbol.deviation_hz
        
        # Крок фази на семпл
        phase_step = two_pi_over_Fs * freq_deviation
        
        # Генеруємо фази для всього символу
        phases = phase + phase_step * np.arange(n_samples, dtype=np.float64)
        iq_samples[idx:idx+n_samples] = np.exp(1j * phases).astype(np.complex64)
        
        # Оновлюємо фазу для безперервності
        phase = float(phases[-1] + phase_step)
        idx += n_samples
    
    # Apply RRC filter if requested
    if apply_rrc:
        from scipy.signal import firwin, lfilter
        
        sps = int(Fs / symbol_rate)
        
        # Create RRC filter
        # Для CPM/FSK краще використати Gaussian filter замість RRC
        # але спробуємо RRC спочатку
        num_taps = rrc_span * sps + 1
        
        # Simplified RRC (approximation via raised cosine window)
        t = np.arange(-rrc_span*sps//2, rrc_span*sps//2 + 1) / sps
        
        # RRC formula
        h = np.zeros(len(t))
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1 - rrc_alpha + 4*rrc_alpha/np.pi
            elif abs(ti) == 1/(4*rrc_alpha):
                h[i] = (rrc_alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*rrc_alpha)) + 
                                                  (1-2/np.pi)*np.cos(np.pi/(4*rrc_alpha)))
            else:
                numerator = np.sin(np.pi*ti*(1-rrc_alpha)) + 4*rrc_alpha*ti*np.cos(np.pi*ti*(1+rrc_alpha))
                denominator = np.pi*ti*(1-(4*rrc_alpha*ti)**2)
                h[i] = numerator / denominator
        
        # Normalize
        h = h / np.sqrt(np.sum(h**2))
        
        # Apply filter separately to I and Q
        iq_samples_real = lfilter(h, 1, iq_samples.real)
        iq_samples_imag = lfilter(h, 1, iq_samples.imag)
        iq_samples = (iq_samples_real + 1j * iq_samples_imag).astype(np.complex64)
    
    return iq_samples


def tst_sync_template(Fs: float = 48e3, id:str = "MS_Voice"):
    # Додай це в look_for_sync_in_file перед циклом:

    # Генеруємо один шаблон для аналізу
    test_sync = SYNC_PATTERNS[id]
    test_template = generate_iq_samples_C(test_sync, Fs)

    print(f"\n Template generation check:")
    print(f"  Bit sequence length: {len(test_sync.bit_seq)} bits")
    print(f"  Expected symbols: {len(test_sync.bit_seq)//2} (dibits)")
    print(f"  Template samples: {len(test_template)}")
    print(f"  Expected samples: {(len(test_sync.bit_seq)//2) * (Fs/4800):.0f}")
    print(f"  SPS calculated: {len(test_template) / (len(test_sync.bit_seq)//2):.1f}")
    print(f"  SPS expected: {Fs/4800:.1f}")

    # Перевір amplitude
    print(f"  Template power: {np.mean(np.abs(test_template)**2):.3f} (має бути ~1.0)")
    print(f"  Template max: {np.max(np.abs(test_template)):.3f}")

    # Подивись на перші 10 семплів
    print(f"  First 10 samples:")
    for i in range(10):
        print(f"    [{i}] {test_template[i]}")

    # FFT шаблону - bandwidth
    template_fft = np.fft.fftshift(np.fft.fft(test_template))
    template_power_spectrum = np.abs(template_fft)**2
    max_power = np.max(template_power_spectrum)
    # 3dB bandwidth = де спектр падає до половини від піку
    threshold_3db = max_power / 2
    above_threshold = template_power_spectrum > threshold_3db
    template_freqs = np.fft.fftshift(np.fft.fftfreq(len(test_template), 1/Fs))
    bw_indices = np.where(above_threshold)[0]
    if len(bw_indices) > 0:
        bw_hz = template_freqs[bw_indices[-1]] - template_freqs[bw_indices[0]]
        print(f"  Template bandwidth (3dB): {bw_hz/1e3:.2f} kHz")
    else:
        print(f"  Template bandwidth: cannot calculate")
    print()


def verify_template_vs_standard(id:str = "MS_Voice"):
    """Перевірка відповідності шаблону стандарту ETSI"""
    
    Fs = 48000
    symbol_rate = 4800
    
    # Беремо MS_Voice SYNC
    sync = SYNC_PATTERNS[id]
    print(f"\nVerifying {id} SYNC pattern:")
    print(f"  Hex: {sync.hex_seq}")
    print(f"  Bin: {sync.bit_seq}")
    
    # Перевіряємо що hex → bin конвертація правильна
    hex_str = sync.hex_seq.replace(" ", "")
    bin_from_hex = bin(int(hex_str, 16))[2:].zfill(len(sync.bit_seq))
    
    print(f"\n  Verification:")
    print(f"    Stored bin: {sync.bit_seq}")
    print(f"    From hex:   {bin_from_hex}")
    print(f"    Match: {sync.bit_seq == bin_from_hex}")
    
    # Перевіряємо перші кілька dibits вручну
    print(f"\n  First 5 dibits decoding:")
    template = generate_iq_samples_C(sync, Fs)
    
    for i in range(5):
        bit1 = int(sync.bit_seq[i*2])
        bit0 = int(sync.bit_seq[i*2 + 1])
        
        # З таблиці стандарту
        expected_dev = DIBIT_TO_DEV_HZ[(bit1, bit0)]
        
        # З шаблону - обчислюємо instantaneous frequency
        samples_start = i * 10
        samples_end = (i+1) * 10
        symbol_samples = template[samples_start:samples_end]
        
        # Обчислюємо середню частоту через фазу
        phases = np.angle(symbol_samples)
        phase_diff = np.diff(np.unwrap(phases))
        inst_freq = np.mean(phase_diff) * Fs / (2 * np.pi)
        
        print(f"    Symbol {i}: bits=({bit1},{bit0})  "
              f"expected_dev={expected_dev:+7.1f} Hz  "
              f"measured_freq={inst_freq:+7.1f} Hz  "
              f"error={abs(inst_freq - expected_dev):.1f} Hz")
    
    # Перевіряємо FSK_LOOKUP vs DIBIT_TO_DEV_HZ
    print(f"\n  Mapping table comparison:")
    for dibit in [(0,1), (0,0), (1,0), (1,1)]:
        fsk_dev = FSK_LOOKUP[dibit].deviation_hz
        direct_dev = DIBIT_TO_DEV_HZ[dibit]
        match = "✓" if fsk_dev == direct_dev else "✗"
        print(f"    {dibit}: FSK_LOOKUP={fsk_dev:+7.1f}  DIBIT_TO_DEV={direct_dev:+7.1f}  {match}")


def estimate_sync_count_in_chunk(sync_type: str, chunk_ln: int, Fs: float) -> int:
    """
    Estimate expected number of SYNC patterns in a chunk.
    
    Args:
        sync_type: SYNC pattern type (e.g., "MS_Voice", "BS_Data")
        chunk_ln: chunk length in samples
        Fs: sample rate in Hz
    
    Returns:
        Expected number of SYNC patterns
    """
    # DMR timing
    slot_duration = 60e-3  # 60 ms per slot
    
    # SYNC appears every 3 slots (360 ms) in both voice and data
    # This is because SYNC is transmitted in both slots of a frame,
    # and frames are 2 slots (120ms), with SYNC pattern every other frame pair
    sync_period = 3 * slot_duration  # 360 ms = 0.36 sec
    
    # Calculate chunk duration
    chunk_duration = chunk_ln / Fs  # seconds
    
    # Expected SYNC count (for continuous transmission)
    expected_count = int(chunk_duration / sync_period)
    
    return expected_count

def analyze_peak_distances(detections: dict, sync_id: str, Fs: float) -> None:
    """
    Аналіз відстаней між детектованими піками SYNC.
    
    Args:
        detections: результат від search_sync_in_phase_chunk
        sync_id: ID SYNC патерну для аналізу
        Fs: sample rate в Hz
    """
    det = detections[sync_id]
    peaks = det['peaks']
    
    if len(peaks) < 2:
        print(f"\nNot enough peaks for distance analysis (only {len(peaks)} peak)")
        return
    
    # Сортуємо піки по позиції
    peaks_sorted = np.sort(peaks)
    
    # Обчислюємо відстані між сусідніми піками
    distances_samples = np.diff(peaks_sorted)
    distances_ms = distances_samples / Fs * 1000
    
    print(f"\n{sync_id} peaks analysis:")
    print(f"  Total peaks: {len(peaks)}")
    if len(peaks_sorted) > 10:
        print(f"  First 10 positions: {peaks_sorted[:10]}")
    else:
        print(f"  Positions: {peaks_sorted}")
    
    print(f"\n  Distances between consecutive peaks:")
    print(f"    Min:    {np.min(distances_samples):7.0f} samples = {np.min(distances_ms):7.1f} ms")
    print(f"    Max:    {np.max(distances_samples):7.0f} samples = {np.max(distances_ms):7.1f} ms")
    print(f"    Mean:   {np.mean(distances_samples):7.0f} samples = {np.mean(distances_ms):7.1f} ms")
    print(f"    Median: {np.median(distances_samples):7.0f} samples = {np.median(distances_ms):7.1f} ms")
    
    # DMR frame timing для порівняння
    dmr_slot = 60  # ms
    dmr_frame = 120  # ms (2 slots)
    dmr_superframe = 720  # ms (6 frames)
    
    print(f"\n  Expected DMR timings:")
    print(f"    Slot:       {dmr_slot:4d} ms = {dmr_slot * Fs / 1000:7.0f} samples")
    print(f"    Frame:      {dmr_frame:4d} ms = {dmr_frame * Fs / 1000:7.0f} samples")
    print(f"    Superframe: {dmr_superframe:4d} ms = {dmr_superframe * Fs / 1000:7.0f} samples")
    
    # Гістограма відстаней
    print(f"\n  Distance distribution:")
    bins = [0, 3000, 6000, 9000, 15000, 30000, np.inf]
    labels = ['<3k', '3-6k', '6-9k', '9-15k', '15-30k', '>30k']
    hist, _ = np.histogram(distances_samples, bins=bins)
    for label, count in zip(labels, hist):
        if count > 0:
            print(f"    {label:8s}: {count:3d} distances")

