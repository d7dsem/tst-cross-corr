import argparse
from pathlib import Path
import numpy as np

from wav import get_iq_wav_prm

# CLI (база для файлів із сигналами IQ int16 LE wav/bin)
def build_cli(_MODULE_MARKER: str = "<placeholder>", description="Bla-Bla-Bla...") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=_MODULE_MARKER,
        description=description
    )
    ap.add_argument("-i", "--file", type=Path, required=True, help="IQ int16 LE file (interleaved I,Q).")
    ap.add_argument("-fs", "--samp-rate", type=float, default=None, help="Sample rate, Hz, for .bin.")
    ap.add_argument("--verbose", action="store_true", help="Details in output.")
    return ap


def resolve_params(args, _verbose: bool = True):
    """
    bin файли вважаються IQ int16 LE.
    Returns:
        file, header_sz, samp_count, Fs, dtype, bps, verbose
    """
    # валідація файлу
    file: Path = args.file
    if not file.exists():
        raise FileNotFoundError(f"file not found: '{file.resolve()}'")

    ext = file.suffix.lower()
    bps = 4
    header_sz = 0
    dtype = np.int16

    if ext == ".wav":
        # Extract WAV parameters
        Fs, dur_sec, samp_count, header_sz, dtype, bps = get_iq_wav_prm(file)
    elif ext == ".bin":
        if args.samp_rate is None:
            raise ValueError("--samp-rate is required for '.bin' file")
        Fs = float(args.samp_rate)
        samp_count = (file.stat().st_size - header_sz) // bps
        dur_sec = samp_count / Fs
    else:
        raise ValueError(f"file type '{ext}' not supported ('{file.resolve()}')")

    if _verbose:
        header_str = f"wav hdr sz = {header_sz}" if header_sz else ""
        print(
            f"File: '{file.resolve()}'"
            f"\n    Fs={Fs:_} Hz  samp_count={samp_count:_}  {header_str}"
            f"\n    bps={bps}  dur≈{dur_sec:.2f} sec"
        )

    verbose = bool(args.verbose)
    return file, header_sz, samp_count, Fs, dtype, bps, verbose
