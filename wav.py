
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict, Union
import struct

import numpy as np

PCM_TAGS = {0x0001, 0xFFFE}  # PCM / WAVE_FORMAT_EXTENSIBLE

class WAVProps(TypedDict):
    codec_tag: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    data_size: int
    data_offset: int
    dur_sec: float

def read_wav_header(p: Union[str, Path]) -> WAVProps:
    with open(p, "rb") as f:
        r = f.read(12)
        if len(r) < 12:
            raise ValueError("Too short for RIFF header")
        riff, size, wave = struct.unpack("<4sI4s", r)
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError("Not RIFF/WAVE")

        fmt_found = False
        data_found = False
        codec_tag = channels = sample_rate = bits = 0
        data_size = 0
        data_offset = 0

        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", hdr)

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                if len(fmt_data) < 16:
                    raise ValueError("fmt chunk too short")
                codec_tag, channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
                    "<HHIIHH", fmt_data[:16]
                )
                fmt_found = True
                if (chunk_size & 1) == 1:
                    f.seek(1, 1)

            elif chunk_id == b"data":
                data_size = chunk_size
                data_offset = f.tell()
                data_found = True
                break

            else:
                f.seek(chunk_size + (chunk_size & 1), 1)

        if not fmt_found:
            raise ValueError("fmt chunk not found")
        if not data_found:
            raise ValueError("data chunk not found")

        bytes_per_sample = channels * (bits // 8)
        dur_sec = data_size / (sample_rate * bytes_per_sample) if bytes_per_sample else 0.0

        return WAVProps(
            codec_tag=codec_tag,
            channels=channels,
            sample_rate=sample_rate,
            bits_per_sample=bits,
            data_size=data_size,
            data_offset=data_offset,
            dur_sec=dur_sec,
        )


def get_iq_wav_prm(file: Path)->Tuple[int, float, int, int, np.dtype, int]:
    """
    Читає WAV-заголовок. Вважаємо що це контейнер для IQ.
    Витягує параметри для читання IQ-даних.
    Якщо вавка не відповідає чи не робиться - виняток
    Returns:
        return Fs, dur_sec, samp_count, header_sz, _dtype, 2 * bytes_per_chan
        
        'sample_rate': int,
        'duration_sec': float,
        'samples_count': int,  # 0 якщо тип не підтримується
        'header_sz': int,
        'dtype': np.dtype
        '2 * bytes_per_chan: complex sample size
    """

    props = read_wav_header(file)

    if props["codec_tag"] not in PCM_TAGS:
        raise ValueError(f"not PCM wav '{file}'")
    if props["channels"] != 2: 
        raise ValueError(f"not 2 chan wav '{file}'")

    Fs = props["sample_rate"]
    dur_sec = props["dur_sec"]
    ctx_sz = props["data_size"]
    bits = props["bits_per_sample"]
    codec = props["codec_tag"]

    # Визначення dtype
    bytes_per_chan = 0

    if codec == 0x0001:  # PCM integer
        if bits == 8:
            _dtype = np.uint8
            bytes_per_chan = 1
        elif bits == 16:
            _dtype = np.int16
            bytes_per_chan = 2
        elif bits == 32:
            _dtype = np.int32
            bytes_per_chan = 4
    elif codec == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE (зазвичай float)
        if bits == 32:
            _dtype = np.float32
            bytes_per_chan = 4
    else:
        raise ValueError(f"unknown dtype wav '{file}'")
    
    # Кількість комплексних семплів
    samp_count = ctx_sz // (2 * bytes_per_chan)
    
    # Offset до даних
    file_size = file.stat().st_size
    header_sz = props["data_offset"] 
    if 0:
        print(f"DEBUG: file_size={file_size}, ctx_sz={ctx_sz}, header_sz={header_sz}")
        print(f"DEBUG: codec={hex(codec)}, bits={bits}, bytes_per_chan={bytes_per_chan}")
        print(f"DEBUG: samp_count={samp_count}")
    return Fs, dur_sec, samp_count, header_sz, _dtype, 2 * bytes_per_chan


def create_wav_header(
    num_samples: int,
    sample_rate: int,
    num_channels: int = 2,
    bits_per_sample: int = 16
) -> bytearray:
    """
    Створює WAV header (44 bytes) для PCM audio.
    
    Args:
        num_samples: Кількість комплексних семплів (для IQ)
        sample_rate: Частота дискретизації (Hz)
        num_channels: 2 для IQ stereo (default)
        bits_per_sample: 16 для int16 PCM (default)
    
    Returns:
        bytearray: 44-byte WAV header готовий до запису
    
    Raises:
        ValueError: Якщо параметри некоректні
    
    Example:
        header = create_wav_header(output_samples_count, 48000)
        with open(out_file, 'r+b') as f:
            f.seek(0)
            f.write(header)
    """
    # import struct
    
    if num_channels not in (1, 2):
        raise ValueError(f"num_channels must be 1 or 2, got {num_channels}")
    if bits_per_sample not in (8, 16, 32):
        raise ValueError(f"bits_per_sample must be 8/16/32, got {bits_per_sample}")
    if sample_rate <= 0 or num_samples < 0:
        raise ValueError(f"Invalid sample_rate={sample_rate} or num_samples={num_samples}")
    
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = num_samples * block_align
    
    header = bytearray(44)
    
    # RIFF/WAVE header
    header[0:4]   = b'RIFF'
    header[4:8]   = struct.pack('<I', 36 + data_size)
    header[8:12]  = b'WAVE'
    
    # fmt chunk
    header[12:16] = b'fmt '
    header[16:20] = struct.pack('<I', 16)
    header[20:22] = struct.pack('<H', 1)               # PCM
    header[22:24] = struct.pack('<H', num_channels)
    header[24:28] = struct.pack('<I', sample_rate)
    header[28:32] = struct.pack('<I', byte_rate)
    header[32:34] = struct.pack('<H', block_align)
    header[34:36] = struct.pack('<H', bits_per_sample)
    
    # data chunk
    header[36:40] = b'data'
    header[40:44] = struct.pack('<I', data_size)
    
    return header

