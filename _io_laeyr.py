#!/usr/bin/env python3
# py/io_laeyr.py



import sys
from typing import Tuple


def get_io_rec_windows(root=u"C:\\"):
    import ctypes, ctypes.wintypes as w
    spc = w.DWORD(); bps = w.DWORD(); nfc = w.DWORD(); tnc = w.DWORD()
    if not ctypes.windll.kernel32.GetDiskFreeSpaceW(w.LPCWSTR(root),
                                                    ctypes.byref(spc),
                                                    ctypes.byref(bps),
                                                    ctypes.byref(nfc),
                                                    ctypes.byref(tnc)):
        # fallback, якщо виклик не вдався
        return (4096, 1<<20)
    align = spc.value * bps.value                  # cluster size (вирівнювання I/O)
    target = (1<<20)                               # 1 MiB базово
    target = max(align, (target//align)*align)     # зробити кратним align
    return (align, target)


def get_io_rec_nix() -> Tuple[int, int]:
    """
    Підбирає вирівнювання і цільовий I/O-блок (align_bytes, target_bytes).
    Лінукс-евристика через /sys; fallback: (4096, 1 MiB).
    """
    import os, re

    def _read(p):
        try:
            with open(p, "r") as f:
                return f.read().strip()
        except:
            return None

    # Визначити блок-девайс кореня
    dev = None
    with open("/proc/mounts") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) >= 2 and parts[1] == "/":
                dev = parts[0]
                break
    if not dev:
        return (4096, 1 << 20)

    base = os.path.basename(dev)  # напр. nvme0n1p1, sda1, dm-0, mmcblk0p2
    # Прибрати суфікси розділів
    # nvmeXnYpZ -> nvmeXnY ; sdxN -> sdx ; mmcblkXpY -> mmcblkX ; dm-* залишити як є
    if base.startswith("nvme"):
        m = re.match(r"(nvme\d+n\d+)", base)
        base_root = m.group(1) if m else base
    elif base.startswith("mmcblk"):
        m = re.match(r"(mmcblk\d+)", base)
        base_root = m.group(1) if m else base
    elif re.match(r"sd[a-z]+", base):
        m = re.match(r"(sd[a-z]+)", base)
        base_root = m.group(1) if m else base
    else:
        base_root = base

    q = f"/sys/block/{base_root}/queue"
    phys = _read(f"{q}/physical_block_size")
    opt  = _read(f"{q}/optimal_io_size")
    rot  = _read(f"{q}/rotational")

    align = int(phys) if (phys and phys.isdigit()) else 4096

    # RAID stripe (якщо є)
    stripe = _read(f"/sys/block/{base_root}/md/stripe_size")
    if stripe and stripe.isdigit():
        align = max(align, int(stripe))

    # Обрати target
    if opt and opt.isdigit() and int(opt) > 0:
        target = int(opt)
    else:
        # NVMe/SSD → 1 MiB; HDD/NFS → 512 KiB
        target = (1 << 20) if (rot == "0") else (512 << 10)

    # Округлити вниз до кратності align, але не менше align
    target = max(align, (target // align) * align)
    return (align, target)

# Платформозалежний вибір джерела рекомендацій I/O
def get_io_rec():
    if sys.platform.startswith("win"):
        _PLATFORM= "win"
        _ALIGN, _CHUNK_SZ_OPT = get_io_rec_windows()
    else:  # Linux/mac/*nix
        _PLATFORM= "nix"
        _ALIGN, _CHUNK_SZ_OPT = get_io_rec_nix()
    return _PLATFORM, _ALIGN, _CHUNK_SZ_OPT
