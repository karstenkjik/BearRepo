from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import struct
import socket
import time

import msgpack
import numpy as np


# ============================================================
# Config
# ============================================================

UDP_PORT = 2122
SAVE_DIR = Path("/home/pi/BearRepo/live_debug_captures_outdoor2")
MAX_SCANS_TO_SAVE = 100
PRINT_EVERY = 1

# Match current live parser logic
VALID_DISTANCE_THRESHOLD_M = 0.01


# Tokenized msgpack keys used by the SICK datagrams
K_DATA         = 0x11
K_NUMELEMS     = 0x12
K_ELEMSZ       = 0x13
K_SEGMENTDATA  = 0x96
K_CHANNELTHETA = 0x50
K_CHANNELPHI   = 0x51
K_DISTVALUES   = 0x52


# ============================================================
# Shared structure
# ============================================================

@dataclass
class RawScan:
    timestamp_ns: int
    dist_m: np.ndarray
    theta_rad: np.ndarray
    phi_rad: float
    valid: np.ndarray


# ============================================================
# UDP parsing helpers
# ============================================================

def getv(d, k_int, k_str=None):
    if isinstance(d, dict):
        if k_int in d:
            return d[k_int]
        if k_str is not None and k_str in d:
            return d[k_str]
    return None


def decode_sick_array(arr_obj):
    if not isinstance(arr_obj, dict):
        return None

    n = getv(arr_obj, K_NUMELEMS, "numOfElems")
    elem_sz = getv(arr_obj, K_ELEMSZ, "elemSz")
    data = getv(arr_obj, K_DATA, "data")

    if n is None or elem_sz is None or data is None:
        return None

    n = int(n)

    if elem_sz == 4:
        return np.frombuffer(data, dtype="<f4", count=n)
    if elem_sz == 2:
        return np.frombuffer(data, dtype="<u2", count=n)
    if elem_sz == 1:
        return np.frombuffer(data, dtype=np.uint8, count=n)

    return None


def parse_stx_framed_msgpack(datagram: bytes):
    if len(datagram) < 12 or datagram[:4] != b"\x02\x02\x02\x02":
        return None

    payload_len = struct.unpack_from("<I", datagram, 4)[0]
    payload = datagram[8:8 + payload_len]

    if len(payload) != payload_len:
        return None

    return msgpack.unpackb(payload, raw=False, strict_map_key=False)


def extract_segment_data(msg: dict):
    seg = getv(msg, K_SEGMENTDATA, "SegmentData")
    if isinstance(seg, list):
        return seg

    d = getv(msg, K_DATA, "data")
    if isinstance(d, dict):
        seg2 = getv(d, K_SEGMENTDATA, "SegmentData")
        if isinstance(seg2, list):
            return seg2

    return None


def scans_from_datagram(datagram: bytes) -> list[RawScan]:
    try:
        msg = parse_stx_framed_msgpack(datagram)
    except Exception:
        return []

    if not isinstance(msg, dict):
        return []

    segment_data = extract_segment_data(msg)
    if not isinstance(segment_data, list) or len(segment_data) == 0:
        return []

    out = []

    for scan in segment_data:
        scan_map = scan
        if isinstance(scan, dict):
            inner = getv(scan, K_DATA, "data")
            if isinstance(inner, dict):
                scan_map = inner

        if not isinstance(scan_map, dict):
            continue

        theta = decode_sick_array(getv(scan_map, K_CHANNELTHETA, "ChannelTheta"))
        phi_arr = decode_sick_array(getv(scan_map, K_CHANNELPHI, "ChannelPhi"))
        dist_list = getv(scan_map, K_DISTVALUES, "DistValues")

        if theta is None or phi_arr is None or not isinstance(dist_list, list) or len(dist_list) == 0:
            continue

        dist0 = decode_sick_array(dist_list[0])
        if dist0 is None:
            continue

        dist_m = dist0.astype(np.float32) / 1000.0
        theta_rad = theta.astype(np.float32)
        phi_rad = float(phi_arr[0])

        valid = (dist_m > VALID_DISTANCE_THRESHOLD_M).astype(np.float32)

        out.append(
            RawScan(
                timestamp_ns=time.time_ns(),
                dist_m=dist_m,
                theta_rad=theta_rad,
                phi_rad=phi_rad,
                valid=valid,
            )
        )

    return out


# ============================================================
# Debug / save helpers
# ============================================================

def save_raw_scan(scan: RawScan, out_dir: Path, index: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"live_scan_{index:05d}.npz"

    np.savez_compressed(
        out_path,
        timestamp_ns=np.int64(scan.timestamp_ns),
        dist_m=scan.dist_m.astype(np.float32),
        theta_rad=scan.theta_rad.astype(np.float32),
        phi_rad=np.float32(scan.phi_rad),
        valid=scan.valid.astype(np.float32),
    )


def print_scan_stats(step: int, scan: RawScan):
    valid_mask = scan.valid > 0
    valid_count = int(np.sum(valid_mask))

    if valid_count > 0:
        valid_dist = scan.dist_m[valid_mask]
        dist_min = float(np.min(valid_dist))
        dist_max = float(np.max(valid_dist))
        dist_mean = float(np.mean(valid_dist))
    else:
        dist_min = 0.0
        dist_max = 0.0
        dist_mean = 0.0

    all_nonzero = scan.dist_m[scan.dist_m > 0]
    if len(all_nonzero) > 0:
        nonzero_mean = float(np.mean(all_nonzero))
    else:
        nonzero_mean = 0.0

    print(
        f"step={step:04d} | "
        f"phi_rad={scan.phi_rad:.6f} | "
        f"valid_count={valid_count} | "
        f"valid_frac={valid_count / len(scan.dist_m):.3f} | "
        f"valid_dist_min={dist_min:.3f} | "
        f"valid_dist_max={dist_max:.3f} | "
        f"valid_dist_mean={dist_mean:.3f} | "
        f"nonzero_mean={nonzero_mean:.3f}"
    )


# ============================================================
# Main
# ============================================================

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    print(f"Listening on UDP {UDP_PORT}")
    print(f"Saving up to {MAX_SCANS_TO_SAVE} scans to {SAVE_DIR}")
    print(f"valid threshold: dist_m > {VALID_DISTANCE_THRESHOLD_M}\n")

    saved = 0
    seen = 0

    while saved < MAX_SCANS_TO_SAVE:
        datagram, addr = sock.recvfrom(65535)
        scans = scans_from_datagram(datagram)

        for scan in scans:
            seen += 1

            if seen % PRINT_EVERY == 0:
                print_scan_stats(seen, scan)

            save_raw_scan(scan, SAVE_DIR, saved)
            saved += 1

            if saved >= MAX_SCANS_TO_SAVE:
                break

    print(f"\nDone. Saved {saved} scans to {SAVE_DIR}")


if __name__ == "__main__":
    main()