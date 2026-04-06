import os
import time
import socket
import struct
import msgpack
import numpy as np

UDP_PORT = 2122
OUT_DIR = "dataset_env_only  sdfg0-"
SAVE_EVERY_N_SCANS = 1          
PRINT_EVERY = 10                

os.makedirs(OUT_DIR, exist_ok=True)

K_DATA         = 0x11
K_NUMELEMS     = 0x12
K_ELEMSZ       = 0x13
K_SEGMENTDATA  = 0x96
K_CHANNELTHETA = 0x50
K_CHANNELPHI   = 0x51
K_DISTVALUES   = 0x52

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
        return np.frombuffer(data, dtype="<f4", count=n)  # float32 LE
    if elem_sz == 2:
        return np.frombuffer(data, dtype="<u2", count=n)  # uint16 LE
    if elem_sz == 1:
        return np.frombuffer(data, dtype=np.uint8, count=n)
    return None

def parse_stx_framed_msgpack(datagram: bytes):
    """
    Your packets start with 0x02 0x02 0x02 0x02.
    Layout: [STX*4][u32 payload_len LE][payload][u32 crc LE]
    """
    if len(datagram) < 12 or datagram[:4] != b"\x02\x02\x02\x02":
        return None
    payload_len = struct.unpack_from("<I", datagram, 4)[0]
    payload = datagram[8:8 + payload_len]
    if len(payload) != payload_len:
        return None
    # IMPORTANT: allow integer map keys
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

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    scan_seen = 0
    scan_saved = 0

    print(f"Listening UDP:{UDP_PORT} -> {OUT_DIR}/frame_*.npz")

    while True:
        datagram, _ = sock.recvfrom(65535)

        try:
            msg = parse_stx_framed_msgpack(datagram)
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue

        segment_data = extract_segment_data(msg)
        if not isinstance(segment_data, list) or len(segment_data) == 0:
            continue

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
            phi = np.float32(phi_arr[0])

            valid = dist_m > 0.01

            scan_seen += 1
            if SAVE_EVERY_N_SCANS > 1 and (scan_seen % SAVE_EVERY_N_SCANS != 0):
                continue

            ts = np.int64(time.time_ns())
            out_path = os.path.join(OUT_DIR, f"frame_{ts}.npz")

            np.savez_compressed(
                out_path,
                timestamp_ns=ts,
                dist_m=dist_m,
                theta_rad=theta.astype(np.float32),
                phi_rad=phi,
                valid=valid
            )

            scan_saved += 1
            if scan_saved % PRINT_EVERY == 0:
                print(f"Saved {scan_saved} scans (latest: {os.path.basename(out_path)}, N={dist_m.size})")

if __name__ == "__main__":
    main()