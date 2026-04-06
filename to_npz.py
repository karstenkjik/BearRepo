import os, time, socket, struct
import msgpack
import numpy as np

UDP_PORT = 2122
OUT_DIR = "dataset_npz"
SAVE_EVERY_N_SCANS = 1
os.makedirs(OUT_DIR, exist_ok=True)

# Tokenized keys :contentReference[oaicite:2]{index=2}
K_DATA         = 0x11
K_NUMELEMS     = 0x12
K_ELEMSZ       = 0x13
K_SEGMENTDATA  = 0x96
K_CHANNELPHI   = 0x51
K_CHANNELTHETA = 0x50
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
        return np.frombuffer(data, dtype="<f4", count=n)
    if elem_sz == 2:
        return np.frombuffer(data, dtype="<u2", count=n)
    return None

def parse_msgpack_stx_framed(datagram: bytes):
    if len(datagram) < 12 or datagram[:4] != b"\x02\x02\x02\x02":
        return None
    payload_len = struct.unpack_from("<I", datagram, 4)[0]
    payload = datagram[8:8 + payload_len]
    return msgpack.unpackb(payload, raw=False, strict_map_key=False)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"Listening UDP:{UDP_PORT} -> {OUT_DIR}/frame_*.npz")

    scan_count = 0
    saved = 0

    while True:
        datagram, _ = sock.recvfrom(65535)
        try:
            msg = parse_msgpack_stx_framed(datagram)
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue

        seg = getv(msg, K_SEGMENTDATA, "SegmentData")
        if seg is None:
            msg_data = getv(msg, K_DATA, "data")
            seg = getv(msg_data, K_SEGMENTDATA, "SegmentData") if isinstance(msg_data, dict) else None
        if not isinstance(seg, list) or len(seg) == 0:
            continue

        for scan in seg:
            scan_map = scan
            if isinstance(scan, dict):
                scan_data = getv(scan, K_DATA, "data")
                if isinstance(scan_data, dict):
                    scan_map = scan_data
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

            scan_count += 1
            if SAVE_EVERY_N_SCANS > 1 and (scan_count % SAVE_EVERY_N_SCANS != 0):
                continue

            ts = np.int64(time.time_ns())
            dist_m = dist0.astype(np.float32) / 1000.0
            phi = np.float32(phi_arr[0])

            out = os.path.join(OUT_DIR, f"frame_{ts}.npz")
            np.savez_compressed(out,
                                timestamp_ns=ts,
                                dist_m=dist_m.astype(np.float32),
                                theta_rad=theta.astype(np.float32),
                                phi_rad=phi)
            saved += 1
            if saved % 10 == 0:
                print(f"Saved {saved} frames")

if __name__ == "__main__":
    main()


