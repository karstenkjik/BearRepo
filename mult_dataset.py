import os
import time
import socket
import struct
import msgpack
import numpy as np

UDP_PORT = 2122
OUT_DIR = "dataset_npz"
SAVE_EVERY_N_SCANS = 1  # set 2/5/10 to downsample

os.makedirs(OUT_DIR, exist_ok=True)

def decode_sick_array(arr_obj):
    """Decode SICK MSGPACK array-serialization dict into a numpy array."""
    if not isinstance(arr_obj, dict):
        return None
    n = arr_obj.get("numOfElems")
    elem_sz = arr_obj.get("elemSz")
    data = arr_obj.get("data")
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

def parse_msgpack_any(datagram: bytes):
    """
    Try common message layouts:
    1) 0x02 0x02 0x02 0x02 + u32len + payload + crc
    2) u32len + payload + crc
    3) whole datagram is msgpack
    4) search for a msgpack map start in first 64 bytes
    """
    # 1) STX framed
    if len(datagram) >= 12 and datagram[:4] == b"\x02\x02\x02\x02":
        try:
            payload_len = struct.unpack_from("<I", datagram, 4)[0]
            payload = datagram[8:8 + payload_len]
            return msgpack.unpackb(payload, raw=False)
        except Exception:
            pass

    # 2) length at start
    if len(datagram) >= 8:
        try:
            payload_len = struct.unpack_from("<I", datagram, 0)[0]
            if 0 < payload_len <= len(datagram) - 4:
                payload = datagram[4:4 + payload_len]
                return msgpack.unpackb(payload, raw=False)
        except Exception:
            pass

    # 3) entire datagram
    try:
        return msgpack.unpackb(datagram, raw=False)
    except Exception:
        pass

    # 4) search for fixmap prefix
    scan_len = min(64, len(datagram))
    for i in range(scan_len):
        b = datagram[i]
        if 0x80 <= b <= 0x8f:
            try:
                return msgpack.unpackb(datagram[i:], raw=False)
            except Exception:
                continue

    return None

def extract_segment_data(msg):
    """SegmentData may be top-level or nested under msg['data']."""
    if not isinstance(msg, dict):
        return None
    seg = msg.get("SegmentData")
    if seg is not None:
        return seg
    d = msg.get("data")
    if isinstance(d, dict):
        return d.get("SegmentData")
    return None

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    scan_count = 0
    saved = 0

    print(f"Listening on UDP {UDP_PORT} and saving frames to {OUT_DIR}/frame_*.npz")

    while True:
        datagram, _ = sock.recvfrom(65535)
        msg = parse_msgpack_any(datagram)
        if not isinstance(msg, dict):
            continue

        segment_data = extract_segment_data(msg)
        if not isinstance(segment_data, list) or len(segment_data) == 0:
            continue

        for scan in segment_data:
            scan_map = scan.get("data", scan)
            if not isinstance(scan_map, dict):
                continue

            theta = decode_sick_array(scan_map.get("ChannelTheta"))
            phi_arr = decode_sick_array(scan_map.get("ChannelPhi"))
            dist_list = scan_map.get("DistValues")

            if theta is None or phi_arr is None or not isinstance(dist_list, list) or len(dist_list) == 0:
                continue

            dist0 = decode_sick_array(dist_list[0])
            if dist0 is None:
                continue

            scan_count += 1
            if SAVE_EVERY_N_SCANS > 1 and (scan_count % SAVE_EVERY_N_SCANS != 0):
                continue

            # Convert distances (mm -> meters)
            dist_m = dist0.astype(np.float32) / 1000.0
            phi = np.float32(phi_arr[0])
            ts = np.int64(time.time_ns())

            out_path = os.path.join(OUT_DIR, f"frame_{ts}.npz")
            np.savez_compressed(
                out_path,
                timestamp_ns=ts,
                dist_m=dist_m.astype(np.float32),
                theta_rad=theta.astype(np.float32),
                phi_rad=phi
            )

            saved += 1
            if saved % 10 == 0:
                print(f"Saved {saved} frames (latest {os.path.basename(out_path)})")

if __name__ == "__main__":
    main()


