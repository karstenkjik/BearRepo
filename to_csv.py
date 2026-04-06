import socket, struct, time, csv
import msgpack
import numpy as np

UDP_PORT = 2122
CSV_FILE = "lidar_scans.csv"
SAVE_EVERY_N_SCANS = 1

# Tokenized msgpack keys used by SICK scansegment msgpack (from sick_scan_xd) :contentReference[oaicite:1]{index=1}
K_CLASS        = 0x10
K_DATA         = 0x11
K_NUMELEMS     = 0x12
K_ELEMSZ       = 0x13
K_SEGMENTDATA  = 0x96
K_CHANNELPHI   = 0x51
K_DISTVALUES   = 0x52

def getv(d, k_int, k_str=None):
    """Get value by integer key (preferred), fallback to string key if present."""
    if isinstance(d, dict):
        if k_int in d:
            return d[k_int]
        if k_str is not None and k_str in d:
            return d[k_str]
    return None

def decode_sick_array(arr_obj):
    """
    Decode SICK array serialization:
      { numOfElems, elemSz, data, ... } where keys may be tokenized ints.
    """
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

def parse_msgpack_stx_framed(datagram: bytes):
    """
    Your packets start with 0x02 0x02 0x02 0x02 and then little-endian payload length.
    Packet len matches: 4 + 4 + payload_len + 4(crc).
    """
    if len(datagram) < 12 or datagram[:4] != b"\x02\x02\x02\x02":
        return None
    payload_len = struct.unpack_from("<I", datagram, 4)[0]
    payload = datagram[8:8 + payload_len]
    # IMPORTANT: allow non-string keys
    return msgpack.unpackb(payload, raw=False, strict_map_key=False)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"Listening UDP:{UDP_PORT} -> {CSV_FILE}")

    scan_count = 0
    written = 0
    header_written = False

    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)

        while True:
            datagram, _ = sock.recvfrom(65535)

            try:
                msg = parse_msgpack_stx_framed(datagram)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue

            # SegmentData can be at top-level or under data
            seg = getv(msg, K_SEGMENTDATA, "SegmentData")
            if seg is None:
                msg_data = getv(msg, K_DATA, "data")
                seg = getv(msg_data, K_SEGMENTDATA, "SegmentData") if isinstance(msg_data, dict) else None
            if not isinstance(seg, list) or len(seg) == 0:
                continue

            for scan in seg:
                # scan map might be wrapped in "data"
                scan_map = scan
                if isinstance(scan, dict):
                    scan_data = getv(scan, K_DATA, "data")
                    if isinstance(scan_data, dict):
                        scan_map = scan_data

                if not isinstance(scan_map, dict):
                    continue

                phi_arr = decode_sick_array(getv(scan_map, K_CHANNELPHI, "ChannelPhi"))
                dist_list = getv(scan_map, K_DISTVALUES, "DistValues")

                if phi_arr is None or not isinstance(dist_list, list) or len(dist_list) == 0:
                    continue

                dist0 = decode_sick_array(dist_list[0])
                if dist0 is None:
                    continue

                scan_count += 1
                if SAVE_EVERY_N_SCANS > 1 and (scan_count % SAVE_EVERY_N_SCANS != 0):
                    continue

                # distances: mm -> meters
                dist_m = dist0.astype(np.float32) / 1000.0
                phi = float(phi_arr[0])
                ts = time.time_ns()

                if not header_written:
                    header = ["timestamp_ns", "phi_rad"] + [f"d{i}_m" for i in range(len(dist_m))]
                    w.writerow(header)
                    f.flush()
                    header_written = True
                    print(f"Header written: {len(dist_m)} distance columns")

                w.writerow([ts, phi] + dist_m.tolist())
                written += 1

                if written % 10 == 0:
                    f.flush()
                    print(f"Wrote {written} scans")

if __name__ == "__main__":
    main()


