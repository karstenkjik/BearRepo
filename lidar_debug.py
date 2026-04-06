import socket, struct, time, csv, binascii
import msgpack
import numpy as np

UDP_PORT = 2122
OUT_CSV = "lidar_scans.csv"
DEBUG_CSV = "lidar_debug.csv"

def decode_sick_array(arr_obj):
    if not isinstance(arr_obj, dict):
        return None
    n = arr_obj.get("numOfElems")
    elem_sz = arr_obj.get("elemSz")
    data = arr_obj.get("data")
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

def try_unpack(payload: bytes):
    return msgpack.unpackb(payload, raw=False)

def parse_msgpack_any(datagram: bytes):
    # A) STX framed: 02 02 02 02 + u32len + payload + crc
    if len(datagram) >= 12 and datagram[:4] == b"\x02\x02\x02\x02":
        try:
            payload_len = struct.unpack_from("<I", datagram, 4)[0]
            payload = datagram[8:8+payload_len]
            return try_unpack(payload), "stx_framed"
        except Exception:
            pass

    # B) length at start (no STX): u32len + payload + crc
    if len(datagram) >= 8:
        try:
            payload_len = struct.unpack_from("<I", datagram, 0)[0]
            if 0 < payload_len <= len(datagram) - 4:
                payload = datagram[4:4+payload_len]
                return try_unpack(payload), "len_framed"
        except Exception:
            pass

    # C) whole datagram is msgpack
    try:
        return try_unpack(datagram), "raw_msgpack"
    except Exception:
        pass

    # D) search for fixmap in first 64 bytes
    scan_len = min(64, len(datagram))
    for i in range(scan_len):
        b = datagram[i]
        if 0x80 <= b <= 0x8f:
            try:
                return try_unpack(datagram[i:]), f"fixmap@{i}"
            except Exception:
                continue

    return None, "no_decode"

def extract_segment_data(msg):
    if not isinstance(msg, dict):
        return None
    if isinstance(msg.get("SegmentData"), list):
        return msg["SegmentData"]
    d = msg.get("data")
    if isinstance(d, dict) and isinstance(d.get("SegmentData"), list):
        return d["SegmentData"]
    return None

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    pkt_count = 0
    ok_scans = 0
    header_written = False

    print(f"Listening UDP:{UDP_PORT}")
    print(f"Writing decoded scans -> {OUT_CSV}")
    print(f"Writing decode failures -> {DEBUG_CSV}")

    with open(OUT_CSV, "w", newline="") as out_f, open(DEBUG_CSV, "w", newline="") as dbg_f:
        out_w = csv.writer(out_f)
        dbg_w = csv.writer(dbg_f)

        dbg_w.writerow(["timestamp_ns", "pkt_len", "parse_mode", "first32_hex"])

        while True:
            datagram, _ = sock.recvfrom(65535)
            pkt_count += 1

            msg, mode = parse_msgpack_any(datagram)
            if not isinstance(msg, dict):
                # log failure
                dbg_w.writerow([time.time_ns(), len(datagram), mode, binascii.hexlify(datagram[:32]).decode()])
                if pkt_count % 50 == 0:
                    print(f"pkts={pkt_count}, decoded_scans={ok_scans} (decode failing; see {DEBUG_CSV})")
                continue

            seg = extract_segment_data(msg)
            if not isinstance(seg, list) or len(seg) == 0:
                dbg_w.writerow([time.time_ns(), len(datagram), f"{mode}:no_segmentdata", binascii.hexlify(datagram[:32]).decode()])
                if pkt_count % 50 == 0:
                    print(f"pkts={pkt_count}, decoded_scans={ok_scans} (no SegmentData; see {DEBUG_CSV})")
                continue

            # For each scan in SegmentData, try to extract distances
            for scan in seg:
                scan_map = scan.get("data", scan) if isinstance(scan, dict) else None
                if not isinstance(scan_map, dict):
                    continue

                dist_list = scan_map.get("DistValues")
                phi_arr = decode_sick_array(scan_map.get("ChannelPhi"))

                if not isinstance(dist_list, list) or len(dist_list) == 0:
                    continue

                dist0 = decode_sick_array(dist_list[0])
                if dist0 is None:
                    continue

                dist_m = dist0.astype(np.float32) / 1000.0
                phi = float(phi_arr[0]) if phi_arr is not None and len(phi_arr) > 0 else 0.0
                ts = time.time_ns()

                if not header_written:
                    header = ["timestamp_ns", "phi_rad"] + [f"d{i}_m" for i in range(len(dist_m))]
                    out_w.writerow(header)
                    header_written = True
                    out_f.flush()
                    print(f"✅ Writing scans: {len(dist_m)} distance columns")

                out_w.writerow([ts, phi] + dist_m.tolist())
                ok_scans += 1

                if ok_scans % 10 == 0:
                    out_f.flush()
                    dbg_f.flush()
                    print(f"pkts={pkt_count}, decoded_scans={ok_scans} (mode={mode})")

if __name__ == "__main__":
    main()


