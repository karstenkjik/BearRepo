from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import deque
import struct
import socket
import time

import msgpack
import numpy as np
import torch

from model import LidarBinaryCNN2D
from test_preprocess_frame import preprocess_scan_to_frame
from test_temporal_buffer import TemporalBuffer
from test_decision_logic import ProbabilityDecisionFilter


# ============================================================
# Config
# ============================================================

UDP_PORT = 2122
SEQUENCE_LENGTH = 5
THRESHOLD = 0.30
SMOOTH_WINDOW = 5
PRINT_EVERY = 1


# Tokenized msgpack keys used by the SICK datagrams
K_DATA         = 0x11
K_NUMELEMS     = 0x12
K_ELEMSZ       = 0x13
K_SEGMENTDATA  = 0x96
K_CHANNELTHETA = 0x50
K_CHANNELPHI   = 0x51
K_DISTVALUES   = 0x52


# ============================================================
# Shared raw scan structure
# ============================================================

@dataclass
class RawScan:
    timestamp_ns: int
    dist_m: np.ndarray
    theta_rad: np.ndarray
    phi_rad: float
    valid: np.ndarray


# ============================================================
# UDP parsing helpers (adapted from log_npz.py)
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
    """
    Expected format:
      [0x02 0x02 0x02 0x02][u32 payload_len LE][payload][u32 crc LE]
    """
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


def scan_from_datagram(datagram: bytes) -> list[RawScan]:
    """
    Parse one UDP datagram into zero or more RawScan objects.
    """
    try:
        msg = parse_stx_framed_msgpack(datagram)
    except Exception:
        return []

    if not isinstance(msg, dict):
        return []

    segment_data = extract_segment_data(msg)
    if not isinstance(segment_data, list) or len(segment_data) == 0:
        return []

    scans_out = []

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

        # Match the live logging script
        valid = (dist_m > 0.01).astype(np.float32)

        scans_out.append(
            RawScan(
                timestamp_ns=time.time_ns(),
                dist_m=dist_m.astype(np.float32),
                theta_rad=theta_rad,
                phi_rad=phi_rad,
                valid=valid,
            )
        )

    return scans_out


# ============================================================
# Model inference
# ============================================================

def safe_load_checkpoint(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except Exception:
        return torch.load(checkpoint_path, map_location=device)


class InferenceRunner:
    def __init__(self, checkpoint_path: str | Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path)

        checkpoint = safe_load_checkpoint(checkpoint_path, self.device)

        self.model = LidarBinaryCNN2D().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_horse_probability(self, window: np.ndarray):

        x = np.expand_dims(window, axis=0).astype(np.float32)

        x_tensor = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs[0].cpu().numpy()
        pred_class = int(np.argmax(probs_np))
        horse_prob = float(probs_np[1])

        return horse_prob, probs_np, pred_class


# ============================================================
# Main live pipeline
# ============================================================

def run_live_pipeline(checkpoint_path: str | Path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    buffer = TemporalBuffer(sequence_length=SEQUENCE_LENGTH)
    inference = InferenceRunner(checkpoint_path)
    decision_filter = ProbabilityDecisionFilter(
        threshold=THRESHOLD,
        smooth_window=SMOOTH_WINDOW,
    )

    print(f"Listening on UDP {UDP_PORT}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Smoothing window: {SMOOTH_WINDOW}")
    print("Waiting for live scans...\n")

    step = 0

    while True:
        datagram, addr = sock.recvfrom(65535)
        scans = scan_from_datagram(datagram)

        for scan in scans:
            step += 1

            frame = preprocess_scan_to_frame(scan)
            buffer.add_frame(frame)

            valid_count = int(np.sum(scan.valid > 0))

            if not buffer.is_full():
                print(
                    f"step={step:04d} | buffering ({len(buffer.buffer)}/{SEQUENCE_LENGTH}) | "
                    f"valid_count={valid_count}"
                )
                continue

            window = buffer.get_window()
            horse_prob, probs_np, pred_class = inference.predict_horse_probability(window)
            result = decision_filter.update(horse_prob)

            if step % PRINT_EVERY == 0:
                print(
                    f"step={step:04d} | "
                    f"valid_count={valid_count} | "
                    f"no_horse={probs_np[0]:.6f} | "
                    f"horse_present={probs_np[1]:.6f} | "
                    f"raw={result.raw_probability:.6f} | "
                    f"smoothed={result.smoothed_probability:.6f} | "
                    f"decision={result.decision} | "
                    f"pred_class={pred_class}"
                )


if __name__ == "__main__":
    # CHANGE THIS PATH
    checkpoint_path = "/home/pi/BearRepo/bestmodel.pt"
    run_live_pipeline(checkpoint_path)