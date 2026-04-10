from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import deque
import struct
import socket
import time
import atexit

import msgpack
import numpy as np
import torch
import pigpio
from gpiozero import PWMOutputDevice

from model import LidarBinaryCNN2D
from test_temporal_buffer import TemporalBuffer


# ============================================================
# Config
# ============================================================

UDP_PORT = 2122
SEQUENCE_LENGTH = 5

# Phase-block model threshold
MODEL_THRESHOLD = 0.50

# Control-side activation logic
ACTIVATION_WINDOW = 5
ACTIVATE_IF_AT_LEAST = 3
DEACTIVATE_IF_AT_MOST = 1

PRINT_EVERY = 1
MAX_RANGE_METERS = 40.0
VALID_DISTANCE_THRESHOLD_M = 0.01

# Deterrent / GPIO config
ULTRASONIC_GPIO = 18
ULTRASONIC_FREQ = 8000
ULTRASONIC_DUTY = 500000  # pigpio hardware_PWM duty range: 0..1_000_000

LED_GPIO = 12
LED_FREQ = 1
LED_DUTY = 0.5


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
# GPIO deterrent controller
# ============================================================

class DeterrentController:
    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError(
                "Failed to connect to pigpio daemon. Make sure pigpiod is running."
            )

        self.led_pwm: PWMOutputDevice | None = None
        self.is_active = False

    def activate(self):
        if self.is_active:
            return

        # Ultrasonic output
        self.pi.hardware_PWM(ULTRASONIC_GPIO, ULTRASONIC_FREQ, ULTRASONIC_DUTY)

        # LED output
        self.led_pwm = PWMOutputDevice(LED_GPIO)
        self.led_pwm.frequency = LED_FREQ
        self.led_pwm.value = LED_DUTY

        self.is_active = True
        print(">>> DETERRENT ACTIVATED <<<")

    def deactivate(self):
        if not self.is_active:
            return

        # Stop ultrasonic PWM
        self.pi.hardware_PWM(ULTRASONIC_GPIO, 0, 0)
        self.pi.write(ULTRASONIC_GPIO, 0)

        # Stop LED PWM
        if self.led_pwm is not None:
            self.led_pwm.close()
            self.led_pwm = None

        self.is_active = False
        print(">>> DETERRENT DEACTIVATED <<<")

    def cleanup(self):
        try:
            self.deactivate()
        finally:
            if self.pi is not None:
                self.pi.stop()


def startup_led_sequence(controller: DeterrentController, flashes: int = 3, delay: float = 0.3):
    print(">>> Running startup LED sequence <<<")

    led = PWMOutputDevice(LED_GPIO)
    led.frequency = LED_FREQ

    for _ in range(flashes):
        led.value = LED_DUTY
        time.sleep(delay)
        led.value = 0
        time.sleep(delay)

    led.close()


# Global deterrent handle so hooks can access it
DETERRENT: DeterrentController | None = None

def activate_deterrent():
    global DETERRENT
    if DETERRENT is not None:
        DETERRENT.activate()


def deactivate_deterrent():
    global DETERRENT
    if DETERRENT is not None:
        DETERRENT.deactivate()


def cleanup_deterrent():
    global DETERRENT
    if DETERRENT is not None:
        DETERRENT.cleanup()


# ============================================================
# Activation controller
# ============================================================

class ActivationController:
    def __init__(
        self,
        window_size: int = ACTIVATION_WINDOW,
        activate_if_at_least: int = ACTIVATE_IF_AT_LEAST,
        deactivate_if_at_most: int = DEACTIVATE_IF_AT_MOST,
    ):
        self.window_size = window_size
        self.activate_if_at_least = activate_if_at_least
        self.deactivate_if_at_most = deactivate_if_at_most
        self.history = deque(maxlen=window_size)
        self.is_active = False

    def update(self, decision: str) -> tuple[str, int]:
        """
        decision:
            'horse_present' or 'no_horse'

        returns:
            action, positive_count

        action:
            'ACTIVATE'
            'DEACTIVATE'
            'HOLD'
        """
        positive = 1 if decision == "horse_present" else 0
        self.history.append(positive)

        positive_count = sum(self.history)

        if len(self.history) < self.window_size:
            return "HOLD", positive_count

        if not self.is_active and positive_count >= self.activate_if_at_least:
            self.is_active = True
            return "ACTIVATE", positive_count

        if self.is_active and positive_count <= self.deactivate_if_at_most:
            self.is_active = False
            return "DEACTIVATE", positive_count

        return "HOLD", positive_count


# ============================================================
# 3-scan rolling phase block processor
# ============================================================

class PhaseBlockProcessor:
    def __init__(self, model_threshold: float = MODEL_THRESHOLD):
        self.buffer = deque(maxlen=3)
        self.model_threshold = model_threshold

    def update(self, horse_prob: float, valid_count: int, mean_dist: float):
        """
        Returns:
            None if the 3-scan block is not full yet

            Otherwise returns:
                (
                    decision,
                    block_prob,
                    avg_valid,
                    avg_dist,
                    zero_count,
                )
        """
        self.buffer.append((horse_prob, valid_count, mean_dist))

        if len(self.buffer) < 3:
            return None

        probs = [x[0] for x in self.buffer]
        valids = [x[1] for x in self.buffer]
        dists = [x[2] for x in self.buffer]

        block_prob = max(probs)
        avg_valid = sum(valids) / 3.0
        avg_dist = sum(dists) / 3.0
        zero_count = sum(1 for v in valids if v == 0)

        # Single veto rule only
        if zero_count >= 2 and avg_valid < 25:
            decision = "no_horse"
        else:
            decision = "horse_present" if block_prob >= self.model_threshold else "no_horse"

        return decision, block_prob, avg_valid, avg_dist, zero_count


# ============================================================
# Preprocessing
# ============================================================

def preprocess_scan_to_frame(scan: RawScan, max_range_m: float = MAX_RANGE_METERS) -> np.ndarray:
    """
    FARFILL version:
    for no-return points (valid == 0), range channel is set to 1.0
    instead of 0.0
    """
    if max_range_m <= 0:
        raise ValueError("max_range_m must be > 0")

    dist_m = scan.dist_m.astype(np.float32)
    valid = scan.valid.astype(np.float32)

    range_img = np.clip(dist_m, 0.0, max_range_m) / max_range_m

    # FARFILL
    range_img[valid == 0] = 1.0

    x = np.stack([range_img, valid], axis=0).astype(np.float32)  # (2, 240)
    x = np.expand_dims(x, axis=1)                                # (2, 1, 240)

    return x


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


def scans_from_datagram(datagram: bytes) -> list[RawScan]:
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

        valid = (dist_m > VALID_DISTANCE_THRESHOLD_M).astype(np.float32)

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
        """
        window shape: (2, 5, 240)
        returns: horse_prob, probs_np, pred_class
        """
        x = np.expand_dims(window, axis=0).astype(np.float32)  # (1, 2, 5, 240)
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

def select_high_valid_scan(scans: list[RawScan]) -> RawScan | None:
    """
    From one UDP datagram's worth of scans, keep only the scan with the
    highest valid_count. If scans is empty, return None.
    """
    if not scans:
        return None

    return max(scans, key=lambda scan: int(np.sum(scan.valid > 0)))

def run_live_pipeline(checkpoint_path: str | Path):
    global DETERRENT

    DETERRENT = DeterrentController()
    startup_led_sequence(DETERRENT)
    atexit.register(cleanup_deterrent)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    buffer = TemporalBuffer(sequence_length=SEQUENCE_LENGTH)
    inference = InferenceRunner(checkpoint_path)
    phase_processor = PhaseBlockProcessor()
    activation_controller = ActivationController()

    print(f"Listening on UDP {UDP_PORT}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Model threshold: {MODEL_THRESHOLD}")
    print(f"Activation window: {ACTIVATION_WINDOW}")
    print(f"Activate if >= {ACTIVATE_IF_AT_LEAST}/{ACTIVATION_WINDOW}")
    print(f"Deactivate if <= {DEACTIVATE_IF_AT_MOST}/{ACTIVATION_WINDOW}")
    print(f"Max range meters: {MAX_RANGE_METERS}")
    print(f"Deterrent ultrasonic GPIO: {ULTRASONIC_GPIO}")
    print(f"Deterrent LED GPIO: {LED_GPIO}")
    print("Preprocessing: FARFILL enabled")
    print("Scan selection: highest-valid-count scan only from each UDP chunk")
    print("Decision logic: rolling 3-scan phase block + 5-vote activation")
    print("Waiting for live scans...\n")

    step = 0

    try:
        while True:
            datagram, addr = sock.recvfrom(65535)
            scans = scans_from_datagram(datagram)

            if not scans:
                continue

            valid_counts_all = [int(np.sum(scan.valid > 0)) for scan in scans]
            selected_scan = select_high_valid_scan(scans)

            if selected_scan is None:
                continue

            step += 1

            frame = preprocess_scan_to_frame(selected_scan)
            buffer.add_frame(frame)

            valid_mask = selected_scan.valid > 0
            valid_count = int(np.sum(valid_mask))
            mean_dist = float(np.mean(selected_scan.dist_m[valid_mask])) if valid_count > 0 else 0.0

            if not buffer.is_full():
                print(
                    f"step={step:04d} | "
                    f"chunk_valid_counts={valid_counts_all} | "
                    f"selected_valid_count={valid_count} | "
                    f"buffering ({len(buffer.buffer)}/{SEQUENCE_LENGTH})"
                )
                continue

            window = buffer.get_window()
            horse_prob, probs_np, pred_class = inference.predict_horse_probability(window)

            block = phase_processor.update(horse_prob, valid_count, mean_dist)

            if block is None:
                if step % PRINT_EVERY == 0:
                    print(
                        f"step={step:04d} | "
                        f"chunk_valid_counts={valid_counts_all} | "
                        f"selected_valid_count={valid_count} | "
                        f"mean_dist={mean_dist:.2f} | "
                        f"no_horse={probs_np[0]:.6f} | "
                        f"horse_present={probs_np[1]:.6f} | "
                        f"pred_class={pred_class} | "
                        f"phase_block=warming"
                    )
                continue

            decision, block_prob, avg_valid, avg_dist, zero_count = block

            activation_action, positive_count = activation_controller.update(decision)

            if activation_action == "ACTIVATE":
                activate_deterrent()
            elif activation_action == "DEACTIVATE":
                deactivate_deterrent()

            if step % PRINT_EVERY == 0:
                print(
                    f"step={step:04d} | "
                    f"chunk_valid_counts={valid_counts_all} | "
                    f"selected_valid_count={valid_count} | "
                    f"mean_dist={mean_dist:.2f} | "
                    f"no_horse={probs_np[0]:.6f} | "
                    f"horse_present={probs_np[1]:.6f} | "
                    f"pred_class={pred_class} | "
                    f"block_prob={block_prob:.6f} | "
                    f"block_avg_valid={avg_valid:.2f} | "
                    f"block_avg_dist={avg_dist:.2f} | "
                    f"block_zero_count={zero_count} | "
                    f"block_decision={decision} | "
                    f"votes={positive_count}/{len(activation_controller.history)} | "
                    f"activation_state={'ACTIVE' if activation_controller.is_active else 'IDLE'} | "
                    f"activation_action={activation_action}"
                )

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, shutting down cleanly...")
    finally:
        try:
            deactivate_deterrent()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        cleanup_deterrent()


if __name__ == "__main__":
    checkpoint_path = "/home/pi/BearRepo/bestmodel.pt"
    run_live_pipeline(checkpoint_path)