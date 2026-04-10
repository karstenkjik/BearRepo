from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import deque
import argparse
import re
import time

import numpy as np
import torch

from model import LidarBinaryCNN2D
from test_temporal_buffer import TemporalBuffer


# ============================================================
# Config
# ============================================================

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
# Offline deterrent controller
# ============================================================

class OfflineDeterrentController:
    """
    Simulates the live deterrent controller, but only prints state changes.
    """
    def __init__(self):
        self.is_active = False

    def startup_sequence(self, flashes: int = 3, delay_s: float = 0.1):
        print(">>> OFFLINE STARTUP LED SEQUENCE <<<")
        for i in range(flashes):
            print(f"    LED FLASH {i+1}/{flashes}: ON")
            time.sleep(delay_s)
            print(f"    LED FLASH {i+1}/{flashes}: OFF")
            time.sleep(delay_s)

    def activate(self):
        if self.is_active:
            return
        self.is_active = True
        print(">>> OFFLINE: DETERRENT ACTIVATED <<<")

    def deactivate(self):
        if not self.is_active:
            return
        self.is_active = False
        print(">>> OFFLINE: DETERRENT DEACTIVATED <<<")

    def cleanup(self):
        self.deactivate()


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

        # Same single-veto rule as live version
        if zero_count >= 2 and avg_valid < 25:
            decision = "no_horse"
        else:
            decision = "horse_present" if block_prob >= self.model_threshold else "no_horse"

        return decision, block_prob, avg_valid, avg_dist, zero_count


# ============================================================
# Preprocessing
# ============================================================

def preprocess_scan_to_frame(scan: RawScan, max_range_m: float = MAX_RANGE_METERS) -> np.ndarray:
    if max_range_m <= 0:
        raise ValueError("max_range_m must be > 0")

    dist_m = scan.dist_m.astype(np.float32)
    valid = scan.valid.astype(np.float32)

    range_img = np.clip(dist_m, 0.0, max_range_m) / max_range_m
    range_img[valid == 0] = 0.0

    x = np.stack([range_img, valid], axis=0).astype(np.float32)  # (2, 240)
    x = np.expand_dims(x, axis=1)                                # (2, 1, 240)
    return x


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
# Folder-of-NPZ loading
# ============================================================

FRAME_TS_RE = re.compile(r"frame_(\d+)\.npz$")


def extract_timestamp_from_name(path: Path) -> int | None:
    m = FRAME_TS_RE.search(path.name)
    if m:
        return int(m.group(1))
    return None


def load_single_frame_npz(npz_path: str | Path) -> RawScan:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    required = ["dist_m", "theta_rad", "phi_rad", "valid"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{npz_path.name} is missing required keys: {missing}")

    timestamp_ns = None
    if "timestamp_ns" in data:
        timestamp_ns = int(np.asarray(data["timestamp_ns"]).item())
    else:
        ts_from_name = extract_timestamp_from_name(npz_path)
        if ts_from_name is None:
            raise ValueError(
                f"{npz_path.name} has no timestamp_ns key and filename does not match frame_<timestamp>.npz"
            )
        timestamp_ns = ts_from_name

    dist_m = np.asarray(data["dist_m"], dtype=np.float32)
    theta_rad = np.asarray(data["theta_rad"], dtype=np.float32)
    phi_rad = float(np.asarray(data["phi_rad"]).item())
    valid = np.asarray(data["valid"]).astype(np.float32)

    if dist_m.shape != (240,):
        raise ValueError(f"{npz_path.name}: dist_m must have shape (240,), got {dist_m.shape}")
    if theta_rad.shape != (240,):
        raise ValueError(f"{npz_path.name}: theta_rad must have shape (240,), got {theta_rad.shape}")
    if valid.shape != (240,):
        raise ValueError(f"{npz_path.name}: valid must have shape (240,), got {valid.shape}")

    return RawScan(
        timestamp_ns=timestamp_ns,
        dist_m=dist_m,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        valid=valid,
    )


def load_scans_from_folder(folder_path: str | Path) -> list[RawScan]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected a folder, got: {folder}")

    npz_files = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix == ".npz"],
        key=lambda p: extract_timestamp_from_name(p) if extract_timestamp_from_name(p) is not None else p.name,
    )

    if not npz_files:
        raise ValueError(f"No .npz files found in {folder}")

    scans = [load_single_frame_npz(p) for p in npz_files]

    # Final sort by actual timestamp_ns from contents
    scans.sort(key=lambda s: s.timestamp_ns)

    print(f"Loaded {len(scans)} frame files from {folder}")
    print(f"First timestamp_ns: {scans[0].timestamp_ns}")
    print(f"Last  timestamp_ns: {scans[-1].timestamp_ns}")

    return scans


# ============================================================
# Main offline pipeline
# ============================================================

def run_offline_pipeline(
    checkpoint_path: str | Path,
    dataset_folder: str | Path,
    sleep_s: float = 0.0,
):
    scans = load_scans_from_folder(dataset_folder)

    deterrent = OfflineDeterrentController()
    deterrent.startup_sequence()

    buffer = TemporalBuffer(sequence_length=SEQUENCE_LENGTH)
    inference = InferenceRunner(checkpoint_path)
    phase_processor = PhaseBlockProcessor()
    activation_controller = ActivationController()

    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Model threshold: {MODEL_THRESHOLD}")
    print(f"Activation window: {ACTIVATION_WINDOW}")
    print(f"Activate if >= {ACTIVATE_IF_AT_LEAST}/{ACTIVATION_WINDOW}")
    print(f"Deactivate if <= {DEACTIVATE_IF_AT_MOST}/{ACTIVATION_WINDOW}")
    print(f"Max range meters: {MAX_RANGE_METERS}")
    print("Mode: OFFLINE FOLDER REPLAY")
    print("Preprocessing: ORIGINAL")
    print("Decision logic: rolling 3-scan phase block + 5-vote activation")
    print()

    summary = {
        "num_scans": 0,
        "num_block_decisions": 0,
        "num_activate": 0,
        "num_deactivate": 0,
        "num_horse_decisions": 0,
        "num_no_horse_decisions": 0,
        "active_steps": 0,
    }

    try:
        for step, scan in enumerate(scans, start=1):
            summary["num_scans"] += 1

            frame = preprocess_scan_to_frame(scan)
            buffer.add_frame(frame)

            valid_mask = scan.valid > 0
            valid_count = int(np.sum(valid_mask))
            mean_dist = float(np.mean(scan.dist_m[valid_mask])) if valid_count > 0 else 0.0

            if not buffer.is_full():
                print(
                    f"step={step:04d} | buffering ({len(buffer.buffer)}/{SEQUENCE_LENGTH}) | "
                    f"valid_count={valid_count}"
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            window = buffer.get_window()
            horse_prob, probs_np, pred_class = inference.predict_horse_probability(window)

            block = phase_processor.update(horse_prob, valid_count, mean_dist)

            if block is None:
                if step % PRINT_EVERY == 0:
                    print(
                        f"step={step:04d} | "
                        f"valid_count={valid_count} | "
                        f"mean_dist={mean_dist:.2f} | "
                        f"no_horse={probs_np[0]:.6f} | "
                        f"horse_present={probs_np[1]:.6f} | "
                        f"pred_class={pred_class} | "
                        f"phase_block=warming"
                    )
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            decision, block_prob, avg_valid, avg_dist, zero_count = block
            summary["num_block_decisions"] += 1

            if decision == "horse_present":
                summary["num_horse_decisions"] += 1
            else:
                summary["num_no_horse_decisions"] += 1

            activation_action, positive_count = activation_controller.update(decision)

            if activation_action == "ACTIVATE":
                deterrent.activate()
                summary["num_activate"] += 1
            elif activation_action == "DEACTIVATE":
                deterrent.deactivate()
                summary["num_deactivate"] += 1

            if activation_controller.is_active:
                summary["active_steps"] += 1

            if step % PRINT_EVERY == 0:
                print(
                    f"step={step:04d} | "
                    f"timestamp_ns={scan.timestamp_ns} | "
                    f"valid_count={valid_count} | "
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

            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping offline replay...")
    finally:
        deterrent.cleanup()

    print("\n================ OFFLINE SUMMARY ================")
    print(f"Total scans processed      : {summary['num_scans']}")
    print(f"Block decisions made       : {summary['num_block_decisions']}")
    print(f"'horse_present' decisions  : {summary['num_horse_decisions']}")
    print(f"'no_horse' decisions       : {summary['num_no_horse_decisions']}")
    print(f"Activation events          : {summary['num_activate']}")
    print(f"Deactivation events        : {summary['num_deactivate']}")
    print(f"Steps active               : {summary['active_steps']}")
    print(f"Final deterrent state      : {'ACTIVE' if activation_controller.is_active else 'IDLE'}")
    print("=================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Offline replay for folder of frame_*.npz files")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/Users/karstenkempfe/Desktop/BearAware/bestmodel.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help="Folder containing per-frame .npz files",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between scans to replay more slowly",
    )

    args = parser.parse_args()
    run_offline_pipeline(
        checkpoint_path=args.checkpoint,
        dataset_folder=args.dataset_folder,
        sleep_s=args.sleep,
    )


if __name__ == "__main__":
    main()