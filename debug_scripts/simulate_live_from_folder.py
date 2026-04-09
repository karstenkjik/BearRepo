from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import deque
import time

import numpy as np
import torch

from model import LidarBinaryCNN2D
from test_temporal_buffer import TemporalBuffer
from test_decision_logic import ProbabilityDecisionFilter


# ============================================================
# Config
# ============================================================

SEQUENCE_LENGTH = 5

# Model-side decision settings
THRESHOLD = 0.40
SMOOTH_WINDOW = 5

# Control-side activation logic
ACTIVATION_WINDOW = 5
ACTIVATE_IF_AT_LEAST = 3
DEACTIVATE_IF_AT_MOST = 1

MAX_RANGE_METERS = 40.0
PRINT_EVERY = 1

# Optional delay between scans to mimic live behavior
SIMULATED_DELAY_S = 0.0


# ============================================================
# Shared raw scan structure
# ============================================================

@dataclass
class RawScan:
    path: Path
    timestamp_ns: int
    dist_m: np.ndarray
    theta_rad: np.ndarray
    phi_rad: float
    valid: np.ndarray


# ============================================================
# Placeholder deterrent hooks
# ============================================================

def activate_deterrent():
    print(">>> PLACEHOLDER: ACTIVATE DETERRENT <<<")


def deactivate_deterrent():
    print(">>> PLACEHOLDER: DEACTIVATE DETERRENT <<<")


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

        if (not self.is_active) and positive_count >= self.activate_if_at_least:
            self.is_active = True
            return "ACTIVATE", positive_count

        if self.is_active and positive_count <= self.deactivate_if_at_most:
            self.is_active = False
            return "DEACTIVATE", positive_count

        return "HOLD", positive_count


# ============================================================
# Raw scan loading
# ============================================================

REQUIRED_KEYS = {"timestamp_ns", "dist_m", "theta_rad", "phi_rad", "valid"}


def load_raw_scan(npz_path: str | Path) -> RawScan:
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"File does not exist: {npz_path}")

    with np.load(npz_path) as data:
        keys = set(data.files)
        missing = REQUIRED_KEYS - keys
        if missing:
            raise KeyError(
                f"Missing required keys in {npz_path.name}: {sorted(missing)}. "
                f"Found keys: {sorted(keys)}"
            )

        timestamp_ns = int(data["timestamp_ns"])
        dist_m = data["dist_m"].astype(np.float32)
        theta_rad = data["theta_rad"].astype(np.float32)
        phi_rad = float(data["phi_rad"])
        valid = data["valid"].astype(np.float32)

    return RawScan(
        path=npz_path,
        timestamp_ns=timestamp_ns,
        dist_m=dist_m,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        valid=valid,
    )


# ============================================================
# Preprocessing (FARFILL)
# ============================================================

def preprocess_scan_to_frame(scan: RawScan, max_range_m: float = MAX_RANGE_METERS) -> np.ndarray:
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
# Simulated live pipeline from folder
# ============================================================

def run_simulated_live_folder(
    scan_folder: str | Path,
    checkpoint_path: str | Path,
    delay_s: float = SIMULATED_DELAY_S,
):
    scan_folder = Path(scan_folder)
    files = sorted(scan_folder.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {scan_folder}")

    buffer = TemporalBuffer(sequence_length=SEQUENCE_LENGTH)
    inference = InferenceRunner(checkpoint_path)
    decision_filter = ProbabilityDecisionFilter(
        threshold=THRESHOLD,
        smooth_window=SMOOTH_WINDOW,
    )
    activation_controller = ActivationController()

    print(f"\nSimulating live run from folder: {scan_folder}")
    print(f"Total scans: {len(files)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Smoothing window: {SMOOTH_WINDOW}")
    print(f"Activation window: {ACTIVATION_WINDOW}")
    print(f"Activate if >= {ACTIVATE_IF_AT_LEAST}/{ACTIVATION_WINDOW}")
    print(f"Deactivate if <= {DEACTIVATE_IF_AT_MOST}/{ACTIVATION_WINDOW}")
    print(f"Max range meters: {MAX_RANGE_METERS}")
    print("Preprocessing: FARFILL enabled\n")

    step = 0

    for npz_path in files:
        step += 1

        scan = load_raw_scan(npz_path)
        frame = preprocess_scan_to_frame(scan)
        buffer.add_frame(frame)

        valid_count = int(np.sum(scan.valid > 0))

        if not buffer.is_full():
            print(
                f"step={step:04d} | file={npz_path.name} | "
                f"buffering ({len(buffer.buffer)}/{SEQUENCE_LENGTH}) | "
                f"valid_count={valid_count}"
            )
            if delay_s > 0:
                time.sleep(delay_s)
            continue

        window = buffer.get_window()
        horse_prob, probs_np, pred_class = inference.predict_horse_probability(window)
        result = decision_filter.update(horse_prob)

        activation_action, positive_count = activation_controller.update(result.decision)

        if activation_action == "ACTIVATE":
            activate_deterrent()
        elif activation_action == "DEACTIVATE":
            deactivate_deterrent()

        if step % PRINT_EVERY == 0:
            print(
                f"step={step:04d} | file={npz_path.name} | "
                f"valid_count={valid_count} | "
                f"no_horse={probs_np[0]:.6f} | "
                f"horse_present={probs_np[1]:.6f} | "
                f"raw={result.raw_probability:.6f} | "
                f"smoothed={result.smoothed_probability:.6f} | "
                f"decision={result.decision} | "
                f"pred_class={pred_class} | "
                f"activation_votes={positive_count}/{len(activation_controller.history)} | "
                f"activation_state={'ACTIVE' if activation_controller.is_active else 'IDLE'} | "
                f"activation_action={activation_action}"
            )

        if delay_s > 0:
            time.sleep(delay_s)


if __name__ == "__main__":
    # CHANGE THESE TWO PATHS
    scan_folder = "/Users/karstenkempfe/Desktop/BearAware/live_debug_captures_DOW2"
    checkpoint_path = "/Users/karstenkempfe/Desktop/BearAware/bestmodel.pt"

    run_simulated_live_folder(
        scan_folder=scan_folder,
        checkpoint_path=checkpoint_path,
        delay_s=0.0,
    )