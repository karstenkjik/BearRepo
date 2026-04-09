from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import deque
import time

import numpy as np
import torch

from model import LidarBinaryCNN2D
from test_temporal_buffer import TemporalBuffer


# ============================================================
# Config
# ============================================================

SEQUENCE_LENGTH = 5
MAX_RANGE_METERS = 40.0

# Model-side probability threshold used inside 3-scan phase block
MODEL_THRESHOLD = 0.50

# Final activation vote settings (over block decisions)
BLOCK_VOTE_WINDOW = 5
ACTIVATE_IF_AT_LEAST = 3
DEACTIVATE_IF_AT_MOST = 1

PRINT_EVERY = 1
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
# Preprocessing (ORIGINAL TRAINING-COMPATIBLE VERSION)
# ============================================================

def preprocess_scan_to_frame(scan: RawScan, max_range_m: float = MAX_RANGE_METERS) -> np.ndarray:
    """
    Original training-style preprocessing:
    - clip distances to max range
    - divide by max range
    - invalid/no-return points become 0.0 in range channel
    - validity mask is preserved as channel 1

    Output shape: (2, 1, 240)
    """
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
# 3-scan rolling phase block processor
# ============================================================

class PhaseBlockProcessor:
    def __init__(self, model_threshold: float = MODEL_THRESHOLD):
        self.buffer = deque(maxlen=3)
        self.model_threshold = model_threshold

    def update(self, horse_prob: float, valid_count: int, mean_valid_dist: float):
        """
        Returns:
            None if block not full yet
            otherwise dict with:
                block_decision
                block_horse_prob
                block_avg_valid
                block_avg_dist
                block_zero_count
        """
        self.buffer.append(
            {
                "horse_prob": float(horse_prob),
                "valid_count": int(valid_count),
                "mean_dist": float(mean_valid_dist),
            }
        )

        if len(self.buffer) < 3:
            return None

        horse_probs = [x["horse_prob"] for x in self.buffer]
        valid_counts = [x["valid_count"] for x in self.buffer]
        mean_dists = [x["mean_dist"] for x in self.buffer]

        block_horse_prob = max(horse_probs)
        block_avg_valid = float(sum(valid_counts) / 3.0)
        block_avg_dist = float(sum(mean_dists) / 3.0)
        block_zero_count = int(sum(1 for v in valid_counts if v == 0))

        # Heuristic vetoes for obvious sparse/open/far background
        if block_zero_count >= 2 and block_avg_valid < 25:
            block_decision = "no_horse"
        else:
            block_decision = "horse_present" if block_horse_prob >= self.model_threshold else "no_horse"

        return {
            "block_decision": block_decision,
            "block_horse_prob": block_horse_prob,
            "block_avg_valid": block_avg_valid,
            "block_avg_dist": block_avg_dist,
            "block_zero_count": block_zero_count,
        }


# ============================================================
# Final activation controller over block decisions
# ============================================================

class BlockActivationController:
    def __init__(
        self,
        window_size: int = BLOCK_VOTE_WINDOW,
        activate_if_at_least: int = ACTIVATE_IF_AT_LEAST,
        deactivate_if_at_most: int = DEACTIVATE_IF_AT_MOST,
    ):
        self.history = deque(maxlen=window_size)
        self.is_active = False
        self.activate_if_at_least = activate_if_at_least
        self.deactivate_if_at_most = deactivate_if_at_most

    def update(self, block_decision: str):
        val = 1 if block_decision == "horse_present" else 0
        self.history.append(val)

        if len(self.history) < self.history.maxlen:
            return "HOLD", sum(self.history)

        count = sum(self.history)

        if (not self.is_active) and count >= self.activate_if_at_least:
            self.is_active = True
            return "ACTIVATE", count

        if self.is_active and count <= self.deactivate_if_at_most:
            self.is_active = False
            return "DEACTIVATE", count

        return "HOLD", count


# ============================================================
# Full pipeline from folder
# ============================================================

def run_full_pipeline_from_folder(
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
    phase_processor = PhaseBlockProcessor(model_threshold=MODEL_THRESHOLD)
    activation_controller = BlockActivationController()

    print(f"\nRunning full pipeline from folder: {scan_folder}")
    print(f"Total scans: {len(files)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Model threshold (inside 3-scan block): {MODEL_THRESHOLD}")
    print(f"Activation vote window: {BLOCK_VOTE_WINDOW}")
    print(f"Activate if >= {ACTIVATE_IF_AT_LEAST}/{BLOCK_VOTE_WINDOW}")
    print(f"Deactivate if <= {DEACTIVATE_IF_AT_MOST}/{BLOCK_VOTE_WINDOW}")
    print(f"Max range meters: {MAX_RANGE_METERS}")
    print("Preprocessing: ORIGINAL (invalid/no-return -> 0.0)\n")

    step = 0
    num_block_decisions = 0
    num_activate = 0
    num_deactivate = 0

    for npz_path in files:
        step += 1

        scan = load_raw_scan(npz_path)
        frame = preprocess_scan_to_frame(scan)
        buffer.add_frame(frame)

        valid_mask = scan.valid > 0
        valid_count = int(np.sum(valid_mask))

        if np.any(valid_mask):
            mean_valid_dist = float(np.mean(scan.dist_m[valid_mask]))
        else:
            mean_valid_dist = 0.0

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

        block_info = phase_processor.update(
            horse_prob=horse_prob,
            valid_count=valid_count,
            mean_valid_dist=mean_valid_dist,
        )

        if block_info is None:
            if step % PRINT_EVERY == 0:
                print(
                    f"step={step:04d} | file={npz_path.name} | "
                    f"valid_count={valid_count} | "
                    f"mean_valid_dist={mean_valid_dist:.3f} | "
                    f"no_horse={probs_np[0]:.6f} | "
                    f"horse_present={probs_np[1]:.6f} | "
                    f"pred_class={pred_class} | "
                    f"phase_block=warming"
                )
            if delay_s > 0:
                time.sleep(delay_s)
            continue

        num_block_decisions += 1
        block_decision = block_info["block_decision"]

        activation_action, vote_count = activation_controller.update(block_decision)

        if activation_action == "ACTIVATE":
            activate_deterrent()
            num_activate += 1
        elif activation_action == "DEACTIVATE":
            deactivate_deterrent()
            num_deactivate += 1

        if step % PRINT_EVERY == 0:
            print(
                f"step={step:04d} | file={npz_path.name} | "
                f"valid_count={valid_count} | "
                f"mean_valid_dist={mean_valid_dist:.3f} | "
                f"no_horse={probs_np[0]:.6f} | "
                f"horse_present={probs_np[1]:.6f} | "
                f"pred_class={pred_class} | "
                f"block_prob={block_info['block_horse_prob']:.6f} | "
                f"block_avg_valid={block_info['block_avg_valid']:.2f} | "
                f"block_avg_dist={block_info['block_avg_dist']:.3f} | "
                f"block_zero_count={block_info['block_zero_count']} | "
                f"block_decision={block_decision} | "
                f"votes={vote_count}/{len(activation_controller.history)} | "
                f"activation_state={'ACTIVE' if activation_controller.is_active else 'IDLE'} | "
                f"activation_action={activation_action}"
            )

        if delay_s > 0:
            time.sleep(delay_s)

    print("\n=== Summary ===")
    print(f"Total scans processed: {step}")
    print(f"Total block decisions made: {num_block_decisions}")
    print(f"Final activation state: {'ACTIVE' if activation_controller.is_active else 'IDLE'}")
    print(f"Number of ACTIVATE events: {num_activate}")
    print(f"Number of DEACTIVATE events: {num_deactivate}")


if __name__ == "__main__":
    # CHANGE THESE TWO PATHS
    scan_folder = "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen"
    checkpoint_path = "/Users/karstenkempfe/Desktop/BearAware/bestmodel.pt"

    run_full_pipeline_from_folder(
        scan_folder=scan_folder,
        checkpoint_path=checkpoint_path,
        delay_s=0.0,
    )