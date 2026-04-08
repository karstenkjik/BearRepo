from __future__ import annotations

from pathlib import Path
import numpy as np
from collections import Counter

from test_preprocess_frame import load_raw_scan
from test_temporal_buffer import TemporalBuffer
from test_model_inference import run_inference
from test_decision_logic import ProbabilityDecisionFilter


MAX_RANGE_METERS = 40.0


def preprocess_original(scan):
    dist_m = scan.dist_m.astype(np.float32)
    valid = scan.valid.astype(np.float32)

    range_img = np.clip(dist_m, 0.0, MAX_RANGE_METERS) / MAX_RANGE_METERS
    range_img[valid == 0] = 0.0

    x = np.stack([range_img, valid], axis=0).astype(np.float32)
    x = np.expand_dims(x, axis=1)  # (2,1,240)
    return x


def preprocess_farfill(scan):
    dist_m = scan.dist_m.astype(np.float32)
    valid = scan.valid.astype(np.float32)

    range_img = np.clip(dist_m, 0.0, MAX_RANGE_METERS) / MAX_RANGE_METERS

    # Key alternate encoding:
    # no-return becomes "far" rather than "zero"
    range_img[valid == 0] = 1.0

    x = np.stack([range_img, valid], axis=0).astype(np.float32)
    x = np.expand_dims(x, axis=1)  # (2,1,240)
    return x


def summarize_folder(folder: Path):
    files = sorted(folder.glob("*.npz"))
    valid_counts = []
    valid_fracs = []
    nonzero_means = []
    nonzero_maxes = []

    for f in files:
        scan = load_raw_scan(f)
        valid_mask = scan.valid > 0
        vc = int(np.sum(valid_mask))
        valid_counts.append(vc)
        valid_fracs.append(vc / len(scan.valid))

        nz = scan.dist_m[scan.dist_m > 0]
        if len(nz) > 0:
            nonzero_means.append(float(np.mean(nz)))
            nonzero_maxes.append(float(np.max(nz)))

    print(f"\n=== {folder} ===")
    print("num scans:", len(files))
    if valid_counts:
        print("avg valid_count:", np.mean(valid_counts))
        print("min valid_count:", np.min(valid_counts))
        print("max valid_count:", np.max(valid_counts))
        print("most common valid_count values:", Counter(valid_counts).most_common(10))
    if nonzero_means:
        print("avg nonzero mean dist:", np.mean(nonzero_means))
        print("avg nonzero max dist:", np.mean(nonzero_maxes))


def replay_folder(folder: Path, checkpoint_path: Path, preprocess_fn, label: str, threshold: float = 0.40):
    files = sorted(folder.glob("*.npz"))
    buffer = TemporalBuffer(sequence_length=5)
    decision_filter = ProbabilityDecisionFilter(threshold=threshold, smooth_window=5)

    horse_probs = []
    decisions = []

    for i, f in enumerate(files, start=1):
        scan = load_raw_scan(f)
        frame = preprocess_fn(scan)
        buffer.add_frame(frame)

        if not buffer.is_full():
            continue

        window = buffer.get_window()
        probs, pred_class = run_inference(window, checkpoint_path)
        horse_prob = float(probs[1])
        result = decision_filter.update(horse_prob)

        horse_probs.append(horse_prob)
        decisions.append(result.decision)

    print(f"\n--- Replay: {label} ---")
    if horse_probs:
        print("num inference steps:", len(horse_probs))
        print("horse_prob mean:", float(np.mean(horse_probs)))
        print("horse_prob min:", float(np.min(horse_probs)))
        print("horse_prob max:", float(np.max(horse_probs)))
        print("horse_present decisions:", sum(d == "horse_present" for d in decisions))
        print("no_horse decisions:", sum(d == "no_horse" for d in decisions))
    else:
        print("No inference steps produced.")


if __name__ == "__main__":
    # CHANGE THESE PATHS
    checkpoint_path = Path("/Users/karstenkempfe/Desktop/BearAware/bestmodel.pt")

    folders = [
        Path("/Users/karstenkempfe/Desktop/BearAware/live_debug_captures"),
        Path("/Users/karstenkempfe/Desktop/BearAware/live_debug_captures_DOW"),
        Path("/Users/karstenkempfe/Desktop/BearAware/live_debug_captures_DOW2"),
        Path("/Users/karstenkempfe/Desktop/BearAware/live_debug_captures_outdoor"),
        Path("/Users/karstenkempfe/Desktop/BearAware/live_debug_captures_outdoor2"),
        # Optional: add DOW, DOW2, outdoor here too
    ]

    for folder in folders:
        summarize_folder(folder)
        replay_folder(folder, checkpoint_path, preprocess_original, f"{folder.name} | original", threshold=0.40)
        replay_folder(folder, checkpoint_path, preprocess_farfill, f"{folder.name} | farfill", threshold=0.40)