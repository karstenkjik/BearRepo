from __future__ import annotations

from pathlib import Path
import numpy as np

from test_preprocess_frame import load_raw_scan, preprocess_scan_to_frame
from test_temporal_buffer import TemporalBuffer
from test_model_inference import run_inference
from test_decision_logic import ProbabilityDecisionFilter


def run_capture_folder(capture_folder: str | Path, checkpoint_path: str | Path):
    capture_folder = Path(capture_folder)
    files = sorted(capture_folder.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {capture_folder}")

    buffer = TemporalBuffer(sequence_length=5)
    decision_filter = ProbabilityDecisionFilter(threshold=0.30, smooth_window=5)

    print(f"\nRunning capture folder: {capture_folder}")
    print(f"Total scans: {len(files)}\n")

    for i, npz_path in enumerate(files, start=1):
        scan = load_raw_scan(npz_path)
        frame = preprocess_scan_to_frame(scan)
        buffer.add_frame(frame)

        valid_count = int(np.sum(scan.valid > 0))

        if not buffer.is_full():
            print(f"step={i:03d} | buffering ({len(buffer.buffer)}/5) | valid_count={valid_count}")
            continue

        window = buffer.get_window()
        probs, pred_class = run_inference(window, checkpoint_path)
        horse_prob = float(probs[1])

        result = decision_filter.update(horse_prob)

        print(
            f"step={i:03d} | "
            f"valid_count={valid_count} | "
            f"no_horse={probs[0]:.6f} | "
            f"horse_present={probs[1]:.6f} | "
            f"raw={result.raw_probability:.6f} | "
            f"smoothed={result.smoothed_probability:.6f} | "
            f"decision={result.decision} | "
            f"pred_class={pred_class}"
        )


if __name__ == "__main__":
    # CHANGE THESE TWO PATHS
    capture_folder = "/Users/karstenkempfe/Desktop/BearAware/live_debug_captures_outdoor2"
    checkpoint_path = "/Users/karstenkempfe/Desktop/BearAware/bestmodel.pt"

    run_capture_folder(capture_folder, checkpoint_path)