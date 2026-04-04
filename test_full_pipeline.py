from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from model import LidarBinaryCNN2D
from test_preprocess_frame import load_raw_scan, preprocess_scan_to_frame
from test_temporal_buffer import TemporalBuffer
from test_decision_logic import ProbabilityDecisionFilter


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

    def predict_horse_probability(self, window: np.ndarray) -> tuple[float, np.ndarray, int]:
        """
        window shape: (2, 5, 240)
        returns:
            horse_prob, probs_np, pred_class
        """
        if window.shape != (2, 5, 240):
            raise ValueError(f"Expected window shape (2,5,240), got {window.shape}")

        x = np.expand_dims(window, axis=0).astype(np.float32)  # (1, 2, 5, 240)
        x_tensor = torch.from_numpy(x).to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs[0].cpu().numpy()
        pred_class = int(np.argmax(probs_np))
        horse_prob = float(probs_np[1])

        return horse_prob, probs_np, pred_class


def run_session_pipeline(
    session_folder: str | Path,
    checkpoint_path: str | Path,
    threshold: float = 0.40,
    smooth_window: int = 5,
    max_files: int | None = 30,
) -> None:
    session_folder = Path(session_folder)
    files = sorted(session_folder.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {session_folder}")

    if max_files is not None:
        files = files[:max_files]

    buffer = TemporalBuffer(sequence_length=5)
    inference = InferenceRunner(checkpoint_path)
    decision_filter = ProbabilityDecisionFilter(
        threshold=threshold,
        smooth_window=smooth_window,
    )

    print(f"\nRunning session: {session_folder}")
    print(f"Total raw scans to process: {len(files)}")
    print(f"Threshold: {threshold}")
    print(f"Smoothing window: {smooth_window}\n")

    for i, npz_path in enumerate(files, start=1):
        scan = load_raw_scan(npz_path)
        frame = preprocess_scan_to_frame(scan)
        buffer.add_frame(frame)

        if not buffer.is_full():
            print(
                f"step={i:02d} | file={npz_path.name} | buffering "
                f"({len(buffer.buffer)}/5)"
            )
            continue

        window = buffer.get_window()
        horse_prob, probs_np, pred_class = inference.predict_horse_probability(window)
        result = decision_filter.update(horse_prob)

        print(
            f"step={i:02d} | file={npz_path.name} | "
            f"no_horse={probs_np[0]:.6f} | horse_present={probs_np[1]:.6f} | "
            f"raw={result.raw_probability:.6f} | "
            f"smoothed={result.smoothed_probability:.6f} | "
            f"decision={result.decision} | "
            f"pred_class={pred_class}"
        )


if __name__ == "__main__":
    # CHANGE THESE TWO PATHS:
    session_folder = "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen"
    checkpoint_path = "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/Training2D/bestmodel.pt"

    run_session_pipeline(
        session_folder=session_folder,
        checkpoint_path=checkpoint_path,
        threshold=0.3,
        smooth_window=5,
        max_files=30,
    )