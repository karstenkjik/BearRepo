from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from model import LidarBinaryCNN2D
from test_preprocess_frame import load_raw_scan, preprocess_scan_to_frame
from test_temporal_buffer import TemporalBuffer


def safe_load_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Matches the repo's checkpoint loading pattern.
    """
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except Exception:
        return torch.load(checkpoint_path, map_location=device)


def build_window_from_folder(folder: Path, num_files_to_scan: int = 20) -> np.ndarray:
    """
    Loads raw scans from a session folder, preprocesses them, fills the temporal buffer,
    and returns the first full temporal window.

    Returns:
        window shape (2, 5, 240)
    """
    buffer = TemporalBuffer(sequence_length=5)

    files = sorted(folder.glob("*.npz"))[:num_files_to_scan]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    for npz_path in files:
        scan = load_raw_scan(npz_path)
        frame = preprocess_scan_to_frame(scan)
        buffer.add_frame(frame)

        if buffer.is_full():
            return buffer.get_window()

    raise RuntimeError(
        f"Could not build a full 5-frame window from first {num_files_to_scan} files in {folder}"
    )


def run_inference(window: np.ndarray, checkpoint_path: str | Path):
    """
    Runs the 2D temporal CNN on one temporal window.

    Args:
        window: np.ndarray of shape (2, 5, 240)
        checkpoint_path: path to Training2D best model checkpoint

    Returns:
        probs_np: np.ndarray of shape (2,)
        pred_class: int
    """
    if window.shape != (2, 5, 240):
        raise ValueError(f"Expected window shape (2,5,240), got {window.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)

    checkpoint = safe_load_checkpoint(checkpoint_path, device)

    model = LidarBinaryCNN2D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = np.expand_dims(window, axis=0).astype(np.float32)  # (1, 2, 5, 240)
    x_tensor = torch.from_numpy(x).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

    probs_np = probs[0].cpu().numpy()
    pred_class = int(np.argmax(probs_np))

    return probs_np, pred_class


def print_inference_summary(probs: np.ndarray, pred_class: int):
    class_names = ["no_horse", "horse_present"]

    print("\nProbabilities:")
    print(f"  no_horse:      {probs[0]:.6f}")
    print(f"  horse_present: {probs[1]:.6f}")
    print(f"Predicted class: {pred_class} ({class_names[pred_class]})")


if __name__ == "__main__":
    # CHANGE THESE TWO PATHS:
    session_folder = Path(
        "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/background/dataset_env_only"
    )
    checkpoint_path = Path(
        "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/Training2D/bestmodel.pt"
    )

    window = build_window_from_folder(session_folder)
    print("Built window shape:", window.shape)

    probs, pred_class = run_inference(window, checkpoint_path)
    print_inference_summary(probs, pred_class)