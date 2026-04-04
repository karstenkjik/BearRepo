from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


MAX_RANGE_METERS = 40.0


@dataclass
class RawScan:
    path: Path
    timestamp_ns: int
    dist_m: np.ndarray
    theta_rad: np.ndarray
    phi_rad: float
    valid: np.ndarray


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

    if not (dist_m.ndim == theta_rad.ndim == valid.ndim == 1):
        raise ValueError(
            f"Expected 1D arrays. Got dist_m={dist_m.shape}, "
            f"theta_rad={theta_rad.shape}, valid={valid.shape}"
        )

    if not (dist_m.shape == theta_rad.shape == valid.shape):
        raise ValueError(
            f"Shape mismatch: dist_m={dist_m.shape}, "
            f"theta_rad={theta_rad.shape}, valid={valid.shape}"
        )

    return RawScan(
        path=npz_path,
        timestamp_ns=timestamp_ns,
        dist_m=dist_m,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        valid=valid,
    )


def preprocess_scan_to_frame(scan: RawScan, max_range_m: float = MAX_RANGE_METERS) -> np.ndarray:
    """
    Convert one raw scan into one processed frame.

    Output shape:
        (2, 1, 240)

    Channel 0:
        normalized range values in [0, 1]
    Channel 1:
        validity mask (float32, usually 0.0 or 1.0)
    """
    if max_range_m <= 0:
        raise ValueError("max_range_m must be > 0")

    dist_m = scan.dist_m.astype(np.float32)
    valid = scan.valid.astype(np.float32)

    # Match repo behavior:
    # clip range to max range, divide by max range, set invalid to 0
    range_img = np.clip(dist_m, 0.0, max_range_m) / max_range_m
    range_img[valid == 0] = 0.0

    # Build 2-channel frame with height=1
    x = np.stack([range_img, valid], axis=0).astype(np.float32)  # (2, 240)
    x = np.expand_dims(x, axis=1)                                # (2, 1, 240)

    return x


def print_frame_summary(frame: np.ndarray, label: str = "frame") -> None:
    print(f"\n{label} shape: {frame.shape}")
    print(f"{label} dtype: {frame.dtype}")

    range_channel = frame[0]
    valid_channel = frame[1]

    print(f"range channel min/max: {range_channel.min():.6f} / {range_channel.max():.6f}")
    print(f"valid channel unique values: {np.unique(valid_channel)}")

    print("first 8 normalized range values:", range_channel[0, :8])
    print("first 8 valid values:", valid_channel[0, :8])

    nonzero_valid = int(np.sum(valid_channel > 0))
    print(f"valid count: {nonzero_valid}")


if __name__ == "__main__":
    example_path = "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen/frame_1772589640580168074.npz"

    scan = load_raw_scan(example_path)
    frame = preprocess_scan_to_frame(scan)
    print_frame_summary(frame)