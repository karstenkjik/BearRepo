from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


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
    """
    Load one raw LiDAR scan .npz file and validate its contents.

    Expected keys:
        - timestamp_ns
        - dist_m
        - theta_rad
        - phi_rad
        - valid
    """
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

    # Basic shape checks
    if dist_m.ndim != 1:
        raise ValueError(f"dist_m should be 1D, got shape {dist_m.shape}")
    if theta_rad.ndim != 1:
        raise ValueError(f"theta_rad should be 1D, got shape {theta_rad.shape}")
    if valid.ndim != 1:
        raise ValueError(f"valid should be 1D, got shape {valid.shape}")

    if not (dist_m.shape == theta_rad.shape == valid.shape):
        raise ValueError(
            "dist_m, theta_rad, and valid must have the same shape. "
            f"Got dist_m={dist_m.shape}, theta_rad={theta_rad.shape}, valid={valid.shape}"
        )

    return RawScan(
        path=npz_path,
        timestamp_ns=timestamp_ns,
        dist_m=dist_m,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        valid=valid,
    )


def print_scan_summary(scan: RawScan) -> None:
    """
    Print a safe, human-readable summary for debugging.
    """
    print(f"\nLoaded: {scan.path}")
    print(f"timestamp_ns: {scan.timestamp_ns}")
    print(f"phi_rad: {scan.phi_rad}")
    print(f"dist_m shape: {scan.dist_m.shape}, dtype: {scan.dist_m.dtype}")
    print(f"theta_rad shape: {scan.theta_rad.shape}, dtype: {scan.theta_rad.dtype}")
    print(f"valid shape: {scan.valid.shape}, dtype: {scan.valid.dtype}")

    print(f"dist_m min/max: {np.nanmin(scan.dist_m):.4f} / {np.nanmax(scan.dist_m):.4f}")
    print(f"theta_rad min/max: {np.nanmin(scan.theta_rad):.4f} / {np.nanmax(scan.theta_rad):.4f}")
    print(f"valid unique values: {np.unique(scan.valid)}")

    print("first 8 dist_m values:", scan.dist_m[:8])
    print("first 8 theta_rad values:", scan.theta_rad[:8])
    print("first 8 valid values:", scan.valid[:8])


if __name__ == "__main__":
    # Replace this with one actual raw scan path from your repo
    example_path = "/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen/frame_1772589904707293620.npz"

    scan = load_raw_scan(example_path)
    print_scan_summary(scan)