from pathlib import Path
import numpy as np

file_path = Path("/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen/frame_1772589946750019836.npz")

with np.load(file_path) as data:
    print("keys:", data.files)
    print("timestamp_ns:", data["timestamp_ns"])
    print("phi_rad:", data["phi_rad"])
    print("dist_m shape:", data["dist_m"].shape)
    print("theta_rad shape:", data["theta_rad"].shape)
    print("valid shape:", data["valid"].shape)

    print("dist first 20:", data["dist_m"][:20])
    print("valid first 20:", data["valid"][:20])

    print("dist min/max:", np.min(data["dist_m"]), np.max(data["dist_m"]))
    print("valid unique:", np.unique(data["valid"]))