from pathlib import Path
import numpy as np

folder = Path("/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen")

files = sorted(folder.glob("*.npz"))[:20]

for path in files:
    with np.load(path) as data:
        dist = data["dist_m"]
        valid = data["valid"]
        nonzero_valid = int(np.sum(valid > 0))
        dist_max = float(np.max(dist))
        print(path.name, "valid_count =", nonzero_valid, "dist_max =", dist_max)