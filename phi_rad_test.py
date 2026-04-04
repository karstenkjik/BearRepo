from pathlib import Path
import numpy as np

folder = Path("/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen")

for path in sorted(folder.glob("*.npz"))[:10]:
    with np.load(path) as data:
        print(path.name, "phi_rad =", float(data["phi_rad"]))