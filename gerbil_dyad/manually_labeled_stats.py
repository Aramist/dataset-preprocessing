import json
from pathlib import Path

import numpy as np
import pandas as pd

with open("consts.json", "r") as f:
    consts = json.load(f)

working_dir = Path(consts["working_dir"])
manual_save_file = working_dir / consts["track_annotation_path"]


df = pd.read_csv(manual_save_file)

print(df.columns)
all_points = df[["p0_x", "p0_y", "p1_x", "p1_y"]].to_numpy().reshape(-1, 2, 2)
mask = ~np.isnan(all_points).any(axis=(-1, -2))
all_points = all_points[mask]
print(all_points)
print(mask.sum())
