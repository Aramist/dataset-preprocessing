import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

audio_sr = int(consts["audio_sr"])
working_dir = Path(consts["working_dir"])
partial_dataset_dir = Path(consts["partial_dataset_dir"])

dset_path = Path(consts["full_dataset_path"])
full_metadata_path = Path(consts["full_metadata_path"])


if __name__ == "__main__":
    with h5py.File(dset_path, "r") as ctx:
        num_vocalizations = len(ctx["length_idx"]) - 1
        vocalization_lengths = (
            np.diff(ctx["length_idx"][:]) / ctx.attrs["audio_sr"] * 1000
        )
        vlen_mean, vlen_std, vlen_min, vlen_max, vlen_q1, vlen_median, vlen_q3 = (
            vocalization_lengths.mean(),
            vocalization_lengths.std(),
            vocalization_lengths.min(),
            vocalization_lengths.max(),
            np.percentile(vocalization_lengths, 25),
            np.median(vocalization_lengths),
            np.percentile(vocalization_lengths, 75),
        )

        locations = ctx["locations"][:]
        x_min, x_max = locations[:, 0].min(), locations[:, 0].max()
        y_min, y_max = locations[:, 1].min(), locations[:, 1].max()
        naive_err = np.linalg.norm(locations - locations.mean(axis=0), axis=1).mean()

    print(f"Number of vocalizations: {num_vocalizations}")
    print(f"Vocalization length (ms):")
    print(f"  Mean: {vlen_mean}")
    print(f"  Std: {vlen_std}")
    print(f"  Min: {vlen_min}")
    print(f"  Max: {vlen_max}")
    print(f"  Q1: {vlen_q1}")
    print(f"  Median: {vlen_median}")
    print(f"  Q3: {vlen_q3}")
    print(f"X range: {x_min}, {x_max}")
    print(f"Y range: {y_min}, {y_max}")
    print(f"Naive error: {naive_err}")
