import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

FULL_DATASET_PATH = Path(consts["full_dataset_path"])
PARTIAL_DATASET_DIR = Path(consts["partial_dataset_dir"])
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]
VIDEO_FRAME_RATE = consts["video_frame_rate"]
NUM_MICROPHONES = consts["audio_num_channels"]


def main():
    if not PARTIAL_DATASET_DIR.exists():
        raise FileNotFoundError(
            "No partial datasets found. Run process_datasets.py first."
        )
    # First pass through partial datasets to get full length
    all_locations = []
    node_names = None
    partial_dataset_paths = list(PARTIAL_DATASET_DIR.glob("*.h5"))
    partial_dataset_paths.sort()
    for partial_dataset_path in partial_dataset_paths:
        with h5py.File(partial_dataset_path, "r") as f:
            all_locations.append(f["locations"][:])
            if node_names is None:
                node_names = f["node_names"][:]

    all_locations = np.concatenate(all_locations)

    print(f"Total num locations: {len(all_locations)}")

    if not FULL_DATASET_PATH.exists():
        raise FileNotFoundError("No full dataset found. Run process_datasets.py first.")

    print(f"Overwriting locations in {FULL_DATASET_PATH}")
    with h5py.File(FULL_DATASET_PATH, "r+") as f:
        # Copy attrs from one dataset
        if "locations" in f:
            del f["locations"]
        f.create_dataset("locations", data=all_locations)

        if "orientations" in f:
            del f["orientations"]
        if "node_names" in f:
            del f["node_names"]
        f.create_dataset("node_names", data=node_names)


if __name__ == "__main__":
    main()
