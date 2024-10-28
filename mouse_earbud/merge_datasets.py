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
    all_lengths = []

    partial_dataset_paths = list(PARTIAL_DATASET_DIR.glob("*.h5"))
    partial_dataset_paths.sort()
    for partial_dataset_path in partial_dataset_paths:
        with h5py.File(partial_dataset_path, "r") as f:
            all_locations.append(f["locations"][:])
            all_lengths.append(np.diff(f["length_idx"][:]))

    all_locations = np.concatenate(all_locations)
    all_lengths = np.concatenate(all_lengths)
    print(f"Total num samples: {all_lengths.sum()}")
    print(f"Total num locations: {len(all_locations)}")

    with h5py.File(FULL_DATASET_PATH, "w") as f:
        # Copy attrs from one dataset
        with h5py.File(partial_dataset_paths[0], "r") as f_partial:
            for key, value in f_partial.attrs.items():
                f.attrs[key] = value
            f["node_names"] = f_partial["node_names"][:]

        full_length_idx = np.cumsum(np.insert(all_lengths, 0, 0))
        f.create_dataset("locations", data=all_locations)
        f.create_dataset("length_idx", data=full_length_idx)
        audio_dset = f.create_dataset(
            "audio",
            shape=(full_length_idx[-1], NUM_MICROPHONES),
            dtype="float16",
            chunks=(AUDIO_SAMPLE_RATE, NUM_MICROPHONES),
        )

        # Second pass through partial datasets to write audio
        start_idx = 0
        for partial_dataset_path in tqdm(partial_dataset_paths):
            with h5py.File(partial_dataset_path, "r") as f_partial:
                audio = f_partial["audio"][:].astype(np.float16)
                end_idx = start_idx + len(audio)
                audio_dset[start_idx:end_idx] = audio
                start_idx = end_idx


if __name__ == "__main__":
    main()
