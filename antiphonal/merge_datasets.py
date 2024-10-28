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


def main():
    partial_datasets = list(partial_dataset_dir.glob("*.h5"))
    partial_datasets.sort()
    if not partial_datasets:
        raise FileNotFoundError(
            "No partial datasets found. Run make_partial_datasets.py first"
        )

    # First pass to get full dataset length
    full_dataset_length = 0
    num_mics = None
    full_dataset_lengths = []
    for partial_dset_path in partial_datasets:
        with h5py.File(partial_dset_path, "r") as ctx:
            full_dataset_length += len(ctx["audio"])
            num_mics = ctx["audio"].shape[1]
            full_dataset_lengths.extend(np.diff(ctx["length_idx"][:]))

    metadatas = []
    # Start merging
    with h5py.File(dset_path, "w") as ctx:
        ctx.create_dataset("audio", (full_dataset_length, num_mics), dtype=np.float32)
        ctx.create_dataset(
            "locations", (len(full_dataset_lengths), 2), dtype=np.float32
        )
        ctx.create_dataset(
            "length_idx",
            (len(full_dataset_lengths) + 1,),
            dtype=np.int64,
            data=np.cumsum([0] + full_dataset_lengths),
        )

        cur_audio_sample = 0
        cur_location_idx = 0
        for partial_dset_path in tqdm(partial_datasets):
            partial_metadata_path = partial_dset_path.with_suffix(".metadata.csv")
            partial_metadata = pd.read_csv(partial_metadata_path)
            metadatas.append(partial_metadata)
            with h5py.File(partial_dset_path, "r") as partial_ctx:
                audio = partial_ctx["audio"][:]
                locations = partial_ctx["locations"][:]

                ctx["audio"][cur_audio_sample : cur_audio_sample + len(audio)] = audio
                ctx["locations"][
                    cur_location_idx : cur_location_idx + len(locations), :
                ] = locations

                cur_audio_sample += len(audio)
                cur_location_idx += len(locations)

        print(
            f"Wrote {cur_location_idx} vocalizations totaling {cur_audio_sample} samples to {dset_path}"
        )
    full_metadata = pd.concat(metadatas)
    print(full_metadata)
    full_metadata.to_csv(full_metadata_path, index=False, mode="w")


if __name__ == "__main__":
    main()
