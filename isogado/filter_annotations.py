import glob
from os import path
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

AUDIO_SR = 125000
# Custom sort key because I might end up with data distributed across multiple folders
path_date = lambda p: "_".join(Path(p).stem.split("_")[1:-1])
annotations_paths = list(
    glob.glob("/mnt/ceph/users/rpeterson/ssl/gerbil/adolescents/das-adolescent/*.csv")
) + list(glob.glob("/mnt/ceph/users/rpeterson/ssl/adolescents/*/*.csv"))
annotations_paths.sort(key=path_date)

# h5transfer only contains dates 6/20 and 6/16
audio_paths = sorted(
    # glob.glob("/mnt/home/atanelus/ceph/h5transfer/adolescents/*/*.h5"), key=path_date
    glob.glob("/mnt/ceph/users/rpeterson/ssl/adolescents/*/*.h5"),
    key=path_date,
)

num_instances_counted = 0


def process_annotation(annotation_path):
    date = path_date(annotation_path)
    print(date)

    onset_path = list(filter(lambda x: date in x, audio_paths))
    if not onset_path:
        print(f"Could not find onset for {date}")
        return None
    else:
        onset_path = onset_path[0]

    df = pd.read_csv(annotation_path).dropna()

    segments = np.stack([df["start_seconds"].values, df["stop_seconds"].values], axis=1)
    segments = (segments * AUDIO_SR).astype(int)
    print(f"Before: {segments.shape}")

    with h5py.File(onset_path, "r") as ctx:
        speaker_onsets = ctx["audio_onset"][:, 0]

    # Filter out onsets that start within 2 seconds after a speaker onset
    # Should have shape (num mic onsets, num speaker onsets)
    invalid_ranges = np.stack(
        [speaker_onsets, speaker_onsets + 2.10 * AUDIO_SR], axis=1
    )
    mask = (segments[:, 0, None] >= invalid_ranges[None, :, 0]) & (
        segments[:, 0, None] <= invalid_ranges[None, :, 1]
    )
    mask = mask.any(axis=1)
    # this mask should have shape (num mic onsets,)

    segments = segments[~mask, :]
    print(f"After: {segments.shape}")
    print(f"Num speaker playbacks: {speaker_onsets.shape[0]}")
    print()

    return segments


if __name__ == "__main__":
    for annotation in annotations_paths:
        new_segments = process_annotation(annotation)
        if new_segments is None:
            continue
        Path("filtered_annotations").mkdir(exist_ok=True)
        new_path = Path(annotation).stem + "_filtered.npy"
        new_path = path.join("filtered_annotations", new_path)
        np.save(new_path, new_segments)
