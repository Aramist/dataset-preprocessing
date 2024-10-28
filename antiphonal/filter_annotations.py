import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Custom sort key because I might end up with data distributed across multiple folders
path_date = lambda p: "_".join(p.stem.split("_")[1:7])

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

audio_sr = consts["audio_sr"]
min_vox_len_sec = consts["min_vox_len_ms"] / 1000
max_vox_len_sec = consts["max_vox_len_ms"] / 1000

vox_buffer_left_samples = int(consts["vox_buffer_left_ms"] * audio_sr // 1000)
vox_buffer_right_samples = int(consts["vox_buffer_right_ms"] * audio_sr // 1000)

working_dir = Path(consts["working_dir"])
audio_paths = sum(
    (
        list(Path(search_dir).glob("*/*.h5"))
        for search_dir in consts["recording_session_dirs"]
    ),
    start=[],
)
unprocessed_annotations_paths = sum(
    (
        list(Path(search_dir).glob("*/*.csv"))
        for search_dir in consts["recording_session_dirs"]
    ),
    start=[],
)
unprocessed_annotations_paths.sort(key=path_date)
processed_annotation_dir = working_dir / consts["processed_annotation_dir"]
processed_annotation_dir.mkdir(parents=True, exist_ok=True)

if not audio_paths:
    raise FileNotFoundError(
        "No audio files found in the provided recording session directory"
    )
if not unprocessed_annotations_paths:
    raise FileNotFoundError(
        "No annotation files found in the provided recording session directory"
    )

banned_words = consts["banned_words"]


def process_annotation(annotation_path: Path):
    date = path_date(annotation_path)

    if any(banned_word in date for banned_word in banned_words):
        print(f"Skipping {date} due to banned word")
        return None

    onset_path = list(filter(lambda x: date in x.name, audio_paths))
    if not onset_path:
        print(f"Could not find onset for {date}")
        return None
    else:
        onset_path = onset_path[0]

    df = pd.read_csv(annotation_path).dropna()

    segments = np.stack([df["start_seconds"].values, df["stop_seconds"].values], axis=1)

    # Find segments that are too short or too long
    segment_lengths = segments[:, 1] - segments[:, 0]
    mask = (segment_lengths >= min_vox_len_sec) & (segment_lengths <= max_vox_len_sec)
    segments = segments[mask, :]

    segments = (segments * audio_sr).astype(int)
    # Add a buffer to the start and end of each segment
    segments[:, 0] -= vox_buffer_left_samples
    segments[:, 1] += vox_buffer_right_samples
    segments = np.clip(segments, 0, None)

    with h5py.File(onset_path, "r") as ctx:
        speaker_onsets = ctx["audio_onset"][:, 0]

    # Filter out onsets that start within 2 seconds after a speaker onset
    # Should have shape (num mic onsets, num speaker onsets)
    invalid_ranges = np.stack(
        [speaker_onsets, speaker_onsets + 2.10 * audio_sr], axis=1
    )

    # Since we only want the audio between -5 minutes of the first stimulus and +5 of the last,
    # add the rest of the session as an invalid range
    invalid_ranges = np.concatenate(
        [
            np.array([0, speaker_onsets[0] - 5 * 60 * audio_sr])[None, :],
            invalid_ranges,
            np.array([speaker_onsets[-1] + 5 * 60 * audio_sr, np.inf])[None, :],
        ]
    )

    mask = (segments[:, 0, None] >= invalid_ranges[None, :, 0]) & (
        segments[:, 0, None] <= invalid_ranges[None, :, 1]
    )
    mask = mask.any(axis=1)
    # this mask should have shape (num mic onsets,)

    segments = segments[~mask, :]

    return segments


if __name__ == "__main__":
    total_num_instances = 0
    for annotation in tqdm(unprocessed_annotations_paths):
        new_segments = process_annotation(annotation)
        if new_segments is None:
            continue
        total_num_instances += len(new_segments)
        new_file_name = annotation.with_suffix(".npy").name
        new_path = processed_annotation_dir / new_file_name
        np.save(new_path, new_segments)
