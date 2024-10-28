import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

DOWNLOAD_PATH = Path(consts["download_path"])
DATASET_PATH = Path(consts["full_dataset_path"])
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]
VIDEO_FRAME_RATE = consts["video_frame_rate"]
NUM_MICROPHONES = consts["audio_num_channels"]


def load_tracks_from_h5(h5_file: Path) -> np.ndarray:
    with h5py.File(h5_file, "r") as f:
        locations = f["tracks"][
            :, :, 0, :
        ]  # n, 2, 3 for n frames, 2 animals, 3 dimensions
        return locations


def to_float(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float32) / np.iinfo(array.dtype).max).astype(np.float16)


def write_to_h5(
    output_path: Path,
    audio: list[np.ndarray],
    locations: np.ndarray,
    length_idx: np.ndarray,
    extra_metadata: Optional[dict] = None,
):
    with h5py.File(output_path, "w") as f:
        if extra_metadata is not None:
            for k, v in extra_metadata.items():
                f.attrs[k] = v
        f.create_dataset("audio", data=np.concatenate(audio, axis=0), dtype=np.float16)
        f.create_dataset("locations", data=locations)
        f.create_dataset("length_idx", data=length_idx)


def merge(A, B):
    # Assumes A and B are sorted by onset and within each series, all onsets
    # are greater than the subsequent offset
    i, j = 0, 0

    cur_onset, cur_offset = None, None
    merged = []

    while i < len(A) and j < len(B):
        if A[i][0] < B[j][0]:
            cur_onset, cur_offset = A[i]
            while B[j][0] < cur_offset:
                cur_offset = max(cur_offset, B[j][1])
                j += 1
                if j >= len(B):
                    break
            merged.append((cur_onset, cur_offset))
            i += 1
        else:
            cur_onset, cur_offset = B[j]
            while A[i][0] < cur_offset:
                cur_offset = max(cur_offset, A[i][1])
                i += 1
                if i >= len(A):
                    break
            merged.append((cur_onset, cur_offset))
            j += 1
    return merged


def merge_segments(segment_files: list[Path]) -> np.ndarray:
    def load_file(fpath: Path) -> np.ndarray:
        df = pd.read_csv(fpath)
        return np.stack(
            [df["start_seconds"].to_numpy(), df["stop_seconds"].to_numpy()], axis=1
        )

    segments = load_file(segment_files[0])
    segments = np.sort(segments, axis=0)
    for seg_file in segment_files[1:]:
        segments = merge(segments, np.sort(load_file(seg_file), axis=0))

    return np.array(segments)


def load_arena_dims(arena_dims_path: Path) -> np.ndarray:
    with h5py.File(arena_dims_path, "r") as f:
        # order north, east, south, west
        four_corners = f["tracks"][0, 0, :4, :]
        awidth = np.max(four_corners[:, 0]) - np.min(four_corners[:, 0])
        aheight = np.max(four_corners[:, 1]) - np.min(four_corners[:, 1])
    return np.array([awidth, aheight])


def make_dataset(data_dir: Path, output_path: Path) -> np.ndarray:
    """Expects a directory containing the following:
    - An h5 file (locations)
    - A txt file (segments)
    - A numpy memmap file (audio of shape TxN)
    """
    track_dir = data_dir / "tracking"
    track_path = next(
        filter(lambda p: "arena" not in p.name, track_dir.glob("*.h5")), None
    )
    arena_dims_path = next(
        filter(lambda p: "arena" in p.name, track_dir.glob("*.h5")), None
    )

    seg_dir = data_dir / "audio" / "das_annotations"
    segment_files = list(seg_dir.glob("*.csv"))

    audio_file = next((data_dir / "audio").glob("*.mmap"), None)
    if track_path is None or not segment_files or audio_file is None:
        raise FileNotFoundError(
            f"Missing files in directory {data_dir}. Consider re-running download_dataset.py"
        )

    adims = load_arena_dims(arena_dims_path)
    segments = merge_segments(segment_files)
    num_samples_in_file = int(audio_file.stem.split("_")[-3])
    handle = np.memmap(
        audio_file,
        dtype="int16",
        mode="r",
        shape=(num_samples_in_file, NUM_MICROPHONES),
    )

    onset_indices = (segments[:, 0] * AUDIO_SAMPLE_RATE).astype(int)
    offset_indices = (segments[:, 1] * AUDIO_SAMPLE_RATE).astype(int)
    # Load the locations
    tracks = load_tracks_from_h5(track_path)
    onsets_in_seconds = segments[:, 0]
    onsets_in_video_frames = (onsets_in_seconds * VIDEO_FRAME_RATE).astype(int)
    locations = tracks[onsets_in_video_frames]
    locations = locations * 1000

    # Read the audio
    audio = [
        to_float(np.array(handle[onset:offset, :]))
        for onset, offset in zip(onset_indices, offset_indices)
    ]
    audio_lengths = offset_indices - onset_indices
    length_idx = np.cumsum(np.insert(audio_lengths, 0, 0))

    # Save the data
    extra_metadata = {
        "arena_dims": adims * 1000,
        "arena_dims_units": "mm",
        "audio_sr": AUDIO_SAMPLE_RATE,
        "video_fps": VIDEO_FRAME_RATE,
    }
    write_to_h5(
        output_path,
        audio,
        locations,
        length_idx,
        extra_metadata,
    )


if __name__ == "__main__":
    make_dataset(DOWNLOAD_PATH, DATASET_PATH)
