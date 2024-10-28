import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

DOWNLOAD_PATH = Path(consts["download_path"])
PARTIAL_DATASET_DIR = Path(consts["partial_dataset_dir"])
ARENA_DIMS = consts["arena_dims"]
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]
VIDEO_FRAME_RATE = consts["video_frame_rate"]
NUM_MICROPHONES = consts["audio_num_channels"]
date_save_format = consts["date_format_for_saving"]


def parse_date_from_filename(filepath: Path) -> datetime:
    num = filepath.name
    # Will look like yyMMddHHmmss
    # Ex: 240307103101
    date = datetime.strptime(num, "%Y%m%d_%H%M%S")
    return date


def load_tracks_from_h5(h5_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns tracks in meters and node names"""
    with h5py.File(h5_file, "r") as f:
        tracks_m = f["tracks"][:][:, 0, :, :]
        names = f["node_names"][:]
    return tracks_m, names


def to_float(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float32) / np.iinfo(array.dtype).max).astype(np.float16)


def write_to_h5(
    output_path: Path,
    audio: list[np.ndarray],
    locations_mm: np.ndarray,
    length_idx: np.ndarray,
    node_names: np.ndarray,
):
    with h5py.File(output_path, "w") as f:
        f.attrs["arena_dims"] = ARENA_DIMS
        f.attrs["arena_dims_units"] = "mm"
        f.attrs["audio_sample_rate"] = AUDIO_SAMPLE_RATE
        f.attrs["video_frame_rate"] = VIDEO_FRAME_RATE

        f.create_dataset("audio", data=np.concatenate(audio, axis=0), dtype=np.float16)
        f.create_dataset("node_names", data=node_names)
        f.create_dataset("locations", data=locations_mm)
        f.create_dataset("length_idx", data=length_idx)


def write_metadata(
    output_path: Path, video_path: Path, onsets_in_video_frames: np.ndarray
):
    metadata = {
        "video_path": [str(video_path)] * len(onsets_in_video_frames),
        "onsets_in_video_frames": onsets_in_video_frames.tolist(),
    }

    df = pd.DataFrame(metadata)
    metadata_path = output_path.with_suffix(".csv")
    df.to_csv(metadata_path, index=False)


def make_partial_dataset(session_dir: Path, skip_if_exists: bool = True) -> np.ndarray:
    """Expects a directory containing the following:
    - An h5 file (locations)
    - A txt file (segments)
    - A numpy memmap file (audio of shape TxN)
    """
    file_date = parse_date_from_filename(session_dir)
    reformatted_date = file_date.strftime(date_save_format)
    output_path = PARTIAL_DATASET_DIR / f"{reformatted_date}.h5"
    if output_path.exists() and skip_if_exists:
        print(f"File already exists at {output_path}")
        return

    loc_file = next(session_dir.glob("*.h5"), None)
    seg_file = next(session_dir.glob("*.txt"), None)
    audio_file = next(session_dir.glob("*.mmap"), None)
    if loc_file is None or seg_file is None or audio_file is None:
        raise FileNotFoundError(
            f"Missing file in directory {session_dir}. Expected h5, txt, and mmap."
        )

    num_samples_in_file = int(audio_file.stem.split("_")[-3])
    handle = np.memmap(
        audio_file,
        dtype="int16",
        mode="r",
        shape=(num_samples_in_file, NUM_MICROPHONES),
    )

    # Given in seconds
    true_start_point = float(
        consts["manually_annotated_start_indices"][file_date.strftime(date_save_format)]
    )
    if math.isnan(true_start_point):
        return
    true_start_point = int(true_start_point * AUDIO_SAMPLE_RATE)

    durations = np.loadtxt(seg_file, dtype=np.int32).reshape(-1)
    # durations has format [delay_length, stim_duration, delay_length, stim_duration, ...]
    # Reformat to onset and offset indices
    cum_index = np.cumsum(np.insert(durations, 0, 0))
    stim_indices = np.arange(len(durations))[1::2]

    ############################
    # for debugging, remove later
    # stim_indices = stim_indices[:100]
    ############################

    onset_indices = cum_index[stim_indices]
    offset_indices = cum_index[stim_indices + 1]
    # Account for the estimated start point
    shift = true_start_point - onset_indices[0]
    onset_indices += shift
    offset_indices += shift

    # Load the locations
    video_path = next(session_dir.glob("*.mp4"), None)
    tracks, node_names = load_tracks_from_h5(loc_file)
    onsets_in_seconds = onset_indices.astype(float) / AUDIO_SAMPLE_RATE
    onsets_in_video_frames = (onsets_in_seconds * VIDEO_FRAME_RATE).astype(int)
    locations_m = tracks[onsets_in_video_frames]
    locations_mm = locations_m * 1000.0

    # Read the audio
    audio = [
        to_float(np.array(handle[onset:offset, :]))
        for onset, offset in zip(onset_indices, offset_indices)
    ]
    audio_lengths = offset_indices - onset_indices
    length_idx = np.cumsum(np.insert(audio_lengths, 0, 0))

    # Save the data
    write_to_h5(output_path, audio, locations_mm, length_idx, node_names)
    # Make metadata file
    write_metadata(output_path, video_path, onsets_in_video_frames)


if __name__ == "__main__":
    PARTIAL_DATASET_DIR.mkdir(exist_ok=True, parents=True)
    sessions = DOWNLOAD_PATH.iterdir()
    sessions = filter(
        lambda p: not set(p.stem).difference(set("0123456789_")), sessions
    )
    sessions = list(sessions)
    sessions.sort()
    for session in tqdm(sessions):
        make_partial_dataset(session, skip_if_exists=False)
