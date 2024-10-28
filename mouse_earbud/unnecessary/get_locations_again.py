import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

DOWNLOAD_PATH = Path(consts["download_path"])
PARTIAL_DATASET_DIR = Path(consts["partial_dataset_dir"])
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


def load_tracks_from_h5(h5_file: Path) -> np.ndarray:
    with h5py.File(h5_file, "r") as f:
        tracks = f["tracks"][:, 0, :, :]  # (n, 15, 3)

        node_names = f["node_names"][:]
        return tracks, node_names


def to_float(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float32) / np.iinfo(array.dtype).max).astype(np.float16)


def overwrite_locations_to_h5(
    output_path: Path, locations: np.ndarray, node_names: np.ndarray
):
    with h5py.File(output_path, "r+") as f:
        f.attrs["arena_dims"] = [615.0, 615.0, 425.0]
        f.attrs["arena_dims_units"] = "mm"
        f.attrs["audio_sample_rate"] = AUDIO_SAMPLE_RATE
        f.attrs["video_frame_rate"] = VIDEO_FRAME_RATE
        if "locations" in f:
            del f["locations"]

        locations_mm = locations * 1000.0  # m -> mm
        f.create_dataset("locations", data=locations_mm)

        if "orientations" in f:
            del f["orientations"]
        if "node_names" in f:
            del f["node_names"]

        f.create_dataset("node_names", data=node_names)


def make_partial_dataset(session_dir: Path, skip_if_exists: bool = True) -> np.ndarray:
    """Expects a directory containing the following:
    - An h5 file (locations)
    - A txt file (segments)
    - A numpy memmap file (audio of shape TxN)
    """
    file_date = parse_date_from_filename(session_dir)
    reformatted_date = file_date.strftime(date_save_format)
    output_path = PARTIAL_DATASET_DIR / f"{reformatted_date}.h5"
    if not output_path.exists():
        print(f"File already exists at {output_path}")
        return

    loc_file = next(
        filter(lambda p: "speaker" not in p.stem, session_dir.glob("*.h5")), None
    )
    seg_file = next(session_dir.glob("*.txt"), None)

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
    tracks, node_names = load_tracks_from_h5(loc_file)
    onsets_in_seconds = onset_indices.astype(float) / AUDIO_SAMPLE_RATE
    onsets_in_video_frames = (onsets_in_seconds * VIDEO_FRAME_RATE).astype(int)
    locations = tracks[onsets_in_video_frames]

    # Save the data
    overwrite_locations_to_h5(output_path, locations, node_names)


if __name__ == "__main__":
    PARTIAL_DATASET_DIR.mkdir(exist_ok=True, parents=True)
    sessions = DOWNLOAD_PATH.iterdir()
    sessions = filter(
        lambda p: not set(p.stem).difference(set("0123456789_")), sessions
    )
    sessions = list(sessions)
    sessions.sort()
    for session in tqdm(sessions):
        make_partial_dataset(session, skip_if_exists=True)

    # Run the merge script too
    from merge_overwrite_locations import main as merge_main

    merge_main()
