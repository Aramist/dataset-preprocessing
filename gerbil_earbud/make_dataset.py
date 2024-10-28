import json
from collections import namedtuple
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# A named tuple to hold a dataset that hasn't been written to disk yet
# The partial datasets correspoding to a single session should be small enough to fit in memory
ram_dataset = namedtuple(
    "ram_dataset", ["audio", "locations_px", "locations_mm", "length_idx"]
)

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

audio_sr = int(consts["audio_sr"])
video_framerate = int(consts["video_framerate"])
buffer_left_ms, buffer_right_ms = int(consts["vox_buffer_left_ms"]), int(
    consts["vox_buffer_right_ms"]
)
buffer_left_samps, buffer_right_samps = int(buffer_left_ms * audio_sr / 1000), int(
    buffer_right_ms * audio_sr / 1000
)
min_duration_ms = int(consts["min_duration_ms"])
min_duration_samps = int(min_duration_ms * audio_sr / 1000)
est_delay_ms = int(consts["estimated_playback_delay_ms"])
est_delay_samps = int(est_delay_ms * audio_sr / 1000)

corner_points_path = Path(consts["corner_points_path"])
corner_points = np.array(consts["corner_points"])  # Unit: px
arena_dims = np.array(consts["arena_dims"])  # Unit: mm
durations = np.load(consts["stim_durations_file"])  # Unit: s

working_dir = Path(consts["working_dir"])
data_dir = Path(consts["recording_session_dir"])
processed_annotation_dir = working_dir / consts["processed_annotation_dir"]
processed_track_dir = working_dir / consts["processed_track_dir"]

output_dataset_path = Path(consts["full_dataset_path"])
output_metadata_path = Path(consts["full_metadata_path"])


def make_homology(image_points: np.ndarray) -> np.ndarray:
    width, height = arena_dims
    half_width, half_height = width / 2, height / 2

    # This will always be the same, so we can hardcode it
    # Ordering is top left, top right, bottom right, bottom left
    target_points = np.array(
        [
            [-half_width, half_height],
            [half_width, half_height],
            [half_width, -half_height],
            [-half_width, -half_height],
        ]
    )

    H, _ = cv2.findHomography(image_points, target_points, method=cv2.RANSAC)
    return H


def interpolate_nans(tracks):
    # tracks have shape (time, coords)
    tracks = tracks.copy()
    n_ts, n_coords = tracks.shape
    for coord in range(n_coords):
        arr_slice = tracks[:, coord]
        nan_mask = np.isnan(arr_slice)
        where_valid = np.flatnonzero(~nan_mask)

        # fill nans at beginning and end with first and last valid value
        if (first_valid := where_valid[0]) > 0:
            arr_slice[:first_valid] = arr_slice[first_valid]
        if (last_valid := where_valid[-1]) < n_ts - 1:
            arr_slice[last_valid + 1 :] = arr_slice[last_valid]

        # update nan mask to account for changed values
        nan_mask = np.isnan(arr_slice)

        # Find long stretches of nans, we will leave these alone because
        # interpolation can be unreliable
        nan_onsets = np.flatnonzero(np.diff(nan_mask.astype(int)) == 1)
        nan_offsets = np.flatnonzero(np.diff(nan_mask.astype(int)) == -1)
        nan_regions = np.stack([nan_onsets, nan_offsets], axis=1)
        nan_regions = nan_regions[
            nan_regions[:, 1] - nan_regions[:, 0] > 30
        ]  # one second of nan

        # interpolate nans in between
        real_xp = np.arange(n_ts)[~nan_mask]
        real_fp = arr_slice[~nan_mask]
        eval_x = np.arange(n_ts)[nan_mask]

        fill_fp = np.interp(eval_x, real_xp, real_fp)
        arr_slice[nan_mask] = fill_fp

        # fill in long nan regions with nans
        for onset, offset in nan_regions:
            arr_slice[onset:offset] = np.nan

        tracks[:, coord] = arr_slice
    return tracks


def convert_points(points, H):
    # Given a stream and a point (in pixels), converts to inches within the global coordinate frame
    # Pixel coordinates should be presented in (x, y) order, with the origin in the top-left corner of the frame
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # System is M * [x_px y_px 1] ~ [x_r y_r 1]
    ones = np.ones((*points.shape[:-1], 1))
    points = np.concatenate([points, ones], axis=-1)
    prod = np.einsum("ij,...j->...i", H, points)[..., :-1]  # remove ones row
    return prod


def load_tracks() -> np.ndarray:
    track_file = next(data_dir.glob("*.analysis.h5"), None)
    if track_file is None:
        raise FileNotFoundError("No track file found")

    with h5py.File(track_file, "r") as f:
        tracks_px = f["tracks"][:].squeeze().T
        tracks_px = interpolate_nans(tracks_px)

    # Convert to mm
    homology = make_homology(corner_points)
    tracks_mm = convert_points(tracks_px, homology)
    return tracks_px, tracks_mm


def get_segments() -> np.ndarray:
    audio_file = next(data_dir.glob("mic_*.h5"), None)
    if audio_file is None:
        raise FileNotFoundError("No audio file found")

    with h5py.File(audio_file, "r") as f:
        recording_length = len(f["ai_channels/ai0"])
        onsets = f["audio_onset"][:, 0]
        onsets += est_delay_samps

        stim_durations = (durations[: len(onsets)] * audio_sr).astype(int)
        offsets = onsets + stim_durations

        onsets -= buffer_left_samps
        offsets += buffer_right_samps

        onsets = np.clip(onsets, 0, None)
        offsets = np.clip(offsets, None, recording_length)
        duration_thresh = min_duration_samps + buffer_left_samps + buffer_right_samps
        valid_mask = (offsets - onsets) >= duration_thresh
        onsets = onsets[valid_mask]
        offsets = offsets[valid_mask]
        segments = np.stack([onsets, offsets], axis=1)

    return segments


def fetch_audio(segments: np.ndarray) -> list[np.ndarray]:
    audio_file = next(data_dir.glob("mic_*.h5"), None)
    if audio_file is None:
        raise FileNotFoundError("No audio file found")

    audio_segments = []
    with h5py.File(audio_file, "r") as f:
        num_channels = len(f["ai_channels"].keys())
        for start, end in tqdm(segments):
            snippet = np.stack(
                [f[f"ai_channels/ai{i}"][start:end] for i in range(num_channels)],
                axis=1,
            )
            audio_segments.append(snippet)

    return audio_segments


def make_dataset() -> tuple[ram_dataset, pd.DataFrame]:
    tracks_px, tracks_mm = load_tracks()
    segments = get_segments()
    audio_segments = fetch_audio(segments)

    track_indices = (segments.mean(axis=1) / audio_sr * video_framerate).astype(int)
    track_indices = np.clip(track_indices, 0, len(tracks_px) - 1)

    tracks_px = tracks_px[track_indices, :]
    tracks_mm = tracks_mm[track_indices, :]
    length_idx = np.cumsum([0] + [len(seg) for seg in audio_segments])
    dset = ram_dataset(
        np.concatenate(audio_segments, axis=0), tracks_px, tracks_mm, length_idx
    )

    video_path = next(data_dir.glob("*.avi"), None)
    metadata = pd.DataFrame(
        {
            "video_path": [video_path] * len(segments),
            "frame_idx": track_indices,
        }
    )
    return dset, metadata


def save_dataset(dset: ram_dataset, output_path: Path):
    with h5py.File(output_path, "w") as f:
        f.create_dataset("audio", data=dset.audio)
        f.create_dataset("locations_px", data=dset.locations_px)
        f.create_dataset("locations", data=dset.locations_mm)
        f.create_dataset("length_idx", data=dset.length_idx)


if __name__ == "__main__":
    processed_annotation_dir.mkdir(parents=True, exist_ok=True)
    processed_track_dir.mkdir(parents=True, exist_ok=True)

    np.save(corner_points_path, corner_points)

    dset, metadata = make_dataset()
    save_dataset(dset, output_dataset_path)
    metadata.to_csv(output_metadata_path, index=False)
