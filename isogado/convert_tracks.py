"""Converts tracks from pixels to mm and saves the results as numpy arrays.
"""

import argparse
import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

# Normally, the expected order of points is: top left, top right, bottom right, bottom left
# However, the adolescent dataset seems to have its camera coordinates rotated 180 degrees relative to other datasets
# Therefore, the order presented here is: bottom right, bottom left, top left, top right
# Prior to August 3, the arena was slightly tilted, so the points change slightly after that date
tilted_date_range = (datetime.datetime(2023, 6, 1), datetime.datetime(2023, 8, 3))
tilted_arena_corner_points = np.array([[581, 386], [87, 400], [75, 79], [572, 65]])
straight_arena_corner_points = np.array([[580, 407], [91, 403], [90, 90], [573, 92]])


def get_video_date(video_path: str) -> datetime.datetime:
    """Extracts the date from a video path. Assumes the path is of the form
    /path/to/dir/yyyy_mm_dd_hh_mm_ss_000000_name.avi
    """
    date_string = "_".join(
        Path(video_path).stem.split("_")[:6]
    )  # drop the milliseconds
    return datetime.datetime.strptime(date_string, "%Y_%m_%d_%H_%M_%S")


def make_homology(image_points: np.ndarray) -> np.ndarray:
    arena_dims = np.array([558.9, 355.6])
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
    # tracks have shape (num_nodes, num_coords, time)
    tracks = tracks.copy()
    n_nodes, n_coords, n_ts = tracks.shape
    for node in range(n_nodes):
        for coord in range(n_coords):
            arr_slice = tracks[node, coord, :]
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

            tracks[node, coord] = arr_slice
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


def process_tracks(track_path: str) -> np.ndarray:
    """Given a sleap analysis file, converts the tracks from pixels to mm and
    returns the results as a numpy array.
    """
    track_date = get_video_date(track_path)
    if track_date >= tilted_date_range[0] and track_date < tilted_date_range[1]:
        arena_corner_points = tilted_arena_corner_points
    else:
        arena_corner_points = straight_arena_corner_points
    H = make_homology(arena_corner_points)

    with h5py.File(track_path, "r") as ctx:
        tracks = ctx["tracks"][0, ...]
        tracks = interpolate_nans(tracks)
        tracks = np.transpose(tracks, (2, 1, 0))  # time, nodes, coords

        orig_shape = tracks.shape
        tracks = tracks.reshape(-1, 2)
        tracks = convert_points(tracks, H)
        tracks = tracks.reshape(orig_shape)

    arena_dims = np.array([558.9, 355.6])
    tracks = clamp_tracks(tracks, arena_dims)
    return tracks


def clamp_tracks(tracks: np.ndarray, arena_dims: tuple[float, float]):
    # Clamps tracks to arena dims, ignoring nan values
    # tracks have shape (time, num_nodes, num_coords)
    orig_shape = tracks.shape
    tracks = tracks.reshape(-1, 2)

    non_nan_mask = ~np.isnan(tracks).any(axis=1)
    tracks[non_nan_mask] = np.clip(
        tracks[non_nan_mask], -arena_dims / 2, arena_dims / 2
    )
    tracks = tracks.reshape(orig_shape)
    return tracks


def run(track_paths: list[str], output_dir: str):
    Path(output_dir).mkdir(exist_ok=True)
    for track_path in track_paths:
        tracks = process_tracks(track_path)
        track_name = Path(track_path).stem
        track_name = track_name[: track_name.find("preds")] + "npy"
        track_path = str(Path(output_dir) / track_name)
        np.save(track_path, tracks)


if __name__ == "__main__":
    # Request a list of track files and convert all of them
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "track_paths", help="Path to the track file to convert", nargs="+"
    )
    parser.add_argument("output_dir", help="Path to save the output numpy arrays")
    args = parser.parse_args()

    run(args.track_paths, args.output_dir)
