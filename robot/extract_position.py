import argparse
import glob
from pathlib import Path
from typing import List

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_homography():
    # pixel coordinates of quad corners
    # determined by manually analyzing the video '2023_07_20_16_07_54_323236_cam_d.avi'
    tl_px = (100, 100)
    tr_px = (560, 100)
    br_px = (560, 400)
    bl_px = (100, 400)
    source_points = np.stack([tl_px, tr_px, br_px, bl_px]).astype(float)
    # halved arena dimensions
    a_hwidth, a_hheight = np.array([572.0, 360.0]) / 2
    dest_points = -np.array(
        [
            [-a_hwidth, a_hheight],
            [a_hwidth, a_hheight],
            [a_hwidth, -a_hheight],
            [-a_hwidth, -a_hheight],
        ]
    )

    H, _ = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC)
    return H


def get_track_path(video_path: str) -> str:
    """Locates the tracks corresponding to a video under the assumption that
    they are in the same directory

    Args:
        video_path (str): Path to video
    """
    vid_dir = Path(video_path).parent
    track_glob = vid_dir / "*.analysis.h5"
    return glob.glob(str(track_glob))[0]


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


def interpolate_nans(tracks: np.ndarray):
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

            # interpolate nans in between
            real_xp = np.arange(n_ts)[~nan_mask]
            real_fp = arr_slice[~nan_mask]
            eval_x = np.arange(n_ts)[nan_mask]

            fill_fp = np.interp(eval_x, real_xp, real_fp)
            arr_slice[nan_mask] = fill_fp

            tracks[node, coord] = arr_slice
    return tracks


def get_tracks_in_metric(track_paths: List[str]) -> List[np.ndarray]:
    """Loads SLEAP tracks for all videos and does basic preprocessing

    Args:
        track_paths (List[str]): paths to all tracks

    Returns:
        List[np.ndarray]:
    """
    tracks, orientations = [], []

    for track_path in track_paths:
        t, o = process_tracks(track_path)
        tracks.append(t)
        orientations.append(o)
    return tracks, orientations


def process_tracks(analysis_fname):
    H = get_homography()
    with h5py.File(analysis_fname, "r") as ctx:
        tracks = ctx["tracks"][0, ...]  # there is only one animal
        tracks = interpolate_nans(tracks)
        tracks = np.transpose(tracks, (2, 1, 0))  # time, nodes, coords

        orig_shape = tracks.shape
        tracks = tracks.reshape(-1, 2)
        tracks = convert_points(tracks, H)
        tracks = tracks.reshape(orig_shape)

        orientations = tracks[:, 2, :] - tracks[:, 1, :]
        orientations /= np.linalg.norm(orientations, axis=-1, keepdims=True)

        tracks = tracks[:, 0, :]
        return tracks, orientations


if __name__ == "__main__":
    # directory structure:
    # video/
    #   video_paths.csv
    #   vid_0.npy
    #   vid_1.npy
    #   ...
    #   vid_n.npy
    # video_paths.csv links video paths to track paths

    # get video paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-paths", nargs="+", type=str)
    args = ap.parse_args()
    if args.video_paths is None:
        video_paths = list(Path("/mnt/ceph/users/rpeterson/ssl/robot").glob("*/*.avi"))
    else:
        video_paths = args.video_paths
    video_paths.sort()

    track_paths = [get_track_path(vp) for vp in video_paths]
    tracks, orientations = get_tracks_in_metric(track_paths)

    new_track_paths = [f"video/{vp.stem}_tracks.npy" for vp in video_paths]
    new_orientation_paths = [f"video/{vp.stem}_orientations.npy" for vp in video_paths]

    df = pd.DataFrame(
        {
            "video_path": video_paths,
            "track_path": new_track_paths,
            "orientation_path": new_orientation_paths,
        }
    )

    Path("video").mkdir(exist_ok=True)
    df.to_csv("video/video_paths.csv")

    for track, path in zip(tracks, new_track_paths):
        np.save(path, track)

    for orientation, path in zip(orientations, new_orientation_paths):
        np.save(path, orientation)
