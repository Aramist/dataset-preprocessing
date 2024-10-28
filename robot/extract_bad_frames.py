import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

VIDEO_SR = 30
pca_corner_size = 2


def video_iterator(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    while ret:
        yield frame
        ret, frame = cap.read()


def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def get_video_framerate(video_path: Path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if np.abs(fps) < 1e-5:
        print(f"Warning: fps is {fps} for {video_path}, falling back to 30")
        fps = 30
    return fps


def arena_pixel_validator():
    # Uses an exponentially moving average to account for drift
    running_mean = None
    tau = 0.999

    while True:
        pixel_value = yield
        if running_mean is None:
            running_mean = pixel_value.astype(float)

        pixel_validity = (np.abs(pixel_value - running_mean) < 6).all()
        if pixel_validity:  # don't adjust to outliers
            running_mean = tau * running_mean + (1 - tau) * pixel_value
        yield pixel_validity


def merge(pairs):
    # yields pairs without overlaps
    # assumes pairs are sorted by onset
    cur = pairs[0]
    working = None
    should_yield_final = True
    i = 1
    while i < len(pairs):
        working = pairs[i]
        w_on, w_off = working
        c_on, c_off = cur

        if w_on < c_off:
            # There is overlap, merge the two pairs
            cur = np.array([c_on, w_off])
            should_yield_final = True
        else:
            # No more overlap, yield the current pair and reset to the next pair
            yield cur
            cur = working
            should_yield_final = (
                False  # if the stream ends here, don't yield the final pair
            )
        i += 1
    if should_yield_final:
        yield cur


def find_invalid_location_frames(track_path: Path) -> np.ndarray:
    """A second pass filter for invalid frames which locates frames containing
    locations outside the expected region. Returns a boolean mask which is True
    for valid frames and False for invalid frames.
    """
    arena_dims = (
        np.load("/mnt/ceph/users/atanelus/speaker_ssl/robot/arena_dims.npy") * 1.05
    )  # add 5% padding
    half_dims = arena_dims / 2
    tracks = np.load(track_path)  # has shape (n, 3, 2)
    valid = (
        (tracks[..., 0] > -half_dims[0])
        & (tracks[..., 0] < half_dims[0])
        & (tracks[..., 1] > -half_dims[1])
        & (tracks[..., 1] < half_dims[1])
    ).all(
        axis=1
    )  # shape: (n,)

    print(f"arena_dims thresh: {half_dims}")
    print(f"num invalid frames: {(~valid).sum()}")
    return valid


def get_invalid_frames(video_path: Path, track_path: Path) -> np.ndarray:
    """Returns a list of pairs of indices for the onsets and offsets of invalid
    regions within the video. Each pair is of the form (onset, offset) where
    onset and offset are integer indices for frames."""
    test_point = (0, 0)
    frames = video_iterator(video_path)
    validator = arena_pixel_validator()
    pixel_valid_frames = []

    for frame, _ in zip(frames, validator):
        pixel = frame[test_point[1], test_point[0], :]
        pixel_valid = validator.send(pixel)
        pixel_valid_frames.append(pixel_valid)
    pixel_valid_frames = np.array(pixel_valid_frames, dtype=bool)

    # Fold in information from the track file
    track_valid_frames = find_invalid_location_frames(track_path)
    # frames which were initially True but now False
    print(f"Num flipped: {(valid_frames & ~track_valid_frames).sum()}")
    # Only accept frames marked true by both criteria
    valid_frames = pixel_valid_frames & track_valid_frames

    onsets = np.flatnonzero(valid_frames[:-1] & ~valid_frames[1:])  # true to false
    offsets = np.flatnonzero(~valid_frames[:-1] & valid_frames[1:])  # false to true

    if len(onsets) == 0 or len(offsets) == 0:
        return np.zeros((0, 2), dtype=int)

    # prune edge cases
    if onsets[0] > offsets[0]:
        offsets = offsets[1:]
    if onsets[-1] > offsets[-1]:
        onsets = onsets[:-1]
    pairs = np.stack([onsets, offsets], axis=1)

    # Add 15s padding and merge resulting onset/offset pairs
    framerate = get_video_framerate(video_path)
    pairs[:, 0] -= 15 * framerate
    pairs[:, 1] += 15 * framerate

    if len(pairs) > 1:
        pairs = list(merge(pairs))
    pairs = np.array(pairs).astype(
        int
    )  # has shape (n, 2), wheret he onsets and offsets are indices for frames

    # Clip between 0 and video length
    vid_length = get_video_length(video_path)
    pairs = np.clip(pairs, 0, vid_length - 1)
    return pairs


if __name__ == "__main__":
    data_fpath = Path("video/video_paths.csv")
    if not data_fpath.exists():
        raise FileNotFoundError(f"Could not find {data_fpath}")

    csv_data = pd.read_csv(data_fpath)
    video_paths = csv_data["video_path"]
    track_paths = csv_data["track_path"]

    expanded_video_paths = []
    invalid_indices_for_videos = []
    for video_path, track_path in zip(video_paths, track_paths):
        print(f"Processing {video_path}")
        try:
            invalid_idx = get_invalid_frames(video_path, track_path)
            if len(invalid_idx) == 0:
                continue
            invalid_indices_for_videos.append(invalid_idx)
            # Expand the video paths into the rows for the csv
            expanded_video_paths.extend([video_path] * len(invalid_idx))
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")
            continue
    # flatten the invalid indices list so it can go into a dataframe
    invalid_array = np.concatenate(invalid_indices_for_videos, axis=0)  # shape (N, 2)
    invalid_onsets = invalid_array[:, 0]
    invalid_offsets = invalid_array[:, 1]

    # For reference: timestamps are given in terms of frame indices, not seconds
    df = pd.DataFrame(
        {
            "video_path": expanded_video_paths,
            "invalid_onset": invalid_onsets,
            "invalid_offset": invalid_offsets,
        }
    )

    """ Note that the following rows were manually appended to the csv. Still
    need to find the bug which causes the invalid frames to be missed. If the
    script is run again, these rows need to be added manually again.

    /mnt/ceph/users/rpeterson/ssl/robot/2023_07_21_14_29_15_032121_robot_3/2023_07_21_14_29_15_032121_cam_d.avi,4747,5647
    /mnt/ceph/users/rpeterson/ssl/robot/2023_07_21_14_29_15_032121_robot_3/2023_07_21_14_29_15_032121_cam_d.avi,45209,46109
    """
    df.to_csv("video/invalid_regions.csv", index=False)
