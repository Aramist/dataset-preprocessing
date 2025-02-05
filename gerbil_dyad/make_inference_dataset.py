import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from correct_rearing_points import correct_rearing_points
from tqdm import tqdm

annotation_path_date = lambda p: "_".join(
    p.stem.split("_")[1:-2]
)  # Drop the microseconds

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

audio_sr = consts["audio_sr"]
video_framerate = consts["video_framerate"]
min_vox_len_sec = consts["min_vox_len_ms"] / 1000
max_vox_len_sec = consts["max_vox_len_ms"] / 1000

vox_buffer_left_samples = int(consts["vox_buffer_left_ms"] * audio_sr // 1000)
vox_buffer_right_samples = int(consts["vox_buffer_right_ms"] * audio_sr // 1000)

working_dir = Path(consts["working_dir"])
processed_annotations_dir = working_dir / consts["processed_annotation_dir"]

video_label = lambda p: p.parent.name.split("_")[-1]

corner_points = np.array(consts["corner_points"])
arena_dims_mm = np.array(consts["arena_dims_mm"])

partial_dataset_dir = Path(consts["partial_dataset_dir"])
output_dataset_path = Path(consts["full_dataset_path"])
output_metadata_path = Path(consts["full_metadata_path"])


def get_audio(audio_path, segments):
    all_audio = []
    with h5py.File(audio_path, "r") as ctx:
        for n, (start, stop) in enumerate(segments):
            num_channels = len(ctx["ai_channels"].keys())
            audio = np.stack(
                [ctx["ai_channels"][f"ai{i}"][start:stop] for i in range(num_channels)],
                axis=1,
            )
            all_audio.append(audio)

    return all_audio


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


def make_homology(image_points: np.ndarray) -> np.ndarray:
    width, height = arena_dims_mm
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

    # found that what I originally wrote was rotated 180 degrees relative to solo gerbil
    # rolled the corner points in consts.json to fix this

    H, _ = cv2.findHomography(image_points, target_points, method=cv2.RANSAC)
    return H


def convert_locations_to_mm(locations_px: np.ndarray) -> np.ndarray:
    H = make_homology(corner_points)
    locations_mm = convert_points(locations_px, H)
    return locations_mm


def main():
    annotations_files = list(processed_annotations_dir.glob("*.npy"))
    audio_files = sum(
        (
            list(Path(search_dir).glob("*/mic_*.h5"))
            for search_dir in consts["recording_session_dirs"]
        ),
        start=[],
    )
    video_files = sum(
        (
            list(Path(search_dir).glob("*/*.avi"))
            for search_dir in consts["recording_session_dirs"]
        ),
        start=[],
    )

    for annotation_file in annotations_files:
        # Figure out which audio and video file go with it
        date = annotation_path_date(annotation_file)

        vid_f = next(filter(lambda p: date in p.stem, video_files), None)
        aud_f = next(filter(lambda p: date in p.stem, audio_files), None)
        if vid_f is None or aud_f is None:
            print(f"Couldn't find matching video or audio file for {annotation_file}")
            continue

        # Populate columns
        segments = np.load(annotation_file)  # unit: samples

        audio = get_audio(aud_f, segments)
        length_idx = np.cumsum([0] + [len(x) for x in audio])
        with h5py.File(partial_dataset_dir / f"{date}_audio.h5", "w") as ctx:
            ctx.create_dataset("audio", data=np.concatenate(audio, axis=0))
            ctx.create_dataset("length_idx", data=length_idx)


if __name__ == "__main__":
    main()
