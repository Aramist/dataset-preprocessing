import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from correct_rearing_points import correct_rearing_points
from tqdm import tqdm

with open("consts.json", "r") as f:
    consts = json.load(f)

working_dir = Path(consts["working_dir"])
manual_save_file = working_dir / consts["track_annotation_path"]
audio_sr = consts["audio_sr"]
video_framerate = consts["video_framerate"]

corner_points = np.array(consts["corner_points"])
arena_dims_mm = np.array(consts["arena_dims_mm"])

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
    manual_data = pd.read_csv(manual_save_file)
    mask = ~np.isnan(
        manual_data[["p0_x", "p0_y", "p1_x", "p1_y"]].to_numpy().reshape(-1, 2, 2)
    ).any(axis=(-1, -2))
    subset = manual_data[mask]

    # Lists for dataset
    audio = []
    locations_px = []

    # Lists for metadata
    video_paths = []
    video_frame_indices = []

    unique_audio_paths = subset["audio_path"].unique()
    for audio_path in tqdm(unique_audio_paths):
        all_with_same_audio = subset[subset["audio_path"] == audio_path]
        segments = all_with_same_audio[["audio_onset", "audio_offset"]].to_numpy()
        snippets = get_audio(audio_path, segments)
        audio.extend(snippets)

        locations_px.extend(
            all_with_same_audio[["p0_x", "p0_y", "p1_x", "p1_y"]]
            .to_numpy()
            .reshape(-1, 2, 2)
        )

        video_paths.extend(all_with_same_audio["video_path"])
        video_frame_indices.extend(all_with_same_audio["video_index"])

    length_idx = np.cumsum([0] + [len(x) for x in audio])
    locations_px = np.array(locations_px)
    audio = np.concatenate(audio, axis=0)

    locations_px = correct_rearing_points(locations_px)

    with h5py.File(output_dataset_path, "w") as ctx:
        ctx.create_dataset("audio", data=audio)
        ctx.create_dataset("length_idx", data=length_idx)
        ctx.create_dataset("locations_px", data=locations_px)
        locations_mm = convert_locations_to_mm(locations_px)
        ctx.create_dataset("locations", data=locations_mm)

    metadata_df = pd.DataFrame(
        {
            "video_path": video_paths,
            "frame_idx": video_frame_indices,
        }
    )
    metadata_df.to_csv(output_metadata_path, index=False)

    vox_lengths = np.diff(length_idx)
    # Print stats
    print(f"Dataset saved to {output_dataset_path}")
    print(f"Number of vocalizations: {len(vox_lengths)}")
    print(
        f"Mean vocalization length: {vox_lengths.mean() / audio_sr * 1000:.1f} milliseconds"
    )
    print(
        f"Median vocalization length: {np.median(vox_lengths) / audio_sr * 1000:.1f} milliseconds"
    )
    gerbil_distance_px = np.linalg.norm(
        np.diff(locations_px, axis=1), axis=-1
    ).squeeze()
    print(f"Mean gerbil distance: {gerbil_distance_px.mean():.1f} pixels")
    print(f"Median gerbil distance: {np.median(gerbil_distance_px):.1f} pixels")

    gerbil_distance_cm = (
        np.linalg.norm(np.diff(locations_mm, axis=1), axis=-1).squeeze() / 10
    )
    print(f"Mean gerbil distance: {gerbil_distance_cm.mean():.1f} cm")
    print(f"Median gerbil distance: {np.median(gerbil_distance_cm):.1f} cm")


if __name__ == "__main__":
    main()
    np.save("corner_points.npy", corner_points)
