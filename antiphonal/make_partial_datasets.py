import json
from collections import namedtuple
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

# A named tuple to hold a dataset that hasn't been written to disk yet
# The partial datasets correspoding to a single session should be small enough to fit in memory
ram_dataset = namedtuple("ram_dataset", ["audio", "locations", "length_idx"])

# Custom sort key because I might end up with data distributed across multiple folders
str_path_date = lambda p: "_".join(p.stem.split("_")[1:-1])

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

audio_sr = int(consts["audio_sr"])
working_dir = Path(consts["working_dir"])
session_dirs = sum(
    (
        list(Path(search_dir).iterdir())
        for search_dir in consts["recording_session_dirs"]
    ),
    start=[],
)
processed_annotation_dir = working_dir / consts["processed_annotation_dir"]
if not processed_annotation_dir.exists():
    raise FileNotFoundError(
        "No processed annotations found. Run filter_annotations.py first"
    )
processed_annotation_paths = sorted(
    list(processed_annotation_dir.glob("*.npy")), key=str_path_date
)

processed_track_dir = working_dir / consts["processed_track_dir"]

track_dir = Path(consts["sleap_track_dir"])
if not track_dir.exists():
    raise FileNotFoundError("No SLEAP tracks found. Run make_tracks.py first")


def get_framerate(video_path: Path):
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    return cap.get(cv2.CAP_PROP_FPS)


def get_audio(audio_path: Path, annotations: np.ndarray) -> list[np.ndarray]:
    if audio_path.suffix == ".h5":
        with h5py.File(audio_path, "r") as ctx:
            file_length = len(ctx["ai_channels/ai0"])
            num_channels = len(ctx["ai_channels"].keys())

            clipped_annotations = np.clip(annotations, 0, file_length)
            audio = [
                np.stack(
                    [
                        ctx["ai_channels"][f"ai{i}"][start:stop]
                        for i in range(num_channels)
                    ],
                    axis=1,
                )
                for start, stop in clipped_annotations
            ]
        return audio
    elif audio_path.suffix == ".wav":
        fs, audio = wavfile.read(audio_path)
        clipped_annotations = np.clip(annotations, 0, len(audio))
        return [audio[start:stop] for start, stop in clipped_annotations]
    else:
        raise ValueError("Audio file must be .wav or .h5")


def make_partial_dataset(annotation_path: Path):
    # Attempt to find the corresponding video file
    date = str_path_date(annotation_path)
    session = next(filter(lambda x: x.stem.startswith(date), session_dirs), None)
    if not session:
        print(f"Could not find session directory for {date}")
        return

    video_path = next(session.glob("*.avi"), None)
    if not video_path:
        print(f"Could not find video for {date}")
        return

    audio_path = next(session.glob("*.h5"), None)
    if not audio_path:
        print(f"Could not find audio for {date}")
        return

    # Accidentally deleted the years from the track file names
    date_without_year = "_".join(date.split("_")[1:])
    track_path = next(
        filter(
            lambda x: date_without_year in x.stem,
            processed_track_dir.glob("*.tracks.npy"),
        ),
        None,
    )
    if not track_path:
        print(f"Could not find tracks for {date}")
        return

    # Load the processed annotations
    annotations = np.load(
        annotation_path
    )  # Shape: (n_segments, 2), unit: samples (int)

    try:
        processed_tracks = np.load(
            track_path
        )  # Shape: (n_frames, 2), unit: pixels (float)
    except Exception as e:
        print(e)
        print(track_path)
        return

    # Convert annotations to video frame indices
    framerate = get_framerate(video_path)
    video_frame_indices = (
        (annotations.astype(float) / audio_sr * framerate).mean(axis=1).astype(int)
    )
    valid_mask = (video_frame_indices >= 0) & (
        video_frame_indices < len(processed_tracks)
    )
    video_frame_indices = video_frame_indices[valid_mask]
    annotations = annotations[valid_mask]

    locations_px = processed_tracks[video_frame_indices]

    # Gather audio
    audio = get_audio(audio_path, annotations)
    if not audio:
        return

    lengths = [len(arr) for arr in audio]
    len_idx = np.cumsum([0] + lengths)

    # Make metadata df
    # metadata cols: video_path,frame_idx,x_px,y_px
    metadata = pd.DataFrame(
        {
            "video_path": [str(video_path)] * len(video_frame_indices),
            "frame_idx": video_frame_indices,
            "x_px": locations_px[:, 0],
            "y_px": locations_px[:, 1],
        }
    )

    return ram_dataset(np.concatenate(audio, axis=0), locations_px, len_idx), metadata


def make_homology(image_points: np.ndarray) -> np.ndarray:
    arena_dims = np.array([572.0, 360.0])
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


def write_dataset(
    dataset: ram_dataset,
    dataset_path: Path,
    converted_locations: Optional[np.ndarray] = None,
):
    with h5py.File(dataset_path, "w") as ctx:
        ctx.create_dataset("audio", data=dataset.audio)
        ctx.create_dataset("locations_px", data=dataset.locations)
        if converted_locations is not None:
            ctx.create_dataset("locations", data=converted_locations)
        ctx.create_dataset("length_idx", data=dataset.length_idx)


if __name__ == "__main__":
    partial_dataset_dir = Path(consts["partial_dataset_dir"])
    partial_dataset_dir.mkdir(exist_ok=True, parents=True)

    corner_points = np.load(consts["corner_points_path"])
    H = make_homology(corner_points)

    for annotation_path in tqdm(processed_annotation_paths):
        date = str_path_date(annotation_path)
        partial_dataset_path = partial_dataset_dir / (date + ".h5")
        metadata_path = partial_dataset_dir / (date + ".metadata.csv")
        if partial_dataset_path.exists() and metadata_path.exists():
            print(f"Skipping {date}")
            continue

        result = make_partial_dataset(annotation_path)
        if not result:
            continue
        dataset, metadata = result

        locations_converted = convert_points(dataset.locations, H)

        write_dataset(
            dataset, partial_dataset_path, converted_locations=locations_converted
        )
        metadata.to_csv(metadata_path, index=False, mode="w")
