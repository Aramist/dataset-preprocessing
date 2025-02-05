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
ram_dataset = namedtuple(
    "ram_dataset", ["audio", "locations", "locations_px", "length_idx", "node_names"]
)

# Custom sort key because I might end up with data distributed across multiple folders
str_path_date = lambda p: "_".join(p.stem.split("_")[:6])
annotation_path_date = lambda p: "_".join(p.stem.split("_")[1:7])

is_video = lambda x: ".mp4" in x.suffixes or ".avi" in x.suffixes

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
video_alias_paths = sum(
    (
        list(filter(is_video, Path(search_dir).iterdir()))
        for search_dir in consts["video_alias_dirs"]
    ),
    start=[],
)
processed_annotation_dir = working_dir / consts["processed_annotation_dir"]
if not processed_annotation_dir.exists():
    raise FileNotFoundError(
        "No processed annotations found. Run filter_annotations.py first"
    )
processed_annotation_paths = sorted(
    list(processed_annotation_dir.glob("*.npy")), key=annotation_path_date
)

processed_track_dir = working_dir / consts["processed_track_dir"]

corner_points = np.array(consts["arena_corner_points"])
corner_points_path = Path(consts["corner_points_path"])
if not corner_points_path.exists():
    np.save(corner_points_path, corner_points)
arena_dims = np.array(consts["arena_dims"])

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


def make_partial_dataset(annotation_path: Path):
    # Attempt to find the corresponding video file
    date = annotation_path_date(annotation_path)
    session = next(filter(lambda x: x.stem.startswith(date), session_dirs), None)
    if not session:
        print(f"Could not find session directory for {date}")
        return

    # Prefer to load the video from an alias (some of the original files are corrupted)
    video_path = next(
        filter(lambda v: v.stem.startswith(date), video_alias_paths), None
    )
    # if this fails, load from the original session directory
    if not video_path:
        print(f"Could not find alias for {date}. Loading from session directory")
        video_path = next(session.glob("*.avi"), None)
    # If this also fails, skip this session
    if not video_path:
        print(f"Could not find video for {date}")
        return

    audio_path = next(session.glob("*.h5"), None)
    if not audio_path:
        print(f"Could not find audio for {date}")
        return

    # Accidentally deleted the years from the track file names
    track_path = next(
        filter(
            lambda x: x.stem.startswith(date),
            processed_track_dir.glob("*.tracks.npy"),
        ),
        None,
    )
    if not track_path:
        print(f"Could not find tracks for {date}")
        print(annotation_path)
        print(video_path)
        print(audio_path)
        return

    # Load the processed annotations
    annotations = np.load(
        annotation_path
    )  # Shape: (n_segments, 2), unit: samples (int)

    try:
        processed_tracks = np.load(
            track_path
        )  # Shape: (n_frames, 2 nodes, 2 coords), unit: pixels (float)
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
            "x_px": locations_px[:, 0, 0],
            "y_px": locations_px[:, 0, 1],
        }
    )

    H = make_homology(corner_points)

    node_names = [b"nose", b"head"]
    dset = ram_dataset(
        audio=np.concatenate(audio, axis=0),
        locations=convert_points(locations_px, H),
        locations_px=locations_px,
        length_idx=len_idx,
        node_names=node_names,
    )

    return dset, metadata


def write_dataset(
    dataset: ram_dataset,
    dataset_path: Path,
):
    with h5py.File(dataset_path, "w") as ctx:
        ctx.create_dataset("audio", data=dataset.audio)
        ctx.create_dataset("length_idx", data=dataset.length_idx)
        ctx.create_dataset("locations", data=dataset.locations)
        ctx.create_dataset("locations_px", data=dataset.locations_px)
        ctx.create_dataset("node_names", data=dataset.node_names)


if __name__ == "__main__":
    partial_dataset_dir = Path(consts["partial_dataset_dir"])
    partial_dataset_dir.mkdir(exist_ok=True, parents=True)

    H = make_homology(corner_points)

    for annotation_path in tqdm(processed_annotation_paths):
        date = annotation_path_date(annotation_path)
        partial_dataset_path = partial_dataset_dir / (date + ".h5")
        metadata_path = partial_dataset_dir / (date + ".metadata.csv")
        if partial_dataset_path.exists() and metadata_path.exists():
            print(f"Skipping {date}")
            continue

        result = make_partial_dataset(annotation_path)
        if not result:
            continue
        dataset, metadata = result

        write_dataset(dataset, partial_dataset_path)
        metadata.to_csv(metadata_path, index=False, mode="w")
