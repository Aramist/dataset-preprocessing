import json
from collections import namedtuple
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import pandas as pd
import soundfile as sf
from librosa import resample
from tqdm import tqdm

ram_dataset = namedtuple(
    "ram_dataset", ["audio", "locations_mm", "locations_px", "length_idx"]
)

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)
data_dir = Path(consts["recording_session_dir"])
recording_sessions = list(filter(lambda p: p.is_dir(), data_dir.glob("*/*")))
recording_sessions.sort()

working_dir = Path(consts["working_dir"])

processed_annotation_dir = working_dir / consts["processed_annotation_dir"]
if not processed_annotation_dir.exists():
    raise FileNotFoundError(
        "No processed annotations found. Run filter_annotations.py first"
    )
processed_annotation_paths = sorted(list(processed_annotation_dir.glob("*.npy")))

processed_track_dir = working_dir / consts["processed_track_dir"]
partial_dataset_dir = Path(consts["partial_dataset_dir"])
audio_sr = consts["audio_sr"]

arena_corner_points = np.array(consts["arena_corner_points_px"])
arena_dims_mm = np.array(consts["arena_dims_mm"])


def get_framerate(video_path: Path) -> int:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_num_frames(video_path: Path) -> int:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames


def get_audio(audio_path: Path, segments: np.ndarray) -> list[np.ndarray]:
    total_audio, _ = sf.read(audio_path)
    audio_segments = []
    for start, end in segments:
        if end > len(total_audio):
            continue
        clip = total_audio[start:end, :]
        # clip = resample(y=clip, orig_sr=audio_sr, target_sr=44100, axis=0)
        audio_segments.append(clip)
    return audio_segments


def convert_locations_to_mm(points: np.ndarray) -> np.ndarray:
    image_points = arena_corner_points
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

    H, _ = cv2.findHomography(image_points, target_points, method=cv2.RANSAC)

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
):
    with h5py.File(dataset_path, "w") as ctx:
        ctx.create_dataset("audio", data=dataset.audio)
        ctx.create_dataset("locations_px", data=dataset.locations_px)
        if dataset.locations_mm is not None:
            ctx.create_dataset("locations", data=dataset.locations_mm)
        ctx.create_dataset("length_idx", data=dataset.length_idx)


def make_partial_dataset(session_dir: Path) -> tuple[ram_dataset, pd.DataFrame]:
    video_path = next(session_dir.glob("*_CamFlir1_*.avi"), None)
    audio_path = next(session_dir.glob("*_audiorec.flac"), None)
    processed_track_path = next(
        processed_track_dir.glob(f"{session_dir.parent.name}*.npy"), None
    )
    processed_segement_path = next(
        processed_annotation_dir.glob(f"{session_dir.parent.name}*.npy"), None
    )
    if not all([video_path, audio_path, processed_track_path, processed_segement_path]):
        return None, None

    tracks = np.load(processed_track_path)
    segments = np.load(processed_segement_path)
    video_framerate = get_framerate(video_path)
    video_length = get_num_frames(video_path)

    audio = get_audio(audio_path, segments)
    audio_lengths = np.array([len(a) for a in audio])
    audio = np.concatenate(audio, axis=0)

    segments = segments[: len(audio_lengths)]
    video_indices = np.mean(segments / audio_sr * video_framerate, axis=1).astype(int)
    video_indices = np.clip(video_indices, 0, video_length - 1)
    locations_px = tracks[video_indices]

    # Convert locations to mm
    locations_mm = convert_locations_to_mm(locations_px)

    metadata = pd.DataFrame(
        {
            "video_path": [str(video_path)] * len(video_indices),
            "frame_idx": video_indices,
            "x_px": locations_px[:, 0],
            "y_px": locations_px[:, 1],
        }
    )

    dset = ram_dataset(
        audio=audio,
        locations_px=locations_px,
        locations_mm=locations_mm,
        length_idx=np.cumsum(np.insert(audio_lengths, 0, 0)),
    )

    return dset, metadata


def main():
    partial_dataset_dir.mkdir(exist_ok=True)

    for session_dir in tqdm(recording_sessions):
        partial_dset_path = partial_dataset_dir / f"{session_dir.parent.name}.h5"
        metadata_path = partial_dataset_dir / f"{session_dir.parent.name}_metadata.csv"
        ram_dset, metadata = make_partial_dataset(session_dir)
        if not ram_dset:
            print(f"Could not make partial dataset for {session_dir}")
            continue

        metadata.to_csv(metadata_path, index=False)
        write_dataset(ram_dset, partial_dset_path)


if __name__ == "__main__":
    main()
