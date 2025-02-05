from multiprocessing.managers import Namespace
from pathlib import Path
from queue import PriorityQueue
from shutil import copyfile
from typing import Generator, Optional, Tuple

import cv2
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from joblib import Parallel, delayed
from scipy.signal import stft
from tqdm import tqdm

torch.set_num_threads(1)  # Prevents deadlock when using multiprocessing. Very important
torch.multiprocessing.set_sharing_strategy("file_system")
# Increase logging level
mp.log_to_stderr(30)

SPECTROGRAM_HEIGHT = 160


def make_homology(
    image_points: np.ndarray, arena_dims=(558.9, 355.6)
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a homology matrix to convert from pixel coordinates to millimeters in the global coordinate frame

    Args:
        image_points (ndarray): 4x2 array of pixel coordinates in the order (top-left, top-right, bottom-right, bottom-left)

    Returns:
        _type_: _description_
    """
    width, height = arena_dims

    if image_points.shape != (4, 2):
        raise ValueError("Image points must be a 4x2 array")

    dest_points = image_points
    # dest_points = np.array([[80.0, 80.0], [580.0, 70.0], [580.0, 390.0], [90.0, 400.0]])

    hh, hw = height / 2, width / 2
    source_points = np.array([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]])

    H, _ = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC)
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


def get_frames_from_video(
    video_path: str, frame_indices: np.ndarray
) -> Generator[np.ndarray, None, None]:
    """Returns a generator that yields frames from a video at the given frame indices
    Indices need not be sorted
    """
    cap = cv2.VideoCapture(video_path)
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {i} from {video_path}")
        yield frame


def draw_point_on_frame(
    frame: np.ndarray, points: np.ndarray, color: Optional[str] = "r"
) -> np.ndarray:
    """Draws a point on a frame as a red circle"""
    if isinstance(points, np.ndarray):
        points = points.squeeze()

    points = points.reshape(-1, 2)
    for point in points:
        point = (int(point[0]), int(point[1]))
        # Get color from matplotlib
        color_rgb = mpl.colors.to_rgb(color)
        color_bgr_int = tuple(int(255 * c) for c in color_rgb[::-1])
        frame = cv2.circle(frame, point, 5, color_bgr_int, -1)

    return frame


def log_progress(
    num_frames_written: int, num_frames_to_generate: int, log_interval: int = 50
) -> None:
    if num_frames_written % log_interval == 0:
        print(f"Generated {num_frames_written}/{num_frames_to_generate} frames")


def iter_audio(
    audio_path: Path, index: Optional[np.ndarray]
) -> Generator[np.ndarray, None, None]:
    """Yields audio samples from a file at the given indices"""

    with h5py.File(audio_path, "r") as ctx:
        len_idx = ctx["length_idx"][:]
        if index is None:
            index = np.arange(len(len_idx) - 1)

        for idx in index:
            start, end = len_idx[idx], len_idx[idx + 1]
            yield ctx["audio"][start:end, :]


def null_iterator(*args, **kwargs):
    while True:
        yield None


def make_spectrograms(audio: np.ndarray, width: int) -> np.ndarray:
    # Draw spectrograms if provided
    if len(audio.shape) == 1:
        audio = audio[:, None]
    _, _, spectrograms = stft(x=audio, axis=0)
    # Shape should be (freq, num_channels, time)
    spectrograms = np.abs(spectrograms)
    spectrograms = np.log(spectrograms + 1e-9)
    spectrograms = spectrograms[::-1, ...]  # Flip the frequency axis
    # concat horizontally
    cmin, cmax = np.quantile(spectrograms, [0.05, 0.95])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], -1)
    spectrograms = (spectrograms - cmin) / (cmax - cmin)
    spectrograms = np.clip(spectrograms, 0, 1)
    # Convert to uint8, resize, and apply colormap
    spectrograms = (255 * spectrograms).astype(np.uint8)
    spectrograms = cv2.resize(spectrograms, (width, SPECTROGRAM_HEIGHT))
    spectrograms = cv2.applyColorMap(spectrograms, cv2.COLORMAP_VIRIDIS)
    return spectrograms


def generate_frame(
    video_frame: np.ndarray,
    ground_truth: Optional[np.ndarray],
    annotation: Optional[str],
    audio: np.ndarray,
    H: np.ndarray,
    floor_corner_points: np.ndarray,
):
    """Within a multiprocessing context, generates a single frame of the output video"""
    try:
        # Annotate frame
        if annotation is not None:
            video_frame = cv2.putText(
                video_frame,
                annotation,
                (10, 30),  # bottom-left corner of text
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                # (255, 255, 255),
                (0, 0, 0),
                2,  # thickness
                cv2.LINE_AA,
            )

        if ground_truth is not None:
            ground_truth = convert_points(ground_truth, H)
            video_frame = draw_point_on_frame(video_frame, ground_truth, "r")

        strongest_channel = np.argmax(np.var(audio, axis=0))
        strongest_channel = (strongest_channel + 2) % 4  # Rotate the channels
        strong_mic_pos = floor_corner_points[strongest_channel]
        video_frame = draw_point_on_frame(video_frame, strong_mic_pos, "b")

        # Concat the spectrogram image to the top of the frame
        spectrograms = make_spectrograms(audio, video_frame.shape[1])
        video_frame = np.concatenate([spectrograms, video_frame], axis=0)

        return video_frame
    except Exception as e:
        # redirect to queue
        print(e)
        return None


def run_on_assess_file(
    audio_path: Path,
    metadata_csv_path: Path,
    output_video_path: Path,
    floor_corner_points: np.ndarray,
    *,
    num_frames_to_generate: int = -1,
    framerate: int = 30,
    split_index: Optional[np.ndarray] = None,
) -> None:
    """Generates a demo video from the results of gerbilizer.assess module
    metadata_csv_path is a csv file with the following columns: video_path, frame_idx
    Each row in the csv corresponds to a vocalization in the full dataset.
    Optionally, a split_index can be provided to only generate a video for a subset of the rows in the csv.
    """
    metadata = pd.read_csv(metadata_csv_path)
    sample_video = metadata["video_path"].iloc[0]
    cap = cv2.VideoCapture(sample_video)
    video_dims = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()

    err_frame = np.full((video_dims[1], video_dims[0], 3), 255, dtype=np.uint8)
    cv2.putText(
        err_frame,
        "Failed to generate frame",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    if audio_path is not None:
        video_dims = (video_dims[0], video_dims[1] + SPECTROGRAM_HEIGHT)
    if output_video_path.suffix != ".mp4":
        raise ValueError(
            f"Output video path must end with .mp4, got {output_video_path}"
        )

    if floor_corner_points.shape != (4, 2):
        raise ValueError("Corner points must be a 4x2 array")

    ground_truth = None
    arena_dims = np.array([572.0, 360.0])
    with h5py.File(audio_path, "r") as f:
        if "arena_dims_mm" in f.attrs:
            print("Using arena_dims from audio file")
            arena_dims = np.array(f.attrs["arena_dims_mm"])
        num_vocalizations = len(f["length_idx"]) - 1
        if split_index is None:
            split_index = np.arange(num_vocalizations)
        if (np.diff(split_index) < 1).any():
            raise ValueError("Split index must be sorted and unique")

        if "locations" in f:
            ground_truth = f["locations"][:]
            ground_truth = ground_truth[split_index]
    if ground_truth is None:
        ground_truth = [None] * len(metadata)

    H = make_homology(image_points=floor_corner_points, arena_dims=arena_dims)

    metadata = metadata.iloc[split_index]  # (N, 2 or 3)
    metadata = metadata.reset_index(drop=True)

    unique_vid_paths = metadata["video_path"].unique()
    all_frame_idx = metadata["frame_idx"]  # (N,)

    if "annotation" in metadata.columns:
        annotations = metadata["annotation"]
    else:
        annotations = [
            f"{Path(vp).stem}     Dataset index:{i}"
            for vp, i in zip(metadata["video_path"], split_index)
        ]

    num_frames_to_generate = (
        len(split_index) if num_frames_to_generate < 0 else num_frames_to_generate
    )

    # Create a new sorting of the outputs based on the video path and the frame index
    video_index_sorting = []
    for unique_video in unique_vid_paths:
        sub_df = metadata[metadata["video_path"] == unique_video]
        # See if it's sorted
        if not np.diff(sub_df["frame_idx"]).all() >= 0:
            raise ValueError("Frame indices must be sorted in metadata csv")
        indices = sub_df.index.to_numpy()
        video_index_sorting.append(indices)

    # Make a generator for the arguments to the worker function
    def argument_generator():
        num_generated = 0
        for video_path, indices_for_video in zip(unique_vid_paths, video_index_sorting):
            audio_iterator = iter_audio(audio_path, indices_for_video)
            frames_in_video = all_frame_idx[indices_for_video]
            for frame, index, audio in zip(
                get_frames_from_video(video_path, frames_in_video),
                indices_for_video,
                audio_iterator,
            ):
                args = (
                    frame,
                    ground_truth[index],
                    annotations[index],
                    audio,
                    H,
                    floor_corner_points,
                )
                yield args
                num_generated += 1
                if num_generated >= num_frames_to_generate:
                    return

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        framerate,
        video_dims,
        isColor=True,
    )
    par = Parallel(n_jobs=-2, return_as="generator")(
        delayed(generate_frame)(*args)
        for args in tqdm(argument_generator(), total=num_frames_to_generate)
    )
    for frame in par:
        writer.write(frame if frame is not None else err_frame)

    writer.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--corner-points", type=Path, required=True)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--split-index", type=Path)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to generate. Useful for debugging",
    )
    args = parser.parse_args()

    split_index = None
    if args.split_index is not None:
        if not args.split_index.exists():
            raise ValueError(f"Split index {args.split_index} does not exist")
        idx_path = args.split_index
        if idx_path.suffix == ".npy":
            split_index = np.load(idx_path)
        elif idx_path.suffix == ".h5":
            with h5py.File(idx_path, "r") as f:
                split_index = f["split_indices"][:]
        else:
            raise ValueError(f"Split index {idx_path} must be either .npy or .h5")
    corner_points = np.load(args.corner_points)

    run_on_assess_file(
        args.audio_path,
        args.metadata_csv,
        args.output_path,
        corner_points,
        num_frames_to_generate=args.num_frames,
        framerate=args.framerate,
        split_index=split_index,
    )
