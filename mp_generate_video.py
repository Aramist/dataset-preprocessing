import json
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
from scipy.signal import stft
from vocalocator.outputs.base import ModelOutput, Unit
from vocalocator.training.models import make_output_factory

torch.set_num_threads(1)  # Prevents deadlock when using multiprocessing. Very important
torch.multiprocessing.set_sharing_strategy("file_system")

SPECTROGRAM_HEIGHT = 160


def get_outputs_from_assess_file(h5_path):
    with h5py.File(h5_path) as f:
        if "model_config" not in f.attrs:
            raise ValueError(
                "Assess file must embed the model's config in the file atttributes for video generation to proceed."
            )
        ensemble_config = f.attrs["model_config"]
        ensemble_config = json.loads(ensemble_config)
        factory = make_output_factory(ensemble_config)

        if "ensemble" in ensemble_config["ARCHITECTURE"].lower():
            constituent_keys = list(
                filter(
                    lambda x: x.startswith("constituent") and x.endswith("raw_output"),
                    f.keys(),
                )
            )
            outputs = []
            for key in constituent_keys:
                raw = torch.from_numpy(f[key][:])
                model_idx = int(key.split("_")[1])

                model_config = ensemble_config["MODEL_PARAMS"]["CONSTITUENT_MODELS"][
                    model_idx
                ]
                model_i_factory = make_output_factory(model_config)

                outputs.append(
                    [model_i_factory.create_output(o.unsqueeze(0)) for o in raw]
                )
            outputs = list(zip(*outputs))
            return [factory.create_output(o) for o in outputs]
        else:
            raw = torch.from_numpy(f["raw_model_output"][:])
            return [factory.create_output(o.unsqueeze(0)) for o in raw]


def make_homology(
    image_points: np.ndarray, arena_dims=(558.9, 355.6), render_dims=(600, 400)
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a homology matrix to convert from pixel coordinates to millimeters in the global coordinate frame

    Args:
        image_points (ndarray): 4x2 array of pixel coordinates in the order (top-left, top-right, bottom-right, bottom-left)

    Returns:
        _type_: _description_
    """
    width, height = arena_dims
    render_width, render_height = render_dims

    if image_points.shape != (4, 2):
        raise ValueError("Image points must be a 4x2 array")

    dest_points = image_points
    # dest_points = np.array([[80.0, 80.0], [580.0, 70.0], [580.0, 390.0], [90.0, 400.0]])

    source_points = np.array([[0, height], [width, height], [width, 0], [0, 0]])

    render_points = np.array(
        [[0, render_height], [render_width, render_height], [render_width, 0], [0, 0]]
    )

    H, _ = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC)
    render_H, _ = cv2.findHomography(render_points, dest_points, method=cv2.RANSAC)
    return H, render_H


def convert_render(render, H):
    return cv2.warpPerspective(render, H, (640, 512))


def scale_and_color_render(pmf, H):
    if isinstance(pmf, torch.Tensor):
        pmf = pmf.numpy()
    pmf /= pmf.sum()

    argsort = np.argsort(-pmf.ravel())  # Index that sorts from high to low
    sums_mask = np.cumsum(pmf.ravel()[argsort]) < 0.95
    pmf_mask = np.zeros_like(sums_mask)
    pmf_mask[argsort] = sums_mask
    pmf_mask = pmf_mask.reshape(pmf.shape)

    pmf = (pmf / pmf.max() * 255).astype(np.uint8)
    pmf_colored = cv2.applyColorMap(pmf, cv2.COLORMAP_VIRIDIS)
    pmf_colored[~pmf_mask, :] = 0
    return convert_render(pmf_colored, H)


def make_xy_grid(arena_dims, render_dims):
    """Generates a grid of points for evaluating a PMF"""
    test_points = np.stack(
        np.meshgrid(
            np.linspace(-arena_dims[0] / 2, arena_dims[0] / 2, render_dims[0]),
            np.linspace(-arena_dims[1] / 2, arena_dims[1] / 2, render_dims[1]),
        ),
        axis=-1,
    )
    return test_points


def draw_rendered_cov(
    frame: np.ndarray, render: np.ndarray, blend_ratio=0.6
) -> np.ndarray:
    """Overlays a rendered PMF image onto a frame"""
    if render.shape[:2] != frame.shape[:2]:
        render = cv2.resize(render, frame.shape[:2][::-1])
    frame = frame.copy()
    density_mask = (render > 0).any(axis=-1)
    frame[density_mask] = render[density_mask] * blend_ratio + frame[density_mask] * (
        1 - blend_ratio
    )
    return frame


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
    frame: np.ndarray, point: np.ndarray, color: Optional[str] = "r"
) -> np.ndarray:
    """Draws a point on a frame as a red circle"""
    return frame
    if isinstance(point, np.ndarray):
        point = point.squeeze()
        point = (int(point[0]), int(point[1]))
    # Get color from matplotlib
    color_rgb = mpl.colors.to_rgb(color)
    color_bgr_int = tuple(int(255 * c) for c in color_rgb[::-1])

    return cv2.circle(frame, point, 5, color_bgr_int, -1)


def log_progress(
    num_frames_written: int, num_frames_to_generate: int, log_interval: int = 50
) -> None:
    if num_frames_written % log_interval == 0:
        print(f"Generated {num_frames_written}/{num_frames_to_generate} frames")


def iter_audio(
    audio_path: Path, index: Optional[np.ndarray]
) -> Generator[np.ndarray, None, None]:
    """Yields audio samples from a file at the given indices"""

    if audio_path is None:
        return null_iterator()

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


def generate_frame(
    video_frame: np.ndarray,
    model_output: ModelOutput,
    ground_truth: Optional[np.ndarray],
    muse_pred: Optional[np.ndarray],
    annotation: Optional[str],
    audio: Optional[np.ndarray],
    global_variables: Namespace,  # Contains arena_dims, render_dims, H, render_H, grid, etc
    output_queue: mp.Queue,
    output_index: int,
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
                (255, 255, 255),
                2,  # thickness
                cv2.LINE_AA,
            )

        grid = global_variables.grid
        render_dims = global_variables.render_dims
        render_H = global_variables.render_H
        # Compute the pmf, I think this is the bottleneck
        pmf = model_output.pmf(grid, units=Unit.MM)
        pmf = pmf.reshape(render_dims[1], render_dims[0])
        rendered_pmf = scale_and_color_render(pmf, render_H)
        drawn_frame = draw_rendered_cov(video_frame, rendered_pmf)

        # Draw the point prediction on the frame
        point_pred = model_output.point_estimate(units=Unit.MM)
        point_pred = convert_points(point_pred, global_variables.H)
        drawn_frame = draw_point_on_frame(drawn_frame, point_pred, "r")
        if ground_truth is not None:
            ground_truth = convert_points(ground_truth, global_variables.H)
            drawn_frame = draw_point_on_frame(drawn_frame, ground_truth, "b")

        if muse_pred is not None:
            muse_pred = convert_points(muse_pred, global_variables.H)
            drawn_frame = draw_point_on_frame(drawn_frame, muse_pred, "g")

        # Draw spectrograms if provided
        if audio is not None:
            if len(audio.shape) == 1:
                audio = audio[:, None]
            num_channels = audio.shape[1]
            _, _, spectrograms = stft(x=audio, axis=0)
            # Shape should be (freq, num_channels, time)
            spectrograms = np.abs(spectrograms)
            spectrograms = np.log(spectrograms + 1e-8)
            spectrograms = spectrograms[::-1, ...]  # Flip the frequency axis
            # concat horizontally
            spectrograms = spectrograms.reshape(spectrograms.shape[0], -1)
            cmin, cmax = np.quantile(spectrograms, [0.05, 0.95])
            spectrograms = (spectrograms - cmin) / (cmax - cmin)
            spectrograms = np.clip(spectrograms, 0, 1)
            # Convert to uint8, resize, and apply colormap
            spectrograms = (255 * spectrograms).astype(np.uint8)
            spectrograms = cv2.resize(
                spectrograms, (drawn_frame.shape[1], SPECTROGRAM_HEIGHT)
            )
            spectrograms = cv2.applyColorMap(spectrograms, cv2.COLORMAP_VIRIDIS)
            # Concat the spectrogram image to the top of the frame
            drawn_frame = np.concatenate([spectrograms, drawn_frame], axis=0)

        # Push to the output queue
        # Index is included because the frames may not be generated in order
        output_queue.put((output_index, drawn_frame))
    except Exception as e:
        # redirect to queue
        print(e)
        output_queue.put((output_index, e))


def run_on_assess_file(
    assess_path: Path,
    video_csv: Path,
    output_video_path: Path,
    corner_points: np.ndarray,
    *,
    num_frames_to_generate: int = -1,
    audio_path: Optional[Path] = None,
    framerate: int = 30,
    split_index: Optional[np.ndarray] = None,
    max_workers: int = 30,
    copy_videos: bool = True,
) -> None:
    """Generates a demo video from the results of gerbilizer.assess module
    video_csv is a csv file with the following columns: video_path, frame_idx
    Each row in the csv corresponds to a vocalization in the full dataset.
    Optionally, a split_index can be provided to only generate a video for a subset of the rows in the csv.
    """
    video_dims = (640, 512)
    if audio_path is not None:
        video_dims = (video_dims[0], video_dims[1] + SPECTROGRAM_HEIGHT)
    render_dims = (150, 100)
    if output_video_path.suffix != ".mp4":
        raise ValueError(
            f"Output video path must end with .mp4, got {output_video_path}"
        )

    if corner_points.shape != (4, 2):
        raise ValueError("Corner points must be a 4x2 array")

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        framerate,
        video_dims,
        isColor=True,
    )

    ground_truth = None
    muse_preds = None
    all_outputs = get_outputs_from_assess_file(assess_path)
    arena_dims_mm = np.array(all_outputs[0].arena_dims[Unit.MM])
    with h5py.File(assess_path, "r") as f:
        if "scaled_locations" in f:
            ground_truth = f["scaled_locations"][:]

    if audio_path is not None:
        with h5py.File(audio_path, "r") as f:
            if "muse_pred" in f:
                muse_preds = f["muse_pred"][:]
                muse_preds += arena_dims_mm / 2

    # One grid can be used for all PMFs
    grid = make_xy_grid(arena_dims_mm, render_dims).reshape(-1, 1, 2)
    grid = torch.from_numpy(grid).float()

    H, render_H = make_homology(render_dims=render_dims, image_points=corner_points)
    df = pd.read_csv(video_csv)

    # Make sure the video csv has the same number of rows as the assess file
    # Utilize the split index if provided
    if len(df) != len(all_outputs) and split_index is None:
        raise ValueError(
            f"Video csv has fewer rows than the number of outputs in the assess file. "
            f"Either provide a split_index or a video csv with {len(all_outputs)} rows"
        )
    elif split_index is not None and len(split_index) != len(all_outputs):
        raise ValueError(
            f"Split index must have the same length as the number of outputs in the assess file. "
            f"Got {len(split_index)} split indices and {len(all_outputs)} outputs"
        )
    elif split_index is not None:
        df = df.iloc[split_index]
        # reset the index of the dataframe
        df = df.reset_index(drop=True)

    print(df)

    # Assume that the video csv ordering is the same as the assess file ordering
    unique_vid_paths = df["video_path"].unique()

    # Copy videos to local disk to avoid spamming the remote disk
    local_vid_paths = []
    if copy_videos:
        print("Copying videos to local disk...")
        for vid_path in unique_vid_paths:
            new_path = Path("/tmp") / Path(vid_path).name
            if not new_path.exists():
                copyfile(vid_path, new_path)
            local_vid_paths.append(str(new_path))
        print("Done copying videos to local disk")
    else:
        print("Skipping video copy")
        local_vid_paths = unique_vid_paths

    all_frame_idx = df["frame_idx"]

    if "annotation" in df.columns:
        annotations = df["annotation"]
    else:
        annotations = [None] * len(df)

    if ground_truth is None:
        ground_truth = [None] * len(df)

    if muse_preds is None:
        muse_preds = [None] * len(df)

    num_frames_to_generate = (
        len(all_outputs) if num_frames_to_generate < 0 else num_frames_to_generate
    )

    # Create a new sorting of the outputs based on the video path and the frame index
    video_index_sorting = []
    for unique_video in unique_vid_paths:
        sub_df = df[df["video_path"] == unique_video]
        # See if it's sorted
        if not np.diff(sub_df["frame_idx"]).all() >= 0:
            sub_df = sub_df.sort_values(by="frame_idx")
        indices = sub_df.index.to_numpy()
        video_index_sorting.append(indices)

    # Create a manager to share variables across processes
    with mp.Manager() as manager:
        global_variables = manager.Namespace()
        global_variables.grid = grid
        global_variables.H = H
        global_variables.render_dims = render_dims
        global_variables.render_H = render_H

        # Create a queue to store the output frames
        output_queue = manager.Queue()

        # Make a generator for the arguments to the worker function
        def argument_generator():
            num_frames_submitted = 0
            for local_video, indices_for_video in zip(
                local_vid_paths, video_index_sorting
            ):
                audio_iterator = (
                    iter_audio(audio_path, indices_for_video)
                    if audio_path
                    else null_iterator()
                )
                frames_in_video = all_frame_idx[indices_for_video]
                for frame, index, audio in zip(
                    get_frames_from_video(local_video, frames_in_video),
                    indices_for_video,
                    audio_iterator,
                ):
                    if annotations[index] is None:
                        annotation = str(index)
                    else:
                        annotation = annotations[index]

                    args = (
                        frame,
                        all_outputs[index],
                        ground_truth[index],
                        muse_preds[index],
                        annotation,
                        audio,
                        global_variables,
                        output_queue,
                        num_frames_submitted,
                    )
                    yield args
                    num_frames_submitted += 1
                    if num_frames_submitted >= num_frames_to_generate:
                        return

        num_workers = min(max_workers, mp.cpu_count())
        arg_generator = argument_generator()
        with mp.Pool(num_workers, maxtasksperchild=1) as pool:
            pool.starmap_async(generate_frame, arg_generator, chunksize=num_workers)

            num_frames_written = 0
            backup_queue = PriorityQueue()

            while num_frames_written < num_frames_to_generate:
                if (
                    not backup_queue.empty()
                    and backup_queue.queue[0][0] == num_frames_written
                ):
                    _, drawn_frame = backup_queue.get()
                    writer.write(drawn_frame)
                    num_frames_written += 1
                    log_progress(num_frames_written, num_frames_to_generate)
                    continue

                try:
                    output_index, drawn_frame = output_queue.get(timeout=1)
                    if isinstance(drawn_frame, Exception):
                        return
                    backup_queue.put((output_index, drawn_frame))
                except:
                    print("Waiting for frames to be generated...")
                    continue

            writer.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--assess-path", type=Path, required=True)
    parser.add_argument("--video-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--split-index", type=Path)
    parser.add_argument("--corner-points", type=Path)
    parser.add_argument("--dont-copy-videos", action="store_true")
    parser.add_argument("--audio-path", type=Path)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to generate. Useful for debugging",
    )
    parser.add_argument(
        "--max-num-workers",
        type=int,
        default=30,
        help="Maximum number of workers to use for multiprocessing",
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
        args.assess_path,
        args.video_csv,
        args.output_path,
        corner_points,
        num_frames_to_generate=args.num_frames,
        audio_path=args.audio_path,
        framerate=args.framerate,
        split_index=split_index,
        max_workers=args.max_num_workers,
        copy_videos=not args.dont_copy_videos,
    )
