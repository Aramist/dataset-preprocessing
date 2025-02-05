import json
from pathlib import Path
from typing import Generator, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from gerbilizer.outputs.base import EnsembleOutput, Unit
from gerbilizer.training.models import make_output_factory


def get_outputs_from_assess_file(h5_path):
    with h5py.File(h5_path) as f:
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


def render_grid(arena_dims, render_dims):
    """Generates a grid of points for evaluating a PMF"""
    test_points = np.stack(
        np.meshgrid(
            np.linspace(0, arena_dims[0], render_dims[0]),
            np.linspace(0, arena_dims[1], render_dims[1]),
        ),
        axis=-1,
    )
    return test_points


def render_cov(pred, H, render_dims=(600, 400), arena_dims=(558.9, 355.6)):
    if len(pred.shape) > 3:
        raise ValueError("no batch")
    mu = pred[0, :]
    cov = pred[1:, :]
    prec = np.linalg.inv(cov)

    test_points = render_grid(arena_dims, render_dims)
    # not going to compute the coefficient term outside the exp
    # can also subtract the min from the exponent to stabilize the computation

    diff = test_points - mu[None, None, :]
    exp_term = -0.5 * np.einsum("...j,jk,...k->...", diff, prec, diff)
    exp_term -= exp_term.max()
    pmf = np.exp(exp_term)
    return scale_and_color_render(pmf, H)


def draw_rendered_cov(frame, render):
    if render.shape[:2] != frame.shape[:2]:
        render = cv2.resize(render, frame.shape[:2][::-1])
    frame = frame.copy()
    density_mask = (render > 0).any(axis=-1)
    frame[density_mask] = render[density_mask] * 0.6 + frame[density_mask] * 0.4
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
    cap = cv2.VideoCapture(video_path)
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {i} from {video_path}")
        yield frame


def draw_point_on_frame(frame: np.ndarray, point: np.ndarray) -> np.ndarray:
    if isinstance(point, np.ndarray):
        point = tuple(point.astype(int))
    return cv2.circle(frame, point, 5, (0, 0, 255), -1)


def log_progress(
    num_frames_written: int, num_frames_to_generate: int, log_interval: int = 50
) -> None:
    if num_frames_written % log_interval == 0:
        print(f"Generated {num_frames_written}/{num_frames_to_generate} frames")


def run_on_assess_file(
    assess_path: Path,
    video_csv: Path,
    output_video_path: Path,
    framerate: int = 30,
    *,
    num_frames_to_generate: int = -1,
    split_index: np.ndarray = None,
    corner_points: np.ndarray = None,
) -> None:
    """Generates a demo video from the results of gerbilizer.assess module
    video_csv is a csv file with the following columns: video_path, frame_idx
    Each row in the csv corresponds to a vocalization in the full dataset.
    Optionally, a split_index can be provided to only generate a video for a subset of the rows in the csv.
    """
    video_dims = (640, 512)
    render_dims = (600, 400)
    if output_video_path.suffix != ".mp4":
        raise ValueError(
            f"Output video path must end with .mp4, got {output_video_path}"
        )

    if corner_points is None:
        raise ValueError("Must provide corner points")

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        framerate,
        video_dims,
        isColor=True,
    )

    outputs = get_outputs_from_assess_file(assess_path)
    grid = render_grid(outputs[0].arena_dims[Unit.MM], render_dims).reshape(-1, 1, 2)
    grid = torch.from_numpy(grid).float()

    H, render_H = make_homology(render_dims=render_dims, image_points=corner_points)
    df = pd.read_csv(video_csv)

    if len(df) < len(outputs) and split_index is None:
        raise ValueError(
            f"Video csv has fewer rows than the number of outputs in the assess file. "
            f"Either provide a split_index or a video csv with {len(outputs)} rows"
        )
    elif split_index is not None and len(split_index) != len(outputs):
        raise ValueError(
            f"Split index must have the same length as the number of outputs in the assess file. "
            f"Got {len(split_index)} split indices and {len(outputs)} outputs"
        )
    elif split_index is not None:
        df = df.iloc[split_index]

    print(df)
    all_vid_paths = df["video_path"]
    unique_vid_paths = df["video_path"].unique()
    all_frame_idx = df["frame_idx"]
    num_frames_written = 0

    if 'annotation' in df.columns:
        annotations = df['annotation']
    else:
        annotations = [None] * len(df)

    num_frames_to_generate = (
        len(outputs) if num_frames_to_generate < 0 else num_frames_to_generate
    )

    for unique_vid in unique_vid_paths:
        vid_idx = all_vid_paths == unique_vid
        frames = all_frame_idx[vid_idx]

        numpy_vid_idx = np.flatnonzero(
            vid_idx.to_numpy()
        )  # indices within the csv where this video appears, in dataset order
        video_outputs = [outputs[i] for i in numpy_vid_idx]

        if not (np.diff(frames) >= 0).all():
            sorted_frames = np.sort(frames)
            sorting = (
                frames.argsort().to_numpy()
            )  # sorting of frames within the video, in chronological order
        else:
            sorted_frames = frames
            sorting = np.arange(len(frames))
        for n, frame in enumerate(get_frames_from_video(unique_vid, sorted_frames)):
            if num_frames_written >= num_frames_to_generate:
                break

            annotation = annotations[vid_idx].iloc[sorting[n]]
            if annotation is not None:
                # Draw the annotation text in the top left corner of the frame
                frame = cv2.putText(
                    frame,
                    annotation,
                    (10, 30),  # bottom-left corner of text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 255, 255),
                    2,  # thickness
                    cv2.LINE_AA,
                )

            pmf = video_outputs[sorting[n]].pmf(grid, units=Unit.MM)
            pmf = pmf.reshape(render_dims[1], render_dims[0])
            rendered_pmf = scale_and_color_render(pmf, render_H)
            drawn_frame = draw_rendered_cov(frame, rendered_pmf)
            writer.write(drawn_frame)
            num_frames_written += 1
            log_progress(num_frames_written, num_frames_to_generate)
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
        args.assess_path,
        args.video_csv,
        args.output_path,
        args.framerate,
        num_frames_to_generate=args.num_frames,
        split_index=split_index,
        corner_points=corner_points,
    )
