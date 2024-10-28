import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from librosa.feature import spectral_flatness
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
processed_annotation_dir = working_dir / consts["processed_annotation_dir"]

video_label = lambda p: p.parent.name.split("_")[-1]

annotations_paths = sum(
    (
        list(Path(search_dir).glob("*/mic_*_annotations.csv"))
        for search_dir in consts["recording_session_dirs"]
    ),
    start=[],
)
onset_paths = sum(
    (
        list(Path(search_dir).glob("*/mic_*.h5"))
        for search_dir in consts["recording_session_dirs"]
    ),
    start=[],
)
video_paths = sum(
    (
        list(Path(search_dir).glob("*/*.avi"))
        for search_dir in consts["recording_session_dirs"]
    ),
    start=[],
)


print(f"Found {len(annotations_paths)} annotation files")
print(f"Found {len(onset_paths)} onset files")
print(f"Found {len(video_paths)} video files")


def get_powers(onset_path: Path, segments: np.ndarray):
    with h5py.File(onset_path, "r") as f:
        powers = []
        num_channels = len(f["ai_channels"].keys())
        for start, stop in tqdm(segments):
            audio = np.stack(
                [
                    f[f"ai_channels/ai{channel}"][start:stop]
                    for channel in range(num_channels)
                ],
                axis=0,
            )
            power = np.mean(audio**2, axis=1)
            powers.append(power)
    return np.array(powers)


def process_annotation(annotation_path):
    date = annotation_path_date(annotation_path)

    # Ensure it has a corresponding onset file
    onset_path = next(filter(lambda x: date in x.stem, onset_paths), None)
    if not onset_path:
        print(f"No audio file found for {date}")
        return None
    video_path = next(filter(lambda x: date in x.stem, video_paths), None)
    if not video_path:
        print(f"No video file found for {date}")
        # return None

    df = pd.read_csv(annotation_path).dropna()

    segments_sec = np.stack(
        [df["start_seconds"].values, df["stop_seconds"].values], axis=1
    )
    segments_samps = (segments_sec * audio_sr).astype(int)

    lengths = segments_sec[:, 1] - segments_sec[:, 0]  # stored in seconds
    mask = (lengths > max_vox_len_sec) | (lengths < min_vox_len_sec)  # bad segments
    segments_samps = segments_samps[~mask, :]

    powers = get_powers(onset_path, segments_samps)

    power_thresh = 2e-4
    good_vocalizations = powers.mean(axis=-1) > power_thresh

    segments_samps = segments_samps[good_vocalizations, :]

    return segments_samps


def num_frames_in_video(video_path: Path) -> int:
    """Gets the number of frames in a video file

    Args:
        video_path (Path)

    Returns:
        int: Number of frames in the video
    """
    cap = cv2.VideoCapture(str(video_path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames


def get_video_frames(annotation_path: Path, segments: np.ndarray):
    # Find the corresponding video file, if it exists
    date = annotation_path_date(annotation_path)

    video_path = list(filter(lambda x: date in x.stem, video_paths))
    if not video_path:
        return None, None
    else:
        video_path = video_path[0]

    frame_indices = (segments.mean(axis=1) * video_framerate / audio_sr).astype(int)
    num_frames = num_frames_in_video(video_path)
    frame_indices = np.clip(frame_indices, 0, num_frames - 1)

    return video_path, frame_indices


if __name__ == "__main__":
    video_fnames = []
    video_frames = []

    processed_annotation_dir.mkdir(exist_ok=True)
    for annotation in annotations_paths:
        new_segments = process_annotation(annotation)
        if new_segments is None:
            continue
        vid_fpath, vid_frames = get_video_frames(annotation, new_segments)
        if vid_fpath is None:
            continue

        new_path = Path(annotation).stem + "_filtered.npy"
        new_path = processed_annotation_dir / new_path
        np.save(new_path, new_segments)

        video_fnames.extend([vid_fpath] * len(vid_frames))
        video_frames.extend(vid_frames)
    if not video_fnames:
        raise ValueError("No video frames found!")
    # Save the video frame index dataframe
    df = pd.DataFrame(
        {
            "video_path": video_fnames,
            "frame_idx": video_frames,
            "annotation": [video_label(p) for p in video_fnames],
        }
    )
    df.to_csv("/mnt/home/atanelus/ceph/datasets/dyad_metadata.csv", index=False)
