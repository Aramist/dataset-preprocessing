"""Script for manually annotating gerbil dyad tracks.

Given a set of annotated onsets and offsets from the audio, will display the
video frame associated with each vocalization and allow the user to manually
select two points and flip through the existing annotations with 'h' and 'l'.

Press 'q' to quit the program and save the annotations.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.signal import stft

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

save_file_path = working_dir / consts["track_annotation_path"]

codes = {
    "h": "reverse",
    "l": "forward",
    "q": "quit",
    "e": "erase",
    "0": "to_start",
    "9": "to_end",
}


def get_audio(audio_path: Path, onset: int, offset: int) -> np.ndarray:
    with h5py.File(audio_path, "r") as f:
        num_channels = len(f["ai_channels"].keys())
        audio = np.stack(
            [
                f[f"ai_channels/ai{channel}"][onset:offset]
                for channel in range(num_channels)
            ],
            axis=1,
        )
    return audio


def populate_save_file() -> pd.DataFrame:
    """Gathers existing audio annotations and creates a save file with no annotated points"""
    audio_onsets = []
    audio_offsets = []
    audio_paths = []
    video_paths = []
    video_indices = []

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
            list(Path(search_dir).glob("*/*.mp4"))
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
        annotations = np.load(annotation_file)  # unit: samples
        audio_onsets.extend(annotations[:, 0])
        audio_offsets.extend(annotations[:, 1])
        audio_paths.extend([aud_f] * len(annotations))
        video_paths.extend([vid_f] * len(annotations))
        vid_idx = (
            annotations.astype(float).mean(axis=-1) * video_framerate / audio_sr
        ).astype(int)
        video_indices.extend(vid_idx)

    # Create the save file
    save_file = pd.DataFrame(
        {
            "audio_onset": audio_onsets,
            "audio_offset": audio_offsets,
            "audio_path": audio_paths,
            "video_path": video_paths,
            "video_index": video_indices,
            "p0_x": [np.nan] * len(audio_onsets),
            "p0_y": [np.nan] * len(audio_onsets),
            "p1_x": [np.nan] * len(audio_onsets),
            "p1_y": [np.nan] * len(audio_onsets),
        }
    )
    save_file.to_csv(save_file_path, index=False)
    return save_file


def make_spectrograms(audio: np.ndarray, height: int, width: int) -> np.ndarray:
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
    spectrograms = cv2.resize(spectrograms, (width, height))
    spectrograms = cv2.applyColorMap(spectrograms, cv2.COLORMAP_VIRIDIS)
    return spectrograms


def get_annotated_points_for_frame(
    frame: np.ndarray,
    frame_idx: int,
    video_label: str,
    previous_points: Optional[np.ndarray] = None,
    audio: Optional[np.ndarray] = None,
) -> np.ndarray:
    cv2.namedWindow("Frame")

    spec_height = 160
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    if audio is not None:
        spectrograms = make_spectrograms(audio, spec_height, frame_width)
        frame = np.concatenate([spectrograms, frame], axis=0)
    else:
        spec_height = 0  # make the math easier

    # Draw the frame number and video label
    annotation_str = f"Frame {frame_idx} - {video_label}"
    frame = cv2.putText(
        frame,
        annotation_str,
        (10, 30),  # bottom-left corner of text
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,  # font scale
        (0, 0, 0),
        2,  # thickness
        cv2.LINE_AA,
    )

    # Get the previous annotations, if present
    points = np.full((2, 2), np.nan)
    if previous_points is not None:
        previous_points = previous_points.reshape(-1, 2)[:2]
        points[: len(previous_points), :] = previous_points

    # callback for mouse input:
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            true_y = y - spec_height
            if true_y < 0:
                return
            clicked = np.array([x, true_y])
            closest = np.argmin(np.linalg.norm(points - clicked[None, :], axis=1))
            points[closest] = clicked

    cv2.setMouseCallback("Frame", mouse_callback)

    # Control loop
    while True:
        drawn_frame = frame.copy()
        # Draw the points
        for x, y in points:
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(drawn_frame, (int(x), int(y) + spec_height), 5, (0, 0, 255), -1)

        cv2.imshow("Frame", drawn_frame)
        key = cv2.waitKey(33)

        if key == ord("q"):
            return points, codes["q"]
        elif key == ord("h"):
            return points, codes["h"]
        elif key == ord("l"):
            return points, codes["l"]
        elif key == ord("e"):
            points = np.full((2, 2), np.nan)
            return points, codes["e"]
        elif key == ord("0"):
            return points, codes["0"]
        elif key == ord("9"):
            return points, codes["9"]
        else:
            continue


def temp_change_path(old_path: Path):
    # new_dir = Path("/Users/atanelus/new_dyads")
    # if old_path.suffix == ".avi":
    #     new_path = new_dir / old_path.parent.name / old_path.with_suffix(".mp4").name
    # else:
    #     new_path = new_dir / old_path.parent.name / old_path.name
    return old_path


def main():
    if save_file_path.exists():
        existing_annotations = pd.read_csv(save_file_path)
    else:
        existing_annotations = populate_save_file()

    # Find index of last non-nan (annotated) point
    cur_idx = np.flatnonzero(~np.isnan(existing_annotations["p0_x"].to_numpy()))[-1]
    print(cur_idx)
    while True:
        # cap = cv2.VideoCapture(
        #     str(Path(existing_annotations["video_path"][cur_idx]))
        # )
        cap = cv2.VideoCapture(
            str(temp_change_path(Path(existing_annotations["video_path"][cur_idx])))
        )
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            existing_annotations["video_index"][cur_idx],
        )
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            cur_idx += 1
            continue
        existing_points = (
            existing_annotations.loc[cur_idx, ["p0_x", "p0_y", "p1_x", "p1_y"]]
            .to_numpy()
            .reshape(2, 2)
        )
        new_points, code = get_annotated_points_for_frame(
            frame,
            cur_idx,
            video_label(Path(existing_annotations["video_path"][cur_idx])),
            previous_points=existing_points,
            # audio=get_audio(
            #     existing_annotations["audio_path"][cur_idx],
            #     existing_annotations["audio_onset"][cur_idx],
            #     existing_annotations["audio_offset"][cur_idx],
            # ),
            audio=get_audio(
                temp_change_path(Path(existing_annotations["audio_path"][cur_idx])),
                existing_annotations["audio_onset"][cur_idx],
                existing_annotations["audio_offset"][cur_idx],
            ),
        )

        if code == "quit":
            existing_annotations.to_csv(save_file_path, index=False)
            break

        existing_annotations.loc[cur_idx, ["p0_x", "p0_y"]] = new_points[0]
        existing_annotations.loc[cur_idx, ["p1_x", "p1_y"]] = new_points[1]

        if code == "reverse":
            cur_idx = max(0, cur_idx - 1)
        elif code == "forward":
            cur_idx = min(cur_idx + 1, len(existing_annotations) - 1)
        elif code == "erase":
            continue
        elif code == "to_start":
            cur_idx = 0
        elif code == "to_end":
            cur_idx = len(existing_annotations) - 1
        else:
            # default to last annotated point
            cur_idx = (~existing_annotations["p0_x"].isna()).idxmax()


if __name__ == "__main__":
    main()
