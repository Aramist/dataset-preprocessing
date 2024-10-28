import argparse
import glob
import os
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

TMP_DIR = Path("/home/atanelus/robot_temp")
# Expected latency between the audio playback signal and the true onset of audio stimulus
# Given in seconds
EST_LATENCY = 118 / 1000
# Sampling rate of the audio in the h5 files given in Hz
AUDIO_SR = 125000
# Fallback framerate of videos given in Hz
DEFAULT_FRAMERATE = 30.0
# Duration of the audio playback signal for each stimulus in milliseconds
audio_durations = np.load("durations.npy")


def get_video_framerate(video_path: Path) -> float:
    """Get the framerate of a video"""
    cap = cv2.VideoCapture(str(video_path))
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Sometimes the method fails and returns 0.0
    if np.abs(framerate) < 1e-5:
        # print(
        #     f"Warning: failed to get framerate for {video_path}, falling back to 30.0"
        # )
        framerate = DEFAULT_FRAMERATE
    return framerate


def get_audio(
    fname: Path,
    indices: np.ndarray,
    return_audio: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the audio from the h5 file, and the lengths of each audio segment"""
    threshold_ms = 15
    concat_audio = []
    with h5py.File(fname, "r") as ctx:
        keys = sorted(list(ctx["ai_channels"].keys()))
        file_len = ctx["ai_channels/ai0"].shape[0]

        indices[:, 1] = np.clip(
            indices[:, 1], 0, file_len
        )  # Ensure none extend past end of file
        # Remove segments that are too short
        valid_onsets = indices[
            indices[:, 1] - indices[:, 0] > threshold_ms * AUDIO_SR / 1000
        ]
        lengths = valid_onsets[:, 1] - valid_onsets[:, 0]

        if return_audio:
            for start, end in valid_onsets:
                audio = np.stack(
                    [ctx[f"ai_channels/{key}"][start:end] for key in keys], axis=1
                )
                concat_audio.append(audio)
    return (
        np.concatenate(concat_audio, axis=0) if return_audio else None,
        np.array(lengths),
        np.array(valid_onsets),
    )


def make_metadata_df(
    temp_file_dir: Path, vid_paths: List[Path], invalid_frames_df: pd.DataFrame
) -> pd.DataFrame:
    """Make a metadata csv containing the following for each vocalization:
    - index of the vocalization in full dataset
    - path to temp file containing the vocalization
    - path to the video file containing the vocalization
    - path to the audio file containing the vocalization
    - start index of the vocalization in the original audio file
    - end index of the vocalization in the original audio file
    - start index of the vocalization in the temp audio file
    - end index of the vocalization in the temp audio file
    """
    metadata_rows = []
    vid_name_df = pd.read_csv("video/video_paths.csv")
    vid_paths = vid_name_df["video_path"]
    track_paths = vid_name_df["track_path"]

    sorted_temp_paths = sorted(
        list(temp_file_dir.glob("*.h5")), key=lambda x: int(x.stem.split("_")[-1])
    )

    for vid_path, track_path, temp_path in tqdm(
        zip(vid_paths, track_paths, sorted_temp_paths)
    ):
        audio_path = audio_path_from_video(vid_path)
        audio_length = get_length_of_audio(audio_path)
        segment_indices = get_onsets_offsets(audio_path)
        invalid_segments = invalid_frames_df[
            invalid_frames_df["video_path"] == vid_path
        ]

        j = -1
        while segment_indices[j, 1] > audio_length:
            j -= 1
        if j < -1:
            segment_indices = segment_indices[: j + 1, :]
            print(
                f"Pruned {-1-j} invalid segments from {vid_path}. Exceeded audio length"
            )

        video_sr = get_video_framerate(vid_path)
        segment_indices = filter_segments(segment_indices, invalid_segments, video_sr)
        _, lengths, valid_onsets = get_audio(
            audio_path, segment_indices, return_audio=False
        )
        len_idx = np.cumsum(np.insert(lengths, 0, 0))
        locations, frames = locations_for_audio(track_path, onsets=valid_onsets)

        for k, (start, end) in enumerate(valid_onsets):
            metadata_rows.append(
                {
                    "temp_path": temp_path,
                    "vid_path": vid_path,
                    "audio_path": audio_path,
                    "full_dataset_start_idx": start,
                    "full_dataset_end_idx": end,
                    "temp_file_start_idx": len_idx[k],
                    "temp_file_end_idx": len_idx[k + 1],
                    "video_frame_idx": frames[k],
                }
            )
    metadata_df = pd.DataFrame(metadata_rows)

    return metadata_df


def extract_background_audio(fname: Path, vox_indices: np.ndarray) -> np.ndarray:
    """Extracts a segment of bacground audio for future computations"""
    # Background audio is defined as a segment of audio outside the provided vocalization indices
    # We will take the first non-vocalization segment of audio as background
    with h5py.File(fname, "r") as ctx:
        keys = sorted(list(ctx["ai_channels"].keys()))
        start = vox_indices[0, 0]
        audio = np.stack([ctx[f"ai_channels/{key}"][:start] for key in keys], axis=1)
    return audio


def get_length_of_audio(fname: Path) -> int:
    """Get the length of the audio in samples"""
    with h5py.File(fname, "r") as ctx:
        audio_length = ctx["ai_channels/ai0"].shape[0]
    return audio_length


def get_onsets_offsets(fname):
    with h5py.File(fname, "r") as ctx:
        audio_onsets = ctx["audio_onset"][:, 0]
    audio_onsets += int(AUDIO_SR * EST_LATENCY)
    if len(audio_onsets) > len(audio_durations):
        audio_onsets = audio_onsets[-len(audio_durations) :]
    lengths = audio_durations[: len(audio_onsets)] * AUDIO_SR

    audio_offsets = audio_onsets + lengths

    audio_onsets = np.round(audio_onsets).astype(int)
    audio_offsets = np.round(audio_offsets).astype(int)
    stacked = np.stack([audio_onsets, audio_offsets], axis=1)
    # filter out degenerate samples (idk how these got here)
    threshold_ms = 5
    bad_mask = np.diff(stacked, axis=1).squeeze() < threshold_ms * AUDIO_SR / 1000
    stacked = stacked[~bad_mask, :]
    return stacked


def locations_for_audio(fname, onsets):
    video_sr = get_video_framerate(fname)
    avg_ts = 0.5 * onsets.sum(axis=1)
    avg_ts = np.round(avg_ts * video_sr / AUDIO_SR).astype(int)
    frames = np.load(fname)
    avg_ts = np.clip(avg_ts, 0, len(frames) - 1)
    return frames[avg_ts], avg_ts


def audio_path_from_video(vid_path):
    parent_path = Path(vid_path).parent
    return glob.glob(str(parent_path / "mic*.h5"))[0]


def write_frame_nums(frame_nums, n):
    fname = f"audio/frames_{n}.npy"
    np.save(fname, frame_nums)


def filter_segments(
    segments: np.ndarray, video_invalid_segments: pd.DataFrame, video_sr: float
) -> np.ndarray:
    i, j = 0, 0
    filtered_segments = []
    i_segments = [
        video_invalid_segments["invalid_onset"].values,
        video_invalid_segments["invalid_offset"].values,
    ]
    i_segments = np.stack(i_segments, axis=1)
    # Convert from video indices to audio indices
    i_segments = (i_segments * AUDIO_SR / video_sr).astype(int)

    # Removes any segments that overlap with the invalid segments
    while i < len(segments) and j < len(i_segments):
        start, end = segments[i]
        i_start, i_end = i_segments[j]
        if start < i_start and end < i_start:
            # segemnt is earlier than the invalid segment and does not overlop
            i += 1
            filtered_segments.append((start, end))
        elif (start < i_start and end > i_start) or (start > i_start and start < i_end):
            # segment overlaps with the invalid segment
            i += 1
        else:
            # segment is later than the invalid segment and no overlap
            j += 1
    # Add the segments that come after the last invalid segment
    filtered_segments.extend(segments[i:])
    return np.array(filtered_segments)


if __name__ == "__main__":
    # Command line flag for whether to make the train set or test set
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", action="store_true")
    args = parser.parse_args()
    make_train_set = not args.test_set

    vid_name_df = pd.read_csv("video/video_paths.csv")
    invalid_frames_df = pd.read_csv("video/invalid_regions.csv")

    Path("audio").mkdir(exist_ok=True)

    temp_files = []
    all_lengths = []
    all_locations = []
    # lazy so I'm keeping everything in RAM

    vids = list(vid_name_df["video_path"])
    tracks = list(vid_name_df["track_path"])

    print("Generating temp files (partial datasets)")
    for n, (vid_path, track_path) in tqdm(
        enumerate(zip(vids, tracks)), total=len(vids)
    ):
        # Preserve ordering from the csv
        tmp_file_path = Path(f"/home/atanelus/robot_temp/n_{n}.h5")
        tmp_file_path.parent.mkdir(exist_ok=True, parents=True)

        # Use cached results whenever possible
        if tmp_file_path.exists():
            print(f"Already processed {vid_path}, skipping")
            temp_files.append(tmp_file_path)
            # add its locations and lengths to the accumulator
            with h5py.File(tmp_file_path, "r") as ctx:
                vox_lengths = np.diff(ctx["length_idx"][:])
                all_lengths.extend(vox_lengths)
                all_locations.extend(ctx["locations"][:])
            continue

        # Start processing the video if the cached results don't exist

        invalid_segments = invalid_frames_df[
            invalid_frames_df["video_path"] == vid_path
        ]
        audio_path = audio_path_from_video(vid_path)
        segment_indices = get_onsets_offsets(audio_path)

        # Sometimes the last onset is invalid, so we remove it if necessary
        audio_length = get_length_of_audio(audio_path)
        # Starting from the last offset, prune backward until they are all valid
        i = -1
        while segment_indices[i, 1] > audio_length:
            i -= 1
        if i < -1:
            segment_indices = segment_indices[: i + 1, :]
            print(f"Removed {-1-i} invalid onsets from {vid_path}")

        # if invalid video segments exist, ensure that no audio is used from them
        if len(invalid_segments) > 0:
            pre_filtration_num_segments = len(segment_indices)
            video_sr = get_video_framerate(vid_path)
            segment_indices = filter_segments(
                segment_indices, invalid_segments, video_sr
            )
            post_filtration_num_segments = len(segment_indices)
            print(
                f"filtered {pre_filtration_num_segments - post_filtration_num_segments} onsets from {vid_path}"
            )
        audio, lengths, segment_indices = get_audio(
            audio_path, segment_indices
        )  # lengths does not include a zero at the beginning

        ######################################################################################
        # background_audio = extract_background_audio(audio_path, onsets)
        # Save to home dir

        # background_audio_path = f"/home/atanelus/robot/background_audio_{n}.npy"
        # np.save(background_audio_path, background_audio)
        ######################################################################################

        locations, frame_nums = locations_for_audio(track_path, segment_indices)
        print(f"Num frames: {len(frame_nums)}")
        write_frame_nums(frame_nums, n)

        len_idx = np.cumsum(np.insert(lengths, 0, 0))

        all_lengths.extend(lengths)
        all_locations.extend(locations)
        with h5py.File(tmp_file_path, "w") as ctx:
            ctx["audio"] = audio
            ctx["length_idx"] = len_idx
            ctx["locations"] = locations
        temp_files.append(tmp_file_path)
        print(f"Finished processing {vid_path}")
        print(f"Found {len(lengths)} vocalizations in {vid_path}")

    # Make metadata file
    metadata = make_metadata_df(
        temp_file_dir=TMP_DIR,
        vid_paths=vids,
        invalid_frames_df=invalid_frames_df,
    )
    print(f"Metadata num rows: {len(metadata)}")
    metadata.to_csv("dataset/metadata.csv", index=True)

    # Merge all the temp files into one big file
    print(f"Writing to final dataset")
    Path("dataset").mkdir(exist_ok=True)

    output_path = "dataset/robot_dataset_full.h5"

    with h5py.File(output_path, "w") as ctx:
        # all_lengths is still a python list
        full_len_index = np.cumsum([0] + all_lengths)
        ctx["audio"] = np.zeros((full_len_index[-1], 4), dtype=np.float32)
        ctx["length_idx"] = full_len_index  # should have shape (n+1,)
        ctx["locations"] = np.array(all_locations)  # should have shape (n, 3, 2)

        vox_idx = 0
        for i, tmp_file_path in tqdm(enumerate(temp_files), total=len(temp_files)):
            with h5py.File(tmp_file_path, "r") as tmp_ctx:
                tmp_len_idx = tmp_ctx["length_idx"][:]
                ctx["audio"][vox_idx : vox_idx + tmp_len_idx[-1], :] = tmp_ctx["audio"][
                    :
                ]
                vox_idx += tmp_len_idx[-1]

    # Clean up the temp files
    # for tmp_file_path in temp_files:
    # os.remove(tmp_file_path)
    print(f"Finished writing to {output_path}")
    print(f"Found {len(all_lengths)} vocalizations in total")
