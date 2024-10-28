import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import stft
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

DOWNLOAD_PATH = Path(consts["download_path"])
PARTIAL_DATASET_DIR = Path(consts["partial_dataset_dir"])
CLIP_DIR = Path(consts["manual_annotation_npy_dir"])
BACKUP_CLIP_DIR = Path(consts["manual_annotation_npy_dir_2"])
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]
VIDEO_FRAME_RATE = consts["video_frame_rate"]
NUM_MICROPHONES = consts["audio_num_channels"]
date_save_format = consts["date_format_for_saving"]


def parse_date_from_filename(filepath: Path) -> datetime:
    num = filepath.name
    # Will look like yyMMddHHmmss
    # Ex: 240307103101
    date = datetime.strptime(num, "%Y%m%d_%H%M%S")
    return date


def to_float(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float32) / np.iinfo(array.dtype).max).astype(np.float64)


def get_npys():
    audio_files = DOWNLOAD_PATH.glob("*/*.mmap")
    audio_files = filter(
        lambda p: not set(p.parent.name).difference(set("0123456789_")), audio_files
    )
    audio_files = list(audio_files)
    audio_files.sort()

    one_minute = int(AUDIO_SAMPLE_RATE * 60)

    for audio_file in tqdm(audio_files):
        try:
            path_date = parse_date_from_filename(audio_file.parent).strftime(
                date_save_format
            )
            new_path = CLIP_DIR / f"{path_date}.npy"
            should_make_backup = math.isnan(
                float(
                    consts["manually_annotated_start_indices"].get(
                        str(path_date), "nan"
                    )
                )
            )
            if new_path.exists() and not should_make_backup:
                continue
        except:
            continue
        num_samples_in_file = int(audio_file.stem.split("_")[-3])
        handle = np.memmap(
            audio_file,
            dtype="int16",
            mode="r",
            shape=(num_samples_in_file, NUM_MICROPHONES),
        )
        # clip = to_float(np.array(handle[:one_minute, :]))
        if should_make_backup:
            clip = np.array(handle[one_minute : 2 * one_minute, :])
        else:
            clip = np.array(handle[:one_minute, :])
        # strongest_channel = (clip**2).sum(axis=0).argmax()
        # clip = clip[:, strongest_channel]
        _, t, clip = stft(clip, axis=0)
        # shape is (freq, channels, time)
        clip = np.log(np.abs(clip) + 1e-12).max(axis=1)
        if should_make_backup:
            new_path = BACKUP_CLIP_DIR / f"{path_date}.npy"
        np.save(new_path, clip.astype(np.float16))
        np.save(new_path.with_suffix(".time.npy"), t)
        del handle


if __name__ == "__main__":
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    get_npys()
