import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import stft
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

DOWNLOAD_PATH = Path(consts["download_path"])
PARTIAL_DATASET_DIR = Path(consts["partial_dataset_dir"])
CLIP_DIR = Path(consts["manual_annotation_npy_dir"])
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
    return array.astype(np.float32) / np.iinfo(array.dtype).max


def get_npys():
    audio_files = DOWNLOAD_PATH.glob("*/*.mmap")
    audio_files = filter(
        lambda p: not set(p.parent.name).difference(set("0123456789_")), audio_files
    )
    audio_files = list(audio_files)
    audio_files.sort()

    one_minute = int(AUDIO_SAMPLE_RATE * 60)

    def proc(audio_file):
        try:
            path_date = parse_date_from_filename(audio_file.parent)
            if path_date < datetime(2024, 9, 1):
                return
            path_date = path_date.strftime(date_save_format)
            txt_file = next(audio_file.parent.glob("*_spacing.txt"), None)

            new_path = CLIP_DIR / f"{path_date}.npy"
            new_txt_path = CLIP_DIR / f"{path_date}_spacing.txt"
            if txt_file is not None:
                shutil.copy(txt_file, new_txt_path)
            if new_path.exists():
                return
        except Exception as e:
            print(e)
            return
        num_samples_in_file = int(audio_file.stem.split("_")[-3])
        handle = np.memmap(
            audio_file,
            dtype="int16",
            mode="r",
            shape=(num_samples_in_file, NUM_MICROPHONES),
        )
        clip = to_float(np.array(handle[: 5 * one_minute, :]))
        strongest_channel = (clip**2).sum(axis=0).argmax()
        clip = clip[:, strongest_channel]
        f, t, clip = stft(clip, axis=0, fs=AUDIO_SAMPLE_RATE)
        # shape is (freq, time)
        clip = clip[(f > 60000) & (f < 80000), :]
        clip = np.log(np.abs(clip) + 1e-12)  # reduce across channels
        power = (clip).sum(axis=0)
        np.save(new_path, power.astype(np.float16))
        # t = t + (num_samples_in_file - 5 * one_minute) / AUDIO_SAMPLE_RATE
        np.save(new_path.with_suffix(".time.npy"), t)
        del handle

    # Parallel(n_jobs=-1)(delayed(proc)(audio_file) for audio_file in tqdm(audio_files))
    # for audio_file in tqdm(audio_files):
    # proc(audio_file)
    proc(sorted(audio_files)[-2])


if __name__ == "__main__":
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    get_npys()
