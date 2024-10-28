import json
from datetime import datetime
from pathlib import Path
from random import choice

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# mpl.use("Qt5Agg")

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

FULL_DATASET_PATH = Path(consts["full_dataset_path"])
NUM_MICROPHONES = consts["audio_num_channels"]
DOWNLOAD_PATH = Path(consts["download_path"])
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]
date_save_format = consts["date_format_for_saving"]


def parse_date_from_filename(filepath: Path) -> datetime:
    num = filepath.name
    # Will look like yyMMddHHmmss
    # Ex: 240307103101
    date = datetime.strptime(num, "%Y%m%d_%H%M%S")
    return date


def get_powers(dset_path: Path) -> np.ndarray:
    if Path("powers.npy").exists():
        return np.load("powers.npy").mean(axis=0)
    with h5py.File(dset_path, "r") as f:
        audio = f["audio"]
        len_idx = f["length_idx"][:]
        rand_subset = np.random.choice(len(len_idx) - 1, size=2000)
        powers = []
        for idx in tqdm(rand_subset):
            start, end = len_idx[idx], len_idx[idx + 1]
            power = np.mean(audio[start:end] ** 2, axis=0)
            powers.append(power)
        powers = np.array(powers)
    np.save("powers.npy", powers)
    return powers.mean(axis=0)


def to_float(array: np.ndarray) -> np.ndarray:
    return (array.astype(np.float64) / np.iinfo(array.dtype).max).astype(np.float64)


def background_power():
    sessions = DOWNLOAD_PATH.iterdir()
    sessions = filter(
        lambda p: not set(p.stem).difference(set("0123456789_")), sessions
    )
    sessions = list(sessions)
    rand_session_dir = choice(sessions)
    audio_file = next(rand_session_dir.glob("*.mmap"), None)

    num_samples_in_file = int(audio_file.stem.split("_")[-3])
    handle = np.memmap(
        audio_file,
        dtype="int16",
        mode="r",
        shape=(num_samples_in_file, NUM_MICROPHONES),
    )

    # Given in seconds
    file_date = parse_date_from_filename(rand_session_dir)
    true_start_point = float(
        consts["manually_annotated_start_indices"][file_date.strftime(date_save_format)]
    )
    true_start_point = int(true_start_point * AUDIO_SAMPLE_RATE)

    num_to_take = min(true_start_point - 500, 10 * AUDIO_SAMPLE_RATE)
    background_audio = to_float(handle[:num_to_take])

    power = np.mean(background_audio**2, axis=0)
    return power


### code for mouse data
sr = AUDIO_SAMPLE_RATE


def load_audio():
    audio_fn = "/mnt/home/atanelus/ceph/murthy-lab-data/orig_dataset/230119183250_concatenated_audio_250000_300119365_24_int16.mmap"
    handle = np.memmap(audio_fn, dtype=np.int16, mode="r")
    handle = handle.reshape(300119365, 24)
    return handle


def load_segments():
    return np.load("/mnt/home/atanelus/ceph/murthy-lab-data/merged_annotations.npy")


def get_samples_for_vox(segments, idx):
    return (segments[idx] * sr).astype(int)


def get_vocalization(segments, audio_handle, idx, sr=sr):
    onset, offset = get_samples_for_vox(segments, idx)
    audio = to_float(audio_handle[onset:offset, :])
    return audio


def mouse_get_background_noise_power():
    audio_handle = load_audio()
    segments = load_segments()
    # first_onset = int(segments[0, 0] * sr)
    last_offset = int(segments[-1, 1] * sr)

    # start_point = 30 * sr
    # end_point = min(first_onset - 500, 10 * sr + start_point)

    start_point = max(len(audio_handle) - 10 * sr, last_offset + 100)
    end_point = len(audio_handle)

    background = to_float(audio_handle[start_point:end_point, :])
    power = np.mean(background**2, axis=0)

    return power


def mouse_get_powers():
    audio_handle = load_audio()
    segments = load_segments()
    if Path("mouse_powers.npy").exists():
        return np.load("mouse_powers.npy").mean(axis=0)
    powers = []
    for idx in tqdm(range(len(segments))):
        audio = get_vocalization(segments, audio_handle, idx)
        power = np.mean(audio**2, axis=0)
        powers.append(power)
    powers = np.array(powers)
    np.save("mouse_powers.npy", powers)
    return powers.mean(axis=0)


def plot_power():
    powers = get_powers(FULL_DATASET_PATH)
    background = background_power()
    ratio = powers / background
    diff = np.log10(ratio)
    sorted_ratios = np.sort(diff)[::-1]

    mouse_powers = mouse_get_powers()
    mouse_background = mouse_get_background_noise_power()
    mouse_ratio = mouse_powers / mouse_background
    mouse_diff = np.log10(mouse_ratio)
    mouse_sorted_ratios = np.sort(mouse_diff)[::-1]

    fig, ax = plt.subplots()
    ax.plot(sorted_ratios, label="Earbud")
    ax.plot(mouse_sorted_ratios, label="Mouse")
    ax.axhline(0, linestyle="--", label="Background noise")
    ax.legend()
    ax.set_title("Avg channel power during vocalization relative to baseline")
    ax.set_xlabel("Channel (sorted)")
    ax.set_ylabel("Power difference (log scale)")
    plt.show()


if __name__ == "__main__":
    plot_power()
