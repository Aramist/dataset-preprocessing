import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

PARTIAL_DATASET_DIR = Path(consts["partial_dataset_dir"])
VERIFICATION_DIR = Path(consts["verification_dir"])
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]
NUM_MICROPHONES = consts["audio_num_channels"]


def make_plot_for_dataset(dset_path: Path):
    # Plots the first five vocalizations

    # Each spectrogram will be 3inx2in
    fig, axes = plt.subplots(NUM_MICROPHONES, 5, figsize=(5 * 3, NUM_MICROPHONES * 2))

    with h5py.File(dset_path, "r") as ctx:
        length_idx = ctx["length_idx"][:]
        for i in range(5):
            start, end = length_idx[i], length_idx[i + 1]
            audio = ctx["audio"][start:end, :]
            for j in range(NUM_MICROPHONES):
                axes[j, i].specgram(audio[:, j], Fs=AUDIO_SAMPLE_RATE)
                axes[j, i].axis("off")

    plt.tight_layout()
    plt.savefig(VERIFICATION_DIR / f"{dset_path.stem}.png")


def main():
    VERIFICATION_DIR.mkdir(exist_ok=True, parents=True)
    for dset_path in tqdm(sorted(list(PARTIAL_DATASET_DIR.glob("*.h5")))):
        make_plot_for_dataset(dset_path)


if __name__ == "__main__":
    main()
