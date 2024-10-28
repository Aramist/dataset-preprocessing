import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]


def get_dir():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d", "--dir", required=True, help="Directory of npy files", type=Path
    )
    args = ap.parse_args()

    return args.dir


def main():
    npy_dir = get_dir()
    npy_files = list(filter(lambda p: "time" not in p.name, npy_dir.glob("*.npy")))
    npy_files.sort()

    results = {}

    for npy_file in npy_files:
        date = npy_file.stem
        spectrogram = np.load(npy_file)
        time_axis = np.load(npy_file.with_suffix(".time.npy")) / AUDIO_SAMPLE_RATE
        plt.title(date)
        plt.imshow(
            spectrogram[::-1],
            aspect="auto",
            extent=[0, time_axis[-1], 0, AUDIO_SAMPLE_RATE / 2],
        )
        plt.show()

        true_start = input("Enter true start time (seconds): ")
        results[date] = float(true_start)

    print(results)


if __name__ == "__main__":
    main()
