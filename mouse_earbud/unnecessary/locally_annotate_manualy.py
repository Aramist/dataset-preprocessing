import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AUDIO_SAMPLE_RATE = 250000

cached_results = {
    "2024_03_07_10_31_00": 35.855,
    "2024_03_07_11_04_03": 62.266,
    "2024_03_07_13_42_44": 18.0146,
    "2024_03_07_16_03_36": 23.299,
    "2024_03_08_10_56_36": 60.000,
    "2024_03_08_14_12_51": 69.9251,
    "2024_03_08_14_42_16": 60.395,
    "2024_03_08_15_20_57": 60.96,
    "2024_03_08_16_28_20": 46.909,
    "2024_03_08_16_57_26": 71.375,
    "2024_03_08_18_18_48": 66.3,
    "2024_03_08_18_52_23": 87.518,
    "2024_03_10_17_27_28": 53.838,
    "2024_03_10_18_12_33": 67.96,
    "2024_03_10_18_57_13": 76.786,
    "2024_03_11_14_06_47": 65.173,
    "2024_03_11_14_38_03": 62.994,
    "2024_03_11_15_06_59": 62.993,
    "2024_03_11_16_54_08": 61.944,
    "2024_03_11_17_25_40": 60.37,
    "2024_10_19_16_37_06": 65.27,
    "2024_10_19_18_15_42": 67.143,
    "2024_10_19_19_00_10": 73.447,
    "2024_10_19_19_33_51": 37.075,
    "2024_10_19_20_08_14": 83.996,
    "2024_10_19_20_38_20": 55.112,
    "2024_10_19_21_13_56": 25.064,
    "2024_10_19_22_25_17": 31.768,
    "2024_10_19_22_56_41": 9.96,
    "2024_10_20_14_21_00": 35.133,
    "2024_10_20_14_51_42": 42.511,
    "2024_10_20_15_23_01": 55.303,
    "2024_10_20_16_10_33": 22.142,
    "2024_10_20_16_42_22": 26.981,
    "2024_10_20_17_14_32": 58.405,
    "2024_10_20_18_30_59": 41.444,
    "2024_10_20_19_03_57": 71.739,
    "2024_10_20_19_35_52": 68.752,
    "2024_10_20_20_10_23": 125.824,
    "2024_10_20_20_39_33": 40.736,
}


def get_dir():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d", "--dir", required=True, help="Directory of npy files", type=Path
    )
    args = ap.parse_args()

    return args.dir


def get_timing_file(audio_fpath: Path):
    date = audio_fpath.stem
    split = date.split("_")
    date_refmt = "".join(split[:3]) + "_" + "".join(split[3:])

    timing_txt = next(
        filter(
            lambda p: date_refmt in p.stem, audio_fpath.parent.glob("*_spacing.txt")
        ),
        None,
    )
    if timing_txt is None:
        raise FileNotFoundError(f"Timing file not found for {date}")
    return timing_txt


def main():
    npy_dir = get_dir()
    npy_files = list(filter(lambda p: "time" not in p.name, npy_dir.glob("*.npy")))
    npy_files.sort()

    txt_files = list(filter(lambda p: "spacing" in p.name, npy_dir.glob("*.txt")))
    txt_files.sort()

    results = {}

    for npy_file in npy_files:
        date = npy_file.stem
        time_axis = np.load(npy_file.with_suffix(".time.npy"))
        timing_file_ms = npy_file.parent / (npy_file.stem + "_spacing.txt")
        timing_samps = np.loadtxt(timing_file_ms, delimiter=",", dtype=np.int64)[
            1:-1
        ]  # remove first silence epoch
        timing_samps = np.insert(timing_samps, 0, 0)
        keypoints = np.cumsum(timing_samps)
        keypoints_sec = keypoints / 250000.0
        # between keypoints idx 0 and 1 is first vocalization,
        # between 1 and 2 is first silence
        # between 2 and 3 is second vocalization, etc.
        vocalization_indices = keypoints_sec.reshape(-1, 2)
        # Now this only contains vocalizations

        power = np.load(npy_file)  # last five minutes of audio

        fig, ax = plt.subplots()
        ax.plot(time_axis, power)
        plt.show()

        true_start = input("Enter true end time (seconds): ")
        if true_start.strip() == "":
            end_time = float("nan")
        else:
            end_time = float(true_start)

        # time between end of last event and start of first event in seconds
        good_playback_duration = (
            vocalization_indices[-1, 1] - vocalization_indices[0, 0]
        )

        start_time = end_time - good_playback_duration  # will propagate nan
        results[date] = start_time

    print(results)


if __name__ == "__main__":
    main()
