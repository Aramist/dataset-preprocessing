import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

FULL_DATASET_PATH = Path(consts["full_dataset_path"])
AUDIO_SAMPLE_RATE = consts["audio_sample_rate"]
VIDEO_FRAME_RATE = consts["video_frame_rate"]
NUM_MICROPHONES = consts["audio_num_channels"]


def fix_locations():
    with h5py.File(FULL_DATASET_PATH, "r+") as ctx:
        if "orig_locations" not in ctx:
            ctx.copy("locations", "orig_locations")

        if "locations" in ctx:
            del ctx["locations"]

        orig_locations = ctx["orig_locations"][:]

        ctx.attrs["arena_dims"] = [615.0, 615.0, 425.0]
        ctx.attrs["arena_dims_units"] = "mm"

        scaled_locations = orig_locations * 1000.0  # m -> mm
        ctx["locations"] = scaled_locations


if __name__ == "__main__":
    fix_locations()
