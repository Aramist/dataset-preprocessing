from pathlib import Path

import h5py
import numpy as np

TEMP_DIR = Path("/home/atanelus/robot_temp")


def get_locations():
    temp_datasets = list(TEMP_DIR.glob("*.h5"))
    if not temp_datasets:
        raise ValueError("No partial datasets found in {}".format(TEMP_DIR))

    temp_datasets.sort(key=lambda p: int(p.stem.split("_")[-1]))

    all_locations = []
    for tp in temp_datasets:
        with h5py.File(tp, "r") as ctx:
            all_locations.extend(ctx["locations"][:])

    all_locations = np.array(all_locations)
    print(all_locations.shape)
    return all_locations


def main():
    dataset_path = Path(
        "/mnt/home/atanelus/ceph/neurips_datasets/audio/edison-4m-e1_audio.h5"
    )

    with h5py.File(dataset_path, "r+") as f:
        if "orientations" in f:
            del f["orientations"]
        if "node_names" in f:
            del f["node_names"]
        if "locations" in f:
            del f["locations"]

        new_locations = get_locations()
        f["locations"] = new_locations

        node_names = np.array(
            list(map(str.encode, ["earbud", "robot_front", "robot_rear"]))
        )
        f["node_names"] = node_names


if __name__ == "__main__":
    main()
