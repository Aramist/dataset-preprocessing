import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from kalman_smoother import KalmanSmoother, get_tracks_from_h5
from tqdm import tqdm

# Custom sort key because I might end up with data distributed across multiple folders
str_path_date = lambda p: "_".join(p.stem.split("_")[:-1])

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

working_dir = Path(consts["working_dir"])

processed_track_dir = working_dir / consts["processed_track_dir"]
processed_track_dir.mkdir(exist_ok=True, parents=True)

unprocessed_track_dir = Path(consts["sleap_track_dir"])
if not unprocessed_track_dir.exists():
    raise FileNotFoundError("No SLEAP tracks found. Run make_tracks.py first")


def simple_process_tracks(tracks):
    # tracks have shape (time, num_nodes, num_coords)
    tracks = tracks.copy()
    n_ts, n_nodes, n_coords = tracks.shape
    for node in range(n_nodes):
        for coord in range(n_coords):
            arr_slice = tracks[:, node, coord]
            nan_mask = np.isnan(arr_slice)
            where_valid = np.flatnonzero(~nan_mask)

            # fill nans at beginning and end with first and last valid value
            if (first_valid := where_valid[0]) > 0:
                arr_slice[:first_valid] = arr_slice[first_valid]
            if (last_valid := where_valid[-1]) < n_ts - 1:
                arr_slice[last_valid + 1 :] = arr_slice[last_valid]

            # update nan mask to account for changed values
            nan_mask = np.isnan(arr_slice)

            # Find long stretches of nans, we will leave these alone because
            # interpolation can be unreliable
            nan_onsets = np.flatnonzero(np.diff(nan_mask.astype(int)) == 1)
            nan_offsets = np.flatnonzero(np.diff(nan_mask.astype(int)) == -1)
            nan_regions = np.stack([nan_onsets, nan_offsets], axis=1)

            # interpolate nans in between
            real_xp = np.arange(n_ts)[~nan_mask]
            real_fp = arr_slice[~nan_mask]
            eval_x = np.arange(n_ts)[nan_mask]

            fill_fp = np.interp(eval_x, real_xp, real_fp)
            arr_slice[nan_mask] = fill_fp

            tracks[:, node, coord] = arr_slice
    return tracks


def main():
    unprocessed_track_files = list(unprocessed_track_dir.glob("*.analysis.h5"))

    def subroutine(track_file: Path):
        date = str_path_date(track_file)
        processed_track_file = processed_track_dir / (date + ".tracks.npy")
        processed_covs_file = processed_track_dir / (date + ".covs.npy")
        if processed_track_file.exists():
            return
        try:
            # has shape (time, num_coords)
            tracks = get_tracks_from_h5(track_file)
        except Exception as e:
            print(track_file)
            print(e)
            return
        bounds = np.array(
            [[0, 0, -100, -100, -500, -500], [640, 512, 100, 100, 500, 500]]
        )
        # smoother = KalmanSmoother(bounds=bounds)
        # smoothed_tracks, covs = smoother.smooth(tracks)
        # if np.isnan(smoothed_tracks).any():
        # print(f"NaNs found in {track_file}. Attempting simple processing")
        reshaped_tracks = tracks[:, None, :]
        smoothed_tracks = simple_process_tracks(reshaped_tracks).squeeze()
        # covs = np.zeros((smoothed_tracks.shape[0], 2, 2))
        np.save(processed_track_file, smoothed_tracks)
        # np.save(processed_covs_file, covs)

    Parallel(n_jobs=-2)(
        delayed(subroutine)(track_file) for track_file in tqdm(unprocessed_track_files)
    )


if __name__ == "__main__":
    main()
