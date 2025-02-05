import json
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Custom sort key because I might end up with data distributed across multiple folders
str_path_date = lambda p: "_".join(p.stem.split("_")[:-1])

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)

working_dir = Path(consts["working_dir"])

processed_track_dir = working_dir / consts["processed_track_dir"]
processed_track_dir.mkdir(exist_ok=True, parents=True)

corner_points = np.array(consts["arena_corner_points"])

unprocessed_track_dir = Path(consts["sleap_track_dir"])
if not unprocessed_track_dir.exists():
    raise FileNotFoundError("No SLEAP tracks found. Run make_tracks.py first")


def get_tracks_from_h5(h5_file: Path):
    """Reads tracks from an h5 file and returns them as an array"""
    with h5py.File(h5_file, "r") as ctx:
        tracks = ctx["tracks"][0, :, :, :]
        # shape is (coords, nodes, time)
        tracks = tracks.transpose(2, 1, 0)
        # shape is (time, nodes, coords)
        node_names = ctx["node_names"][:]

    # Remove tail, keep nose and head
    # tracks = tracks[:, :2, :]
    return tracks, node_names


def simple_process_tracks(tracks):
    # tracks have shape (time, num_nodes, num_coords)

    pooled_nan = np.isnan(tracks)
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

    # print(f"Proportion of NaNs in tracks: {pooled_nan.any(axis=-1).mean(axis=0)}")
    return tracks


def triangle_area(points: np.ndarray) -> float:
    """Returns the area of the triangle defined by the three points

    Args:
        points: A 3x2 numpy array of the three points
    """
    a, b, c = points
    return 0.5 * np.abs(np.cross(b - a, c - a))


def project_point_to_line(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Projects a point onto a line defined by two points

    Args:
        point: A 2D numpy array of the point to project
        line: A 2x2 numpy array of the two points defining the line
    """

    # vec from a to b
    a, b = line
    p = point
    ab = b - a
    ap = p - a
    return a + np.dot(ap, ab) / np.dot(ab, ab) * ab


def project_to_bounds(tracks: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    """Gets the projection of tracks onto the quadrilateral defined by corner_points
    This removes changes in the observed x and y coordinates that are due to climbing
    along the border wall.
    """

    centroid = corner_points.mean(axis=0)

    orig_shape = tracks.shape
    tracks = tracks.reshape(-1, 2)

    # assumes the points are ordered clockwise or counterclockwise
    quad_area = triangle_area(corner_points[(0, 1, 2), :]) + triangle_area(
        corner_points[(0, 2, 3), :]
    )

    # Approx len of the 0-1 and 2-3 edges
    edge_1_len = min(
        np.linalg.norm(corner_points[0] - corner_points[1]),
        np.linalg.norm(corner_points[3] - corner_points[2]),
    )
    # Approx len of the 1-2 and 3-0 edges
    edge_2_len = min(
        np.linalg.norm(corner_points[1] - corner_points[2]),
        np.linalg.norm(corner_points[0] - corner_points[3]),
    )

    proj_tracks = []
    for t in tracks:
        # Determine if the point is outside the quadrilateral
        # If it is, project it onto the nearest edge
        tri_areas = (
            triangle_area([t, *corner_points[(0, 1), :]]),
            triangle_area([t, *corner_points[(1, 2), :]]),
            triangle_area([t, *corner_points[(2, 3), :]]),
            triangle_area([t, *corner_points[(3, 0), :]]),
        )

        in_quad = np.sum(tri_areas) <= quad_area
        if in_quad:
            proj_tracks.append(t)
            continue

        # Project the point onto the nearest edge
        edge_dists = [
            np.linalg.norm(np.cross(t - corner_points[i], t - corner_points[j]))
            / np.linalg.norm(corner_points[i] - corner_points[j])
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]
        ]
        if edge_dists[0] > edge_2_len:
            # The point is far from the 0-1 edge, project to the 2-3 line
            t = project_point_to_line(t, corner_points[(2, 3), :])
        if edge_dists[1] > edge_1_len:
            # The point is far from the 1-2 edge, project to the 3-0 line
            t = project_point_to_line(t, corner_points[(3, 0), :])
        if edge_dists[2] > edge_2_len:
            # The point is far from the 2-3 edge, project to the 0-1 line
            t = project_point_to_line(t, corner_points[(0, 1), :])
        if edge_dists[3] > edge_1_len:
            # The point is far from the 3-0 edge, project to the 1-2 line
            t = project_point_to_line(t, corner_points[(1, 2), :])

        proj_tracks.append(t)
    proj_tracks = np.array(proj_tracks)
    proj_tracks = proj_tracks.reshape(orig_shape)

    return proj_tracks


def main():
    unprocessed_track_files = list(unprocessed_track_dir.glob("*.analysis.h5"))

    def subroutine(track_file):
        date = str_path_date(track_file)
        processed_track_file = processed_track_dir / (date + ".tracks.npy")
        tracks, node_names = get_tracks_from_h5(track_file)
        tracks = simple_process_tracks(tracks)
        if b"nose" not in node_names:
            # 0 is head, 1 is tail
            # tail -> head vector
            diff = tracks[:, 0, :] - tracks[:, 1, :]
            est_nose = tracks[:, 0, :] + diff * 0.25
            tracks = np.concatenate([est_nose[:, None, :], tracks], axis=1)
        tracks = tracks[:, :2, :]  # remove tail
        # tracks = project_to_bounds(tracks, corner_points)
        np.save(processed_track_file, tracks)

    Parallel(n_jobs=-1)(
        delayed(subroutine)(track_file) for track_file in tqdm(unprocessed_track_files)
    )


if __name__ == "__main__":
    main()
