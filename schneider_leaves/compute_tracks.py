import json
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)
data_dir = Path(consts["recording_session_dir"])
working_dir = Path(consts["working_dir"])
processed_track_dir = working_dir / consts["processed_track_dir"]

recording_sessions = list(filter(lambda p: p.is_dir(), data_dir.glob("*/*")))
recording_sessions.sort()
audio_sample_rate = consts["audio_sr"]
arena_corner_points = np.array(consts["arena_corner_points_px"])


def simple_interpolate_tracks(tracks):
    # tracks have shape (time, num_coords)
    tracks = tracks.copy()
    n_ts, n_coords = tracks.shape
    for coord in range(n_coords):
        arr_slice = tracks[:, coord]
        nan_mask = np.isnan(arr_slice)
        where_valid = np.flatnonzero(~nan_mask)

        # fill nans at beginning and end with first and last valid value
        if (first_valid := where_valid[0]) > 0:
            arr_slice[:first_valid] = arr_slice[first_valid]
        if (last_valid := where_valid[-1]) < n_ts - 1:
            arr_slice[last_valid + 1 :] = arr_slice[last_valid]

        # update nan mask to account for changed values
        nan_mask = np.isnan(arr_slice)

        # interpolate nans in between
        real_xp = np.arange(n_ts)[~nan_mask]
        real_fp = arr_slice[~nan_mask]
        eval_x = np.arange(n_ts)[nan_mask]

        fill_fp = np.interp(eval_x, real_xp, real_fp)
        arr_slice[nan_mask] = fill_fp

        tracks[:, coord] = arr_slice
    return tracks


def get_pos_for_frame(frame: np.ndarray, thresh_value=90) -> np.ndarray:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = hsv_frame[..., 2] < thresh_value

    # Erode and dilate to remove noise from leaves on the ground
    for ksize in (5, 11):
        kernel = np.ones((ksize, ksize), np.uint8)
        thresh = cv2.erode(thresh.astype(np.uint8), kernel, iterations=2)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Remove anything outside the arena roi
    min_x, max_x = arena_corner_points[:, 0].min(), arena_corner_points[:, 0].max()
    min_y, max_y = arena_corner_points[:, 1].min(), arena_corner_points[:, 1].max()
    thresh[:min_y, :] = 0
    thresh[max_y:, :] = 0
    thresh[:, :min_x] = 0
    thresh[:, max_x:] = 0

    # Get contours
    contours, _ = cv2.findContours(
        thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [cv2.convexHull(contour) for contour in contours]
    contours = list(filter(lambda c: cv2.contourArea(c) > 1000, contours))

    if not contours and thresh_value < 120:
        return get_pos_for_frame(frame, thresh_value=thresh_value + 10)
    elif not contours:
        return np.array([np.nan, np.nan])

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return np.array([cx, cy])


def process_tracks_for_session(session_dir: Path):
    vid_path = next(session_dir.glob("*_CamFlir1_*.avi"), None)
    if not vid_path:
        raise ValueError(f"No video file found in {session_dir}")

    cap = cv2.VideoCapture(str(vid_path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    tracks = []

    def frame_iterator(vid_path):
        cap = cv2.VideoCapture(str(vid_path))
        for _ in tqdm(range(num_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

    tracks = Parallel(n_jobs=-4)(
        delayed(get_pos_for_frame)(frame) for frame in frame_iterator(vid_path)
    )

    tracks = np.array(tracks)
    tracks = simple_interpolate_tracks(tracks)
    return tracks


def main():
    processed_track_dir.mkdir(exist_ok=True, parents=True)

    for session_dir in recording_sessions:
        session_name = session_dir.parent.name
        tracks = process_tracks_for_session(session_dir)
        np.save(processed_track_dir / f"{session_name}.npy", tracks)


if __name__ == "__main__":
    main()
