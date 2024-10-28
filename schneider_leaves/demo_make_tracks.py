from collections import deque
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# top left, top right, bottom right, bottom left
arena_corner_points = np.array([[270, 70], [1215, 82], [1204, 1011], [263, 1007]])


def load_tracks(track_file: Path, process_tracks: bool = True):
    """Reads tracks from an h5 file and returns them as an array"""
    point_a_label = b"spine1"
    point_b_label = b"spine2"
    with h5py.File(track_file, "r") as ctx:
        node_names = ctx["node_names"][:]
        point_a_idx = np.where(node_names == point_a_label)[0][0]
        point_b_idx = np.where(node_names == point_b_label)[0][0]

        # shape is initially (coords, nodes, time)
        tracks = ctx["tracks"][0, :, [point_a_idx, point_b_idx], :]
        tracks = tracks.mean(axis=1)  # Want the midpoint of spine1 and spine2
        tracks = tracks.T
        # shape is now (time, coords)
    if not process_tracks:
        return tracks
    # Attempt to filter out jumpy predictions
    velocity_est = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
    velocity_est = np.insert(velocity_est, 0, 0)  # Recover original shape
    smoothed_velocity = np.convolve(velocity_est, np.ones(10) / 10, mode="same")
    non_nan_velocities = smoothed_velocity[~np.isnan(smoothed_velocity)]
    thresh = np.quantile(non_nan_velocities, 0.80)
    tracks[smoothed_velocity > thresh] = np.nan  # will be linearly interpolated later

    return simple_process_tracks(tracks)


def simple_process_tracks(tracks):
    # tracks have shape (time, num_nodes, num_coords)
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


def video_feed():
    writer = cv2.VideoWriter(
        "demo.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (1440, 1080),
        isColor=True,
    )
    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # while cv2.waitKey(1) != ord("q"):
    while True:
        package = yield
        if package is None:
            break
        frame, points = package
        for point in points:
            if np.isnan(point).any():
                frame = cv2.putText(
                    frame,
                    "Lost track",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                break
            frame = cv2.circle(
                frame, tuple(point.squeeze().astype(int)), 5, (0, 0, 255), -1
            )
        # cv2.imshow("frame", frame)
        writer.write(frame)
    # cv2.destroyWindow("frame")
    writer.release()


class OpenCVTracker:
    def get_pos_for_frame(self, frame: np.ndarray, thresh_value=90) -> np.ndarray:
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
            return self.get_pos_for_frame(frame, thresh_value=thresh_value + 10)
        elif not contours:
            return np.array([np.nan, np.nan])

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return np.array([cx, cy])


class PointFlowOrganizer:
    lk_params = {
        "winSize": (30, 30),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }

    def __init__(
        self, max_population: int, velocity_window_len: int, speed_thresh: float
    ):
        self.max_population = max_population
        self.velocity_window_len = velocity_window_len
        self.speed_thresh = speed_thresh
        self.pos_history: list[deque] = [
            deque(maxlen=velocity_window_len) for _ in range(max_population)
        ]

        self.last_frame = None

    @property
    def existing_points(self):
        return (
            np.array([deq[0] for deq in self.pos_history if deq])
            .astype(np.float32)
            .reshape(-1, 1, 2)
        )

    def to_gray(self, im):
        if len(im.shape) == 3:
            return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return im

    def initialize(self, first_frame: np.ndarray, initial_guess: np.ndarray):
        # Create many guesses in a cloud around the initial guess
        noise = np.random.normal(0, 30, (self.max_population, 1, 2)).astype(np.float32)
        noise[0, ...] = 0
        initial_guess = (
            np.repeat(initial_guess[None, :], self.max_population, axis=0)[:, None, :]
            + noise
        )

        for guess, deq in zip(initial_guess, self.pos_history):
            deq.append(guess.squeeze())

        self.last_frame = self.to_gray(first_frame)
        return initial_guess

    def update(self, new_video_frame: np.ndarray, est_pos: np.ndarray):
        new_frame_gray = self.to_gray(new_video_frame)
        new_points, st, err = cv2.calcOpticalFlowPyrLK(
            self.last_frame,
            new_frame_gray,
            self.existing_points,
            None,
            **self.lk_params,
        )

        # At the start new_points should have the same len as point_history
        to_remove = ~(st.astype(bool).reshape(-1))

        # Update pos history
        for deq, point in zip(self.pos_history, new_points):
            deq.append(point.squeeze())

        # Compute velocity
        for n, deq in enumerate(self.pos_history):
            if to_remove[n] or len(deq) < self.velocity_window_len:
                continue
            speed = np.linalg.norm(np.diff(deq, axis=0), axis=-1)
            avg_speed = speed.mean()
            if avg_speed < self.speed_thresh:
                to_remove[n] = True

        # Remove points
        # Important that this is done in reverse order
        for n in np.flatnonzero(to_remove)[::-1]:
            del self.pos_history[n]
        new_points = np.delete(new_points, np.flatnonzero(to_remove), axis=0)

        # Add new points if space permits
        if len(new_points) < self.max_population:
            new_points = np.concatenate([new_points, est_pos.reshape(1, 1, 2)], axis=0)
            self.pos_history.append(deque(maxlen=self.velocity_window_len))
            self.pos_history[-1].append(est_pos)

        self.last_frame = new_frame_gray
        return new_points


def make_tracks_from_of(
    vid_path: Path,
    mouse_est_tracks: np.ndarray,
    max_population: int = 30,
    velocity_window: int = 5,
    speed_thresh: float = 2,  # pixels per frame
):

    tracker = PointFlowOrganizer(max_population, velocity_window, speed_thresh)
    cap = cv2.VideoCapture(str(vid_path))
    ret, old_frame = cap.read()
    new_points = tracker.initialize(old_frame, mouse_est_tracks[0])

    vid_feed = video_feed()
    next(vid_feed)
    vid_feed.send((old_frame, new_points))

    for idx in range(1, len(mouse_est_tracks)):
        ret, new_frame = cap.read()
        if not ret:
            break

        new_points = tracker.update(new_frame, mouse_est_tracks[idx])

        # Visualize
        vid_feed.send((new_frame, new_points))


def process_with_simple_tracker(vid_path: Path) -> np.ndarray:

    cap = cv2.VideoCapture(str(vid_path))
    tracker = OpenCVTracker()
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"resolution: ({frame_width},{frame_height})")

    vid_feed = video_feed()
    next(vid_feed)

    tracks = []

    for _ in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Visualize
        point = tracker.get_pos_for_frame(frame)
        tracks.append(point)
        vid_feed.send((frame, [point]))

    vid_feed.send(None)
    return np.array(tracks)


def demo():
    local_vid_path = Path("/Users/atanelus/video_samples.mp4")
    # local_track_path = Path(
    #     "/Users/atanelus/20240424_vocalizations_m5_2024-04-24_001_CamFlir1_20240424_161448_.predictions.000_20240424_vocalizations_m5_2024-04-24_001_CamFlir1_20240424_161448.analysis.h5"
    # )
    # tracks = load_tracks(local_track_path, process_tracks=False)
    # make_tracks_from_of(
    #     local_vid_path,
    #     tracks,
    #     max_population=60,
    #     velocity_window=10,
    #     speed_thresh=5,
    # )
    process_with_simple_tracker(local_vid_path)
