from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class KalmanSmoother:
    def __init__(
        self,
        bounds: np.ndarray = None,
    ):
        """A class for smoothing SLEAP tracks using a Kalman filter.
        All equations derived from "Feedback Systems: An Introduction for Scientists and Engineers" by Karl J. Åström and Richard M. Murray
        """

        self.x = None  # Initial state estimate with velocity
        # Transition matrix
        self.A = np.eye(6)
        self.A[0:2, 2:4] = np.eye(2)  # velocity -> position
        self.A[0:2, 4:6] = np.eye(2)  # acceleration -> velocity
        self.A[0:2, 4:6] = 0.5 * np.eye(2)  # acceleration -> position
        self.R_v = np.diag(
            [0, 0, 0, 0, 5, 5]
        )  # Process noise. We allow for random velocity changes

        self.P = np.diag([5, 5, 5, 5, 5, 5]) ** 2  # Initial state covariance

        self.C = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        )  # Measurement matrix. Assume no velocity measurements
        self.R_w = np.diag([10, 10, 5, 5]) ** 2  # Measurement noise covariance
        self.nan_R_w = (
            np.diag([100, 100, 100, 100]) ** 2
        )  # Measurement noise covariance for NaNs

        # Measure and process noise are initially sampled from standard normal distribution
        # Covaried noise is then obtained through R_w * noise * R_w.T and R_v * noise * R_v.T
        self.bounds = bounds
        self.last_measurement = None

    @property
    def L(self):
        """Computes the observer gain matrix L. Equation 7.22"""
        return (
            self.A
            @ self.P
            @ self.C.T
            @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R_w)
        )

    @property
    def L_nan(self):
        """Computes the observer gain matrix L when the measurement is NaN. Equation 7.22"""
        return (
            self.A
            @ self.P
            @ self.C.T
            @ np.linalg.inv(self.C @ self.P @ self.C.T + self.nan_R_w)
        )

    def get_new_P(self, cur_L: np.ndarray):
        """Computes the state covariance matrix P for the next time step without mutation. Equation 7.22"""
        mod_transition = self.A - cur_L @ self.C
        return (
            mod_transition @ self.P @ mod_transition.T
            + self.R_v @ self.R_v.T
            + cur_L @ self.R_w @ cur_L.T
        )

    def get_new_x(self, measurement: np.ndarray, L: np.ndarray):
        """Computes the state estimate x for the next time step without mutation. Equation 7.21"""

        if self.last_measurement is None:
            measurement_with_velocity = np.concatenate([measurement, np.zeros(2)])
        else:
            velocity = measurement - self.last_measurement[:2]
            measurement_with_velocity = np.concatenate([measurement, velocity])
        cand = self.A @ self.x + L @ (measurement_with_velocity - self.C @ self.x)
        if self.bounds is not None:
            cand = np.clip(cand, self.bounds[0], self.bounds[1])
        self.last_measurement = measurement_with_velocity
        return cand

    def smooth(self, tracks: np.ndarray, return_cov: bool = True):
        """Smoothes the input tracks using a Kalman filter. Returns a list of smoothed tracks.
        Args:
            tracks: An array of shape (n_frames, 2) containing a single node's x and y coordinates. May contain NaNs after the first entry
        """
        if tracks.shape[1] != 2 or len(tracks.shape) != 2:
            raise ValueError("Input tracks must have shape (n_frames, 2)")

        states = np.empty((tracks.shape[0], self.A.shape[0]))
        states[0, :2] = tracks[0]

        if return_cov:
            covs = np.empty((tracks.shape[0], 2, 2))

        self.x = states[0]

        # Order: Compute L, Compute X_hat_t+1, compute and update p_t+1
        for t, track in tqdm(enumerate(tracks[1:])):
            if np.isnan(track).any():
                cur_L = self.L_nan
                y = states[t, :2]
            else:
                cur_L = self.L
                y = track

            if return_cov:
                covs[t - 1] = self.P[:2, :2]
            cur_L = self.L
            self.x = self.get_new_x(y, cur_L)
            states[t + 1] = self.x
            self.P = self.get_new_P(cur_L)

        if return_cov:
            return states[:, :2], covs
        return states[:, :2]


def get_tracks_from_h5(h5_file: Path):
    """Reads tracks from an h5 file and returns them as an array"""
    with h5py.File(h5_file, "r") as ctx:
        tracks = ctx["tracks"][0, :, :, :]
        # shape is (coords, nodes, time)
        tracks = tracks.transpose(2, 1, 0)
        # shape is (time, nodes, coords)
        node_names = ctx["node_names"][:]

    return tracks, node_names


if __name__ == "__main__":
    sample_track_file = Path("2023_08_15_12_03_50_315836_cam_d.preds.analysis.h5")
    tracks = get_tracks_from_h5(sample_track_file)[:10000, :]
    bounds = np.array([[0, 0, -100, -100, -500, -500], [640, 512, 100, 100, 500, 500]])
    smoother = KalmanSmoother(bounds=bounds)
    smoothed_tracks = smoother.smooth(tracks)
    plt.plot(tracks[:, 0], tracks[:, 1], label="Original")
    plt.plot(smoothed_tracks[:, 0], smoothed_tracks[:, 1], label="Smoothed")
    plt.legend()
    plt.show()
