""" Makes an inference dataset for the isolated mother dataset, collection date: 2022-03-30
"""

from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

AUDIO_SR = 125000
data_dir = Path("data")
video_path = data_dir / "cam_b.avi"
segments_path = data_dir / "mic_2022_03_30_15_51_37_777910_segments.npy"
dataset_path = data_dir / "mother_inference_dataset.h5"


def make_video_frames_csv():
    """Makes a csv file with the frame numbers of the video corresponding to
    the onset times of the vocalizations
    """
    global video_path, segments_path, data_dir

    video = cv2.VideoCapture(str(video_path))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    framerate = video.get(cv2.CAP_PROP_FPS)
    video.release()

    segments = np.load(segments_path)
    mean_segment = np.mean(segments, axis=1)  # Currently given in samples
    # Convert to video frames
    mean_segment = mean_segment / AUDIO_SR * framerate
    # Convert to int and clip to bounds
    mean_segment = np.clip(mean_segment.astype(int), 0, n_frames - 1)

    # Make csv file
    cluster_path = (
        Path("/mnt/home/atanelus/ceph/speaker_ssl/isolate_mother/data/")
        / video_path.name
    )
    video_path = [str(cluster_path)] * len(mean_segment)
    frame_idx = mean_segment

    df = pd.DataFrame({"video_path": video_path, "frame_idx": frame_idx})
    df.to_csv(data_dir / "mother_inference_frames.csv", index=False)

    corner_points_hardcoded = np.array(
        [
            [85, 103],  # top left
            [577, 92],  # top right
            [586, 410],  # bottom right
            [92, 422],  # bottom left
        ]
    )
    np.save(data_dir / "mother_inference_corner_points.npy", corner_points_hardcoded)


if __name__ == "__main__":
    make_video_frames_csv()
