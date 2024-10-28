import json
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from tqdm import tqdm

from schneider_leaves.demo_make_tracks import load_tracks

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)
data_dir = Path(consts["recording_session_dir"])
working_dir = Path(consts["working_dir"])
recording_sessions = list(filter(lambda p: p.is_dir(), data_dir.glob("*/*")))
recording_sessions.sort()
audio_sample_rate = consts["audio_sr"]


def get_snippet(session_dir: Path):
    """Grabs a small segment of video and audio from one session

    Args:
        session_dir (Path): Path to session directory. Expected to contain a video
        file (*_CamFlir1_*.avi), an audio file (_audiorec.flac), and tracks (*.analysis.h5)
    """

    # Ensure all files are present
    video_file = next(session_dir.glob("*_CamFlir1_*.avi"), None)
    audio_file = next(session_dir.glob("*_audiorec.flac"), None)
    track_file = next(session_dir.glob("*.analysis.h5"), None)
    if not all([video_file, audio_file, track_file]):
        return None

    # Load tracks
    tracks = load_tracks(track_file)
    # Estimate velocity
    velocity_est = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
    smoothed_velocity = np.convolve(velocity_est, np.ones(10) / 10, mode="same")
    # save smoothed and raw velocity

    a_fast_frame = (-smoothed_velocity).argsort()[20]  # Avoid potential outliers
    # Will grab 10s before and 10s after this frame

    cap = cv2.VideoCapture(str(video_file))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    midpoint_frame_idx = a_fast_frame
    start_frame = int(max(0, midpoint_frame_idx - 10 * frame_rate))
    end_frame = int(min(frame_count, midpoint_frame_idx + 10 * frame_rate))
    track_subset = tracks[start_frame:end_frame, :]

    # Get frames in range
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # draw the track on the frame
        frame = cv2.circle(
            frame,
            tuple(track_subset[i].astype(int)),
            5,
            (0, 255, 0),
            -1,
        )
        # Annotate with the session name
        session_name = session_dir.parent.name.split("_")[-1]  # mouse id
        frame = cv2.putText(
            frame,
            session_name,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)
    cap.release()

    # Get the corresponding audio snippet
    audio, _ = sf.read(audio_file)
    audio_midpoint_frame_idx = midpoint_frame_idx * audio_sample_rate / frame_rate
    audio_start = int(max(0, audio_midpoint_frame_idx - 10 * audio_sample_rate))
    audio_end = int(min(len(audio), audio_midpoint_frame_idx + 10 * audio_sample_rate))
    audio_snippet = audio[audio_start:audio_end]

    return frames, audio_snippet


def concat_video_and_audio(
    output_video_path: Path,
    output_audio_path: Path,
    frames: list[np.ndarray],
    audio_snippets: list[np.ndarray],
):
    """Concatenates frames and audio snippets into a single video and audio file

    Args:
        output_video_path (Path): Path to output video file
        output_audio_path (Path): Path to output audio file
        frames (list[np.ndarray]): List of frames
        audio_snippets (list[np.ndarray]): List of audio snippets
    """

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_video_path), fourcc, 30, (frames[0].shape[1], frames[0].shape[0])
    )
    for frame in tqdm(frames):
        out.write(frame)
    out.release()

    # Write audio
    audio = np.concatenate(audio_snippets)
    sf.write(output_audio_path, audio, audio_sample_rate)


if __name__ == "__main__":
    all_frames, all_audio = [], []
    for session_dir in tqdm(recording_sessions):
        frames, audio = get_snippet(session_dir)
        all_frames.extend(frames)
        all_audio.append(audio)

    output_video_path = working_dir / "video_samples.mp4"
    output_audio_path = working_dir / "audio_samples.wav"

    concat_video_and_audio(output_video_path, output_audio_path, all_frames, all_audio)
