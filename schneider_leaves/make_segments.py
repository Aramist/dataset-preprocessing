import json
from pathlib import Path

import numpy as np
import soundfile as sf
from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)
data_dir = Path(consts["recording_session_dir"])
working_dir = Path(consts["working_dir"])
processed_track_dir = working_dir / consts["processed_track_dir"]
unprocessed_annotation_dir = working_dir / consts["unprocessed_annotation_dir"]

recording_sessions = list(filter(lambda p: p.is_dir(), data_dir.glob("*/*")))
recording_sessions.sort()
audio_sample_rate = consts["audio_sr"]

seg_params = {
    "min_freq": 10,  # minimum frequency
    "max_freq": audio_sample_rate / 2,  # maximum frequency
    "nperseg": 512,  # FFT
    "noverlap": 256,  # FFT
    "spec_min_val": -3,  # minimum log-spectrogram value
    "spec_max_val": 1,  # maximum log-spectrogram value
    "fs": audio_sample_rate,  # audio samplerate
    "th_1": 5,  # segmenting threshold 1
    "th_2": 5,  # segmenting threshold 2
    "th_3": 10,  # segmenting threshold 3
    "min_dur": 0.01,  # minimum syllable duration
    "max_dur": 10.0,  # maximum syllable duration
    "smoothing_timescale": 0.007,  # amplitude
    "softmax": False,  # apply softmax to the frequency bins to calculate
    # amplitude
    "temperature": 0.5,  # softmax temperature parameter
    "algorithm": get_onsets_offsets,  # (defined above)
}


def get_segments_for_session(session_dir: Path):
    # get onsets, offsets, and amplitude trace
    audio_path = next(session_dir.glob("*_audiorec.flac"), None)
    if not audio_path:
        raise FileNotFoundError(f"No audio file found in {session_dir}")

    audio, _ = sf.read(audio_path)
    audio = audio.mean(axis=-1)
    audio = (audio - audio.mean()) / audio.std()

    on, off = get_onsets_offsets(audio, seg_params, return_traces=False)

    on = np.array(on)
    off = np.array(off)
    segments = np.stack([on, off], axis=-1)
    if len(segments) == 0:
        raise Exception(f"No segments found in {session_dir}")
    return segments


def main():
    unprocessed_annotation_dir.mkdir(exist_ok=True, parents=True)
    for session_dir in tqdm(recording_sessions):
        segments = get_segments_for_session(session_dir)
        session_name = session_dir.parent.name
        np.save(unprocessed_annotation_dir / f"{session_name}_segments.npy", segments)


if __name__ == "__main__":
    main()
