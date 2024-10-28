import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from librosa import resample
from scipy.io import wavfile
from tqdm import tqdm

AUDIO_SR = 125000
# hardcoded pulse durations for each stimulus
pulse_durations = {
    "dfm-loud.wav": 10,
    "dfm-medium.wav": 15,
    "dfm-soft.wav": 20,
    "sc-loud.wav": 25,
    "sc-medium.wav": 30,
    "sc-soft.wav": 35,
    "stack-loud.wav": 40,
    "stack-medium.wav": 45,
    "stack-soft.wav": 50,
    "upfm-loud.wav": 55,
    "upfm-medium.wav": 60,
    "upfm-soft.wav": 65,
    "warble-loud.wav": 70,
    "warble-medium.wav": 75,
    "warble-soft.wav": 80,
    "white-loud.wav": 85,
    "white-medium.wav": 90,
    "white-soft.wav": 95,
}

# hardcoded stimulus lengths (in ms)
default_stimulus_durations = {
    "dfm-loud.wav": 73.78645833333333,
    "dfm-medium.wav": 73.78645833333333,
    "dfm-soft.wav": 73.78645833333333,
    "sc-loud.wav": 73.78645833333333,
    "sc-medium.wav": 73.78645833333333,
    "sc-soft.wav": 73.78645833333333,
    "stack-loud.wav": 73.78645833333333,
    "stack-medium.wav": 73.78645833333333,
    "stack-soft.wav": 73.78645833333333,
    "upfm-loud.wav": 73.78645833333333,
    "upfm-medium.wav": 73.78645833333333,
    "upfm-soft.wav": 73.78645833333333,
    "warble-loud.wav": 73.78645833333333,
    "warble-medium.wav": 73.78645833333333,
    "warble-soft.wav": 73.78645833333333,
    "white-loud.wav": 73.78645833333333,
    "white-medium.wav": 73.78645833333333,
    "white-soft.wav": 73.78645833333333,
}

stimulus_durations = {}
stimulus_dir = Path("/mnt/home/atanelus/ceph/ssl_stimuli")
if stimulus_dir.exists():
    for fn in default_stimulus_durations.keys():
        fs, audio = wavfile.read(stimulus_dir / fn)
        if fs != AUDIO_SR:
            audio = resample(audio, orig_sr=fs, target_sr=AUDIO_SR)
        thresh = 0.0002
        crossings = np.flatnonzero(np.abs(audio) > thresh)
        audio = audio[crossings[0] : crossings[-1]]
        stimulus_durations[fn] = len(audio) / AUDIO_SR * 1000
else:
    stimulus_durations = default_stimulus_durations


pulse_lookup = {pulse: stimulus_durations[fn] for fn, pulse in pulse_durations.items()}


def get_audio(fname, indices):
    concat_audio = []
    with h5py.File(fname, "r") as ctx:
        keys = sorted(list(ctx["ai_channels"].keys()))
        for start, end in indices:
            audio = np.stack(
                [ctx[f"ai_channels/{key}"][start:end] for key in keys], axis=1
            )
            concat_audio.append(audio)
    lengths = [end - start for start, end in indices]
    return np.concatenate(concat_audio, axis=0), lengths


def get_onsets_offsets(fname):
    with h5py.File(fname, "r") as ctx:
        audio_onsets = ctx["audio_onset"][:, 0]
        audio_identity = ctx["audio_onset"][:, 1]
    # mult by 1.1 to allow for some travel time
    audio_offsets = np.array(
        [
            o + pulse_lookup.get(p, 70) * AUDIO_SR / 1000 + 0.005 * AUDIO_SR
            for o, p in zip(audio_onsets, audio_identity)
        ]
    ).astype(int)

    return np.stack([audio_onsets, audio_offsets], axis=1), audio_identity


def audio_path_from_video(vid_path):
    parent_path = Path(vid_path).parent
    return glob.glob(str(parent_path / "mic*.h5"))[0]


def make_dataset(tmp_dir: Path):
    vid_csv = pd.read_csv("video_positions.csv")

    # lazy so I'm keeping everything in RAM
    audio_blocks = []
    audio_lengths = []
    location_blocks = []
    identity_blocks = []

    for n, (vid_path, x, y) in tqdm(
        enumerate(
            zip(vid_csv["vid_paths"], vid_csv["speaker_x_px"], vid_csv["speaker_y_px"])
        ),
        total=len(vid_csv),
    ):
        if (tmp_dir / f"audio_{n}.h5").exists():
            with h5py.File(tmp_dir / f"audio_{n}.h5", "r") as ctx:
                audio = ctx["audio"][:]
                lengths_part = np.diff(ctx["length_idx"][:])
                identity = ctx["stimulus_identities"][:]
                location = ctx["locations_px"][:]
        else:
            location = np.array((x, y))[None, :]
            audio_path = audio_path_from_video(vid_path)
            onsets, identity = get_onsets_offsets(audio_path)
            audio, lengths_part = get_audio(audio_path, onsets)
            with h5py.File(tmp_dir / f"audio_{n}.h5", "w") as ctx:
                ctx["audio"] = audio
                ctx["length_idx"] = np.cumsum(np.insert(lengths_part, 0, 0))
                ctx["locations_px"] = location
                ctx["stimulus_identities"] = identity

        # write tmp file

        identity_blocks.append(identity)
        audio_blocks.append(audio)
        audio_lengths.extend(lengths_part)
        repeated_locations = np.repeat(location, len(lengths_part), axis=0)
        location_blocks.append(repeated_locations)
    concat_audio = np.concatenate(audio_blocks, axis=0)
    concat_lengths = np.cumsum(np.insert(audio_lengths, 0, 0))
    concat_locations = np.concatenate(location_blocks, axis=0)
    identity = np.concatenate(identity_blocks, axis=0)

    # rescale identities
    identity = (identity - 10) // 5
    identity = identity.astype(int)
    stimulus_names = list(pulse_durations.items())
    stimulus_names = list(
        map(lambda x: x[0].split(".")[0], sorted(stimulus_names, key=lambda x: x[1]))
    )

    print("audio shape: ", concat_audio.shape)
    print("length index shape: ", concat_lengths.shape)
    print("locations shape: ", concat_locations.shape)
    print("identity shape: ", identity.shape)
    dset_path = tmp_dir / "speaker_dataset_full.h5"
    dset_path.parent.mkdir(exist_ok=True)
    with h5py.File(dset_path, "w") as ctx:
        ctx["audio"] = concat_audio
        ctx["length_idx"] = concat_lengths
        ctx["locations_px"] = concat_locations
        ctx["stimulus_identities"] = identity
        ctx["stimulus_names"] = np.array(
            list(map(lambda x: x.encode(), stimulus_names))
        )


if __name__ == "__main__":
    # tmp_dir = Path("/home/atanelus/speaker_dataset_tmp")
    tmp_dir = Path("/tmp/speaker_dataset_tmp")
    tmp_dir.mkdir(exist_ok=True)
    make_dataset(tmp_dir)
