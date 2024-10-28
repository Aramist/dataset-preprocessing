import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


AUDIO_SR = 125000
# hardcoded pulse durations for each stimulus
pulse_durations = {
    'dfm-loud.wav': 10,
    'dfm-medium.wav': 15,
    'dfm-soft.wav': 20,
    'sc-loud.wav': 25,
    'sc-medium.wav': 30,
    'sc-soft.wav': 35,
    'stack-loud.wav': 40,
    'stack-medium.wav': 45,
    'stack-soft.wav': 50,
    'upfm-loud.wav': 55,
    'upfm-medium.wav': 60,
    'upfm-soft.wav': 65,
    'warble-loud.wav': 70,
    'warble-medium.wav': 75,
    'warble-soft.wav': 80,
    'white-loud.wav': 85,
    'white-medium.wav': 90,
    'white-soft.wav': 95
}

# hardcoded stimulus lengths (in ms)
stimulus_durations = {
    'dfm-loud.wav': 73.78645833333333,
    'dfm-medium.wav': 73.78645833333333,
    'dfm-soft.wav': 73.78645833333333,
    'sc-loud.wav': 73.78645833333333,
    'sc-medium.wav': 73.78645833333333,
    'sc-soft.wav': 73.78645833333333,
    'stack-loud.wav': 73.78645833333333,
    'stack-medium.wav': 73.78645833333333,
    'stack-soft.wav': 73.78645833333333,
    'upfm-loud.wav': 73.78645833333333,
    'upfm-medium.wav': 73.78645833333333,
    'upfm-soft.wav': 73.78645833333333,
    'warble-loud.wav': 73.78645833333333,
    'warble-medium.wav': 73.78645833333333,
    'warble-soft.wav': 73.78645833333333,
    'white-loud.wav': 73.78645833333333,
    'white-medium.wav': 73.78645833333333,
    'white-soft.wav': 73.78645833333333
}
pulse_lookup = {pulse: stimulus_durations[fn]
                for fn, pulse in pulse_durations.items()}


def get_audio(fname, indices):
    concat_audio = []
    with h5py.File(fname, 'r') as ctx:
        keys = sorted(list(ctx['ai_channels'].keys()))
        for start, end in indices:
            audio = np.stack(
                [ctx[f'ai_channels/{key}'][start:end] for key in keys], axis=1)
            concat_audio.append(audio)
    lengths = [end - start for start, end in indices]
    return np.concatenate(concat_audio, axis=0), lengths


def get_onsets_offsets(fname):
    with h5py.File(fname, 'r') as ctx:
        audio_onsets = ctx['audio_onset'][:, 0]
        audio_identity = ctx['audio_onset'][:, 1]
    # mult by 1.1 to allow for some travel time
    audio_offsets = np.array([
        o + pulse_lookup.get(p, 10) * 1.1 * AUDIO_SR / 1000
        for o, p in zip(audio_onsets, audio_identity)
    ]).astype(int)

    return np.stack([audio_onsets, audio_offsets], axis=1), audio_identity

def audio_path_from_video(vid_path):
    parent_path = Path(vid_path).parent
    print(parent_path)
    return glob.glob(str(parent_path / 'mic*.h5'))[0]

if __name__ == '__main__':
    vid_csv = pd.read_csv('video_positions.csv')

    # lazy so I'm keeping everything in RAM
    audio_blocks = []
    lengths = [0]
    location_blocks = []
    identity_blocks = []

    for n, (vid_path, x, y) in enumerate(zip(vid_csv['vid_paths'], vid_csv['speaker_x_px'], vid_csv['speaker_y_px'])):
        # location = np.array((x, y))[None, :]
        audio_path = audio_path_from_video(vid_path)
        onsets, identity = get_onsets_offsets(audio_path)
        identity_blocks.append(identity)
        # audio, lengths_part = get_audio(audio_path, onsets)
        # locations = np.tile(location, (len(lengths_part), 1))

        # audio_blocks.append(audio)
        # lengths.extend(lengths_part)
        # location_blocks.append(locations)
    '''
    concat_audio = np.concatenate(audio_blocks, axis=0)
    concat_lengths = np.cumsum(lengths)
    concat_locations = np.concatenate(location_blocks, axis=0)

    print(concat_audio.shape)
    print(concat_lengths.shape)
    print(concat_locations.shape)
    print(concat_lengths)
    '''

    identity = np.concatenate(identity_blocks, axis=0)
    #with h5py.File('speaker_dataset.h5', 'w') as ctx:
        # ctx['vocalizations'] = concat_audio
        # ctx['len_idx'] = concat_lengths
        # ctx['locations_px'] = concat_locations
    np.save('identity.npy', identity)
    
