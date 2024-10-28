import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

id_to_idx = {15 + 5 * i : i for i in range(18)}
stim_waveforms = 

def get_specgram(ctx, idx):
    id = ctx['identities'][idx]

def run():
    audio_path = '/mnt/home/atanelus/ceph/speaker_dataset.h5'
    with h5py.File(audio_path, 'r') as ctx:
        vid_csv = pd.read_csv('video_positions.csv')
        for n, (frame_path, x, y) in enumerate(zip(vid_csv['frame_paths'], vid_csv['speaker_x_px'], vid_csv['speaker_y_px'])):
            matching_locs = 

if __name__ == '__main__':
    run()
