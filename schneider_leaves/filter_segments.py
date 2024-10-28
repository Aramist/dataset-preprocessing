import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

with open("consts.json", "r") as ctx:
    consts = json.load(ctx)
data_dir = Path(consts["recording_session_dir"])
working_dir = Path(consts["working_dir"])
unprocessed_annotation_dir = working_dir / consts["unprocessed_annotation_dir"]
raw_segments = list(unprocessed_annotation_dir.glob("*.npy"))
raw_segments.sort()

processed_annotation_dir = working_dir / consts["processed_annotation_dir"]

audio_sample_rate = consts["audio_sr"]
min_len_ms = consts["min_instance_len_ms"]
max_len_ms = consts["max_instance_len_ms"]
min_len_samples = int(min_len_ms * audio_sample_rate / 1000)
max_len_samples = int(max_len_ms * audio_sample_rate / 1000)


def split_segments(orig_segments: np.ndarray):
    """Splits long segments into smaller ones and removes too-short segments."""
    # originally stored in seconds
    segments_samps = (orig_segments * audio_sample_rate).astype(int)
    segment_lengths = segments_samps[:, 1] - segments_samps[:, 0]
    long_segments = segments_samps[segment_lengths > max_len_samples]
    orig_segments = segments_samps[segment_lengths <= max_len_samples]

    chopped_segments = []
    for seg in long_segments:
        start, end = seg
        while end - start > max_len_samples:
            new_end = start + max_len_samples
            chopped_segments.append([start, new_end])
            start = new_end
    chopped_segments = np.array(chopped_segments)
    segments_samps = np.concatenate([orig_segments, chopped_segments])
    # somehow, some of the onsets are duplicated
    unique_segments_idx = np.unique(segments_samps[:, 0], return_index=True)[1]
    segments_samps = segments_samps[unique_segments_idx]
    segments_samps = np.sort(segments_samps, axis=0)
    return segments_samps


def main():
    processed_annotation_dir.mkdir(exist_ok=True)
    total_num_instances = 0
    for seg_file in tqdm(raw_segments):
        segments = np.load(seg_file)
        segments = split_segments(segments)
        total_num_instances += len(segments)
        np.save(processed_annotation_dir / seg_file.name, segments)
    print(f"Total number of instances: {total_num_instances}")


if __name__ == "__main__":
    main()
