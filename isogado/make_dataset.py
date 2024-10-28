import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

AUDIO_SR = 125000.0
VIDEO_SR = 30.0
annotation_paths = sorted(glob.glob("filtered_annotations/*.npy"))
track_paths = glob.glob("adolescent_tracks/*.npy")
video_paths = glob.glob("/mnt/ceph/users/rpeterson/ssl/adolescents/*/*.avi")

# h5transfer only contains dates 6/20 and 6/16
audio_paths = sorted(
    # glob.glob("/mnt/home/atanelus/ceph/h5transfer/adolescents/*/*.h5"), key=path_date
    glob.glob("/mnt/ceph/users/rpeterson/ssl/adolescents/*/*.h5"),
)


def process_annotation(annotation_path):
    date = "_".join(Path(annotation_path).stem.split("_")[1:7])
    video_path = list(filter(lambda x: date in Path(x).stem, video_paths))
    if not video_path:
        return None
    else:
        video_path = video_path[0]

    segments = np.load(annotation_path)
    frame_idx = (segments.astype(int).mean(axis=1) / AUDIO_SR * VIDEO_SR).astype(int)

    return [video_path] * len(frame_idx), segments, frame_idx


def get_track_file_path(annotation_path):
    date = "_".join(Path(annotation_path).stem.split("_")[1:7])
    track_path = list(filter(lambda x: date in Path(x).stem, track_paths))
    if not track_path:
        return None
    return track_path[0]


def get_audio_file_path(annotation_path):
    date = "_".join(Path(annotation_path).stem.split("_")[1:7])
    audio_path = list(filter(lambda x: date in Path(x).stem, audio_paths))
    if not audio_path:
        return None
    return audio_path[0]


def get_audio(annotation_path, segments):
    audio_path = get_audio_file_path(annotation_path)
    if audio_path is None:
        return None
    with h5py.File(audio_path, "r") as ctx:
        file_length = ctx["ai_channels/ai0"].shape[0]
        num_channels = len(ctx["ai_channels"].keys())

        for start, stop in segments:
            stop = min(stop, file_length)
            if (stop - start < AUDIO_SR * 0.020) or (stop - start > AUDIO_SR * 0.250):
                yield None
                continue

            audio = np.stack(
                [ctx["ai_channels"][f"ai{i}"][start:stop] for i in range(num_channels)],
                axis=1,
            )
            yield audio


def get_tracks(annotation_path, frame_idx):
    track_path = get_track_file_path(annotation_path)
    if track_path is None:
        return None
    tracks = np.load(track_path)
    return tracks[frame_idx]


def run(output_path: Path):
    valid_annotation_paths = []
    videos = []
    frames = []
    locations = []
    unfiltered_audio_segment_batches = []

    for annotation_path in annotation_paths:
        # Holds video paths, audio segments (units: audio idx), and video frames (units: video idx)
        # Holds None if the corresponding video is not found
        result = process_annotation(annotation_path)
        if result is None:
            print(f"Skipping {annotation_path} because video not found")
            continue
        video_paths, audio_segments, video_frame_indices = result
        # Holds ndarray of animal locations (already converted to mm from pixels)
        # Holds None if the corresponding track file is not found
        processed_tracks = get_tracks(annotation_path, video_frame_indices)
        if processed_tracks is None:
            print(f"Skipping {annotation_path} because track file not found")
            continue

        if (
            len(video_paths) != len(video_frame_indices)
            or len(video_paths) != len(processed_tracks)
            or len(video_frame_indices) != len(audio_segments)
        ):
            raise ValueError(
                "Number of video paths, audio segments, and processed tracks do not match"
            )

        valid_annotation_paths.append(annotation_path)
        videos.extend(video_paths)
        frames.extend(video_frame_indices)
        locations.extend(processed_tracks)
        unfiltered_audio_segment_batches.append(audio_segments)

    print(
        f"Num unfiltered segments: {sum(len(arr) for arr in unfiltered_audio_segment_batches)}"
    )
    frames = np.array(frames)
    locations = np.array(locations)[:, 0, :]

    # for future video generation, save a df linking stimulus idx to video path and frame idx
    metadata = pd.DataFrame(
        {
            "video_path": videos,
            "frame_idx": frames,
            "x_mm": locations[:, 0],
            "y_mm": locations[:, 1],
        }
    )
    # Not writing this yet because some rows will be dropped after seeing the audio

    # start processing audio
    with h5py.File(output_path, "w") as ctx:
        # sum of offset - onset for all segments
        expected_length = sum(
            np.diff(arr, axis=1).sum() for arr in unfiltered_audio_segment_batches
        )
        vox_dset = ctx.create_dataset(
            "audio",
            shape=(expected_length, 4),
            chunks=(128, 4),
            dtype=np.float32,
        )
        n_samples_added = 0
        lengths_added = []

        # Make an iterator for tqdm to be usable
        def audio_iterator():
            for annotation_path, segments_for_file in zip(
                valid_annotation_paths, unfiltered_audio_segment_batches
            ):
                for audio in get_audio(annotation_path, segments_for_file):
                    yield audio

        indices_dropped = set()
        indices_dropped.update(np.flatnonzero(np.isnan(locations).any(axis=1)))
        print(f"Dropping {len(indices_dropped)} rows with NaN locations")
        for cur_idx, audio in tqdm(enumerate(audio_iterator()), total=len(locations)):
            if cur_idx in indices_dropped:
                continue
            if audio is None:
                indices_dropped.add(cur_idx)
                continue
            lengths_added.append(len(audio))
            vox_dset[n_samples_added : n_samples_added + len(audio), :] = audio
            n_samples_added += len(audio)

        print("dropped", len(indices_dropped))
        vox_dset.resize(n_samples_added, axis=0)

        # Filter out unused rows before writing
        indices_dropped = sorted(list(indices_dropped))
        metadata = metadata.drop(indices_dropped)
        locations = np.delete(locations, indices_dropped, axis=0)

        ctx["locations"] = locations
        len_idx = np.cumsum([0] + lengths_added)
        ctx["length_idx"] = len_idx

        print(
            f"Total audio length: {n_samples_added/AUDIO_SR:.0f}s / {expected_length/AUDIO_SR:.0f}s expected"
        )
        print(f"Num vocalizations: {len(lengths_added)}")
        print(f"Mean vocalization length: {np.mean(lengths_added) / 125:.1f}ms")
        print(
            f"Median vocalization length: {np.quantile(lengths_added, 0.5)/125:.1f}ms"
        )
        print(f"Max vocalization length: {np.max(lengths_added) / 125:.1f}ms")
        print(f"Min vocalization length: {np.min(lengths_added) / 125:.1f}ms")
        print(f"Locations shape: {locations.shape}")
        print(f"Len idx shape: {len_idx.shape}")
    metadata.to_csv("adolescent_metadata.csv")


if __name__ == "__main__":
    run("/home/atanelus/adolescent_dataset.h5")
