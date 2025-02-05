from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.signal import stft


def get_vocalization(dset, idx):
    start, end = dset["length_idx"][idx : idx + 2]
    audio = dset["audio"][start:end]
    return audio


def make_spectrogram_image(
    signal: np.ndarray,
    size_per_spectrogram=(400, 200),
    sample_rate=125000,
) -> np.ndarray:
    f, t, dft = stft(signal, sample_rate, nfft=256, axis=0)
    # dft initially has shape (frequencies, channels, time)
    # Change to (channels, frequencies, time)
    dft = np.einsum("fct->cft", dft)
    dft = dft[:, 1:, :]  # remove 0-frequency component
    dft = np.log(np.abs(dft) + 1e-12)
    # rescale to [0,255]

    dmin, dmax = np.quantile(dft, (0.05, 0.95))
    scaled_dft = (dft - dmin) * 255 / (dmax - dmin)
    scaled_dft = np.clip(scaled_dft, 0, 255)
    # flip frequencies so low freqs are at the bottom of the image
    scaled_dft = scaled_dft[:, ::-1, :]
    scaled_dft = scaled_dft.astype(np.uint8)
    # Temporarily concat images to apply colormap
    n_chan, n_freq, n_time = scaled_dft.shape
    concat_dft = scaled_dft.reshape(n_chan * n_freq, n_time)  # vertical stacking
    colored_dft = cv2.applyColorMap(concat_dft, cv2.COLORMAP_MAGMA).reshape(
        n_chan, n_freq, n_time, 3
    )

    # Add an annotation with the channel index to each spectrogram
    for i in range(n_chan):
        colored_dft[i] = cv2.putText(
            colored_dft[i],
            f"Ch: {i}",
            (10, 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
        )

    # Now stack the images into a grid
    num_grid_rows = int(np.ceil(n_chan / 2))
    full_im = np.zeros(
        (num_grid_rows * size_per_spectrogram[1], 2 * size_per_spectrogram[0], 3),
        dtype=np.uint8,
    )
    for i in range(n_chan):
        row = i // 2
        col = i % 2
        full_im[
            row * size_per_spectrogram[1] : (row + 1) * size_per_spectrogram[1],
            col * size_per_spectrogram[0] : (col + 1) * size_per_spectrogram[0],
        ] = cv2.resize(colored_dft[i], size_per_spectrogram)

    return full_im


def make_and_show_im(ctx, idx):
    vox = get_vocalization(ctx, idx)
    spec_image = make_spectrogram_image(vox)
    spec_image = cv2.putText(
        spec_image,
        f"Idx: {idx}",
        (10, 100),
        cv2.FONT_HERSHEY_PLAIN,
        5,
        (0, 0, 0),
        5,
    )
    cv2.imshow("spectrogram", spec_image)
    cv2.waitKey(1)


def visualize_dataset(dset_path, index=None):
    idx = 0
    cv2.namedWindow("spectrogram", cv2.WINDOW_NORMAL)

    with h5py.File(dset_path, "r") as ctx:
        if index is None:
            index = np.arange(len(ctx["length_idx"]) - 1)

        make_and_show_im(ctx, index[idx])

        while idx < len(index) and (key := cv2.waitKey(0)) != ord("q"):
            if key == ord("l"):
                idx += 1
            elif key == ord("h"):
                idx -= 1
            else:
                continue
            make_and_show_im(ctx, index[idx])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dset_path", type=str)
    parser.add_argument("--index", type=Path, required=False)
    args = parser.parse_args()
    index = None
    if args.index is not None:
        index = np.load(args.index)
    visualize_dataset(args.dset_path, index)
