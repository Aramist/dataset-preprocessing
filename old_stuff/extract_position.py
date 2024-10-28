import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_position_in_frame(frame):
    if len(frame.shape) != 3 or frame.shape[-1] != 3:
        raise ValueError('Expects frame to be a 3-channel image')
    # assume frame is BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh_bin = (hsv[..., 2] < 70) & (hsv[..., 1] < 70)  # Search for black pixels. The speaker is black
    thresh = np.full(frame.shape[:2], 255, dtype=np.uint8)
    thresh[~thresh_bin] = 0

    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT,
        ksize=(9, 9),  # should be large enough to fill in any holes inside the speaker
    )
    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )  # Dilate holes away
    thresh = cv2.erode(
        thresh,
        kernel,
        iterations=2
    )  # Undo dilation on edges


    # get largest contour
    contours, _ = cv2.findContours(
        thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    # expand to list to index into it later
    contours = list(filter(lambda c: cv2.contourArea(c) > 3000.0, contours))

    # find the most circle-like contour.
    # circleness is determined by the ratio of the best-fitting circle's area and contour area
    circleness = map(
        lambda c: cv2.contourArea(c) / (cv2.minEnclosingCircle(c)[1]**2 * np.pi),
        contours
    )

    contour = contours[np.argmax(list(circleness))]

    moments = cv2.moments(contour)
    centroid_x = moments['m10'] / moments['m00']
    centroid_y = moments['m01'] / moments['m00']
    return np.array([centroid_x, centroid_y])



def position_for_video(vid_path) -> np.ndarray:
    reader = cv2.VideoCapture(vid_path)
    ret, frame = reader.read()
    if not ret:
        raise ValueError(f"Could not read provided video {vid_path}")
    # positions = []
    num_frames_total = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    reader.set(cv2.CAP_PROP_POS_FRAMES, num_frames_total // 2)
    ret, frame = reader.read()
    '''
    while ret:
        positions.append(get_position_in_frame(frame))
        ret, frame = reader.read()
    return np.quantile(np.stack(positions), 0.5, axis=0)
    '''
    pos = get_position_in_frame(frame)
    return pos


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('video_paths', nargs='+', type=str)
    video_paths = ap.parse_args().video_paths
    positions = []

    '''
    writer = cv2.VideoWriter(
        'speaker_pos.mp4',
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,  # This is the framerate of the video
        (640, 512),
        isColor=True
    )
    '''

    for vid_path in video_paths:
        # speaker_pos, frame = position_for_video(vid_path)
        speaker_pos = position_for_video(vid_path)
        positions.append(speaker_pos)
        # writer.write(frame)
    positions = np.stack(positions).astype(int)
    # writer.release()
    df = pd.DataFrame({
        'vid_paths': video_paths,
        'speaker_x_px': list(positions[:, 0]),
        'speaker_y_px': list(positions[:, 1])
    })

    df.to_csv('video_positions.csv')
