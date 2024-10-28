import argparse

import cv2
import h5py
import numpy as np


def get_homography():
    # pixel coordinates of quad corners
    tl_px = (72, 78)
    tr_px = (585, 70)
    br_px = (587, 396)
    bl_px = (91, 402)
    source_points = np.stack([tl_px, tr_px, br_px, bl_px]).astype(float)
    # halved arena dimensions
    a_hwidth, a_hheight = np.array([558.9, 355.6]) / 2
    """  These are the original locations, but they got rotated 180˚ for this experiment
    dest_points = np.array([
        [-a_hwidth, a_hheight],
        [a_hwidth, a_hheight],
        [a_hwidth, -a_hheight],
        [-a_hwidth, -a_hheight],
    ])
    """
    dest_points = -np.array([
        [-a_hwidth, a_hheight],
        [a_hwidth, a_hheight],
        [a_hwidth, -a_hheight],
        [-a_hwidth, -a_hheight],
    ])

    H, _ = cv2.findHomography(source_points, dest_points, method=cv2.RANSAC)
    return H


def convert_points(points, H):
    # Given a stream and a point (in pixels), converts to inches within the global coordinate frame
    # Pixel coordinates should be presented in (x, y) order, with the origin in the top-left corner of the frame
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    # System is M * [x_px y_px 1] ~ [x_r y_r 1]
    ones = np.ones((*points.shape[:-1], 1))
    points = np.concatenate([points, ones], axis=-1)
    prod = np.einsum('ij,...j->...i', H, points)[..., :-1]  # remove ones row
    return prod


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_path', type=str, help='Path to dataset containing pixel coordinates of speaker location')
    dset_path = ap.parse_args().dataset_path
    
    H = get_homography()
    
    with h5py.File(dset_path, 'r+') as ctx:
        source_points = ctx['locations_px'][:]
        dest_points = convert_points(source_points, H)
        if 'locations' in ctx:
            del ctx['locations']
        ctx['locations'] = dest_points

