import json
from pathlib import Path

import numpy as np

with open("consts.json", "r") as f:
    consts = json.load(f)

working_dir = Path(consts["working_dir"])
corner_points = np.array(consts["corner_points"])
arena_dims_mm = np.array(consts["arena_dims_mm"])
video_dims = np.array(consts["video_dims"])


def point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    """Returns True if the point is inside the triangle defined by the corner points

    point: (2,) np.ndarray
    triangle: (3, 2) np.ndarray
    """
    v_0 = triangle[2] - triangle[0]
    v_1 = triangle[1] - triangle[0]
    v_2 = point - triangle[0]

    dot_00 = np.dot(v_0, v_0)
    dot_01 = np.dot(v_0, v_1)
    dot_02 = np.dot(v_0, v_2)
    dot_11 = np.dot(v_1, v_1)
    dot_12 = np.dot(v_1, v_2)

    inv_denom = 1 / (dot_00 * dot_11 - dot_01 * dot_01)
    u = (dot_11 * dot_02 - dot_01 * dot_12) * inv_denom
    v = (dot_00 * dot_12 - dot_01 * dot_02) * inv_denom
    return u >= 0 and v >= 0 and u + v < 1


def is_in_quadrilateral(point: np.ndarray) -> bool:
    """Returns True if the point is inside the quadrilateral defined by the corner points"""
    tri_indices = [(0, 1, 2), (0, 2, 3)]
    for indices in tri_indices:
        if point_in_triangle(point, corner_points[indices, :]):
            return True
    return False


def line_segment_intersection(
    a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """Computes the point of intersection between two line segments"""

    # Assume the lines extend infinitely
    # Ax = b
    A = np.stack([a2 - a1, b1 - b2], axis=1)
    b = b1 - a1
    try:
        x = np.linalg.solve(A, b)
    except:
        return None

    # Check if the intersection point is within the line segments
    if 0 <= x[0] <= 1 and 0 <= x[1] <= 1:
        return a1 + x[0] * (a2 - a1)
    return None


def contract_toward_center(point: np.ndarray) -> np.ndarray:
    """Takes a point outside the arena floor and moves it to the nearest quadrilateral edge"""
    screen_center = video_dims / 2
    edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edge_indices:
        intersection = line_segment_intersection(
            point, screen_center, corner_points[i], corner_points[j]
        )
        if intersection is not None:
            return intersection

    # If the point is not near any edge, return the point itself
    return point


def correct_rearing_points(rearing_points: np.ndarray) -> np.ndarray:
    result = []
    for point in rearing_points:
        point = point.copy()
        if not is_in_quadrilateral(point[0]):
            point[0] = contract_toward_center(point[0])
        if not is_in_quadrilateral(point[1]):
            point[1] = contract_toward_center(point[1])
        result.append(point)
    return np.array(result)
