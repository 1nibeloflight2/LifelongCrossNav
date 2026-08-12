"""
Planning utilities
"""
# numpy
import time

import numpy as np

# queue
from queue import PriorityQueue
from heapq import *

from dataclasses import dataclass, field
from typing import Any, List
import matplotlib.pyplot as plt
import random

# numba
from numba import jit, njit

# CPP planning utils
from planning_utils_cpp import dijkstra, compute_reachable_area, a_star_range

__all__ = ['compute_reachable_area_score', 'compute_best_path', 'generate_large_map']


def compute_reachable_area_score(start: np.ndarray,
                                 mask_coverage: np.ndarray,
                                 scores: np.ndarray,
                                 max_depth: int
                                 ) -> (float, int, np.ndarray):
    """
    Compute the score and number of visited nodes for a given start position (optimized version)
    :param start: start position in pixel coordinates
    :param mask_coverage: binary mask indicating which points are reachable
    :param scores: scoring map
    :param max_depth: maximum depth to explore
    :return: score, number of visited nodes, reachable area
    """

    return compute_reachable_area(start, mask_coverage, scores, max_depth)


def compute_to_goal(start: np.ndarray, mask_coverage: np.ndarray, feasible_goal_pts: np.ndarray, goal_pt: np.ndarray,
                    obstcl_kernel_size, min_goal_dist):
    best_path = a_star_range(mask_coverage, feasible_goal_pts, tuple(start), (goal_pt[0], goal_pt[1]),
                             obstcl_kernel_size, min_goal_dist)
    if best_path is None or len(best_path) == 0:
        best_path = a_star_range(mask_coverage, feasible_goal_pts, tuple(start), (goal_pt[0], goal_pt[1]),
                                 obstcl_kernel_size, min_goal_dist * 2)
    if best_path and len(best_path):
        for i in range(len(best_path)):
            best_path[i] = np.array(best_path[i])
        return best_path
    return None


def compute_best_path(start: np.ndarray, mask_coverage: np.ndarray, scores: np.ndarray, kernel_width: int):
    goal_pts = np.where(scores > 0)
    distance_and_paths = dijkstra(mask_coverage, tuple(start), [(a, b) for (a, b) in zip(goal_pts[0], goal_pts[1])],
                                  kernel_width)
    best_score = 0.0
    best_path = None
    for i, d_p in enumerate(distance_and_paths):
        d, path = d_p
        if len(path):
            pt = (goal_pts[0][i], goal_pts[1][i])
            score = scores[pt]
            # ratio = score/ d
            # ratio = 1.0 / d
            ratio = score
            if ratio > best_score:
                best_score = score
                best_path = path
    if best_path:
        for i in range(len(best_path)):
            best_path[i] = np.array(best_path[i])
        return best_path, best_score
    return None, None


def simplify_path(path):
    # Compute differences between consecutive points
    diffs = path[1:] - path[:-1]

    # Find where the direction changes
    changes = np.any(diffs[1:] != diffs[:-1], axis=1)

    # Create a boolean mask for points to keep
    mask = np.zeros(len(path), dtype=bool)
    mask[0] = True  # Always keep the start point
    mask[-1] = True  # Always keep the end point
    mask[1:-1] = changes  # Keep points where direction changes

    # Return the simplified path
    return path[mask]


def generate_large_map(size, num_rectangles):
    # Initialize mask coverage with all ones (reachable)
    mask_coverage = np.ones((size, size))

    # Add random rectangular occluded areas
    for _ in range(num_rectangles):
        x1, y1 = random.randint(0, size - 1), random.randint(0, size - 1)
        x2, y2 = random.randint(x1, size - 1), random.randint(y1, size - 1)
        mask_coverage[y1:y2 + 1, x1:x2 + 1] = 0

    # Create scores array using a sine function (positive values)
    x = np.linspace(0, 2 * np.pi, size)
    y = np.linspace(0, 2 * np.pi, size)
    xv, yv = np.meshgrid(x, y)
    scores = np.abs(np.sin(xv) * np.sin(yv)) + 1  # Ensure all values are positive

    return mask_coverage, scores


if __name__ == "__main__":
    size = 1000  # Size of the map
    num_rectangles = 50  # Number of random rectangular occluded areas
    start = np.array([1, 1])

    # Sample mask coverage (1 = reachable, 0 = occluded)
    mask_coverage, scores = generate_large_map(size, num_rectangles)

    mask_coverage = np.pad(mask_coverage, pad_width=1, mode='constant', constant_values=0)

    scores = np.pad(scores, pad_width=1, mode="constant", constant_values=0)
    scores_, num, reachable = compute_reachable_area_score(start, mask_coverage, scores, 40000)

    a = time.time()
    scores_a, num, reachable = compute_reachable_area_score(start, mask_coverage, scores, 40000)
    print(time.time() - a)
    a = time.time()
    scores_b, num, reachable = compute_reachable_area_score_optimized(start, mask_coverage, scores, 40000)
    print(time.time() - a)
    a = time.time()
    scores_c, num, reachable = compute_reachable_area_score(start, mask_coverage, scores, 40000)
    print(time.time() - a)
    a = time.time()
    print(scores_a, scores_b, scores_c)
    goal_positions = np.array([[size - 2, size - 2]])

    # Call the build_tree function
    # path = find_path(start, mask_coverage, goal_positions)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot mask coverage
    ax[0].imshow(mask_coverage, cmap='gray')
    ax[0].set_title('Mask Coverage')
    ax[0].scatter(start[1], start[0], color='red', label='Start')
    ax[0].legend()

    # Plot path
    ax[1].imshow(mask_coverage, cmap='gray')
    ax[1].set_title('Path')
    path = np.array(path)
    ax[1].plot(path[:, 1], path[:, 0], label='Path')
    ax[1].scatter(start[1], start[0], color='blue', label='Start')
    # ax[2].legend()

    plt.show()
