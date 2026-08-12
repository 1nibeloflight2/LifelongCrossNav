"""
Contains utility functions for clustering
"""

from dataclasses import dataclass

import numpy as np

from scipy.ndimage import maximum_filter
from skimage import filters, morphology, feature
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
from dataclasses import dataclass
from typing import List
from scipy.ndimage import maximum_filter
from skimage.segmentation import expand_labels

# typing
from typing import List, Union

# torch
import torch

# navgoal
from mapping.nav_goals.navigation_goals import NavGoal


@dataclass
class Cluster(NavGoal):
    """
    Represents a cluster of data points
    """
    center: np.ndarray
    points: np.ndarray
    cluster_score: float

    def __repr__(self):
        return f"Cluster(center={self.center}, points={self.points}, score={self.cluster_score})"

    def __eq__(self, other):
        return np.all(self.center == other.center)

    def get_score(self):
        return self.cluster_score

    def get_descr_point(self):
        return self.center

    def compute_score(self, score_map):
        # score is the max score in the cluster
        self.cluster_score = np.max(score_map[self.points[:, 0], self.points[:, 1]])


# Include the previous clustering functions here
def find_local_maxima(similarity_map, mask, neighborhood_size=10):
    local_max = maximum_filter(similarity_map, size=neighborhood_size)
    return (similarity_map == local_max) & mask  # & (similarity_map > 0.5)


def cluster_high_similarity_regions(
        similarity_map: np.ndarray,
        mask: np.ndarray,
        neighborhood_size: int = 10,
        min_cluster_size: int = 2,
        relative_threshold: float = 0.8
) -> List[Cluster]:
    # Find local maxima
    # blur the similarity map
    # similarity_map = filters.gaussian(similarity_map, sigma=0.02)
    local_maxima = find_local_maxima(similarity_map, mask, neighborhood_size)
    maxima_coords = np.column_stack(np.where(local_maxima))
    maxima_with_scores = [(coord, similarity_map[tuple(coord)]) for coord in maxima_coords]
    maxima_with_scores.sort(key=lambda x: x[1], reverse=True)

    if len(maxima_coords) == 0:
        return np.zeros_like(similarity_map, dtype=int)

    # Initialize clusters
    clusters = []
    processed = np.zeros_like(similarity_map, dtype=bool)

    for coord, score in maxima_with_scores:
        # check if already processed
        if processed[tuple(coord)]:
            continue

        # Get local threshold
        local_max_value = similarity_map[tuple(coord)]
        threshold = local_max_value * relative_threshold

        # Region growing
        stack = [coord]
        cluster_score = -np.inf  # Initialize with negative infinity
        cluster_points = []
        max_similarity_point = None

        while stack:
            current = stack.pop()
            if processed[tuple(current)]:
                continue
            if similarity_map[tuple(current)] >= threshold and mask[tuple(current)]:
                processed[tuple(current)] = True
                cluster_points.append(current)

                # Update max similarity point
                current_similarity = similarity_map[tuple(current)]
                if current_similarity > cluster_score:
                    cluster_score = current_similarity
                    max_similarity_point = current

                # Add neighbors to stack
                neighbors = np.array([(current[0] + dx, current[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
                valid_neighbors = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < similarity_map.shape[0]) & \
                                  (neighbors[:, 1] >= 0) & (neighbors[:, 1] < similarity_map.shape[1])
                for v in neighbors[valid_neighbors]:
                    stack.append(v)

        if len(cluster_points) >= min_cluster_size:
            cluster_points = np.array(cluster_points)
            # Set center as the point with maximum similarity
            center = np.array(max_similarity_point)
            clusters.append(Cluster(center=center, points=cluster_points, cluster_score=cluster_score))

    return clusters


def watershed_clustering(
        similarity_map: np.ndarray,
        mask: np.ndarray,
        neighborhood_size: int = 30,
        min_cluster_size: int = 5,
        relative_threshold: float = 0.95
) -> List[Cluster]:
    # Invert the similarity map
    inverted_map = np.max(similarity_map) - similarity_map

    # Find local maxima (which become minima in the inverted map)
    local_maxima = find_local_maxima(similarity_map, mask, neighborhood_size)
    maxima_coords = np.column_stack(np.where(local_maxima))

    if len(maxima_coords) == 0:
        return []

    # Create markers for watershed
    markers = np.zeros_like(similarity_map, dtype=int)
    for i, coord in enumerate(maxima_coords, start=1):
        markers[tuple(coord)] = i

    # Apply watershed
    labels = watershed(inverted_map, markers, mask=mask)

    # Process clusters
    clusters = []
    for i in range(1, np.max(labels) + 1):
        cluster_mask = labels == i
        cluster_points = np.argwhere(cluster_mask)

        if len(cluster_points) < min_cluster_size:
            continue

        cluster_values = similarity_map[cluster_mask]
        max_similarity_index = np.argmax(cluster_values)
        max_similarity_point = cluster_points[max_similarity_index]
        cluster_score = cluster_values[max_similarity_index]

        # Filter points based on relative threshold
        threshold = cluster_score * relative_threshold
        valid_points = cluster_points
        # valid_points = cluster_points[cluster_values >= threshold]

        if len(valid_points) >= min_cluster_size:
            clusters.append(Cluster(
                center=np.array(max_similarity_point),
                points=valid_points,
                cluster_score=cluster_score
            ))

    return clusters


def gradient_based_clustering(
        similarity_map: np.ndarray,
        mask: np.ndarray,
        neighborhood_size: int = 10,
        min_cluster_size: int = 5,
        gradient_threshold: float = 0.05,
        min_similarity: float = 0.5
) -> List[Cluster]:
    # Compute gradients
    gy, gx = np.gradient(similarity_map)
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Find local maxima
    local_maxima = find_local_maxima(similarity_map, mask, neighborhood_size)
    maxima_coords = np.column_stack(np.where(local_maxima))

    clusters = []
    processed = np.zeros_like(similarity_map, dtype=bool)

    for coord in maxima_coords:
        if processed[tuple(coord)]:
            continue

        # Region growing
        stack = [coord]
        cluster_points = []
        max_similarity_point = coord
        max_similarity = similarity_map[tuple(coord)]

        while stack:
            current = stack.pop()
            if processed[tuple(current)]:
                continue

            if (gradient_magnitude[tuple(current)] <= gradient_threshold and
                    similarity_map[tuple(current)] >= min_similarity and
                    mask[tuple(current)]):

                processed[tuple(current)] = True
                cluster_points.append(current)

                # Update max similarity point
                current_similarity = similarity_map[tuple(current)]
                if current_similarity > max_similarity:
                    max_similarity = current_similarity
                    max_similarity_point = current

                # Add neighbors to stack
                neighbors = np.array([(current[0] + dx, current[1] + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
                valid_neighbors = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < similarity_map.shape[0]) & \
                                  (neighbors[:, 1] >= 0) & (neighbors[:, 1] < similarity_map.shape[1])
                for neighbor in neighbors[valid_neighbors]:
                    if not processed[tuple(neighbor)]:
                        stack.append(neighbor)

        if len(cluster_points) >= min_cluster_size:
            clusters.append(Cluster(
                center=np.array(max_similarity_point),
                points=np.array(cluster_points),
                cluster_score=max_similarity
            ))

    return clusters


def apply_dbscan(coords, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    return db.labels_


def cluster_thermal_image(
        similarity_map: np.ndarray,
        mask: np.ndarray,
        min_distance: int = 10,
        threshold_abs: float = 0.5,
        expansion_distance: int = 5,
        min_cluster_size: int = 5
) -> List[Cluster]:
    # Find local maxima
    local_max = maximum_filter(similarity_map, size=min_distance)
    peaks = ((similarity_map == local_max) & (similarity_map > threshold_abs) & mask)

    # Label the peaks
    num_peaks, labeled_peaks = cv2.connectedComponents(peaks.astype(np.uint8))

    # Expand the labels
    expanded_labels = expand_labels(labeled_peaks, distance=expansion_distance)

    # Create Cluster objects
    cluster_objects = []
    for i in range(1, num_peaks + 1):  # Skip background (0)
        cluster_mask = expanded_labels == i
        cluster_points = np.argwhere(cluster_mask)

        if len(cluster_points) >= min_cluster_size:
            cluster_scores = similarity_map[cluster_mask]
            max_score_idx = np.argmax(cluster_scores)
            center = cluster_points[max_score_idx]

            cluster_objects.append(Cluster(
                center=center,
                points=cluster_points,
                cluster_score=cluster_scores[max_score_idx]
            ))

    # Sort clusters by score in descending order
    cluster_objects.sort(key=lambda x: x.cluster_score, reverse=True)

    return cluster_objects


