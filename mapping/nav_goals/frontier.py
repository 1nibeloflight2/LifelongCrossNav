"""
Frontier detection and processing functions, adapted from https://github.com/bdaiinstitute/vlfm.
"""
# numpy
import numpy as np

# cv2
import cv2

# typing
from typing import List, Optional

# rerun
import rerun as rr

# mapping
from mapping.nav_goals.navigation_goals import NavGoal

from dataclasses import dataclass

@dataclass
class Frontier(NavGoal):
    frontier_midpoint: np.ndarray
    points: np.ndarray
    frontier_score: float

    def __eq__(self, other):
        return np.all(self.frontier_midpoint == other.frontier_midpoint)

    def get_score(self):
        return self.frontier_score

    def get_descr_point(self):
        return self.frontier_midpoint



def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)


def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end) by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def filter_out_small_unexplored(
        full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int
):
    """Edit the explored map to add small unexplored areas, which ignores their
    frontiers."""
    if area_thresh == -1:
        return explored_mask

    unexplored_mask = full_map.copy()
    unexplored_mask[explored_mask > 0] = 0

    # Find contours in the unexplored mask
    contours, _ = cv2.findContours(
        unexplored_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Add small unexplored areas to the explored map
    small_contours = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < area_thresh:
            mask = np.zeros_like(explored_mask)
            mask = cv2.drawContours(mask, [contour], 0, 1, -1)
            # masked_values = unexplored_mask[mask.astype(bool)]
            # values = set(masked_values.tolist())
            # if 1 in values and len(values) == 1:
            #     small_contours.append(contour)
            small_contours.append(contour)
    new_explored_mask = explored_mask.copy()
    cv2.drawContours(new_explored_mask, small_contours, -1, 255, -1)
    return new_explored_mask


def detect_frontiers(
        full_map: np.ndarray, explored_mask: np.ndarray, known_th, area_thresh: Optional[int] = -1
) -> List[np.ndarray]:
    """Detects frontiers in a map.

    Args:
        full_map (np.ndarray): White polygon on black image, where white is navigable.
        Mono-channel mask.
        explored_mask (np.ndarray): Portion of white polygon that has been seen already.
        This is also a mono-channel mask.
        area_thresh (int, optional): Minimum unexplored area (in pixels) needed adjacent
        to a frontier for that frontier to be valid. Defaults to -1.

    Returns:
        np.ndarray: A mono-channel mask where white contours represent each frontier.
    """
    # Find the contour of the explored area
    full_map *= 255
    explored_mask *= 255
    # full_map[known_th == 0] = 0
    # explored_mask = cv2.dilate(
    #     explored_mask.astype(np.uint8),
    #     np.ones((2, 2), np.uint8),
    #     iterations=1,
    # )
    explored_mask[full_map == 0] = 0
    filtered_explored_mask = filter_out_small_unexplored(
        full_map, explored_mask, area_thresh
    )
    contours, _ = cv2.findContours(
        filtered_explored_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    unexplored_mask = np.where(filtered_explored_mask > 0, 0, full_map)
    # unexplored_mask[known_th == 0] = 0
    unexplored_mask = cv2.blur(  # blurring for some leeway
        np.where(unexplored_mask > 0, 255, unexplored_mask), (5, 5)
    )

    frontiers = []
    contour_areas = [cv2.contourArea(contour) for contour in contours]

    # get largest
    cont_ret = None
    if len(contour_areas) > 0:
        largest_contour = contours[np.argmax(contour_areas)]
        cont_log = largest_contour.transpose(1, 0, 2)
        cont_log = np.flip(cont_log, axis=-1)
        cont_ret = np.flip(largest_contour, axis=-1)
        rr.log("map/largest_contour", rr.LineStrips2D(cont_log))
        frontiers = contour_to_frontiers(
            interpolate_contour(largest_contour), cv2.blur(  # blurring for some leeway
                    full_map, (3, 3)
                )
        )
    else:
        print("No frontiers found")


    return frontiers, unexplored_mask, cont_ret


def interpolate_contour(contour):
    """Given a cv2 contour, this function will add points in between each pair of
    points in the contour using the bresenham algorithm to make the contour more
    continuous.
    :param contour: A cv2 contour of shape (N, 1, 2)
    :return:
    """
    # First, reshape and expand the frontier to be a 2D array of shape (N-1, 2, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate((contour[:-1], contour[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # Also add a segment connecting the last point to the first point
    line_segments = np.concatenate(
        (line_segments, np.array([contour[-1], contour[0]]).reshape((1, 2, 2)))
    )
    pts = []
    for (x0, y0), (x1, y1) in line_segments:
        pts.append(
            bresenhamline(np.array([[x0, y0]]), np.array([[x1, y1]]), max_iter=-1)
        )
    pts = np.concatenate(pts).reshape((-1, 1, 2))
    return pts


def contour_to_frontiers(contour, unexplored_mask):
    """Given a contour from OpenCV, return a list of numpy arrays. Each array contains
    contiguous points forming a single frontier. The contour is assumed to be a set of
    contiguous points, but some of these points are not on any frontier, indicated by
    having a value of 0 in the unexplored mask. This function will split the contour
    into multiple arrays that exclude such points."""
    bad_inds = []
    num_contour_points = len(contour)
    for idx in range(num_contour_points):
        x, y = contour[idx][0]
        if unexplored_mask[y, x] != 255:
            bad_inds.append(idx)
    frontiers = np.split(contour, bad_inds)
    # np.split is fast but does NOT remove the element at the split index
    filtered_frontiers = []
    front_last_split = (
            0 not in bad_inds
            and len(bad_inds) > 0
            and max(bad_inds) < num_contour_points - 2
    )
    for idx, f in enumerate(frontiers):
        # a frontier must have at least 2 points (3 with bad ind)
        if len(f) > 2 or (idx == 0 and front_last_split):
            if idx == 0:
                filtered_frontiers.append(f)
            else:
                filtered_frontiers.append(f[1:])
    # Combine the first and last frontier if the first point of the first frontier and
    # the last point of the last frontier are the first and last points of the original
    # contour. Only check if there are at least 2 frontiers.
    if len(filtered_frontiers) > 1 and front_last_split:
        last_frontier = filtered_frontiers.pop()
        filtered_frontiers[0] = np.concatenate((last_frontier, filtered_frontiers[0]))
    return filtered_frontiers


def get_frontier_midpoint(frontier) -> np.ndarray:
    """Given a list of contiguous points (numpy arrays) representing a frontier, first
    calculate the total length of the frontier, then find the midpoint of the
    frontier"""
    # First, reshape and expand the frontier to be a 2D array of shape (X, 2, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate((frontier[:-1], frontier[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # Calculate the length of each line segment
    line_lengths = np.sqrt(
        np.square(line_segments[:, 0, 0] - line_segments[:, 1, 0])
        + np.square(line_segments[:, 0, 1] - line_segments[:, 1, 1])
    )
    cum_sum = np.cumsum(line_lengths)
    total_length = cum_sum[-1]
    # Find the midpoint of the frontier
    half_length = total_length / 2
    # Find the line segment that contains the midpoint
    line_segment_idx = np.argmax(cum_sum > half_length)
    # Calculate the coordinates of the midpoint
    line_segment = line_segments[line_segment_idx]
    line_length = line_lengths[line_segment_idx]
    # Use the difference between the midpoint length and cumsum
    # to find the proportion of the line segment that the midpoint is at
    length_up_to = cum_sum[line_segment_idx - 1] if line_segment_idx > 0 else 0
    proportion = (half_length - length_up_to) / line_length
    # Calculate the midpoint coordinates
    midpoint = line_segment[0] + proportion * (line_segment[1] - line_segment[0])
    return midpoint