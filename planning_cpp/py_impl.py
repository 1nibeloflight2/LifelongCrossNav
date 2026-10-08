import numpy as np
from numba import jit


@jit(nopython=True)
def compute_reachable_area(start: np.ndarray,
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
    height, width = mask_coverage.shape
    reachable = np.zeros_like(scores, dtype=np.float32)
    depth_map = np.full_like(mask_coverage, fill_value=max_depth + 1, dtype=np.int32)
    queue = [(start[0], start[1], 0)]  # (x, y, depth)
    score = 0.0
    num_visited = 0

    while queue:
        x, y, depth = queue.pop(0)

        if depth >= max_depth or depth >= depth_map[x, y] or not mask_coverage[x, y]:
            continue

        depth_map[x, y] = depth
        num_visited += 1
        current_score = scores[x, y]
        reachable[x, y] = current_score
        score = max(score, current_score)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width:
                new_depth = depth + 1
                if new_depth < depth_map[nx, ny] and mask_coverage[nx, ny]:
                    queue.append((nx, ny, new_depth))

    return score, num_visited, reachable