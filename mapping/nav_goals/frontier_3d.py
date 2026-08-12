"""
3D Frontier detection for cross-floor voxel navigation.

Implements TravExplorer-style traversable frontier detection:
  - Headroom check: no TRAVERSABLE above within K_h layers
  - Column boundary check: horizontal 4-neighbor must have fully non-traversable vertical column
  - 26-connected clustering + minimum size filtering (eliminates corner/dead-end traps)
  - Cluster centroid as navigation target (stable, avoids edge voxels)
  - Incremental update via AABB intersection

Also includes:
  - CLIP similarity-guided scoring (BFS behind frontier)
  - Shared physical-distance ordering for basic/descend/ascent categories
  - POI clustering channel (high-similarity OCCUPIED voxels)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from mapping.nav_goals.navigation_goals import NavGoal

if TYPE_CHECKING:
    from mapping.voxel_map import VoxelData, VoxelMap, VoxelType


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

FRONTIER_HEADROOM_LAYERS = 4      # K_h: layers above that must be free of TRAVERSABLE
FRONTIER_MIN_CLUSTER_SIZE = 5     # minimum voxels in a frontier cluster (filters corners)
FRONTIER_MAX_CLUSTERS = 30        # max clusters to return
FRONTIER_COLUMN_SEARCH_RANGE = 6  # how many z-levels to check in vertical column
FRONTIER_VERTICAL_SHADOW_LAYERS = 3  # suppress vertical duplicate frontiers if another floor exists


def _metric_euclidean_priority(
    voxel_map: "VoxelMap",
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
) -> float:
    """Monotonic, non-saturating compatibility score for nearest-first frontiers."""
    dx_m = (goal[0] - start[0]) * float(voxel_map.voxel_size)
    dy_m = (goal[1] - start[1]) * float(voxel_map.voxel_size)
    dz_m = (goal[2] - start[2]) * float(voxel_map.voxel_size_z)
    distance_m = (dx_m * dx_m + dy_m * dy_m + dz_m * dz_m) ** 0.5
    return 1.0 / (1.0 + distance_m)


def _classify_frontier_candidate(
    vid: Tuple[int, int, int],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    occlusion_layers: int = 0,
    agent_ray_iz: Optional[int] = None,
    allow_inflated_connector: bool = False,
) -> Optional[str]:
    """Single source of truth for full and incremental frontier detection.

    Vertical-duplicate suppression is only needed away from the agent's current
    2D raycasting layer.  A frontier on that layer is directly observable and
    can be expanded when reached, whereas an off-layer frontier may be a stale
    traversable/fake projection left by a floor transition.
    """
    from mapping.voxel_map import VoxelType, is_occluded_unknown_column

    vd = voxels.get(vid)
    if vd is None or vd.voxel_type not in (
        VoxelType.TRAVERSABLE,
        VoxelType.TRAVERSABLE_STAIR,
    ):
        return None
    # Inflated floor voxels may be admitted by the detector only as geometric
    # connector members.  They are never frontier seeds and remain forbidden
    # as navigation targets.
    if vd.inflated and not allow_inflated_connector:
        return None

    has_unknown_n4 = False
    has_fake_n8 = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = (vid[0] + dx, vid[1] + dy, vid[2])
            neighbor_vd = voxels.get(neighbor)
            if neighbor_vd is None:
                if abs(dx) + abs(dy) == 1:
                    # A hole in only one horizontal z slice is not genuine
                    # unknown space.  Traversability is often represented on
                    # two adjacent air layers; without this check, whichever
                    # layer currently equals ``agent_ray_iz`` can turn the
                    # other layer's sparse holes into a basic frontier and the
                    # two layers repeatedly alternate.  Keep the current-ray
                    # exception for real observed frontiers, but require the
                    # missing N4 column to be absent from both adjacent floor
                    # layers as well.
                    vertically_covered = any(
                        (
                            (vertical_vd := voxels.get(
                                (neighbor[0], neighbor[1], neighbor[2] + dz)
                            )) is not None and
                            vertical_vd.voxel_type in (
                                VoxelType.TRAVERSABLE,
                                VoxelType.TRAVERSABLE_STAIR,
                            )
                        )
                        for dz in (-1, 1)
                    )
                    if vertically_covered:
                        continue
                    # A missing voxel at the ray layer is genuine unknown only
                    # when the same XY column is also free of known blockers in
                    # the vertical interval used by 2D obstacle aggregation.
                    # Otherwise the ray stopped because of a higher wall/sofa
                    # voxel and this boundary must not become a basic frontier.
                    column_occluded = is_occluded_unknown_column(
                        neighbor,
                        voxels,
                        occlusion_layers,
                    )
                    if not column_occluded:
                        has_unknown_n4 = True
            elif neighbor_vd.voxel_type == VoxelType.TRAVERSABLE_FAKE:
                has_fake_n8 = True

    # Descend geometry wins when both conditions hold. Vertical duplicate
    # suppression applies only to ordinary unknown-space frontiers.
    if has_fake_n8:
        return "descend"
    # TRAVERSABLE_STAIR participates only in descend recovery at a
    # TRAVERSABLE_FAKE boundary; it must not become an ordinary basic frontier.
    if vd.voxel_type == VoxelType.TRAVERSABLE_STAIR:
        return None
    if has_unknown_n4:
        on_agent_ray_layer = agent_ray_iz is not None and vid[2] == agent_ray_iz
        if on_agent_ray_layer or not _is_vertical_duplicate_frontier_candidate(vid, voxels):
            return "basic"
    return None


# ---------------------------------------------------------------------------
# NavGoal subclasses
# ---------------------------------------------------------------------------

@dataclass
class FrontierCluster3D(NavGoal):
    """
    A cluster of traversable frontier voxels (TravExplorer-style).
    Navigation target is the centroid projected to nearest TRAVERSABLE.
    """
    nav_voxel_id: Tuple[int, int, int]     # TRAVERSABLE voxel to path-plan to (nearest to centroid)
    centroid_metric: Tuple[float, float, float]  # geometric centroid in metric coords
    cluster_voxels: Set[Tuple[int, int, int]]    # all voxels in the cluster
    frontier_score: float                  # combined score
    cluster_size: int                      # number of voxels
    floor_iz: int                          # representative z-level
    aabb: Tuple[Tuple[int, int, int], Tuple[int, int, int]]  # (min_corner, max_corner)
    is_stair_adjacent: bool = False
    is_stair_frontier: bool = False        # True if frontier is triggered by fake TRAVERSABLE neighbor
                                           # (indicates potential downstairs — triggers look-down on arrival)
    fake_triggered_voxels: Set[Tuple[int, int, int]] = field(default_factory=set)
    fake_anchor_aabb: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None

    def __eq__(self, other):
        if not isinstance(other, FrontierCluster3D):
            return NotImplemented
        return self.nav_voxel_id == other.nav_voxel_id

    def __hash__(self):
        return hash(self.nav_voxel_id)

    def get_score(self) -> float:
        return self.frontier_score

    def get_descr_point(self) -> np.ndarray:
        return np.array(self.nav_voxel_id)


# Keep Frontier3D as a simple alias for backward compatibility
Frontier3D = FrontierCluster3D


@dataclass
class Cluster3D(NavGoal):
    """
    A cluster of high-similarity OCCUPIED voxels (POI exploitation).
    Navigation target is the nearest TRAVERSABLE to the cluster center.
    """
    nav_voxel_id: Tuple[int, int, int]
    center_voxel: Tuple[int, int, int]
    cluster_score: float
    floor_iz: int
    cluster_size: int = 1
    max_similarity: float = 0.0
    topk_mean_similarity: float = 0.0
    mean_similarity: float = 0.0
    confidence_weighted_similarity: float = 0.0
    bbox_min: Tuple[int, int, int] = (0, 0, 0)
    bbox_max: Tuple[int, int, int] = (0, 0, 0)
    evidence_score: float = 0.0
    poi_type: str = "history"
    last_seen_iter: int = 0

    def __eq__(self, other):
        if not isinstance(other, Cluster3D):
            return NotImplemented
        return self.center_voxel == other.center_voxel

    def __hash__(self):
        return hash(self.center_voxel)

    def get_score(self) -> float:
        return self.cluster_score

    def get_descr_point(self) -> np.ndarray:
        return np.array(self.nav_voxel_id)


@dataclass
class StairSemanticFrontier3D(NavGoal):
    """
    Semantic stair frontier: generated in stair_sem mode.
    Navigation target is the farthest (max z-difference) STAIR cluster edge,
    projected to a traversable/stair voxel for path planning.

    Always gets highest priority score among all nav goals.
    """
    nav_voxel_id: Tuple[int, int, int]     # STAIR/TRAVERSABLE voxel to path-plan to
    stair_cluster_center: Tuple[int, int, int]  # center of the target stair cluster
    z_difference: float                    # z difference from agent (positive = below)
    semantic_score: float                  # always very high to ensure top priority

    def __eq__(self, other):
        if not isinstance(other, StairSemanticFrontier3D):
            return NotImplemented
        return self.nav_voxel_id == other.nav_voxel_id

    def __hash__(self):
        return hash(self.nav_voxel_id)

    def get_score(self) -> float:
        return self.semantic_score

    def get_descr_point(self) -> np.ndarray:
        return np.array(self.nav_voxel_id)


@dataclass
class RiseStairFrontier3D(NavGoal):
    """
    Upstairs stair candidate produced by semantic proposal + depth geometry.

    This is a pre-semantic-mode navigation goal: reaching it lets Navigator3D run
    SAM on the verified stair box and enter stair semantic mode with direction up.
    """
    nav_voxel_id: Tuple[int, int, int]
    candidate_box: Tuple[float, float, float, float]
    confidence: float
    geometry_score: float
    semantic_score: float  # legacy field; stores current distance priority
    bottom_xy: Optional[Tuple[float, float]] = None
    approach_xy: Optional[Tuple[float, float]] = None

    def __eq__(self, other):
        if not isinstance(other, RiseStairFrontier3D):
            return NotImplemented
        return self.nav_voxel_id == other.nav_voxel_id

    def __hash__(self):
        return hash(self.nav_voxel_id)

    def get_score(self) -> float:
        return self.semantic_score

    def get_descr_point(self) -> np.ndarray:
        return np.array(self.nav_voxel_id)


# ---------------------------------------------------------------------------
# TravExplorer-style frontier detection (full scan)
# ---------------------------------------------------------------------------

def detect_frontiers_3d_full(
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    search_radius: Optional[int] = 60,
    frontier_sim_depth: int = 8,
    sim_weight: float = 30.0,
    min_cluster_size: int = FRONTIER_MIN_CLUSTER_SIZE,
    z_range: Optional[int] = None,
    max_clusters: Optional[int] = FRONTIER_MAX_CLUSTERS,
) -> List[FrontierCluster3D]:
    """
    Full-scan 3D frontier detection following TravExplorer paper (Section IV-C).

    Steps:
      1. Identify candidate frontier voxels (headroom check + column boundary check)
      2. Cluster candidates via 26-connected region growing
      3. Filter clusters by minimum size
      4. Score each cluster and return sorted list

    Returns:
        List of FrontierCluster3D sorted by score descending
    """
    from mapping.voxel_map import VoxelType

    voxels = voxel_map.voxels
    n_cells = voxel_map.n_cells
    max_z = voxel_map.max_z_voxels
    agent_ix, agent_iy, agent_iz = agent_voxel

    # ---- Step 1: Identify candidate frontier voxels ----
    # Seed sets contain real, non-inflated frontier voxels. Inflated voxels
    # can connect neighbouring seeds into one physical cluster, but a component
    # containing only inflated connectors is discarded and cannot create a
    # frontier by itself.
    basic_seeds: Set[Tuple[int, int, int]] = set()
    descend_seeds: Set[Tuple[int, int, int]] = set()
    basic_connectors: Set[Tuple[int, int, int]] = set()
    descend_connectors: Set[Tuple[int, int, int]] = set()
    # Track which candidates were triggered by a fake neighbor (stair frontier)
    fake_triggered: Set[Tuple[int, int, int]] = set()

    for vid, vd in voxels.items():
        ix, iy, iz = vid

        # Optional XY pre-filter.  ``None`` is used by the current-floor full
        # rescan: iterate every known voxel in the requested z band rather than
        # incorrectly treating the agent-local 60-voxel window as the floor.
        if (
            search_radius is not None and
            (abs(ix - agent_ix) > search_radius or abs(iy - agent_iy) > search_radius)
        ):
            continue

        # Z-range filter (限制只检测 agent 附近楼层)
        if z_range is not None and abs(iz - agent_iz) > z_range:
            continue

        candidate_kind = _classify_frontier_candidate(
            vid,
            voxels,
            getattr(voxel_map, "frontier_occlusion_layers", 0),
            agent_ray_iz=agent_iz,
            allow_inflated_connector=True,
        )
        if candidate_kind is None:
            continue

        if candidate_kind == "descend":
            if vd.inflated:
                descend_connectors.add(vid)
            else:
                descend_seeds.add(vid)
                fake_triggered.add(vid)
        elif vd.inflated:
            basic_connectors.add(vid)
        else:
            basic_seeds.add(vid)

    # ---- Step 2: 26-connected clustering ----
    basic_components = [
        component
        for component in _cluster_26_connected(basic_seeds | basic_connectors)
        if component & basic_seeds
    ]
    descend_components = [
        component
        for component in _cluster_26_connected(descend_seeds | descend_connectors)
        if component & descend_seeds
    ]
    clusters_voxels = basic_components + descend_components

    # ---- Step 3: Filter by minimum size + compute properties ----
    result: List[FrontierCluster3D] = []
    sim_cache = voxel_map.similarity_cache

    for cluster_set in clusters_voxels:
        # For clusters containing fake-triggered voxels (stair geometry),
        # use a lower minimum size (2) since stair edges are narrow
        has_fake = bool(cluster_set & fake_triggered)
        effective_min_size = 2 if has_fake else min_cluster_size
        # Inflated voxels cannot create a frontier component by themselves
        # (the component is retained only when it contains a real seed), but
        # once connected to a real seed they contribute to the physical size
        # of that frontier cluster.
        if len(cluster_set) < effective_min_size:
            continue

        # Compute AABB
        xs = [v[0] for v in cluster_set]
        ys = [v[1] for v in cluster_set]
        zs = [v[2] for v in cluster_set]
        aabb = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

        # Compute centroid in voxel coords, then convert to metric
        centroid_voxel = (
            int(round(np.mean(xs))),
            int(round(np.mean(ys))),
            int(round(np.mean(zs))),
        )
        centroid_metric = voxel_map.voxel_to_metric(
            centroid_voxel[0], centroid_voxel[1], centroid_voxel[2]
        )

        # The navigation representative must belong to this frontier cluster.
        # A global centroid snap can land on an unrelated safe island separated
        # by inflated connectors/walls, after which one failed A* incorrectly
        # makes the whole cluster look unreachable.
        nav_vid = _find_best_nav_voxel_in_cluster(cluster_set, voxels, agent_voxel)
        if nav_vid is None:
            continue

        # Representative floor level (mode of z values)
        floor_iz = int(round(np.median(zs)))

        # Stair adjacency check
        is_stair_adj = _check_stair_adjacency_cluster(cluster_set, voxels)



        # Descend components are a separate frontier type.  They must never
        # fall through to the normal-frontier append below when validation
        # fails, otherwise one fake seed plus an inflated connector can bypass
        # the normal basic-frontier minimum size.
        fake_count = len(cluster_set & fake_triggered)
        is_stair_front = False

        # ---- Step 4: Scoring ----
        score = _metric_euclidean_priority(voxel_map, agent_voxel, nav_vid)

        if has_fake:
            # Inflated connectors count toward the component's physical size,
            # but cannot replace the two real fake-triggered descend seeds.
            if fake_count < 2:
                continue
            local_fake_frontiers = _make_local_fake_frontier_clusters(
                cluster_set=cluster_set,
                fake_triggered=cluster_set & fake_triggered,
                voxel_map=voxel_map,
                agent_voxel=agent_voxel,
                base_score=score,
                sim_weight=sim_weight,
                frontier_sim_depth=frontier_sim_depth,
            )
            if local_fake_frontiers:
                result.extend(local_fake_frontiers)
            # Whether local descend validation succeeds or fails, this parent
            # component must not be emitted as a basic frontier.
            continue

        result.append(FrontierCluster3D(
            nav_voxel_id=nav_vid,
            centroid_metric=centroid_metric,
            cluster_voxels=cluster_set,
            frontier_score=score,
            cluster_size=len(cluster_set),
            floor_iz=floor_iz,
            aabb=aabb,
            is_stair_adjacent=is_stair_adj,
            is_stair_frontier=is_stair_front,
        ))

    # Do not stop at the first N dictionary/set traversal results.  A
    # current-floor full scan must evaluate the complete z band, then retain
    # the globally best clusters deterministically.
    result.sort(key=lambda c: c.frontier_score, reverse=True)
    if max_clusters is None:
        return result
    return result[:max(0, int(max_clusters))]


# ---------------------------------------------------------------------------
# Incremental frontier detection (AABB-based)
# ---------------------------------------------------------------------------

def detect_frontiers_3d_incremental(
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    frontier_clusters: List[FrontierCluster3D],
    search_radius: int = 60,
    frontier_sim_depth: int = 8,
    sim_weight: float = 30.0,
    min_cluster_size: int = FRONTIER_MIN_CLUSTER_SIZE,
    z_range: Optional[int] = None,
    max_clusters: Optional[int] = FRONTIER_MAX_CLUSTERS,
) -> List[FrontierCluster3D]:
    """
    Incremental frontier update (TravExplorer Section IV-C).

    Strategy:
      1. Compute AABB of newly updated voxels this frame
      2. Remove existing clusters whose AABB intersects the update region
         (they may have been invalidated by new observations)
      3. Re-detect frontiers within the update region + margin
      4. Merge with unaffected clusters
      5. Re-score all clusters (distance changed as agent moved)

    Args:
        voxel_map:          VoxelMap instance
        agent_voxel:        Agent's current (ix, iy, iz)
        frontier_clusters:  Previous frame's cluster list (mutated in place)
        search_radius:      Kept for call compatibility; dirty-region rebuilds
                            are not clipped by distance from the agent.
        frontier_sim_depth: BFS depth for similarity scoring
        sim_weight:         Similarity score weight

    Returns:
        Updated list of FrontierCluster3D (sorted by score)
    """
    from mapping.voxel_map import VoxelType

    new_voxels = voxel_map.new_traversable_voxels
    frontier_dirty = getattr(voxel_map, "frontier_dirty_voxels", set())
    if not new_voxels and not frontier_dirty and frontier_clusters:
        # No new voxels — just re-score existing clusters for updated agent position
        _rescore_clusters(frontier_clusters, voxel_map, agent_voxel, frontier_sim_depth, sim_weight)
        frontier_clusters.sort(key=lambda c: c.frontier_score, reverse=True)
        return frontier_clusters

    # ---- Compute update AABB from new voxels ----
    # Include all updated_voxel_ids (which includes OCCUPIED voxels too)
    all_updated = set(new_voxels)
    all_updated.update(voxel_map.updated_voxel_ids)
    all_updated.update(frontier_dirty)

    if not all_updated:
        if not frontier_clusters:
            return []
        _rescore_clusters(frontier_clusters, voxel_map, agent_voxel, frontier_sim_depth, sim_weight)
        frontier_clusters.sort(key=lambda c: c.frontier_score, reverse=True)
        return frontier_clusters

    xs = [v[0] for v in all_updated]
    ys = [v[1] for v in all_updated]
    zs = [v[2] for v in all_updated]
    update_aabb = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    # ---- Separate affected vs unaffected clusters ----
    margin = int(getattr(voxel_map, "frontier_dirty_margin", 5))
    expanded_aabb = (
        (update_aabb[0][0] - margin, update_aabb[0][1] - margin, update_aabb[0][2] - margin),
        (update_aabb[1][0] + margin, update_aabb[1][1] + margin, update_aabb[1][2] + margin),
    )

    unaffected = []
    for cluster in frontier_clusters:
        if not _aabb_intersect(cluster.aabb, expanded_aabb):
            unaffected.append(cluster)
        # else: cluster is affected, will be re-detected

    # ---- Re-detect in the affected region ----
    # We do a full detection but limit to the expanded region
    new_clusters = _detect_in_region(
        voxel_map, agent_voxel, expanded_aabb,
        frontier_sim_depth, sim_weight, min_cluster_size=min_cluster_size,
        z_range=z_range,
    )

    # ---- Merge and re-score ----
    merged = unaffected + new_clusters
    _rescore_clusters(merged, voxel_map, agent_voxel, frontier_sim_depth, sim_weight)
    merged.sort(key=lambda c: c.frontier_score, reverse=True)

    if max_clusters is None:
        return merged
    return merged[:max(0, int(max_clusters))]


def _detect_in_region(
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    region_aabb: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    frontier_sim_depth: int,
    sim_weight: float,
    min_cluster_size: int = FRONTIER_MIN_CLUSTER_SIZE,
    z_range: Optional[int] = None,
) -> List[FrontierCluster3D]:
    """Detect frontiers only within a specific AABB region."""
    from mapping.voxel_map import VoxelType

    voxels = voxel_map.voxels
    n_cells = voxel_map.n_cells
    max_z = voxel_map.max_z_voxels
    agent_ix, agent_iy, agent_iz = agent_voxel
    (rmin, rmax) = region_aabb

    basic_seeds: Set[Tuple[int, int, int]] = set()
    descend_seeds: Set[Tuple[int, int, int]] = set()
    basic_connectors: Set[Tuple[int, int, int]] = set()
    descend_connectors: Set[Tuple[int, int, int]] = set()
    fake_triggered: Set[Tuple[int, int, int]] = set()

    for vid, vd in voxels.items():
        ix, iy, iz = vid

        # Region filter
        if not (rmin[0] <= ix <= rmax[0] and rmin[1] <= iy <= rmax[1] and rmin[2] <= iz <= rmax[2]):
            continue

        # Z-range filter
        if z_range is not None and abs(iz - agent_iz) > z_range:
            continue

        candidate_kind = _classify_frontier_candidate(
            vid,
            voxels,
            getattr(voxel_map, "frontier_occlusion_layers", 0),
            agent_ray_iz=agent_iz,
            allow_inflated_connector=True,
        )
        if candidate_kind is None:
            continue

        if candidate_kind == "descend":
            if vd.inflated:
                descend_connectors.add(vid)
            else:
                descend_seeds.add(vid)
                fake_triggered.add(vid)
        elif vd.inflated:
            basic_connectors.add(vid)
        else:
            basic_seeds.add(vid)

    # Cluster and filter
    basic_components = [
        component
        for component in _cluster_26_connected(basic_seeds | basic_connectors)
        if component & basic_seeds
    ]
    descend_components = [
        component
        for component in _cluster_26_connected(descend_seeds | descend_connectors)
        if component & descend_seeds
    ]
    clusters_voxels = basic_components + descend_components
    sim_cache = voxel_map.similarity_cache
    result: List[FrontierCluster3D] = []

    for cluster_set in clusters_voxels:
        # For clusters containing fake-triggered voxels (stair geometry),
        # use a lower minimum size (2) since stair edges are narrow
        has_fake = bool(cluster_set & fake_triggered)
        effective_min_size = 2 if has_fake else min_cluster_size
        # Keep the same rule as the full scan: inflated connector voxels count
        # toward cluster size, but cannot seed a frontier on their own.
        if len(cluster_set) < effective_min_size:
            continue

        xs = [v[0] for v in cluster_set]
        ys = [v[1] for v in cluster_set]
        zs = [v[2] for v in cluster_set]
        aabb = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

        centroid_voxel = (int(round(np.mean(xs))), int(round(np.mean(ys))), int(round(np.mean(zs))))
        centroid_metric = voxel_map.voxel_to_metric(
            centroid_voxel[0], centroid_voxel[1], centroid_voxel[2]
        )

        # Keep the representative inside the rebuilt cluster. Reachability is
        # checked later, with one fallback representative per non-inflated
        # seed component when inflated connectors join separate safe regions.
        nav_vid = _find_best_nav_voxel_in_cluster(cluster_set, voxels, agent_voxel)
        if nav_vid is None:
            continue

        floor_iz = int(round(np.median(zs)))
        is_stair_adj = _check_stair_adjacency_cluster(cluster_set, voxels)

        score = _metric_euclidean_priority(voxel_map, agent_voxel, nav_vid)

        # Match the full scan: descend components are type-exclusive and must
        # not be downgraded to basic frontiers after failed validation.
        fake_count = len(cluster_set & fake_triggered)
        is_stair_front = False

        if has_fake:
            if fake_count < 2:
                continue
            local_fake_frontiers = _make_local_fake_frontier_clusters(
                cluster_set=cluster_set,
                fake_triggered=cluster_set & fake_triggered,
                voxel_map=voxel_map,
                agent_voxel=agent_voxel,
                base_score=score,
                sim_weight=sim_weight,
                frontier_sim_depth=frontier_sim_depth,
            )
            if local_fake_frontiers:
                result.extend(local_fake_frontiers)
            continue

        result.append(FrontierCluster3D(
            nav_voxel_id=nav_vid,
            centroid_metric=centroid_metric,
            cluster_voxels=cluster_set,
            frontier_score=score,
            cluster_size=len(cluster_set),
            floor_iz=floor_iz,
            aabb=aabb,
            is_stair_adjacent=is_stair_adj,
            is_stair_frontier=is_stair_front,
        ))

    return result


# ---------------------------------------------------------------------------
# Helper: re-score existing clusters (agent moved, similarity updated)
# ---------------------------------------------------------------------------

def _rescore_clusters(
    clusters: List[FrontierCluster3D],
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    frontier_sim_depth: int,
    sim_weight: float,
) -> None:
    """Re-score clusters in place (distance and similarity may have changed)."""
    from mapping.voxel_map import VoxelType

    voxels = voxel_map.voxels
    sim_cache = voxel_map.similarity_cache
    n_cells = voxel_map.n_cells
    max_z = voxel_map.max_z_voxels
    agent_ix, agent_iy, agent_iz = agent_voxel

    to_remove = []
    for i, cluster in enumerate(clusters):
        # Validate: does the cluster still have frontier voxels?
        # Quick check: is nav_voxel still PATHABLE?
        nav_vid = cluster.nav_voxel_id
        if nav_vid not in voxels or voxels[nav_vid].voxel_type not in (
            VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR,
            VoxelType.STAIR, VoxelType.STAIR_EXPANDED
        ):
            to_remove.append(i)
            continue

        # Validate with the same classifier used by full and local detection.
        still_valid = False
        # Connector-only samples must not invalidate an otherwise live cluster.
        # Validate from non-inflated seeds; connectors never trigger validity.
        sample_vids = [
            sv for sv in cluster.cluster_voxels
            if (sv in voxels and not voxels[sv].inflated)
        ]
        for sv in sample_vids:
            kind = _classify_frontier_candidate(
                sv,
                voxels,
                getattr(voxel_map, "frontier_occlusion_layers", 0),
                agent_ray_iz=agent_iz,
            )
            expected = "descend" if cluster.is_stair_frontier else "basic"
            if kind == expected:
                still_valid = True
                break

        if not still_valid:
            to_remove.append(i)
            continue

        cluster.frontier_score = _metric_euclidean_priority(
            voxel_map, agent_voxel, nav_vid
        )

    # Remove invalid clusters (reverse order to keep indices valid)
    for i in sorted(to_remove, reverse=True):
        clusters.pop(i)


# ---------------------------------------------------------------------------
# Helper: rescore single frontier (backward compat with navigator_3d)
# ---------------------------------------------------------------------------

def rescore_frontier(
    frontier: FrontierCluster3D,
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    frontier_sim_depth: int = 8,
    sim_weight: float = 30.0,
) -> FrontierCluster3D:
    """Re-compute score for a single frontier cluster."""
    _rescore_clusters([frontier], voxel_map, agent_voxel, frontier_sim_depth, sim_weight)
    return frontier


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_vertical_shadowed_frontier_candidate(
    vid: Tuple[int, int, int],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    layers: int = FRONTIER_VERTICAL_SHADOW_LAYERS,
) -> bool:
    """
    Return True when a TRAVERSABLE frontier candidate is likely an upper ghost layer.

    Downstairs scans can leave a sparse upper TRAVERSABLE layer above the actual
    floor. If the same xy column already has a real pathable voxel directly below,
    the upper voxel should not become a normal geometric frontier. FAKE voxels are
    intentionally ignored so stair/drop geometry can still trigger look-down logic.
    """
    from mapping.voxel_map import PATHABLE_TYPES, VoxelType

    ix, iy, iz = vid
    for dz in range(1, layers + 1):
        below = (ix, iy, iz - dz)
        below_vd = voxels.get(below)
        if below_vd is None:
            continue
        if below_vd.voxel_type == VoxelType.TRAVERSABLE_FAKE:
            continue
        if below_vd.voxel_type in PATHABLE_TYPES:
            return True
    return False


def _is_vertical_covered_frontier_candidate(
    vid: Tuple[int, int, int],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    layers: int = FRONTIER_VERTICAL_SHADOW_LAYERS,
) -> bool:
    """
    Return True when a TRAVERSABLE frontier candidate is likely a lower stale layer.

    Upstairs/platform scans can confirm a navigable layer above while leaving
    sparse unknown holes on the lower z slice. If the same xy column already has
    a real pathable voxel directly above, the lower voxel should not remain a
    normal frontier. FAKE voxels are ignored for the same reason as the lower
    shadow check: stair/drop geometry must still be able to trigger separately.
    """
    from mapping.voxel_map import PATHABLE_TYPES, VoxelType

    ix, iy, iz = vid
    for dz in range(1, layers + 1):
        above = (ix, iy, iz + dz)
        above_vd = voxels.get(above)
        if above_vd is None:
            continue
        if above_vd.voxel_type == VoxelType.TRAVERSABLE_FAKE:
            continue
        if above_vd.voxel_type in PATHABLE_TYPES:
            return True
    return False


def _is_vertical_duplicate_frontier_candidate(
    vid: Tuple[int, int, int],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    layers: int = FRONTIER_VERTICAL_SHADOW_LAYERS,
) -> bool:
    """Suppress ordinary frontiers duplicated by nearby pathable layers above or below."""
    return (
        _is_vertical_shadowed_frontier_candidate(vid, voxels, layers)
        or _is_vertical_covered_frontier_candidate(vid, voxels, layers)
    )


def _compute_aabb(
    voxels_set: Set[Tuple[int, int, int]]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    xs = [v[0] for v in voxels_set]
    ys = [v[1] for v in voxels_set]
    zs = [v[2] for v in voxels_set]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _split_large_fake_trigger_cluster(
    fake_cluster: Set[Tuple[int, int, int]],
    tile_size: int = 8,
) -> List[Set[Tuple[int, int, int]]]:
    """Split long fake edges into local anchors so confirmation happens nearby."""
    if not fake_cluster:
        return []
    xs = [v[0] for v in fake_cluster]
    ys = [v[1] for v in fake_cluster]
    if max(xs) - min(xs) <= tile_size and max(ys) - min(ys) <= tile_size:
        return [fake_cluster]

    groups: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
    for vid in fake_cluster:
        key = (vid[0] // tile_size, vid[1] // tile_size, vid[2])
        groups.setdefault(key, set()).add(vid)
    return [group for group in groups.values() if group]


def _has_downward_fake_space(
    fake_cluster: Set[Tuple[int, int, int]],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    depth_voxels: int = 6,
) -> bool:
    """
    Conservative local gate for downstairs anchors.

    We do not require visible stairs here, only that the fake edge has some
    downward/unknown space locally. This keeps real stair mouths eligible while
    letting the later semantic confirmation reject texture leaks and cliffs.
    """
    from mapping.voxel_map import VoxelType

    N8 = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    for sv in fake_cluster:
        ix, iy, iz = sv
        for dx, dy in N8:
            fake = (ix + dx, iy + dy, iz)
            fake_vd = voxels.get(fake)
            if fake_vd is None or fake_vd.voxel_type != VoxelType.TRAVERSABLE_FAKE:
                continue
            for dz in range(1, depth_voxels + 1):
                below = (fake[0], fake[1], fake[2] - dz)
                below_vd = voxels.get(below)
                if below_vd is None or below_vd.voxel_type == VoxelType.TRAVERSABLE_FAKE:
                    return True
    return False


def _find_local_fake_nav_voxel(
    fake_cluster: Set[Tuple[int, int, int]],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    agent_voxel: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int]]:
    from mapping.voxel_map import TARGETABLE_TYPES

    if not fake_cluster:
        return None
    cx = float(np.mean([v[0] for v in fake_cluster]))
    cy = float(np.mean([v[1] for v in fake_cluster]))
    cz = float(np.mean([v[2] for v in fake_cluster]))
    best = None
    best_cost = float("inf")
    for vid in fake_cluster:
        vd = voxels.get(vid)
        if vd is None or vd.voxel_type not in TARGETABLE_TYPES or vd.inflated:
            continue
        cost = (vid[0] - cx) ** 2 + (vid[1] - cy) ** 2 + 0.25 * (vid[2] - cz) ** 2
        cost += 0.05 * (abs(vid[0] - agent_voxel[0]) + abs(vid[1] - agent_voxel[1]))
        if cost < best_cost:
            best_cost = cost
            best = vid
    return best


def _make_local_fake_frontier_clusters(
    cluster_set: Set[Tuple[int, int, int]],
    fake_triggered: Set[Tuple[int, int, int]],
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    base_score: float,
    sim_weight: float,
    frontier_sim_depth: int,
) -> List[FrontierCluster3D]:
    """Create local downstairs confirmation anchors from fake-triggered voxels."""
    if len(fake_triggered) < 2:
        return []

    voxels = voxel_map.voxels
    n_cells = voxel_map.n_cells
    max_z = voxel_map.max_z_voxels
    agent_ix, agent_iy, agent_iz = agent_voxel
    sim_cache = voxel_map.similarity_cache

    inflated_connectors = {
        vid for vid in cluster_set
        if vid in voxels and voxels[vid].inflated
    }
    anchors: List[FrontierCluster3D] = []
    # Cluster real descend seeds together with inflated connectors so the
    # connector semantics survive this local-anchor construction stage too.
    # Components and tiles without enough real seeds remain unable to trigger.
    for connected in _cluster_26_connected(fake_triggered | inflated_connectors):
        if not connected & fake_triggered:
            continue
        for cluster_members in _split_large_fake_trigger_cluster(connected):
            seed_members = cluster_members & fake_triggered
            if len(seed_members) < 2:
                continue
            if not _has_downward_fake_space(seed_members, voxels):
                continue

            nav_vid = _find_local_fake_nav_voxel(seed_members, voxels, agent_voxel)
            if nav_vid is None:
                continue

            aabb = _compute_aabb(cluster_members)
            centroid_voxel = (
                int(round(np.mean([v[0] for v in cluster_members]))),
                int(round(np.mean([v[1] for v in cluster_members]))),
                int(round(np.mean([v[2] for v in cluster_members]))),
            )
            centroid_metric = voxel_map.voxel_to_metric(
                centroid_voxel[0], centroid_voxel[1], centroid_voxel[2]
            )
            floor_iz = int(round(np.median([v[2] for v in cluster_members])))
            local_score = _metric_euclidean_priority(
                voxel_map, agent_voxel, nav_vid
            )

            anchors.append(FrontierCluster3D(
                nav_voxel_id=nav_vid,
                centroid_metric=centroid_metric,
                cluster_voxels=cluster_members,
                frontier_score=local_score,
                cluster_size=len(cluster_members),
                floor_iz=floor_iz,
                aabb=aabb,
                is_stair_adjacent=True,
                is_stair_frontier=True,
                fake_triggered_voxels=set(seed_members),
                fake_anchor_aabb=aabb,
            ))

    return anchors


def check_fake_frontier_has_stair_below(
    cluster: FrontierCluster3D,
    voxel_map: "VoxelMap",
    depth_voxels: int = 6,
) -> bool:
    """
    Check if a stair frontier has STAIR/STAIR_EXPANDED voxels below its TRAVERSABLE_FAKE neighbors.
    If so, the frontier is no longer "potential" — stairs are already confirmed nearby.

    Returns True if STAIR voxels found below (frontier should be removed).
    """
    from mapping.voxel_map import VoxelType, LOCKED_TYPES

    if not cluster.is_stair_frontier:
        return False

    voxels = voxel_map.voxels
    N4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Sample some cluster voxels and check their TRAVERSABLE_FAKE neighbors
    sample = list(cluster.cluster_voxels)[:10]
    for sv in sample:
        if sv not in voxels:
            continue
        ix, iy, iz = sv
        for dx, dy in N4:
            nb = (ix + dx, iy + dy, iz)
            if nb not in voxels:
                continue
            nb_vd = voxels[nb]
            if nb_vd.voxel_type == VoxelType.TRAVERSABLE_FAKE:
                # Check below this fake voxel for STAIR/STAIR_EXPANDED
                for dz in range(1, depth_voxels + 1):
                    below = (nb[0], nb[1], nb[2] - dz)
                    if below in voxels and voxels[below].voxel_type in LOCKED_TYPES:
                        return True
    return False


def create_stair_semantic_frontier(
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    direction: str = "down",
    exclude_regions: Optional[List[Dict[str, object]]] = None,
    include_voxels: Optional[Set[Tuple[int, int, int]]] = None,
    lateral_corridor: Optional[Dict[str, object]] = None,
) -> Optional[StairSemanticFrontier3D]:
    """
    Create a semantic stair frontier in stair_sem mode.

    Strategy: cluster all STAIR voxels, find the cluster layer with maximum
    z-progress in the requested direction. For downstairs this is the lowest
    layer below the agent; for upstairs this is the highest layer above it.

    Returns None if no suitable STAIR cluster found.
    """
    from mapping.voxel_map import VoxelType

    voxels = voxel_map.voxels
    agent_ix, agent_iy, agent_iz = agent_voxel

    from mapping.voxel_map import LOCKED_TYPES

    def is_excluded(vid: Tuple[int, int, int]) -> bool:
        if not exclude_regions:
            return False
        for region in exclude_regions:
            center = region.get("center")
            if center is None:
                continue
            z_min = int(region.get("z_min", -10**9))
            z_max = int(region.get("z_max", 10**9))
            if not (z_min <= vid[2] <= z_max):
                continue
            radius_xy = int(region.get("radius_xy", 0))
            dx = int(vid[0]) - int(center[0])
            dy = int(vid[1]) - int(center[1])
            # Recent downstairs regions are created and matched elsewhere as
            # circular XY neighborhoods.  Using Manhattan distance here left
            # diagonal points from the completed flight unfiltered, allowing a
            # nearby landing remnant to outrank the genuinely lower stair
            # component.  Compare squared Euclidean distance to preserve the
            # same circular region without an unnecessary square root.
            if dx * dx + dy * dy <= radius_xy * radius_xy:
                return True
        return False

    def is_inside_lateral_corridor(vid: Tuple[int, int, int]) -> bool:
        if not lateral_corridor:
            return True
        width_axis = lateral_corridor.get("width_axis")
        if width_axis is None or len(width_axis) != 2:
            return True
        lateral = (
            float(vid[0]) * float(width_axis[0]) +
            float(vid[1]) * float(width_axis[1])
        )
        low = float(lateral_corridor.get("low", -float("inf")))
        high = float(lateral_corridor.get("high", float("inf")))
        margin = float(lateral_corridor.get("margin", 0.0))
        return low - margin <= lateral <= high + margin

    # Collect all STAIR + STAIR_EXPANDED voxels
    stair_voxels: Set[Tuple[int, int, int]] = set()
    for vid, vd in voxels.items():
        if include_voxels is not None and vid not in include_voxels:
            continue
        if vd.voxel_type in LOCKED_TYPES:
            if is_excluded(vid) or not is_inside_lateral_corridor(vid):
                continue
            stair_voxels.add(vid)

    if len(stair_voxels) < 3:
        return None

    # Cluster stairs without allowing STAIR_EXPANDED voxels to bridge separate
    # SAM-confirmed STAIR bodies. Expanded voxels are support/context, not
    # evidence that two stair segments belong to the same semantic component.
    clusters = cluster_stair_voxels_without_expanded_bridges(stair_voxels, voxels)
    if not clusters:
        return None

    # Select the nearest directionally valid stair segment. A previously mapped
    # flight may contain many more voxels than the newly observed next flight;
    # cluster size therefore cannot be the primary key in a multi-floor scene.
    # Require a reliable target layer before ranking, then use physical 3-D
    # distance to the direction-valid part of the cluster. Size is only a
    # deterministic stability tie-breaker.
    STAIR_LAYER_MIN_SIZE = 6
    directional_clusters = []
    for cluster_set in clusters:
        if len(cluster_set) < 3:
            continue
        if direction == "up":
            zs_in_direction = sorted(
                set(v[2] for v in cluster_set if v[2] > agent_iz),
                reverse=True,
            )
        else:
            zs_in_direction = sorted(set(v[2] for v in cluster_set if v[2] < agent_iz))
        if not zs_in_direction:
            continue

        layer_counts: Dict[int, int] = {}
        for _, _, iz in cluster_set:
            layer_counts[iz] = layer_counts.get(iz, 0) + 1
        reliable_zs = [
            z_val for z_val in zs_in_direction
            if layer_counts.get(z_val, 0) >= STAIR_LAYER_MIN_SIZE
        ]
        if not reliable_zs:
            continue

        directional_voxels = [
            v for v in cluster_set
            if v[2] in reliable_zs
        ]
        distance_m = min(
            (
                ((v[0] - agent_ix) * voxel_map.voxel_size) ** 2 +
                ((v[1] - agent_iy) * voxel_map.voxel_size) ** 2 +
                ((v[2] - agent_iz) * voxel_map.voxel_size_z) ** 2
            ) ** 0.5
            for v in directional_voxels
        )
        directional_clusters.append((
            distance_m,
            -len(cluster_set),
            min(cluster_set),
            cluster_set,
            reliable_zs,
        ))

    best_cluster = None
    candidate_zs = None
    if directional_clusters:
        _, _, _, best_cluster, candidate_zs = min(
            directional_clusters,
            key=lambda item: (item[0], item[1], item[2]),
        )

    if best_cluster is None or candidate_zs is None:
        return None

    # From the cluster, find the target layer in the requested z direction.
    # This avoids the median-based approach which fails when the cluster spans
    # both above and below the agent (median ≈ agent_z).
    target_layer = None
    target_z = None
    for z_val in candidate_zs:
        layer = [v for v in best_cluster if v[2] == z_val]
        if len(layer) >= STAIR_LAYER_MIN_SIZE:
            target_layer = layer
            target_z = z_val
            break

    if target_layer is None:
        return None

    # Use the extreme layer together with the layer two voxels toward the
    # agent for XY centering. A one-layer offset often samples the same stair
    # edge; two layers provide a wider cross-section without allowing all z
    # layers to bias the projected center.
    companion_z = target_z - 2 if direction == "up" else target_z + 2
    companion_layer = [v for v in best_cluster if v[2] == companion_z]
    if len(companion_layer) >= 3:
        target_region = list(target_layer) + companion_layer
    else:
        target_region = list(target_layer)

    # z_diff: positive = below, negative = above
    best_z_diff = float(agent_iz - target_z)

    # Project only SAM-confirmed STAIR voxels to XY when computing the target
    # center.  STAIR_EXPANDED remains part of layer reliability and can still
    # be the final target voxel, but wall/step expansion must not pull the
    # semantic frontier away from the observed stair body's middle.
    center_region = [
        v for v in target_region
        if voxels.get(v) is not None and voxels[v].voxel_type == VoxelType.STAIR
    ]
    # Preserve the existing expanded-only compatibility path when no confirmed
    # STAIR voxel exists in either selected layer.
    if not center_region:
        center_region = target_region

    # Restrict the STAIR points used by the 2-D center to the current flight's
    # direction corridor.  Per-height medians describe how the stair center
    # moves in XY as z changes; their z-correlated axis is the travel direction
    # and the perpendicular axis is the stair width.  Median layer bounds make
    # the corridor insensitive to a few side/riser mislabels.  Sparse, turning,
    # or weakly directional stairs automatically retain the unfiltered region.
    stair_cluster_points = [
        v for v in best_cluster
        if (
            voxels.get(v) is not None and
            voxels[v].voxel_type == VoxelType.STAIR and
            min(target_z, agent_iz) <= v[2] <= max(target_z, agent_iz)
        )
    ]
    stair_layers: Dict[int, Set[Tuple[int, int]]] = {}
    for x_val, y_val, z_val in stair_cluster_points:
        stair_layers.setdefault(z_val, set()).add((x_val, y_val))

    layer_records = []
    for z_val, xy_values in stair_layers.items():
        if len(xy_values) < 3:
            continue
        xy_array = np.asarray(sorted(xy_values), dtype=np.float64)
        layer_records.append((
            int(z_val),
            xy_array,
            np.median(xy_array, axis=0),
        ))
    layer_records.sort(key=lambda item: item[0])

    if len(layer_records) >= 3:
        layer_zs = np.asarray([item[0] for item in layer_records], dtype=np.float64)
        layer_centers = np.stack([item[2] for item in layer_records], axis=0)
        z_centered = layer_zs - float(np.mean(layer_zs))
        z_energy = float(np.dot(z_centered, z_centered))
        if z_energy > 1e-6:
            xy_centered = layer_centers - np.mean(layer_centers, axis=0, keepdims=True)
            slope_xy = (z_centered[:, None] * xy_centered).sum(axis=0) / z_energy
            slope_norm = float(np.linalg.norm(slope_xy))
            if slope_norm >= 0.25:
                travel_axis = slope_xy / slope_norm
                width_axis = np.asarray(
                    [-travel_axis[1], travel_axis[0]],
                    dtype=np.float64,
                )
                longitudinal = layer_centers @ travel_axis
                if float(np.std(longitudinal)) > 1e-6:
                    direction_corr = float(abs(np.corrcoef(layer_zs, longitudinal)[0, 1]))
                else:
                    direction_corr = 0.0
                travel_span = float(np.ptp(longitudinal))

                layer_lateral_centers = layer_centers @ width_axis
                lateral_center = float(np.median(layer_lateral_centers))
                center_residual = np.abs(layer_lateral_centers - lateral_center)

                # A straight local flight should show ordered z progress along
                # one XY axis and only modest lateral drift.  If this evidence
                # is weak, preserve the existing STAIR-only median behavior.
                direction_reliable = (
                    direction_corr >= 0.60 and
                    travel_span >= 2.0 and
                    float(np.percentile(center_residual, 90)) <= 3.0
                )
                if direction_reliable:
                    layer_lows = []
                    layer_highs = []
                    # Prefer middle height layers for width estimation: the
                    # extreme layers are the most likely to include a landing
                    # or an incomplete first/last SAM observation.
                    width_records = (
                        layer_records[1:-1]
                        if len(layer_records) >= 5
                        else layer_records
                    )
                    for _, xy_array, _ in width_records:
                        lateral = xy_array @ width_axis
                        layer_lows.append(float(np.percentile(lateral, 10)))
                        layer_highs.append(float(np.percentile(lateral, 90)))

                    corridor_low = float(np.median(layer_lows))
                    corridor_high = float(np.median(layer_highs))
                    if corridor_high < corridor_low:
                        corridor_low, corridor_high = corridor_high, corridor_low
                    if corridor_high - corridor_low < 2.0:
                        corridor_mid = 0.5 * (corridor_low + corridor_high)
                        corridor_low = corridor_mid - 1.0
                        corridor_high = corridor_mid + 1.0

                    # One voxel of tolerance retains genuine edge steps while
                    # rejecting side observations outside the consensus width.
                    corridor_margin = 1.0
                    filtered_center_region = [
                        v for v in center_region
                        if (
                            corridor_low - corridor_margin <=
                            float(np.dot(np.asarray(v[:2], dtype=np.float64), width_axis)) <=
                            corridor_high + corridor_margin
                        )
                    ]
                    if len({(v[0], v[1]) for v in filtered_center_region}) >= 3:
                        center_region = filtered_center_region

    unique_xy_target = set((v[0], v[1]) for v in center_region)
    # Coordinate-wise median is deliberately used instead of the arithmetic
    # mean so a few edge/outlier STAIR observations cannot pull the semantic
    # target away from the main stair band.
    cx = int(round(np.median([xy[0] for xy in unique_xy_target])))
    cy = int(round(np.median([xy[1] for xy in unique_xy_target])))
    centroid = (cx, cy, target_z)
    # Snap to the actual stair voxel nearest to the projected 2-D center.
    # Z is only a deterministic tie-breaker; it no longer contributes a
    # continuous weight to target selection.
    target_vid = min(
        target_region,
        key=lambda v: (
            (v[0] - cx) ** 2 + (v[1] - cy) ** 2,
            abs(v[2] - target_z),
            v,
        ),
    )

    from mapping.voxel_map import TARGETABLE_TYPES as _TT

    def is_directional_target(vid: Tuple[int, int, int]) -> bool:
        if direction == "up":
            return vid[2] > agent_iz
        return vid[2] < agent_iz

    # Find nearest non-FAKE targetable voxel that A* can reach in the requested
    # z direction. This prevents upstairs mode from snapping to a lower-floor
    # stair/nav voxel, and downstairs mode from snapping back upward.
    if (
        target_vid in voxels
        and voxels[target_vid].voxel_type in _TT
        and not voxels[target_vid].inflated
        and is_directional_target(target_vid)
    ):
        nav_vid = target_vid
    else:
        nav_vid = None
        best_dist = float("inf")
        max_radius = 10
        tx, ty, tz = target_vid
        for dx in range(-max_radius, max_radius + 1):
            for dy in range(-max_radius, max_radius + 1):
                for dz in range(-max_radius, max_radius + 1):
                    cand = (tx + dx, ty + dy, tz + dz)
                    if (
                        not is_directional_target(cand) or
                        not is_inside_lateral_corridor(cand)
                    ):
                        continue
                    vd = voxels.get(cand)
                    if vd is None or vd.voxel_type not in _TT or vd.inflated:
                        continue
                    dist = dx * dx + dy * dy + dz * dz
                    if dist < best_dist:
                        best_dist = dist
                        nav_vid = cand
        if nav_vid is None:
            return None

    # Cluster center — 投影到2D平面，每个(x,y)只计一次，避免z堆叠拉偏
    unique_xy_cluster = set((v[0], v[1]) for v in best_cluster)
    center_x = int(round(np.mean([xy[0] for xy in unique_xy_cluster])))
    center_y = int(round(np.mean([xy[1] for xy in unique_xy_cluster])))
    # z取目标层（而非全簇均值，因为全簇可能跨多层）
    center = (center_x, center_y, target_z)

    # Very high score to ensure top priority
    semantic_score = 10000.0 + abs(best_z_diff) * 100.0

    return StairSemanticFrontier3D(
        nav_voxel_id=nav_vid,
        stair_cluster_center=center,
        z_difference=best_z_diff,
        semantic_score=semantic_score,
    )


def _cluster_26_connected(
    candidates: Set[Tuple[int, int, int]]
) -> List[Set[Tuple[int, int, int]]]:
    """26-connected region growing to cluster frontier candidates."""
    unvisited = set(candidates)
    clusters = []

    neighbors_26 = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    while unvisited:
        seed = unvisited.pop()
        queue = deque([seed])
        cluster = {seed}

        while queue:
            current = queue.popleft()
            for dx, dy, dz in neighbors_26:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    cluster.add(neighbor)
                    queue.append(neighbor)

        clusters.append(cluster)

    return clusters


def cluster_stair_voxels_without_expanded_bridges(
    candidates: Set[Tuple[int, int, int]],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    min_seed_size: int = 12,
    merge_small_seed_radius_xy: int = 3,
    merge_small_seed_radius_z: int = 3,
) -> List[Set[Tuple[int, int, int]]]:
    """
    Cluster STAIR/STAIR_EXPANDED while preventing expansion-only bridges.

    The primary components are built only from SAM-confirmed STAIR voxels.
    Small STAIR fragments are merged into a nearby larger seed when possible.
    STAIR_EXPANDED voxels are then assigned to one component they touch, but
    never merge two STAIR components together.
    """
    from mapping.voxel_map import VoxelType

    candidates = set(candidates)
    stair_voxels = {
        vid for vid in candidates
        if voxels.get(vid) is not None and voxels[vid].voxel_type == VoxelType.STAIR
    }
    expanded_voxels = {
        vid for vid in candidates
        if voxels.get(vid) is not None and voxels[vid].voxel_type == VoxelType.STAIR_EXPANDED
    }

    if not stair_voxels:
        clusters = _cluster_26_connected(expanded_voxels)
        clusters.sort(key=len, reverse=True)
        return clusters

    seed_clusters = _cluster_26_connected(stair_voxels)
    seed_clusters.sort(key=len, reverse=True)

    def aabb(cluster: Set[Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        xs = [v[0] for v in cluster]
        ys = [v[1] for v in cluster]
        zs = [v[2] for v in cluster]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def aabb_gap(
        a: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        b: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    ) -> Tuple[int, int]:
        xy_gap = max(
            max(a[0][0] - b[1][0] - 1, b[0][0] - a[1][0] - 1, 0),
            max(a[0][1] - b[1][1] - 1, b[0][1] - a[1][1] - 1, 0),
        )
        z_gap = max(a[0][2] - b[1][2] - 1, b[0][2] - a[1][2] - 1, 0)
        return xy_gap, z_gap

    large_seeds: List[Set[Tuple[int, int, int]]] = [
        set(c) for c in seed_clusters if len(c) >= min_seed_size
    ]
    small_seeds: List[Set[Tuple[int, int, int]]] = [
        set(c) for c in seed_clusters if len(c) < min_seed_size
    ]
    if not large_seeds:
        large_seeds = [set(seed_clusters[0])]
        small_seeds = [set(c) for c in seed_clusters[1:]]

    for small in small_seeds:
        small_aabb = aabb(small)
        best_idx = None
        best_key = None
        for idx, large in enumerate(large_seeds):
            xy_gap, z_gap = aabb_gap(small_aabb, aabb(large))
            if xy_gap > merge_small_seed_radius_xy or z_gap > merge_small_seed_radius_z:
                continue
            key = (xy_gap + z_gap, -len(large))
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            large_seeds.append(set(small))
        else:
            large_seeds[best_idx].update(small)

    result = [set(c) for c in large_seeds]
    component_index: Dict[Tuple[int, int, int], int] = {}
    for idx, cluster in enumerate(result):
        for vid in cluster:
            component_index[vid] = idx

    neighbors_26 = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    assigned_expanded: Set[Tuple[int, int, int]] = set()
    for vid in expanded_voxels:
        touching = set()
        x, y, z = vid
        for dx, dy, dz in neighbors_26:
            nb = (x + dx, y + dy, z + dz)
            if nb in component_index:
                touching.add(component_index[nb])
        if not touching:
            continue
        # Expanded voxels that touch multiple components attach to one component
        # only. This is the key rule that prevents expanded-expanded or
        # expanded-mediated bridges from fusing separate stair/wall regions.
        best_idx = max(touching, key=lambda idx: len(result[idx]))
        result[best_idx].add(vid)
        assigned_expanded.add(vid)

    orphan_expanded = expanded_voxels - assigned_expanded
    if orphan_expanded:
        result.extend(_cluster_26_connected(orphan_expanded))

    result.sort(key=len, reverse=True)
    return result


def _check_stair_adjacency_cluster(
    cluster_voxels: Set[Tuple[int, int, int]],
    voxels: dict,
) -> bool:
    """Return True if any voxel in the cluster is adjacent to a STAIR/STAIR_EXPANDED voxel."""
    from mapping.voxel_map import VoxelType, LOCKED_TYPES

    # Only check a sample for performance
    sample = list(cluster_voxels)[:10]
    neighbors_6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    for sv in sample:
        for dx, dy, dz in neighbors_6:
            nb = (sv[0]+dx, sv[1]+dy, sv[2]+dz)
            if nb in voxels and voxels[nb].voxel_type in LOCKED_TYPES:
                return True
    return False


def _find_best_nav_voxel_in_cluster(
    cluster_set: Set[Tuple[int, int, int]],
    voxels: dict,
    agent_voxel: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int]]:
    """Find a non-inflated non-FAKE targetable voxel in/near cluster closest to agent."""
    from mapping.voxel_map import TARGETABLE_TYPES

    # First try: cluster voxels that are non-inflated targetable
    best_vid = None
    best_dist = float('inf')
    for vid in cluster_set:
        if vid in voxels and voxels[vid].voxel_type in TARGETABLE_TYPES:
            if not voxels[vid].inflated:
                dist = abs(vid[0] - agent_voxel[0]) + abs(vid[1] - agent_voxel[1])
                if dist < best_dist:
                    best_dist = dist
                    best_vid = vid

    # No inflated fallback: inflated voxels are cluster connectors only and
    # must never become nav_voxel_id.
    return best_vid


def _aabb_intersect(
    a: Tuple[Tuple[int,int,int], Tuple[int,int,int]],
    b: Tuple[Tuple[int,int,int], Tuple[int,int,int]],
) -> bool:
    """Check if two AABBs intersect."""
    return (a[0][0] <= b[1][0] and a[1][0] >= b[0][0] and
            a[0][1] <= b[1][1] and a[1][1] >= b[0][1] and
            a[0][2] <= b[1][2] and a[1][2] >= b[0][2])


def _bfs_similarity_score(
    start: Tuple[int, int, int],
    voxels: dict,
    sim_cache: Dict[Tuple[int, int, int], float],
    max_depth: int,
    n_cells: int,
    max_z: int,
) -> float:
    """
    BFS from `start` into UNKNOWN space (not in voxels dict).
    Accumulates similarity from OCCUPIED voxels adjacent to each explored unknown cell.

    Mirrors C++ compute_reachable_area() in planning_utils_cpp:
      - BFS through unknown (unexplored) space up to max_depth steps
      - At each unknown cell, check 6-neighbors for OCCUPIED voxels
      - Sum their similarity_cache scores

    This scores "how much target-related content is likely behind this frontier".
    """
    from mapping.voxel_map import VoxelType

    neighbors_6 = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    total_score = 0.0
    visited: Set[Tuple[int, int, int]] = {start}
    queue = [start]
    depth = 0

    while queue and depth < max_depth:
        next_queue = []
        for cell in queue:
            for dx, dy, dz in neighbors_6:
                nb = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                if nb in visited:
                    continue
                if not (0 <= nb[0] < n_cells and 0 <= nb[1] < n_cells and 0 <= nb[2] < 2 * max_z):
                    continue
                visited.add(nb)
                if nb not in voxels:
                    # Unknown cell: accumulate similarity from its occupied neighbors
                    for dx2, dy2, dz2 in neighbors_6:
                        occ = (nb[0] + dx2, nb[1] + dy2, nb[2] + dz2)
                        if occ in voxels and voxels[occ].voxel_type in (
                            VoxelType.OCCUPIED, VoxelType.STAIR, VoxelType.STAIR_EXPANDED
                        ):
                            sim = sim_cache.get(occ, 0.0)
                            if sim > 0:
                                total_score += sim
                    next_queue.append(nb)
                # Don't pass through OCCUPIED/TRAVERSABLE voxels (they block BFS)
        queue = next_queue
        depth += 1

    return total_score


# ---------------------------------------------------------------------------
# 3D similarity clustering (POI exploitation channel)
# ---------------------------------------------------------------------------

def cluster_high_similarity_3d(
    voxel_map: "VoxelMap",
    agent_voxel: Tuple[int, int, int],
    min_similarity: float = 0.15,
    candidate_percentile: Optional[float] = None,
    candidate_min_similarity: float = 0.04,
    seed_percentile: Optional[float] = None,
    seed_min_similarity: float = 0.06,
    neighborhood_radius: int = 3,
    min_cluster_size: int = 2,
    relative_threshold: float = 0.80,
    max_clusters: int = 15,
    search_radius: int = 80,
    seed_voxels: Optional[Set[Tuple[int, int, int]]] = None,
    poi_type: str = "history",
) -> List[Cluster3D]:
    """
    3D region-growing clustering of high-similarity OCCUPIED voxels.

    Mirrors cluster_high_similarity_regions() from clustering.py but in 3D:
      1. Find local maxima in 3D similarity_cache (max in Chebyshev neighborhood)
      2. Region-grow from each maximum using relative threshold
      3. Return Cluster3D objects sorted by cluster_score descending
    """
    from mapping.voxel_map import VoxelType

    sim_cache = voxel_map.similarity_cache
    voxels = voxel_map.voxels
    agent_ix, agent_iy, agent_iz = agent_voxel

    if not sim_cache:
        return []

    eligible_scores = []
    for vid, score in sim_cache.items():
        if vid not in voxels:
            continue
        if voxels[vid].voxel_type != VoxelType.OCCUPIED:
            continue
        if (abs(vid[0] - agent_ix) > search_radius or
                abs(vid[1] - agent_iy) > search_radius):
            continue
        eligible_scores.append(float(score))

    if not eligible_scores:
        return []

    if candidate_percentile is not None:
        candidate_threshold = max(
            float(np.percentile(eligible_scores, candidate_percentile)),
            float(candidate_min_similarity),
        )
    else:
        candidate_threshold = float(min_similarity)
    if seed_percentile is not None:
        seed_threshold = max(
            float(np.percentile(eligible_scores, seed_percentile)),
            float(seed_min_similarity),
        )
    else:
        seed_threshold = max(float(min_similarity), float(seed_min_similarity))

    # Collect OCCUPIED voxels with sufficient similarity
    candidates = []
    for vid, score in sim_cache.items():
        if score < candidate_threshold:
            continue
        if vid not in voxels:
            continue
        if voxels[vid].voxel_type != VoxelType.OCCUPIED:
            continue
        if (abs(vid[0] - agent_ix) > search_radius or
                abs(vid[1] - agent_iy) > search_radius):
            continue
        candidates.append((vid, score))

    if not candidates:
        return []

    # Find local maxima in 3D
    candidate_dict = {vid_: score_ for vid_, score_ in candidates}
    local_maxima = []
    for vid_, score_ in candidates:
        if score_ < seed_threshold:
            continue
        if seed_voxels is not None and vid_ not in seed_voxels:
            close_to_seed = False
            ix0, iy0, iz0 = vid_
            for dx in range(-neighborhood_radius, neighborhood_radius + 1):
                for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                    for dz in range(-neighborhood_radius, neighborhood_radius + 1):
                        if (ix0 + dx, iy0 + dy, iz0 + dz) in seed_voxels:
                            close_to_seed = True
                            break
                    if close_to_seed:
                        break
                if close_to_seed:
                    break
            if not close_to_seed:
                continue
        is_max = True
        ix, iy, iz = vid_
        for dx in range(-neighborhood_radius, neighborhood_radius + 1):
            for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                for dz in range(-neighborhood_radius, neighborhood_radius + 1):
                    nb = (ix + dx, iy + dy, iz + dz)
                    if nb != vid_ and nb in candidate_dict:
                        if candidate_dict[nb] > score_:
                            is_max = False
                            break
                if not is_max:
                    break
            if not is_max:
                break
        if is_max:
            local_maxima.append((vid_, score_))

    local_maxima.sort(key=lambda x: x[1], reverse=True)

    # Region growing from each local maximum
    processed: Set[Tuple[int, int, int]] = set()
    clusters: List[Cluster3D] = []
    neighbors_26 = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for seed_vid, seed_score in local_maxima:
        if seed_vid in processed:
            continue

        threshold = max(candidate_threshold, seed_score * relative_threshold)
        stack = [seed_vid]
        cluster_vids = []
        max_score = seed_score
        best_vid = seed_vid

        while stack:
            curr = stack.pop()
            if curr in processed:
                continue
            curr_sim = candidate_dict.get(curr, -1.0)
            if curr_sim < threshold:
                continue
            processed.add(curr)
            cluster_vids.append(curr)

            if curr_sim > max_score:
                max_score = curr_sim
                best_vid = curr

            for dx, dy, dz in neighbors_26:
                nb = (curr[0] + dx, curr[1] + dy, curr[2] + dz)
                if nb not in processed and nb in candidate_dict:
                    stack.append(nb)

        if len(cluster_vids) < min_cluster_size:
            continue

        nav_vid = voxel_map._find_nearest_traversable(best_vid, max_radius=15)
        if nav_vid is None:
            continue

        scores = np.array([float(sim_cache.get(vid, 0.0)) for vid in cluster_vids], dtype=np.float32)
        topk = min(8, scores.size)
        topk_mean = float(np.mean(np.sort(scores)[-topk:])) if topk > 0 else 0.0
        mean_score = float(np.mean(scores)) if scores.size else 0.0
        confs = np.array([
            float(getattr(voxels[vid], "confidence", 0.0)) for vid in cluster_vids
        ], dtype=np.float32)
        if float(np.sum(confs)) > 1e-6:
            conf_weighted = float(np.average(scores, weights=np.maximum(confs, 1e-6)))
        else:
            conf_weighted = mean_score

        pts = np.array(cluster_vids, dtype=np.int32)
        bbox_min_arr = pts.min(axis=0)
        bbox_max_arr = pts.max(axis=0)
        xy_coverage = len({(int(v[0]), int(v[1])) for v in cluster_vids})
        coverage_score = min(1.0, xy_coverage / 20.0)
        span = np.maximum(bbox_max_arr - bbox_min_arr + 1, 1)
        volume = float(span[0] * span[1] * span[2])
        fill_ratio = min(1.0, len(cluster_vids) / max(volume, 1.0))
        aspect_penalty = max(float(span.max()) / max(float(span.min()), 1.0), 1.0)
        compactness_score = float(fill_ratio / min(aspect_penalty, 6.0))
        evidence_score = (
            0.35 * float(max_score) +
            0.35 * topk_mean +
            0.15 * conf_weighted +
            0.10 * coverage_score +
            0.05 * compactness_score
        )

        clusters.append(Cluster3D(
            nav_voxel_id=nav_vid,
            center_voxel=best_vid,
            cluster_score=evidence_score,
            floor_iz=best_vid[2],
            cluster_size=len(cluster_vids),
            max_similarity=float(max_score),
            topk_mean_similarity=topk_mean,
            mean_similarity=mean_score,
            confidence_weighted_similarity=conf_weighted,
            bbox_min=tuple(int(v) for v in bbox_min_arr),
            bbox_max=tuple(int(v) for v in bbox_max_arr),
            evidence_score=evidence_score,
            poi_type=poi_type,
            last_seen_iter=getattr(voxel_map, "_iters", 0),
        ))

        if len(clusters) >= max_clusters:
            break

    clusters.sort(key=lambda c: c.evidence_score, reverse=True)
    return clusters
