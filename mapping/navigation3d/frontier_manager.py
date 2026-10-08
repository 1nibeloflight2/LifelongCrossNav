from __future__ import annotations

import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from config import Conf, SpotControllerConf
from mapping.voxel_map import VoxelMap, VoxelType, VoxelData
from mapping.nav_goals.frontier_3d import (
    Frontier3D, FrontierCluster3D, Cluster3D, StairSemanticFrontier3D,
    RiseStairFrontier3D, detect_frontiers_3d_incremental,
    detect_frontiers_3d_full, cluster_high_similarity_3d, rescore_frontier,
    check_fake_frontier_has_stair_below, create_stair_semantic_frontier,
    cluster_stair_voxels_without_expanded_bridges,
)
from mapping.nav_goals.navigation_goals import NavGoal
from planning import Planning, Controllers
from .poi_manager import history_poi_enabled

try:
    import rerun as rr
    _RERUN = True
except ImportError:
    _RERUN = False


def _frontier_component_representatives(navigator, goal, agent_voxel):
    """Return one nearby real seed for each safe component of a frontier cluster.

    Inflated frontier voxels are deliberately allowed to connect seeds during
    clustering, but they are not A* targets. Removing those connectors can
    reveal several independent safe components. Trying one representative per
    component avoids both a false whole-cluster failure and redundant A* calls
    to many points in the same unreachable component. For descend clusters,
    ``fake_triggered_voxels`` is the authoritative set of real geometry seeds;
    inflated connector members must not become alternative targets.
    """
    voxels = navigator.one_map.voxels
    cluster_voxels = set(getattr(goal, "cluster_voxels", set()) or set())
    if getattr(goal, "is_stair_frontier", False):
        descend_seeds = set(
            getattr(goal, "fake_triggered_voxels", set()) or set()
        )
        if descend_seeds:
            cluster_voxels = descend_seeds
    seeds = {
        tuple(int(v) for v in vid)
        for vid in cluster_voxels
        if (
            tuple(int(v) for v in vid) in voxels and
            voxels[tuple(int(v) for v in vid)].voxel_type == VoxelType.TRAVERSABLE and
            not voxels[tuple(int(v) for v in vid)].inflated
        )
    }
    if not seeds:
        return []

    neighbors_26 = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    remaining = set(seeds)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for dx, dy, dz in neighbors_26:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)

    ax, ay, az = navigator.one_map.voxel_to_metric(*agent_voxel)

    def metric_distance(vid):
        x, y, z = navigator.one_map.voxel_to_metric(*vid)
        return float(((x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2) ** 0.5)

    representatives = []
    for component in components:
        representative = min(
            component,
            key=lambda vid: (metric_distance(vid), vid),
        )
        representatives.append((metric_distance(representative), representative))
    representatives.sort(key=lambda item: (item[0], item[1]))
    return representatives


def rank_frontier_candidates(navigator, candidates, agent_voxel):
    """Select the nearest reachable goal inside one frontier category.

    Basic, descend, and ascent retain distinct candidate validation and global
    category priority, but share one internal ordering: physical 3-D Euclidean
    distance from the agent. A* is attempted in that order.
    """
    ax, ay, az = navigator.one_map.voxel_to_metric(*agent_voxel)
    ordered = []
    for goal in candidates:
        goal_vid = tuple(int(v) for v in goal.nav_voxel_id)
        gx, gy, gz = navigator.one_map.voxel_to_metric(*goal_vid)
        distance_m = float(
            ((gx - ax) ** 2 + (gy - ay) ** 2 + (gz - az) ** 2) ** 0.5
        )
        diagnostic_score = 1.0 / (1.0 + distance_m)
        if isinstance(goal, FrontierCluster3D):
            goal.frontier_score = diagnostic_score
        elif isinstance(goal, RiseStairFrontier3D):
            goal.semantic_score = diagnostic_score
        ordered.append((distance_m, goal_vid, goal))

    ordered.sort(key=lambda item: (item[0], item[1]))
    if any(isinstance(goal, RiseStairFrontier3D) for _, _, goal in ordered):
        kind = "ascent"
    elif any(
        isinstance(goal, FrontierCluster3D) and goal.is_stair_frontier
        for _, _, goal in ordered
    ):
        kind = "descend"
    else:
        kind = "basic"

    for attempt, (distance_m, goal_vid, goal) in enumerate(ordered, start=1):
        path = navigator.one_map.a_star_3d(
            agent_voxel,
            goal_vid,
            max_iterations=25000,
            stair_mode=False,
            # Preserve ascent's stair-access planning preference; only the
            # candidate score/order is unified in this change.
            prefer_traversable=(kind != "ascent"),
            allow_inflated_from_inflated_start=True,
        )
        selected_vid = goal_vid
        selected_distance_m = distance_m

        if path is None and isinstance(goal, FrontierCluster3D):
            alternatives = _frontier_component_representatives(
                navigator,
                goal,
                agent_voxel,
            )
            for alternative_distance_m, alternative_vid in alternatives:
                if alternative_vid == goal_vid:
                    continue
                alternative_path = navigator.one_map.a_star_3d(
                    agent_voxel,
                    alternative_vid,
                    max_iterations=25000,
                    stair_mode=False,
                    prefer_traversable=True,
                    allow_inflated_from_inflated_start=True,
                )
                if alternative_path is None:
                    continue
                path = alternative_path
                selected_vid = alternative_vid
                selected_distance_m = alternative_distance_m
                goal.nav_voxel_id = alternative_vid
                goal.frontier_score = 1.0 / (1.0 + alternative_distance_m)
                print(
                    f"[Nav3D] {kind.capitalize()} frontier component fallback: "
                    f"failed={goal_vid}, using={alternative_vid}, "
                    f"components={len(alternatives)}, path_len={len(path)}"
                )
                break

        if path is None:
            # Keep the current candidate before moving to the next one. Distant
            # exploration frontiers can look disconnected while the map is
            # still sparse; an inflated air-class bridge lets the agent approach
            # and replan after nearby observations widen the safe region.
            path = navigator.one_map.a_star_3d(
                agent_voxel,
                goal_vid,
                max_iterations=25000,
                stair_mode=False,
                prefer_traversable=(kind != "ascent"),
                allow_inflated_traversable=True,
                allow_inflated_traversable_stair=True,
                allow_inflated_from_inflated_start=True,
                inflated_cost_multiplier=2.0,
            )
            if path is not None:
                print(
                    f"[Nav3D] Nearest-{kind} inflated-air fallback: "
                    f"goal={goal_vid}, penalty=2.0, path_len={len(path)}"
                )

        if path is None:
            continue

        goal._validated_path = path
        goal._validated_path_start = tuple(int(v) for v in agent_voxel)
        print(
            f"[Nav3D] Nearest-{kind} selected: goal={selected_vid}, "
            f"euclidean={selected_distance_m:.2f}m, "
            f"attempt={attempt}/{len(ordered)}, path_len={len(path)}"
        )
        return [goal]

    if ordered:
        print(
            f"[Nav3D] Nearest-{kind}: all {len(ordered)} candidates "
            f"failed A* reachability"
        )
    return []

class FrontierManagerMixin:
    def _blacklist_basic_frontier_region(
        self,
        goal: FrontierCluster3D,
    ) -> int:
        """Blacklist a failed basic frontier cluster so selection moves to the next one."""
        existing = set(tuple(int(v) for v in b) for b in self.blacklisted_goals)
        to_add: Set[Tuple[int, int, int]] = set()

        nav = tuple(int(v) for v in goal.nav_voxel_id)
        to_add.add(nav)

        for voxel in getattr(goal, "cluster_voxels", []) or []:
            try:
                to_add.add(tuple(int(v) for v in voxel))
            except Exception:
                continue

        added = 0
        for vid in to_add:
            if vid not in existing:
                self.blacklisted_goals.append(vid)
                existing.add(vid)
                added += 1
        return added

    @staticmethod
    def _is_basic_frontier_cluster_blacklisted(
        cluster: FrontierCluster3D,
        blacklist_set: Set[Tuple[int, int, int]],
    ) -> bool:
        if cluster.nav_voxel_id in blacklist_set:
            return True
        for voxel in getattr(cluster, "cluster_voxels", []) or []:
            try:
                if tuple(int(v) for v in voxel) in blacklist_set:
                    return True
            except Exception:
                continue
        return False

    def _check_frontier_arrival(
        self,
        agent_voxel: Tuple[int, int, int],
        stair_dets: Optional[dict] = None,
        image: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        odometry: Optional[np.ndarray] = None,
    ) -> None:
        """
        When agent is close to the current nav_goal (FrontierCluster3D), verify
        the cluster is still valid using TravExplorer column boundary check.
        If all horizontal 4-neighbors now have traversable columns, this frontier
        is no longer a boundary → agent has fully explored this area.

        Also handles StairSemanticFrontier3D arrival → force forward briefly.
        """
        if self.last_nav_goal is None:
            return

        if isinstance(self.last_nav_goal, Cluster3D):
            if self.navigation_mode == "navigate":
                if self._is_history_poi(self.last_nav_goal):
                    arrived, xy_dist_m, xyz_dist_m = self._history_poi_arrived(
                        agent_voxel, self.last_nav_goal
                    )
                    if arrived:
                        print(
                            f"[Nav3D] History POI arrival: nav={self.last_nav_goal.nav_voxel_id}, "
                            f"nav_xy_dist={xy_dist_m:.2f}m, nav_xyz_dist={xyz_dist_m:.2f}m"
                        )
                        self._start_poi_confirm_scan(self.last_nav_goal, odometry)
                else:
                    dist_xy = self._poi_cluster_xy_distance(agent_voxel, self.last_nav_goal)
                    if dist_xy <= self._poi_blacklist_radius_xy:
                        self._start_poi_confirm_scan(self.last_nav_goal, odometry)
            return

        # Handle RiseStairFrontier3D arrival: confirm with SAM and enter upward stair sem mode.
        if isinstance(self.last_nav_goal, RiseStairFrontier3D):
            goal_vid = self.last_nav_goal.nav_voxel_id
            dist_xy = self._xy_distance(agent_voxel, goal_vid)
            z_progressed = agent_voxel[2] >= goal_vid[2] + 2
            arrival_reached = dist_xy <= 6 or (z_progressed and dist_xy <= 8)
            if dist_xy <= 8:
                print(
                    f"[Nav3D] Rise frontier arrival check: agent={agent_voxel}, "
                    f"goal={goal_vid}, dist_xy={dist_xy}, "
                    f"z_progressed={z_progressed}, reached={arrival_reached}"
                )
            if arrival_reached and stair_dets is not None and image is not None and depth is not None and odometry is not None:
                confirm_result = self._confirm_rise_stair_frontier(
                    self.last_nav_goal, stair_dets, image, depth, odometry, agent_voxel
                )
                if confirm_result is True:
                    print(f"[Nav3D] === StairSem-ascent === at ascent frontier {goal_vid}")
                elif confirm_result is False:
                    self.blacklisted_goals.append(goal_vid)
                    self.rise_stair_candidates.pop(goal_vid, None)
                    self.path = None
                    self.last_nav_goal = None
                else:
                    self.path = None
            return

        # Handle StairSemanticFrontier3D arrival
        if isinstance(self.last_nav_goal, StairSemanticFrontier3D):
            goal_vid = self.last_nav_goal.nav_voxel_id
            dx_m = (agent_voxel[0] - goal_vid[0]) * float(self.one_map.voxel_size)
            dy_m = (agent_voxel[1] - goal_vid[1]) * float(self.one_map.voxel_size)
            dz_m = (agent_voxel[2] - goal_vid[2]) * float(self.one_map.voxel_size_z)
            dist_3d_m = float(np.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m))
            if dist_3d_m <= 0.40:
                # Arrived at semantic stair frontier → force forward before rotating/replanning.
                if self._stair_sem_force_forward == 0:
                    if self._stair_sem_direction == "up":
                        steps = int(getattr(self, "_stair_sem_force_forward_steps_up", 3))
                    else:
                        steps = int(getattr(self, "_stair_sem_force_forward_steps", 3))
                    print(f"[Nav3D] Arrived at stair semantic frontier {goal_vid} — forcing {steps} forward steps")
                    self._stair_sem_force_forward = steps
            return

        if not isinstance(self.last_nav_goal, FrontierCluster3D):
            return

        goal_vid = self.last_nav_goal.nav_voxel_id
        # Check proximity — within 4 voxels Euclidean XY distance
        arrival_radius = 4
        dist_xy = self._xy_distance(agent_voxel, goal_vid)
        if dist_xy > arrival_radius:
            return

        # Agent is near the goal — verify column boundary condition still holds
        voxels = self.one_map.voxels
        n_cells = self.one_map.n_cells
        max_z = self.one_map.max_z_voxels
        N4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Check 1: Is nav_voxel still PATHABLE?
        from mapping.voxel_map import TARGETABLE_TYPES
        if goal_vid not in voxels or voxels[goal_vid].voxel_type not in TARGETABLE_TYPES:
            self._invalidate_current_goal("nav_voxel no longer traversable")
            return

        # Stair fake frontiers are confirmed by the selected-anchor look-down
        # flow. A remembered stair entrance may no longer be a literal unknown
        # boundary by the time it is revisited, so do not invalidate it with the
        # ordinary frontier boundary test.
        if self.last_nav_goal.is_stair_frontier:
            return

        # Check 2: Does the cluster still have an undefined or TRAVERSABLE_FAKE 4-neighbor?
        still_boundary = False
        ix, iy, iz = goal_vid
        cluster_voxels = self.last_nav_goal.cluster_voxels
        sample = list(cluster_voxels)[:10]
        for sv in sample:
            if sv not in voxels:
                continue
            if voxels[sv].voxel_type != VoxelType.TRAVERSABLE:
                continue
            if voxels[sv].inflated:
                continue
            for dx, dy in N4:
                nx, ny = sv[0] + dx, sv[1] + dy
                if not (0 <= nx < n_cells and 0 <= ny < n_cells):
                    still_boundary = True
                    break
                neighbor = (nx, ny, sv[2])
                if neighbor not in voxels:
                    still_boundary = True
                    break
                # TRAVERSABLE_FAKE neighbor also counts as boundary
                if neighbor in voxels and voxels[neighbor].voxel_type == VoxelType.TRAVERSABLE_FAKE:
                    still_boundary = True
                    break
            if still_boundary:
                break

        if not still_boundary:
            self._invalidate_current_goal("no undefined neighbors remaining on arrival")
            return

    def _find_alt_nav_voxel(
        self,
        cluster: FrontierCluster3D,
        blacklist_set: set,
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find an alternative navigation voxel for a cluster whose nav_voxel_id
        is blacklisted. Searches cluster voxels for a non-inflated PATHABLE
        one that is not blacklisted.
        """
        from mapping.voxel_map import TARGETABLE_TYPES
        voxels = self.one_map.voxels

        # Try cluster voxels directly
        for vid in cluster.cluster_voxels:
            if vid in blacklist_set:
                continue
            if vid in voxels and voxels[vid].voxel_type in TARGETABLE_TYPES \
               and not voxels[vid].inflated:
                return vid

        # Try neighbors of cluster voxels
        for vid in list(cluster.cluster_voxels)[:20]:  # limit search
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nb = (vid[0] + dx, vid[1] + dy, vid[2])
                    if nb in blacklist_set:
                        continue
                    if nb in voxels and voxels[nb].voxel_type in TARGETABLE_TYPES \
                       and not voxels[nb].inflated:
                        return nb
        return None

    def _debug_no_nav_goals(self, agent_voxel: Tuple[int, int, int]) -> None:
        """Print compact diagnostics when frontier/POI selection yields no goal."""
        try:
            basic = sum(1 for c in self.frontier_clusters if not c.is_stair_frontier)
            down = sum(1 for c in self.frontier_clusters if c.is_stair_frontier)
            visible_history = self._visible_history_pois() if self.allow_poi else []
            live = list(self.live_pois.values())
            mem_counts = {"active": 0, "consumed": 0, "blocked": 0, "other": 0}
            for mem in self.stair_frontier_memory.values():
                status = str(mem.get("status", "other"))
                mem_counts[status if status in mem_counts else "other"] += 1

            blacklist_set = set(tuple(b) if not isinstance(b, tuple) else b for b in self.blacklisted_goals)
            poi_failed = []
            poi_reachable = 0
            for poi in (live + visible_history)[:8]:
                if poi.nav_voxel_id in blacklist_set or self._is_poi_cluster_blacklisted(poi):
                    continue
                is_history_poi = "history" in str(getattr(poi, "poi_type", "")).lower()
                path = self.one_map.a_star_3d(
                    agent_voxel,
                    poi.nav_voxel_id,
                    max_iterations=25000,
                    stair_mode=False,
                    prefer_traversable=False,
                    allow_inflated_traversable=is_history_poi,
                    allow_inflated_stair=not is_history_poi,
                    allow_inflated_traversable_stair=True,
                    connect_traversable_stair_drop=True,
                )
                if path is None:
                    poi_failed.append((getattr(poi, "poi_type", "?"), poi.nav_voxel_id))
                else:
                    poi_reachable += 1
            active_rise = sum(1 for c in self.rise_stair_candidates.values() if c.get("active", False))
            print(
                "[Nav3D][NoGoalDebug] "
                f"raw_basic={basic}, raw_down={down}, "
                f"live_poi={len(live)}, history_poi_visible={len(visible_history)}, "
                f"rise_active={active_rise}, "
                f"ascent_memory(active/blocked/other)="
                f"{mem_counts['active']}/{mem_counts['blocked']}/{mem_counts['other']}, "
                f"poi_reachable={poi_reachable}, "
                f"blacklisted={len(blacklist_set)}"
            )
            if poi_failed:
                print(f"[Nav3D][NoGoalDebug] poi_failed={poi_failed[:4]}")
            if self.frontier_clusters:
                samples = [
                    (
                        c.nav_voxel_id,
                        "down" if c.is_stair_frontier else "basic",
                        round(float(c.frontier_score), 1),
                    )
                    for c in self.frontier_clusters[:5]
                ]
                print(f"[Nav3D][NoGoalDebug] frontier_samples={samples}")
        except Exception as exc:
            print(f"[Nav3D][NoGoalDebug] failed: {exc}")

    def _refresh_current_floor_frontiers(
        self,
        agent_voxel: Tuple[int, int, int],
        blacklist_set: Set[Tuple[int, int, int]],
        reason: str,
    ) -> None:
        refreshed = detect_frontiers_3d_full(
            self.one_map,
            agent_voxel,
            # Current-floor fallback means the complete known floor.  The
            # sparse voxel dictionary and z_range still bound the work; no
            # agent-centred XY window is applied here.
            search_radius=None,
            frontier_sim_depth=self.frontier_sim_depth,
            sim_weight=self.sim_weight,
            min_cluster_size=self._frontier_min_cluster_size,
            z_range=self._frontier_full_check_z_range,
            # Preserve every current-floor cluster for the basic-exhaustion
            # decision; per-category ranking will cheaply reduce it afterward.
            max_clusters=None,
        )
        # Always commit the full current-floor result.  A scan containing only
        # descend geometry is still actionable and must not be discarded just
        # because the basic category is empty.
        self.frontier_clusters = refreshed
        has_basic = any(
            (not c.is_stair_frontier) and
            not self._is_basic_frontier_cluster_blacklisted(c, blacklist_set)
            for c in refreshed
        )
        basic_count = sum(1 for c in refreshed if not c.is_stair_frontier)
        descend_count = sum(1 for c in refreshed if c.is_stair_frontier)
        if has_basic:
            print(
                f"[Nav3D] Current-floor full frontier check ({reason}) found "
                f"{basic_count} basic clusters and {descend_count} descend clusters "
                f"(z_range={self._frontier_full_check_z_range})"
            )
            return
        if descend_count:
            print(
                f"[Nav3D] Current-floor full frontier check ({reason}) found no basic "
                f"clusters; retained {descend_count} descend clusters "
                f"(z_range={self._frontier_full_check_z_range})"
            )
            return
        print(
            f"[Nav3D] Current-floor full frontier check ({reason}) found no basic clusters "
            f"(z_range={self._frontier_full_check_z_range})"
        )
        return

    def _compute_frontiers_and_POIs(
        self,
        agent_voxel: Tuple[int, int, int],
        allow_current_floor_rescan: bool = True,
    ) -> None:
        """
        Build sorted nav_goals list with formal ObjectNav priority:
          POI (if enabled) > normal frontiers > down fake stair > rise stair > memory return.

        Stair semantic frontiers are only active inside stair_sem_mode. Rise/down
        stair frontiers remain explore-mode fallback goals, not navigate-mode goals.
        """
        self.nav_goals = []

        # ---- Frontier clusters from TravExplorer-style detection ----
        blacklist_set = set(
            tuple(b) if not isinstance(b, tuple) else b
            for b in self.blacklisted_goals
        )

        # Use the same support depth as traversability validation so fake
        # frontier stair checks stay aligned with pathability.
        support_check_depth = getattr(
            self.one_map,
            "support_check_depth",
            max(1, round(0.3 / self.one_map.voxel_size_z)),
        )

        basic_frontiers = []      # 基本前沿（未知邻居触发）
        stair_geo_frontiers = []  # 楼梯几何前沿（fake邻居触发）
        rise_stair_frontiers = [] # 上楼候选前沿（语义提议 + 深度台阶验证）

        for cluster in self.frontier_clusters:
            # In stair_sem_mode: skip stair geometric frontiers (prevents spinning)
            if self.stair_sem_mode and cluster.is_stair_frontier:
                continue

            if cluster.is_stair_frontier and self._is_fake_frontier_region_blacklisted(cluster):
                continue

            if (
                not cluster.is_stair_frontier and
                self._is_basic_frontier_cluster_blacklisted(cluster, blacklist_set)
            ):
                continue

            # Skip stair frontiers that have STAIR below (already confirmed stairs)
            if cluster.is_stair_frontier:
                if check_fake_frontier_has_stair_below(cluster, self.one_map, support_check_depth):
                    continue

            # 处理黑名单
            if cluster.nav_voxel_id in blacklist_set:
                if cluster.is_stair_frontier:
                    continue
                alt_vid = self._find_alt_nav_voxel(cluster, blacklist_set)
                if alt_vid is not None:
                    cluster = FrontierCluster3D(
                        nav_voxel_id=alt_vid,
                        centroid_metric=cluster.centroid_metric,
                        cluster_voxels=cluster.cluster_voxels,
                        frontier_score=cluster.frontier_score * 0.9,
                        cluster_size=cluster.cluster_size,
                        floor_iz=cluster.floor_iz,
                        aabb=cluster.aabb,
                        is_stair_adjacent=cluster.is_stair_adjacent,
                        is_stair_frontier=cluster.is_stair_frontier,
                        fake_triggered_voxels=cluster.fake_triggered_voxels,
                        fake_anchor_aabb=cluster.fake_anchor_aabb,
                    )
                else:
                    continue

            # 分类：基本前沿 vs 楼梯几何前沿
            if cluster.is_stair_frontier:
                if self._is_consumed_or_blocked_stair_region(cluster.nav_voxel_id, direction="down"):
                    continue
                self._remember_stair_frontier("down", cluster.nav_voxel_id, cluster=cluster)
                hint_bonus = self._down_stair_hint_bonus(cluster)
                if hint_bonus:
                    cluster.frontier_score += hint_bonus
                stair_geo_frontiers.append(cluster)
            else:
                basic_frontiers.append(cluster)

        for nav_vid, cand in list(self.rise_stair_candidates.items()):
            goal_vid = tuple(cand.get("nav_voxel_id", nav_vid))
            if self._is_in_recent_down_stair_region(goal_vid):
                print(
                    f"[Nav3D][RiseDebug] drop rise candidate in recent_down_region: "
                    f"key={nav_vid}, goal={goal_vid}, active={cand.get('active', False)}, "
                    f"seen={cand.get('seen_count', 0)}"
                )
                self.rise_stair_candidates.pop(nav_vid, None)
                continue
            if goal_vid in blacklist_set:
                print(
                    f"[Nav3D][RiseDebug] skip rise candidate blacklisted: "
                    f"key={nav_vid}, goal={goal_vid}"
                )
                continue
            if not cand.get("active", False):
                continue
            if goal_vid not in self.one_map.voxels:
                alt_vid = None
                if cand.get("bottom_xy") is not None:
                    alt_vid = self._find_floor_nav_voxel_near_metric(cand["bottom_xy"], agent_voxel, radius=10)
                if alt_vid is None and cand.get("approach_xy") is not None:
                    alt_vid = self._find_floor_nav_voxel_near_metric(cand["approach_xy"], agent_voxel, radius=10)
                if alt_vid is None:
                    print(
                        f"[Nav3D][RiseDebug] skip active rise candidate no nav voxel/alt: "
                        f"key={nav_vid}, goal={goal_vid}, seen={cand.get('seen_count', 0)}"
                    )
                    continue
                print(
                    f"[Nav3D][RiseDebug] active rise goal missing, using alt this frame: "
                    f"key={nav_vid}, old={goal_vid}, alt={alt_vid}, "
                    f"persist={'no' if cand.get('active', False) else 'yes'}"
                )
                goal_vid = alt_vid
                if not cand.get("active", False):
                    cand["nav_voxel_id"] = goal_vid
            if goal_vid in blacklist_set:
                print(
                    f"[Nav3D][RiseDebug] skip rise candidate alt blacklisted: "
                    f"key={nav_vid}, goal={goal_vid}"
                )
                continue
            committed_rise_goal = (
                isinstance(self.last_nav_goal, RiseStairFrontier3D) and
                self._xy_distance(self.last_nav_goal.nav_voxel_id, goal_vid) <= max(2, round(0.5 / self.one_map.cell_size))
            )
            if (
                self._post_upstairs_probe_state == "inactive" and
                not self._post_upstairs_prioritize_rise and
                not committed_rise_goal
            ):
                matched_up_mem = self._matching_stair_memory_id(goal_vid, direction="up")
                matched_up_status = (
                    self.stair_frontier_memory[matched_up_mem].get("status")
                    if matched_up_mem is not None and matched_up_mem in self.stair_frontier_memory else None
                )
                if matched_up_status in ("consumed", "blocked"):
                    mem = self.stair_frontier_memory[matched_up_mem]
                    print(
                        f"[Nav3D][RiseDebug] drop rise candidate consumed/blocked up region: "
                        f"key={nav_vid}, goal={goal_vid}, active={cand.get('active', False)}, "
                        f"seen={cand.get('seen_count', 0)}, mem_id={matched_up_mem}, "
                        f"mem_status={matched_up_status}, entry={mem.get('entry_voxel')}, "
                        f"center={mem.get('region_center')}, radius={mem.get('radius_xy')}, "
                        f"z=[{mem.get('z_min')},{mem.get('z_max')}]"
                    )
                    self.rise_stair_candidates.pop(nav_vid, None)
                    continue
            nav_vd = self.one_map.voxels[goal_vid]
            if nav_vd.voxel_type != VoxelType.TRAVERSABLE or nav_vd.inflated:
                alt_vid = None
                if cand.get("bottom_xy") is not None:
                    alt_vid = self._find_floor_nav_voxel_near_metric(cand["bottom_xy"], agent_voxel, radius=10)
                if alt_vid is None and cand.get("approach_xy") is not None:
                    alt_vid = self._find_floor_nav_voxel_near_metric(cand["approach_xy"], agent_voxel, radius=10)
                if alt_vid is None or alt_vid in blacklist_set:
                    print(
                        f"[Nav3D][RiseDebug] skip active rise candidate invalid nav and no valid alt: "
                        f"key={nav_vid}, goal={goal_vid}, type={nav_vd.voxel_type.name}, "
                        f"inflated={nav_vd.inflated}, alt={alt_vid}"
                    )
                    continue
                print(
                    f"[Nav3D][RiseDebug] active rise nav invalid, using alt this frame: "
                    f"key={nav_vid}, old={goal_vid}, alt={alt_vid}, "
                    f"type={nav_vd.voxel_type.name}, inflated={nav_vd.inflated}, "
                    f"persist={'no' if cand.get('active', False) else 'yes'}"
                )
                goal_vid = alt_vid
                if not cand.get("active", False):
                    cand["nav_voxel_id"] = goal_vid
            committed_rise_goal = (
                isinstance(self.last_nav_goal, RiseStairFrontier3D) and
                self._xy_distance(self.last_nav_goal.nav_voxel_id, goal_vid) <= max(2, round(0.5 / self.one_map.cell_size))
            )
            if (
                self._post_upstairs_probe_state == "inactive" and
                not self._post_upstairs_prioritize_rise and
                not committed_rise_goal
            ):
                matched_up_mem = self._matching_stair_memory_id(goal_vid, direction="up")
                matched_up_status = (
                    self.stair_frontier_memory[matched_up_mem].get("status")
                    if matched_up_mem is not None and matched_up_mem in self.stair_frontier_memory else None
                )
                if matched_up_status in ("consumed", "blocked"):
                    mem = self.stair_frontier_memory[matched_up_mem]
                    print(
                        f"[Nav3D][RiseDebug] drop rise candidate alt consumed/blocked up region: "
                        f"key={nav_vid}, goal={goal_vid}, active={cand.get('active', False)}, "
                        f"seen={cand.get('seen_count', 0)}, mem_id={matched_up_mem}, "
                        f"mem_status={matched_up_status}, entry={mem.get('entry_voxel')}, "
                        f"center={mem.get('region_center')}, radius={mem.get('radius_xy')}, "
                        f"z=[{mem.get('z_min')},{mem.get('z_max')}]"
                    )
                    self.rise_stair_candidates.pop(nav_vid, None)
                    continue
            source_box = tuple(float(v) for v in cand["box"])
            self._remember_stair_frontier("up", goal_vid, source_box=source_box)
            rise_stair_frontiers.append(RiseStairFrontier3D(
                nav_voxel_id=goal_vid,
                candidate_box=source_box,
                confidence=float(cand["confidence"]),
                geometry_score=float(cand["geometry_score"]),
                semantic_score=float(cand["semantic_score"]),
                bottom_xy=cand.get("bottom_xy"),
                approach_xy=cand.get("approach_xy"),
            ))

        memory_rise_frontiers, memory_down_frontiers = self._active_stair_memory_frontiers(
            agent_voxel, blacklist_set
        )

        committed_frontier_goal: Optional[NavGoal] = None
        if (
            self.navigation_mode != "navigate" and
            not self.stair_sem_mode and
            isinstance(self.last_nav_goal, (FrontierCluster3D, RiseStairFrontier3D))
        ):
            if isinstance(self.last_nav_goal, RiseStairFrontier3D):
                committed_candidates: List[NavGoal] = rise_stair_frontiers + memory_rise_frontiers
            elif self.last_nav_goal.is_stair_frontier:
                committed_candidates = stair_geo_frontiers + memory_down_frontiers
            else:
                committed_candidates = basic_frontiers

            last_descr = self.last_nav_goal.get_descr_point()
            threshold = max(2, round(0.5 / self.one_map.cell_size))
            for goal in committed_candidates:
                goal_descr = goal.get_descr_point()
                if np.array_equal(goal_descr, last_descr):
                    committed_frontier_goal = goal
                    break
                if np.linalg.norm(goal_descr[:2] - last_descr[:2]) <= threshold:
                    committed_frontier_goal = goal
                    break

        # ---- Stair semantic frontier (in stair_sem_mode, highest priority) ----
        if self.stair_sem_mode:
            # Refresh semantic stair frontier while keeping upstairs targets stable.
            self._refresh_stair_sem_frontier(agent_voxel)
            if (
                self._stair_sem_frontier is not None and
                self._is_valid_stair_sem_frontier_for_direction(self._stair_sem_frontier, agent_voxel)
            ):
                # 语义前沿在stair_sem_mode中最高优先级
                self.nav_goals.insert(0, self._stair_sem_frontier)
            else:
                self._stair_sem_frontier = None

        # ---- Cluster3D POIs: incremental live evidence + target-switch history prior ----
        poi_goals: List[Cluster3D] = []
        poi_select_debug: List[Tuple[str, Tuple[int, int, int], int, str]] = []
        if not self.stair_sem_mode:
            poi_source = list(self.live_pois.values())
            if history_poi_enabled(self):
                poi_source.extend(self._visible_history_pois())
            for cluster in poi_source:
                if cluster.nav_voxel_id in blacklist_set or self._is_poi_cluster_blacklisted(cluster):
                    poi_select_debug.append((
                        str(getattr(cluster, "poi_type", "?")),
                        tuple(int(v) for v in cluster.nav_voxel_id),
                        int(cluster.nav_voxel_id[2] - agent_voxel[2]),
                        "blocked",
                    ))
                    continue
                # POIs are observation targets. If the agent is already near a
                # POI, keeping it in nav_goals lets the arrival check start the
                # confirm scan and then blacklist it if no target is found.
                # Filtering by no_nav_radius here can hide the remaining POIs
                # after one candidate fails, causing a false NO NAV GOALS.
                poi_goals.append(cluster)

        if not self.stair_sem_mode:
            if poi_goals:
                same_floor_live: List[Tuple[int, Cluster3D]] = []
                same_floor_history: List[Tuple[int, Cluster3D]] = []
                other_floor_live: List[Tuple[int, Cluster3D]] = []
                other_floor_history: List[Tuple[int, Cluster3D]] = []
                for goal in poi_goals:
                    is_history_poi = "history" in str(getattr(goal, "poi_type", "")).lower()
                    if is_history_poi:
                        path, path_source = self._plan_history_poi_path_bidirectional(
                            agent_voxel,
                            goal.nav_voxel_id,
                            max_iterations=25000,
                        )
                    else:
                        path = self.one_map.a_star_3d(
                            agent_voxel, goal.nav_voxel_id,
                            max_iterations=25000,
                            stair_mode=False,
                            prefer_traversable=False,
                            allow_inflated_traversable=False,
                            allow_inflated_stair=True,
                            allow_inflated_traversable_stair=True,
                            connect_traversable_stair_drop=True,
                        )
                        path_source = "forward" if path is not None else "no_path"
                    if path is None:
                        poi_select_debug.append((
                            str(getattr(goal, "poi_type", "?")),
                            tuple(int(v) for v in goal.nav_voxel_id),
                            int(goal.nav_voxel_id[2] - agent_voxel[2]),
                            "no_path",
                        ))
                        continue
                    # Preserve the exact validated route selected here.  In
                    # particular, a reversed drop-connector path cannot be
                    # reconstructed by a second forward-only A* call.
                    if is_history_poi:
                        goal._validated_path = list(path)
                        goal._validated_path_start = tuple(int(v) for v in agent_voxel)
                    poi_select_debug.append((
                        str(getattr(goal, "poi_type", "?")),
                        tuple(int(v) for v in goal.nav_voxel_id),
                        int(goal.nav_voxel_id[2] - agent_voxel[2]),
                        f"{'reverse_' if path_source == 'reverse' else ''}path={len(path)}",
                    ))
                    item = (len(path), goal)
                    is_live = "live" in getattr(goal, "poi_type", "")
                    if abs(goal.nav_voxel_id[2] - agent_voxel[2]) <= self._frontier_z_range:
                        if is_live:
                            same_floor_live.append(item)
                        else:
                            same_floor_history.append(item)
                    else:
                        if is_live:
                            other_floor_live.append(item)
                        else:
                            other_floor_history.append(item)
                ranked = (
                    sorted(same_floor_live, key=lambda x: (x[0], -x[1].evidence_score)) or
                    sorted(same_floor_history, key=lambda x: (x[0], -x[1].evidence_score)) or
                    sorted(other_floor_live, key=lambda x: (x[0], -x[1].evidence_score)) or
                    sorted(other_floor_history, key=lambda x: (x[0], -x[1].evidence_score))
                )
                self.nav_goals.extend([goal for _, goal in ranked])
                if poi_select_debug:
                    print(f"[Nav3D][POISelect] {poi_select_debug[:8]}")
            if not self.nav_goals:
                if committed_frontier_goal is not None:
                    self.nav_goals.append(committed_frontier_goal)
                elif self._post_upstairs_prioritize_rise and (rise_stair_frontiers or memory_rise_frontiers):
                    rise_candidates = rise_stair_frontiers + memory_rise_frontiers
                    selected_rise = self._rank_frontier_candidates(
                        rise_candidates, agent_voxel
                    )
                    self.nav_goals.extend(selected_rise)
                    self._post_upstairs_prioritize_rise = False
                    if selected_rise:
                        best = selected_rise[0]
                        print(
                            f"[Nav3D] Post-upstairs selecting ascent frontier: "
                            f"nav={best.nav_voxel_id}, conf={best.confidence:.2f}, "
                            f"geom={best.geometry_score:.2f}, total={len(rise_candidates)}"
                        )
                elif self._post_upstairs_prioritize_rise:
                    print("[Nav3D] Post-upstairs rise priority expired: no active rise frontier available")
                    self._post_upstairs_prioritize_rise = False
                else:
                    # Category priority is applied to usable goals.  When the
                    # committed basic frontier disappears, keep following its
                    # old path during the configured grace period.  Only when
                    # grace expires do one current-floor full rescan before
                    # allowing the category to fall through to descend/ascent.
                    missing_basic_in_grace = (
                        isinstance(self.last_nav_goal, FrontierCluster3D) and
                        not self.last_nav_goal.is_stair_frontier and
                        committed_frontier_goal is None and
                        self.path is not None and
                        self._goal_missing_counter < self._goal_missing_grace
                    )

                    # Do not spend A* work on replacement candidates during
                    # grace: the old committed path still owns these frames.
                    # Recompute Euclidean order from the agent pose on the
                    # actual switch frame after grace expires.
                    if basic_frontiers and not missing_basic_in_grace:
                        self.nav_goals.extend(
                            self._rank_frontier_candidates(basic_frontiers, agent_voxel)
                        )
                    should_rescan_basic = (
                        allow_current_floor_rescan and
                        not self.stair_sem_mode and
                        not missing_basic_in_grace and
                        (self._force_current_floor_frontier_check or not self.nav_goals)
                    )
                    if should_rescan_basic:
                        reason = (
                            "forced" if self._force_current_floor_frontier_check
                            else "basic_exhausted_before_stair"
                        )
                        self._force_current_floor_frontier_check = False
                        self._refresh_current_floor_frontiers(agent_voxel, blacklist_set, reason)
                        # Always rebuild from the committed full-scan result,
                        # including descend-only and empty scans.  The false
                        # flag prevents this recursive assembly pass from
                        # launching another full rescan.
                        self._compute_frontiers_and_POIs(
                            agent_voxel,
                            allow_current_floor_rescan=False,
                        )
                        return

                    # Normal cross-floor fallback keeps a direction preference
                    # only after a successful StairSem traversal.  Basic
                    # frontiers and the current-floor full rescan always remain
                    # ahead of stairs.  The fixed post-upstairs probe branch
                    # above is intentionally unaffected.
                    down_candidates = stair_geo_frontiers + memory_down_frontiers
                    rise_candidates = rise_stair_frontiers + memory_rise_frontiers
                    if self._explore_stair_direction_preference == "up":
                        stair_categories = (rise_candidates, down_candidates)
                    else:
                        # Historical/default order, also used after descent.
                        stair_categories = (down_candidates, rise_candidates)

                    if not missing_basic_in_grace and not self.nav_goals:
                        for candidates in stair_categories:
                            if not candidates:
                                continue
                            ranked = self._rank_frontier_candidates(candidates, agent_voxel)
                            if ranked:
                                self.nav_goals.extend(ranked)
                                break

        # A downstairs landing probe is an active navigation task even though
        # its pre-scan/recenter phases intentionally own no frontier goal and
        # may temporarily have no A* path.  Do not let the evaluator's
        # five-frame ALL_EXPLORED watchdog terminate that probe.  The ascent
        # path and every ordinary no-goal case retain their existing behavior.
        downstairs_probe_active = (
            getattr(self, "_post_stair_probe_direction", None) == "down" and
            self._post_upstairs_probe_active()
        )
        self._no_goal_available = (
            not bool(self.nav_goals) and not downstairs_probe_active
        )

        if self.log and _RERUN:
            try:
                if self.nav_goals:
                    pts = np.array([g.get_descr_point() for g in self.nav_goals[:30]])
                    pts_metric = np.array([
                        self.one_map.voxel_to_metric(int(p[0]), int(p[1]), int(p[2]))[:2]
                        for p in pts
                    ])
                    rr.log("map/nav_goals", rr.Points2D(
                        pts_metric, radii=[0.15] * len(pts_metric)
                    ))
                    for entity in (
                        "map/live_poi_nav_targets",
                        "map/live_poi_centers",
                        "map/history_poi_nav_targets",
                        "map/history_poi_centers",
                    ):
                        rr.log(entity, rr.Clear(recursive=False))
            except Exception:
                pass

    def _rank_frontier_candidates(
        self,
        candidates: List[NavGoal],
        agent_voxel: Tuple[int, int, int],
    ) -> List[NavGoal]:
        return rank_frontier_candidates(self, candidates, agent_voxel)

    def _has_available_basic_frontier(self) -> bool:
        blacklist_set = set(tuple(b) if not isinstance(b, tuple) else b for b in self.blacklisted_goals)
        for cluster in self.frontier_clusters:
            if cluster.is_stair_frontier:
                continue
            if cluster.nav_voxel_id in blacklist_set:
                continue
            vd = self.one_map.voxels.get(cluster.nav_voxel_id)
            if vd is not None and vd.voxel_type == VoxelType.TRAVERSABLE and not vd.inflated:
                return True
        return False
