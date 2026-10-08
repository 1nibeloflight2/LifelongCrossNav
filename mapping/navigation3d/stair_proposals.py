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

try:
    import rerun as rr
    _RERUN = True
except ImportError:
    _RERUN = False

from .state import StairDirection

def proposal_is_persistent(direction) -> bool:
    return StairDirection.normalize(direction) is StairDirection.ASCENT

class StairProposalMixin:
    def _reset_rise_confirm_scan(self) -> None:
        self._rise_confirm_scan_remaining = 0
        self._rise_confirm_scan_goal = None
        self._rise_confirm_scan_turn = "L"

    def _start_rise_confirm_scan(
        self,
        frontier: RiseStairFrontier3D,
        odometry: np.ndarray,
    ) -> None:
        """Turn toward the stored stair base so a narrow stair can be re-observed."""
        if self._rise_confirm_scan_goal == frontier.nav_voxel_id and self._rise_confirm_scan_remaining > 0:
            return

        turn = "L"
        if frontier.bottom_xy is not None:
            dx = float(frontier.bottom_xy[0]) - float(odometry[0, 3])
            dy = float(frontier.bottom_xy[1]) - float(odometry[1, 3])
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                target_yaw = float(np.arctan2(dy, dx))
                current_yaw = float(np.arctan2(odometry[1, 0], odometry[0, 0]))
                turn = "L" if self._wrap_angle(target_yaw - current_yaw) >= 0.0 else "R"

        self._rise_confirm_scan_goal = frontier.nav_voxel_id
        self._rise_confirm_scan_remaining = int(getattr(self, "_rise_confirm_scan_steps", 4))
        self._rise_confirm_scan_turn = turn
        self.path = None
        print(
            f"[Nav3D] Rise stair confirm scan: goal={frontier.nav_voxel_id}, "
            f"turn={turn}, steps={self._rise_confirm_scan_remaining}"
        )

    def _matching_stair_memory_id(
        self,
        nav_voxel: Tuple[int, int, int],
        direction: Optional[str] = None,
        include_active: bool = True,
    ) -> Optional[int]:
        for mem_id, mem in self.stair_frontier_memory.items():
            if direction is not None and mem.get("direction") != direction:
                continue
            status = mem.get("status", "active")
            if status == "active" and not include_active:
                continue
            center = mem.get("region_center", mem.get("entry_voxel"))
            if center is None:
                continue
            center = tuple(center)
            radius_xy = int(mem.get("radius_xy", self._recent_down_stair_radius_xy))
            z_min = int(mem.get("z_min", center[2] - 6))
            z_max = int(mem.get("z_max", center[2] + 6))
            if (
                self._xy_distance(nav_voxel, center) <= radius_xy and
                z_min <= nav_voxel[2] <= z_max
            ):
                return mem_id
        return None

    def _remember_stair_frontier(
        self,
        direction: str,
        nav_voxel: Tuple[int, int, int],
        source_box: Optional[Tuple[float, float, float, float]] = None,
        cluster: Optional[FrontierCluster3D] = None,
    ) -> int:
        # Descend proposals are persistent geometric frontiers and do not need
        # a semantic memory copy. Only ascent proposals are remembered.
        if not proposal_is_persistent(direction):
            return -1
        existing = self._matching_stair_memory_id(nav_voxel, direction=direction)
        if existing is not None:
            mem = self.stair_frontier_memory[existing]
            if mem.get("status") == "active":
                if direction == "down":
                    mem["entry_voxel"] = nav_voxel
                mem["source_box"] = source_box if source_box is not None else mem.get("source_box")
            return existing

        radius_xy = int(self._recent_down_stair_radius_xy if direction == "down" else 10)
        z_min = nav_voxel[2] - 8
        z_max = nav_voxel[2] + 8
        if cluster is not None and cluster.aabb is not None:
            mn, mx = cluster.aabb
            z_min = min(z_min, mn[2])
            z_max = max(z_max, mx[2])
            if cluster.fake_anchor_aabb is not None:
                amn, amx = cluster.fake_anchor_aabb
                radius_xy = max(
                    radius_xy,
                    max(abs(amx[0] - amn[0]), abs(amx[1] - amn[1])) + 4,
                )

        mem_id = self._stair_memory_next_id
        self._stair_memory_next_id += 1
        self.stair_frontier_memory[mem_id] = {
            "direction": direction,
            "entry_voxel": nav_voxel,
            "exit_voxel": None,
            "floor_z": nav_voxel[2],
            "region_center": nav_voxel,
            "radius_xy": radius_xy,
            "z_min": z_min,
            "z_max": z_max,
            "source_box": source_box,
            "status": "active",
        }
        return mem_id

    def _block_stair_memory(self, mem_id: int, reason: str) -> None:
        mem = self.stair_frontier_memory.get(mem_id)
        if mem is None:
            return
        mem["status"] = "blocked"
        print(f"[Nav3D] Stair memory {mem_id} blocked: {reason}")

    def _nearest_floor_nav_for_memory(
        self,
        entry: Tuple[int, int, int],
        radius_xy: int,
        z_min: int,
        z_max: int,
        blacklist_set: Set[Tuple[int, int, int]],
    ) -> Optional[Tuple[int, int, int]]:
        voxels = self.one_map.voxels
        target_types = {VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR}
        best = None
        best_dist = float("inf")
        for vid, vd in voxels.items():
            if vd.voxel_type not in target_types or vd.inflated:
                continue
            if vid in blacklist_set:
                continue
            if not (z_min <= vid[2] <= z_max):
                continue
            if abs(vid[0] - entry[0]) > radius_xy or abs(vid[1] - entry[1]) > radius_xy:
                continue
            dist = self._xy_distance(vid, entry) + 0.25 * abs(vid[2] - entry[2])
            if dist < best_dist:
                best_dist = dist
                best = vid
        return best

    def _validate_active_down_stair_memory(
        self,
        mem_id: int,
        mem: Dict[str, Any],
        entry: Tuple[int, int, int],
        blacklist_set: Set[Tuple[int, int, int]],
    ) -> Optional[FrontierCluster3D]:
        radius_xy = int(mem.get("radius_xy", self._recent_down_stair_radius_xy))
        z_min = int(mem.get("z_min", entry[2] - 4))
        z_max = int(mem.get("z_max", entry[2] + 4))
        nav_voxel = self._nearest_floor_nav_for_memory(entry, radius_xy, z_min, z_max, blacklist_set)
        if nav_voxel is None:
            self._block_stair_memory(mem_id, "down memory has no local traversable nav target")
            return None

        fake_voxels = {
            vid for vid, vd in self.one_map.voxels.items()
            if vd.voxel_type == VoxelType.TRAVERSABLE_FAKE and
            abs(vid[0] - entry[0]) <= radius_xy and
            abs(vid[1] - entry[1]) <= radius_xy and
            z_min <= vid[2] <= z_max
        }
        if not fake_voxels:
            self._block_stair_memory(mem_id, "down memory has no local TRAVERSABLE_FAKE anchor")
            return None

        fxs = [v[0] for v in fake_voxels]
        fys = [v[1] for v in fake_voxels]
        fzs = [v[2] for v in fake_voxels]
        fake_aabb = ((min(fxs), min(fys), min(fzs)), (max(fxs), max(fys), max(fzs)))
        aabb = (
            (min(nav_voxel[0], fake_aabb[0][0]), min(nav_voxel[1], fake_aabb[0][1]), min(nav_voxel[2], fake_aabb[0][2])),
            (max(nav_voxel[0], fake_aabb[1][0]), max(nav_voxel[1], fake_aabb[1][1]), max(nav_voxel[2], fake_aabb[1][2])),
        )
        centroid_metric = tuple(float(v) for v in self.one_map.voxel_to_metric(*nav_voxel))
        cluster = FrontierCluster3D(
            nav_voxel_id=nav_voxel,
            centroid_metric=centroid_metric,
            cluster_voxels={nav_voxel},
            frontier_score=float(mem.get("frontier_score", 250.0)),
            cluster_size=max(1, len(fake_voxels)),
            floor_iz=nav_voxel[2],
            aabb=aabb,
            is_stair_adjacent=True,
            is_stair_frontier=True,
            fake_triggered_voxels=fake_voxels,
            fake_anchor_aabb=fake_aabb,
        )
        if self._is_fake_frontier_region_blacklisted(cluster):
            self._block_stair_memory(mem_id, "down memory fake region blacklisted")
            return None

        support_check_depth = getattr(
            self.one_map,
            "support_check_depth",
            max(1, round(0.3 / self.one_map.voxel_size_z)),
        )
        if check_fake_frontier_has_stair_below(cluster, self.one_map, support_check_depth):
            self._block_stair_memory(mem_id, "down memory fake anchor already has stair below")
            return None
        return cluster

    def _active_stair_memory_frontiers(
        self,
        agent_voxel: Tuple[int, int, int],
        blacklist_set: Set[Tuple[int, int, int]],
    ) -> Tuple[List[RiseStairFrontier3D], List[FrontierCluster3D]]:
        """Recover active stair-entry memories as explore-mode fallback frontiers.

        These are not known stair-return goals. They represent rise/down stair
        entrances that were observed earlier, then disappeared from incremental
        frontier detection before the current floor's normal frontiers were
        exhausted. Recreating them here lets the normal confirmation logic run
        again from the saved local entrance.
        """
        rise_goals: List[RiseStairFrontier3D] = []
        down_goals: List[FrontierCluster3D] = []
        voxels = self.one_map.voxels

        for mem_id, mem in self.stair_frontier_memory.items():
            if mem.get("status") != "active":
                continue
            direction = mem.get("direction")
            entry = mem.get("entry_voxel")
            if direction not in ("up", "down") or entry is None:
                continue
            entry = tuple(int(v) for v in entry)
            if entry in blacklist_set:
                continue

            if direction == "down":
                # Active descend proposal memory is intentionally disabled;
                # geometry frontiers remain derivable from TRAVERSABLE_FAKE.
                continue

            nav_voxel = entry
            vd = voxels.get(nav_voxel)
            if vd is None or vd.inflated or vd.voxel_type == VoxelType.FAKE_STAIR:
                alt = self.one_map._find_nearest_safe_traversable(nav_voxel, max_radius=8)
                if alt is None or alt in blacklist_set:
                    continue
                nav_voxel = alt
                vd = voxels.get(nav_voxel)
            if vd is None or vd.inflated or vd.voxel_type == VoxelType.FAKE_STAIR:
                continue

            source_box = mem.get("source_box")
            if source_box is None:
                source_box = (0.0, 0.0, 0.0, 0.0)
            else:
                source_box = tuple(float(v) for v in source_box)

            if direction == "up":
                if self._is_consumed_or_blocked_stair_region(nav_voxel, direction="up"):
                    continue
                rise_goals.append(RiseStairFrontier3D(
                    nav_voxel_id=nav_voxel,
                    candidate_box=source_box,
                    confidence=float(mem.get("confidence", 0.0)),
                    geometry_score=float(mem.get("geometry_score", 0.0)),
                    semantic_score=float(mem.get("semantic_score", 0.0)),
                ))
                continue

        if (rise_goals or down_goals) and getattr(self, "debug", False):
            print(
                f"[Nav3D] Active stair memory fallback: "
                f"rise={len(rise_goals)}, down={len(down_goals)}"
            )
        return rise_goals, down_goals

    def _consume_active_stair_memory(self, agent_voxel: Optional[Tuple[int, int, int]]) -> None:
        active_mem_id = self._active_stair_memory_id
        if active_mem_id is None or active_mem_id not in self.stair_frontier_memory:
            self._active_stair_memory_id = None
            return
        # Successful traversal consumes the ascent proposal. Confirmed stair
        # geometry remains in the global map; no connector record is retained.
        self.stair_frontier_memory.pop(active_mem_id, None)
        self._active_stair_memory_id = None

    def _reset_post_upstairs_probe(self) -> None:
        self._post_upstairs_probe_state = "inactive"
        self._post_upstairs_probe_target = None
        self._post_upstairs_probe_scan_remaining = 0
        self._post_upstairs_probe_pending = False
        self._post_stair_probe_direction = None
        self._post_upstairs_probe_settle_remaining = 0
        self._post_upstairs_probe_settle_attempts = 0
        self._post_upstairs_prioritize_rise = False

    def _recompute_probe_center(
        self, agent_voxel: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int]]:
        """重新计算平台对侧观察目标（A* 失败时调用，此时可能有更多体素数据）。"""
        cluster = self._localize_probe_cluster(
            self._current_floor_traversable_cluster(agent_voxel),
            agent_voxel,
        )
        if not self._is_small_traversable_cluster(cluster):
            return None
        return self._find_cluster_center_voxel(cluster, agent_voxel)

    def _has_active_rise_stair_candidate(self) -> bool:
        return any(cand.get("active", False) for cand in self.rise_stair_candidates.values())

    def _current_floor_traversable_cluster(
        self,
        agent_voxel: Tuple[int, int, int],
        max_nodes: int = 400,
    ) -> Set[Tuple[int, int, int]]:
        """Return connected platform traversable voxels on agent_z +/- 1."""
        from collections import deque

        voxels = self.one_map.voxels
        platform_types = {VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR}
        targetable_types = {VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR}
        base_z = agent_voxel[2]
        start = agent_voxel
        if (
            start not in voxels or
            voxels[start].voxel_type not in targetable_types or
            voxels[start].inflated or
            abs(start[2] - base_z) > 1
        ):
            start = self.one_map._find_nearest_safe_traversable(agent_voxel, max_radius=4)
            if (
                start is None or
                start not in voxels or
                voxels[start].voxel_type not in targetable_types or
                abs(start[2] - base_z) > 1
            ):
                return set()

        cluster: Set[Tuple[int, int, int]] = set()
        queue = deque([start])
        cluster.add(start)
        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        while queue and len(cluster) < max_nodes:
            vid = queue.popleft()
            for dx, dy, dz in neighbors:
                nb = (vid[0] + dx, vid[1] + dy, vid[2] + dz)
                if nb in cluster or abs(nb[2] - base_z) > 1:
                    continue
                vd = voxels.get(nb)
                if vd is None or vd.voxel_type not in platform_types or vd.inflated:
                    continue
                cluster.add(nb)
                queue.append(nb)
        return cluster

    def _is_small_traversable_cluster(self, cluster: Set[Tuple[int, int, int]]) -> bool:
        if not cluster:
            return False
        unique_xy = set((v[0], v[1]) for v in cluster)
        xs = [xy[0] for xy in unique_xy]
        ys = [xy[1] for xy in unique_xy]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        return span <= self._post_upstairs_probe_max_span

    def _localize_probe_cluster(
        self,
        cluster: Set[Tuple[int, int, int]],
        agent_voxel: Tuple[int, int, int],
    ) -> Set[Tuple[int, int, int]]:
        """Limit upstairs probe centering to the local landing around the stair exit."""
        radius = int(getattr(self, "_post_upstairs_probe_local_radius_xy", 16))
        ax, ay, _ = agent_voxel
        return {
            vid for vid in cluster
            if abs(vid[0] - ax) <= radius and abs(vid[1] - ay) <= radius
        }

    def _find_cluster_center_voxel(
        self,
        cluster: Set[Tuple[int, int, int]],
        agent_voxel: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int, int]]:
        """Pick an opposite-side targetable voxel for a better landing view.

        投影到2D平面计算覆盖范围中心，不考虑z轴体素密度：
        1. 优先仅用普通 TRAVERSABLE 计算平台中心和对称点
        2. 若普通 TRAVERSABLE 不能生成落点，再回退到
           TRAVERSABLE + TRAVERSABLE_STAIR 混合计算
        3. 每个(x,y)只计一次，并保留该位置的最高层体素
        4. 以“agent位置与目标点的中点=平台中心”为目标，选对侧最近可通行点
        """
        if not cluster:
            return None
        voxels = self.one_map.voxels

        def select_symmetric_target(
            allowed_types: Set[VoxelType],
        ) -> Optional[Tuple[int, int, int]]:
            # Build the center and the selectable target set from the same
            # semantic subset. This prevents stair-supported platform cells
            # from pulling a valid ordinary-floor center back toward the stairs.
            xy_to_top: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
            for vid in cluster:
                vd = voxels.get(vid)
                if vd is None or vd.voxel_type not in allowed_types or vd.inflated:
                    continue
                xy = (vid[0], vid[1])
                if xy not in xy_to_top or vid[2] > xy_to_top[xy][2]:
                    xy_to_top[xy] = vid

            if not xy_to_top:
                return None

            unique_xy = set(xy_to_top)
            cx = float(np.mean([xy[0] for xy in unique_xy]))
            cy = float(np.mean([xy[1] for xy in unique_xy]))
            desired_x = 2.0 * cx - float(agent_voxel[0])
            desired_y = 2.0 * cy - float(agent_voxel[1])

            return min(
                xy_to_top.values(),
                key=lambda vid: (
                    (vid[0] - desired_x) ** 2 +
                    (vid[1] - desired_y) ** 2
                ),
            )

        ordinary_target = select_symmetric_target({VoxelType.TRAVERSABLE})
        if ordinary_target is not None:
            return ordinary_target

        return select_symmetric_target({
            VoxelType.TRAVERSABLE,
            VoxelType.TRAVERSABLE_STAIR,
        })

    def _configure_post_upstairs_probe_target(
        self,
        agent_voxel: Tuple[int, int, int],
        allow_settle: bool,
    ) -> bool:
        full_cluster = self._current_floor_traversable_cluster(agent_voxel)
        if not full_cluster:
            print(f"[Nav3D] Post-upstairs probe target failed: empty platform cluster at agent={agent_voxel}")
            return False
        cluster = self._localize_probe_cluster(full_cluster, agent_voxel)
        if not cluster:
            print(
                f"[Nav3D] Post-upstairs probe target failed: empty local platform cluster "
                f"full_size={len(full_cluster)} radius={self._post_upstairs_probe_local_radius_xy} "
                f"agent={agent_voxel}"
            )
            return False
        if not self._is_small_traversable_cluster(cluster):
            unique_xy = set((v[0], v[1]) for v in cluster)
            xs = [xy[0] for xy in unique_xy]
            ys = [xy[1] for xy in unique_xy]
            span = max(max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else 0
            print(
                f"[Nav3D] Post-upstairs probe target failed: local platform span too large "
                f"full_size={len(full_cluster)} local_size={len(cluster)} local_xy={len(unique_xy)} "
                f"span={span} limit={self._post_upstairs_probe_max_span} "
                f"agent={agent_voxel}"
            )
            return False
        target = self._find_cluster_center_voxel(cluster, agent_voxel)
        if target is None:
            unique_xy = set((v[0], v[1]) for v in cluster)
            print(
                f"[Nav3D] Post-upstairs probe target failed: no targetable opposite voxel "
                f"full_size={len(full_cluster)} local_size={len(cluster)} local_xy={len(unique_xy)} "
                f"agent={agent_voxel}"
            )
            return False

        self._post_upstairs_probe_target = target
        dist_xy = self._xy_distance(target, agent_voxel)
        unique_xy = set((v[0], v[1]) for v in cluster)
        cluster_too_small = len(unique_xy) < int(getattr(self, "_post_upstairs_probe_min_center_cluster_size", 50))
        target_too_close = dist_xy <= 2
        can_settle = (
            allow_settle and
            self._post_upstairs_probe_settle_attempts < self._post_upstairs_probe_max_settle_attempts and
            (cluster_too_small or target_too_close)
        )
        if can_settle:
            self._post_upstairs_probe_settle_attempts += 1
            self._post_upstairs_probe_settle_remaining = 1
            self._post_upstairs_probe_state = "settle_forward"
            self._post_upstairs_probe_scan_remaining = 0
            print(
                f"[Nav3D] Post-upstairs probe settle: full_size={len(full_cluster)}, "
                f"local_size={len(cluster)}, local_xy={len(unique_xy)}, target={target}, dist_xy={dist_xy}, "
                f"attempt={self._post_upstairs_probe_settle_attempts}"
            )
            return True

        if dist_xy <= 2:
            self._post_upstairs_probe_state = "scan"
            self._post_upstairs_probe_scan_remaining = self._post_upstairs_probe_scan_steps
            print(
                f"[Nav3D] Post-upstairs probe target too close: full_size={len(full_cluster)}, "
                f"local_size={len(cluster)}, local_xy={len(unique_xy)}, target={target}, "
                f"dist_xy={dist_xy}, state=scan"
            )
        else:
            self._post_upstairs_probe_state = "recenter"
            self._post_upstairs_probe_scan_remaining = 0
            print(
                f"[Nav3D] Post-upstairs probe target updated: full_size={len(full_cluster)}, "
                f"local_size={len(cluster)}, local_xy={len(unique_xy)}, target={target}, "
                f"state={self._post_upstairs_probe_state}"
            )
        return True

    def _start_post_upstairs_probe(
        self,
        agent_voxel: Tuple[int, int, int],
        direction: str = "up",
    ) -> bool:
        """
        For small upstairs landing-like traversable clusters, move to the cluster
        center and scan once before declaring the stair traversal complete.
        """
        if self._post_upstairs_probe_state != "inactive":
            return True

        # The landing has already passed StairSem's directional exit checks.
        # Record the completed direction before clearing StairSem state.
        direction = "up" if direction == "up" else "down"
        self._post_stair_probe_direction = direction
        self._explore_stair_direction_preference = direction
        if direction == "down":
            self._remember_recent_down_stair_region(agent_voxel)
        self._consume_active_stair_memory(agent_voxel)
        self.stair_sem_mode = False
        self._stair_sem_frontier = None
        self._stair_sem_entry_z = None
        self._reset_stair_sem_stability()
        self._stair_sem_direction = "down"
        self._stair_sem_force_forward = 0
        self._reset_fake_frontier_state()
        if direction == "up":
            self._upstairs_transition_cooldown = self._upstairs_transition_cooldown_steps
        self.one_map.mapping_mode = "geometric"
        self.desired_pitch_level = 0
        self.path = None
        self.path_id = 0
        self.last_nav_goal = None
        self.nav_goals.clear()
        self.frontier_clusters.clear()
        self.blacklisted_goals.clear()
        self.rise_stair_candidates.clear()
        self._post_upstairs_probe_settle_attempts = 0
        self._post_upstairs_probe_settle_remaining = 0
        self._post_upstairs_probe_target = None
        self._post_upstairs_probe_state = "pre_scan"
        self._post_upstairs_probe_scan_remaining = self._post_upstairs_probe_scan_steps
        print(
            f"[Nav3D] Post-{'upstairs' if direction == 'up' else 'downstairs'} probe started: "
            f"state=pre_scan, scan_steps={self._post_upstairs_probe_scan_steps}"
        )
        return True

    def _handle_upstairs_stair_sem_exit(self, agent_voxel: Tuple[int, int, int]) -> None:
        """Exit upstairs stair sem through a small-cluster probe when useful."""
        if self._start_post_upstairs_probe(agent_voxel, direction="up"):
            return
        if self._has_active_rise_stair_candidate():
            self._consume_active_stair_memory(agent_voxel)
            self.stair_sem_mode = False
            self.one_map.mapping_mode = "geometric"
            self.path = None
            self.last_nav_goal = None
            self._stair_sem_frontier = None
            self._stair_sem_entry_z = None
            self._reset_stair_sem_stability()
            self._stair_sem_direction = "down"
            self._stair_sem_force_forward = 0
            self._reset_fake_frontier_state()
            self._upstairs_transition_cooldown = self._upstairs_transition_cooldown_steps
            # 上楼完成：清空旧前沿，触发全量重新检测，避免回头选旧楼梯噪点
            self.frontier_clusters.clear()
            self.nav_goals.clear()
            self.blacklisted_goals.clear()
            self.rise_stair_candidates.clear()
            self.first_obs = True
            self._frontier_min_cluster_size = 2
            self.request_reinit = True
            return
        self._exit_stair_sem_mode()

    def _handle_downstairs_stair_sem_exit(self, agent_voxel: Tuple[int, int, int]) -> None:
        """Scan and cross a narrow landing before rebuilding descend frontiers."""
        if self._start_post_upstairs_probe(agent_voxel, direction="down"):
            return
        self._exit_stair_sem_mode(agent_voxel)

    def _compute_post_upstairs_probe_path(self, agent_voxel: Tuple[int, int, int]) -> bool:
        """Return True when post-upstairs probe owns planning for this frame."""
        if self._post_upstairs_probe_state == "inactive":
            return False
        if self._post_upstairs_probe_state == "pre_scan":
            self.path = None
            return True
        if self._post_upstairs_probe_state == "compute_center":
            self.path = None
            if self._configure_post_upstairs_probe_target(agent_voxel, allow_settle=False):
                return True
            self._post_upstairs_probe_state = "scan"
            self._post_upstairs_probe_scan_remaining = self._post_upstairs_probe_scan_steps
            return True
        if self._post_upstairs_probe_state == "scan":
            self.path = None
            return True

        target = self._post_upstairs_probe_target
        if target is None:
            self._post_upstairs_probe_state = "scan"
            self._post_upstairs_probe_scan_remaining = self._post_upstairs_probe_scan_steps
            return True
        dist_xy = self._xy_distance(agent_voxel, target)
        if dist_xy <= 2:
            self._post_upstairs_probe_state = "scan"
            self._post_upstairs_probe_scan_remaining = self._post_upstairs_probe_scan_steps
            self.path = None
            print(f"[Nav3D] Post-upstairs probe reached target {target}; scanning")
            return True

        self.path = self.one_map.a_star_3d(agent_voxel, target, max_iterations=30000)
        if self.path is None:
            # A* 失败：重新计算平台中心（此时可能有更多体素数据）
            new_target = self._recompute_probe_center(agent_voxel)
            if new_target is not None and new_target != target:
                self._post_upstairs_probe_target = new_target
                self.path = self.one_map.a_star_3d(agent_voxel, new_target, max_iterations=30000)
                if self.path is not None:
                    self.path_id = 0
                    print(f"[Nav3D] Post-upstairs probe target recomputed: {target}->{new_target}, path found")
                    return True
            print(f"[Nav3D] Post-upstairs probe target path failed: {agent_voxel}->{target}; scanning in place")
            self._post_upstairs_probe_state = "scan"
            self._post_upstairs_probe_scan_remaining = self._post_upstairs_probe_scan_steps
        else:
            self.path_id = 0
        return True

    def _expire_rise_stair_candidates(self) -> None:
        """Drop stale upstairs stair candidates."""
        now = self.one_map._iters
        stale = [
            nav_vid for nav_vid, cand in self.rise_stair_candidates.items()
            if now - cand.get("last_seen_step", now) > self._rise_stair_candidate_ttl
        ]
        for nav_vid in stale:
            self.rise_stair_candidates.pop(nav_vid, None)

    def _expire_recent_down_stair_regions(self) -> None:
        now = self.one_map._iters
        self._recent_down_stair_regions = [
            region for region in self._recent_down_stair_regions
            if now - region.get("last_seen_step", now) <= self._recent_down_stair_region_ttl
        ]

    def _remember_recent_down_stair_region(self, agent_voxel: Tuple[int, int, int]) -> None:
        """Remember the just-descended stair so it is not selected as an immediate rise frontier."""
        from mapping.voxel_map import LOCKED_TYPES

        entry_z = self._stair_sem_entry_z if self._stair_sem_entry_z is not None else agent_voxel[2]
        z_min = min(agent_voxel[2], entry_z) - 3
        z_max = max(agent_voxel[2], entry_z) + 6
        stair_voxels = []
        for vid, vd in self.one_map.voxels.items():
            if vd.voxel_type not in LOCKED_TYPES:
                continue
            if not (z_min <= vid[2] <= z_max):
                continue
            if self._xy_distance(vid, agent_voxel) <= self._recent_down_stair_radius_xy:
                stair_voxels.append(vid)

        if stair_voxels:
            center = (
                int(round(np.mean([v[0] for v in stair_voxels]))),
                int(round(np.mean([v[1] for v in stair_voxels]))),
                int(round(np.mean([v[2] for v in stair_voxels]))),
            )
            z_min = min(v[2] for v in stair_voxels) - 3
            z_max = max(v[2] for v in stair_voxels) + 6
        else:
            center = agent_voxel

        self._recent_down_stair_regions.append({
            "center": center,
            "z_min": z_min,
            "z_max": z_max,
            "radius_xy": self._recent_down_stair_radius_xy,
            "last_seen_step": self.one_map._iters,
        })
        print(f"[Nav3D] Recent down stair region remembered: center={center}, z=[{z_min},{z_max}]")

    def _is_in_recent_down_stair_region(self, nav_vid: Tuple[int, int, int]) -> bool:
        self._expire_recent_down_stair_regions()
        for region in self._recent_down_stair_regions:
            center = region["center"]
            if not (region["z_min"] <= nav_vid[2] <= region["z_max"]):
                continue
            dist_xy = self._xy_distance(nav_vid, center)
            if dist_xy <= int(region.get("radius_xy", self._recent_down_stair_radius_xy)):
                return True
        return False

    def _purge_recent_down_stair_rise_candidates(self) -> None:
        for nav_vid in list(self.rise_stair_candidates):
            if self._is_in_recent_down_stair_region(nav_vid):
                self.rise_stair_candidates.pop(nav_vid, None)

    def _find_matching_rise_candidate(
        self, nav_vid: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int]]:
        """Find an existing rise candidate close enough to merge with nav_vid."""
        for key in self.rise_stair_candidates:
            dist_xy = self._xy_distance(key, nav_vid)
            dist_z = abs(key[2] - nav_vid[2])
            if dist_xy <= 3 and dist_z <= 1:
                return key
        return None

    def _is_rise_candidate_wall_like(
        self,
        forward: np.ndarray,
        lateral: np.ndarray,
        height: np.ndarray,
    ) -> bool:
        """Reject obvious vertical wall/rail planes without tightening stair shape rules."""
        if forward.size < 40:
            return False

        forward_span = float(np.percentile(forward, 95) - np.percentile(forward, 5))
        height_span = float(np.percentile(height, 95) - np.percentile(height, 5))
        if height_span >= 0.55 and forward_span <= 0.35:
            return True

        f_bins = np.floor((forward - forward.min()) / 0.12).astype(np.int32)
        l_bins = np.floor((lateral - lateral.min()) / 0.12).astype(np.int32)
        columns: Dict[Tuple[int, int], List[float]] = {}
        for fb, lb, h in zip(f_bins, l_bins, height):
            columns.setdefault((int(fb), int(lb)), []).append(float(h))
        if len(columns) < 6:
            return False

        tall_columns = sum(1 for vals in columns.values() if max(vals) - min(vals) >= 0.45)
        vertical_ratio = tall_columns / max(1, len(columns))
        # Real stair proposals can include some risers/rail points, but a wall
        # has most columns vertically stacked at nearly the same forward depth.
        return vertical_ratio >= 0.60 and forward_span <= 0.70

    def _validate_rise_stair_candidate(
        self,
        box: np.ndarray,
        score: float,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Tuple[int, int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        Validate an upstairs stair proposal using depth geometry.

        The verifier looks for a wide, forward, monotonically rising stepped
        structure inside the detector box, then anchors the candidate at the
        current-floor traversable voxel near the candidate base.
        """
        if score < self._rise_stair_proposal_threshold:
            return None
        if self.one_map.fx is None or self.one_map.agent_height_0 is None:
            return None

        H, W = depth.shape
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
        y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        ys, xs = np.mgrid[y1:y2 + 1, x1:x2 + 1]
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        depths = depth[ys, xs]
        valid = np.isfinite(depths) & (depths > 0.05)
        if valid.sum() < 80:
            return None
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        depths = depths[valid].astype(np.float32)
        if xs.size > 2500:
            idx = np.linspace(0, xs.size - 1, 2500).astype(np.int64)
            xs = xs[idx]
            ys = ys[idx]
            depths = depths[idx]

        x_cam = (xs - self.one_map.cx) * depths / self.one_map.fx
        y_cam = (ys - self.one_map.cy) * depths / self.one_map.fy
        pts_cam = np.stack([depths, -x_cam, -y_cam], axis=-1)
        R_mat = odometry[:3, :3]
        t_vec = odometry[:3, 3]
        pts_world = (R_mat @ pts_cam.T).T + t_vec

        rel = pts_world - t_vec
        forward_axis = R_mat[:, 0]
        lateral_axis = R_mat[:, 1]
        forward = rel @ forward_axis
        lateral = rel @ lateral_axis
        height = pts_world[:, 2] - self.one_map.agent_height_0

        in_range = (
            (forward >= 0.6) & (forward <= 6.0) &
            (height >= -0.05) & (height <= 1.8)
        )
        if in_range.sum() < 80:
            return None
        pts_world = pts_world[in_range]
        forward = forward[in_range]
        lateral = lateral[in_range]
        height = height[in_range]

        width = float(np.percentile(lateral, 90) - np.percentile(lateral, 10))
        if width < 0.45:
            return None
        if self._is_rise_candidate_wall_like(forward, lateral, height):
            return None

        bin_width = 0.25
        min_tread_thickness = 0.10
        bin_ids = np.floor((forward - forward.min()) / bin_width).astype(np.int32)
        layer_samples = []
        for bin_id in sorted(set(bin_ids)):
            mask = bin_ids == bin_id
            if mask.sum() < 12:
                continue
            f_span = float(np.percentile(forward[mask], 90) - np.percentile(forward[mask], 10))
            if f_span < min_tread_thickness:
                continue
            f_med = float(np.median(forward[mask]))
            h_low = float(np.percentile(height[mask], 25))
            layer_samples.append((f_med, h_low, mask.sum(), f_span))

        if len(layer_samples) < 3:
            return None

        layer_samples.sort(key=lambda item: item[0])
        f_vals = np.array([v[0] for v in layer_samples], dtype=np.float32)
        h_vals = np.array([v[1] for v in layer_samples], dtype=np.float32)
        ascent = float(h_vals[-1] - h_vals[0])
        if ascent < 0.20:
            return None

        diffs = np.diff(h_vals)
        if np.any(diffs < -0.08):
            return None
        step_like = (diffs >= 0.06) & (diffs <= 0.28)
        if int(step_like.sum()) < 2:
            return None
        if len(f_vals) >= 3 and abs(float(np.corrcoef(f_vals, h_vals)[0, 1])) < 0.45:
            return None

        bottom_limit = f_vals[min(1, len(f_vals) - 1)] + 0.05
        bottom_pts = pts_world[forward <= bottom_limit]
        if bottom_pts.shape[0] == 0:
            bottom_pts = pts_world[np.argsort(forward)[: min(30, pts_world.shape[0])]]
        bottom_xy = (float(np.mean(bottom_pts[:, 0])), float(np.mean(bottom_pts[:, 1])))
        agent_x, agent_y, _ = self.one_map.voxel_to_metric(*agent_voxel)
        approach_vec = np.array([bottom_xy[0] - agent_x, bottom_xy[1] - agent_y], dtype=np.float32)
        approach_norm = float(np.linalg.norm(approach_vec))
        if approach_norm < 1e-3:
            forward_xy = np.array([forward_axis[0], forward_axis[1]], dtype=np.float32)
            forward_norm = float(np.linalg.norm(forward_xy))
            approach_vec = forward_xy / max(forward_norm, 1e-3)
        else:
            approach_vec = approach_vec / approach_norm
        approach_distance = float(getattr(self, "_rise_stair_approach_distance_m", 0.60))
        approach_xy = (
            float(bottom_xy[0] - approach_vec[0] * approach_distance),
            float(bottom_xy[1] - approach_vec[1] * approach_distance),
        )
        nav_vid = self._find_floor_nav_voxel_near_metric(bottom_xy, agent_voxel, radius=10)
        if nav_vid is None:
            nav_vid = self._find_floor_nav_voxel_near_metric(approach_xy, agent_voxel, radius=10)
        if nav_vid is None:
            return None

        geometry_score = min(1.0, ascent / 0.60) * 0.45
        geometry_score += min(1.0, width / 1.20) * 0.25
        geometry_score += min(1.0, float(step_like.sum()) / 3.0) * 0.30

        return {
            "nav_voxel_id": nav_vid,
            "box": tuple(float(v) for v in box),
            "confidence": float(score),
            "geometry_score": float(geometry_score),
            "bottom_xy": bottom_xy,
            "approach_xy": approach_xy,
        }

    def _validate_rise_stair_view_hint(
        self,
        box: np.ndarray,
        score: float,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Tuple[int, int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        Weaker upstairs stair geometry used only to navigate to a better view.

        This does not confirm stairs and is never used to run SAM directly. It
        exists for narrow/short stair entrances where only the lower few steps
        are visible from the current pose.
        """
        if score < self._rise_stair_view_hint_threshold:
            return None
        if self.one_map.fx is None or self.one_map.agent_height_0 is None:
            return None

        H, W = depth.shape
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
        y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        ys, xs = np.mgrid[y1:y2 + 1, x1:x2 + 1]
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        depths = depth[ys, xs]
        valid = np.isfinite(depths) & (depths > 0.05)
        if valid.sum() < 50:
            return None
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        depths = depths[valid].astype(np.float32)
        if xs.size > 2500:
            idx = np.linspace(0, xs.size - 1, 2500).astype(np.int64)
            xs = xs[idx]
            ys = ys[idx]
            depths = depths[idx]

        x_cam = (xs - self.one_map.cx) * depths / self.one_map.fx
        y_cam = (ys - self.one_map.cy) * depths / self.one_map.fy
        pts_cam = np.stack([depths, -x_cam, -y_cam], axis=-1)
        R_mat = odometry[:3, :3]
        t_vec = odometry[:3, 3]
        pts_world = (R_mat @ pts_cam.T).T + t_vec

        rel = pts_world - t_vec
        forward_axis = R_mat[:, 0]
        forward = rel @ forward_axis
        lateral = rel @ R_mat[:, 1]
        height = pts_world[:, 2] - self.one_map.agent_height_0

        in_range = (
            (forward >= 0.45) & (forward <= 6.0) &
            (height >= -0.05) & (height <= 1.8)
        )
        if in_range.sum() < 50:
            return None
        pts_world = pts_world[in_range]
        forward = forward[in_range]
        lateral = lateral[in_range]
        height = height[in_range]

        width = float(np.percentile(lateral, 90) - np.percentile(lateral, 10))
        if width < 0.30:
            return None
        if self._is_rise_candidate_wall_like(forward, lateral, height):
            return None

        bin_width = 0.25
        bin_ids = np.floor((forward - forward.min()) / bin_width).astype(np.int32)
        layer_samples = []
        for bin_id in sorted(set(bin_ids)):
            mask = bin_ids == bin_id
            if mask.sum() < 8:
                continue
            f_span = float(np.percentile(forward[mask], 90) - np.percentile(forward[mask], 10))
            if f_span < 0.06:
                continue
            f_med = float(np.median(forward[mask]))
            h_low = float(np.percentile(height[mask], 25))
            layer_samples.append((f_med, h_low, mask.sum(), f_span))

        if len(layer_samples) < 2:
            return None

        layer_samples.sort(key=lambda item: item[0])
        f_vals = np.array([v[0] for v in layer_samples], dtype=np.float32)
        h_vals = np.array([v[1] for v in layer_samples], dtype=np.float32)
        ascent = float(h_vals[-1] - h_vals[0])
        if ascent < 0.12:
            return None

        diffs = np.diff(h_vals)
        if np.any(diffs < -0.10):
            return None
        step_like = (diffs >= 0.06) & (diffs <= 0.22)
        if int(step_like.sum()) < 1:
            return None
        if len(f_vals) >= 3 and abs(float(np.corrcoef(f_vals, h_vals)[0, 1])) < 0.30:
            return None

        bottom_limit = f_vals[min(1, len(f_vals) - 1)] + 0.05
        bottom_pts = pts_world[forward <= bottom_limit]
        if bottom_pts.shape[0] == 0:
            bottom_pts = pts_world[np.argsort(forward)[: min(30, pts_world.shape[0])]]
        bottom_xy = (float(np.mean(bottom_pts[:, 0])), float(np.mean(bottom_pts[:, 1])))

        agent_x, agent_y, _ = self.one_map.voxel_to_metric(*agent_voxel)
        approach_vec = np.array([bottom_xy[0] - agent_x, bottom_xy[1] - agent_y], dtype=np.float32)
        approach_norm = float(np.linalg.norm(approach_vec))
        if approach_norm < 1e-3:
            forward_xy = np.array([forward_axis[0], forward_axis[1]], dtype=np.float32)
            forward_norm = float(np.linalg.norm(forward_xy))
            approach_vec = forward_xy / max(forward_norm, 1e-3)
        else:
            approach_vec = approach_vec / approach_norm
        approach_distance = float(getattr(self, "_rise_stair_approach_distance_m", 0.60))
        approach_xy = (
            float(bottom_xy[0] - approach_vec[0] * approach_distance),
            float(bottom_xy[1] - approach_vec[1] * approach_distance),
        )
        nav_vid = self._find_floor_nav_voxel_near_metric(bottom_xy, agent_voxel, radius=10)
        if nav_vid is None:
            nav_vid = self._find_floor_nav_voxel_near_metric(approach_xy, agent_voxel, radius=10)
        if nav_vid is None:
            return None

        geometry_score = 0.35
        geometry_score += min(1.0, ascent / 0.35) * 0.25
        geometry_score += min(1.0, width / 0.80) * 0.15
        geometry_score += min(1.0, float(step_like.sum()) / 2.0) * 0.15

        return {
            "nav_voxel_id": nav_vid,
            "box": tuple(float(v) for v in box),
            "confidence": float(score),
            "geometry_score": float(min(0.70, geometry_score)),
            "bottom_xy": bottom_xy,
            "approach_xy": approach_xy,
            "view_hint": True,
        }

    def _validate_contextual_rise_stair_candidate(
        self,
        box: np.ndarray,
        score: float,
        agent_voxel: Tuple[int, int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        Conservative fallback for small upstairs landings.

        Sometimes the next stair flight is visible semantically from a tiny
        landing, but the depth box is too partial/occluded to pass the stepped
        geometry verifier. In that context, a strong stairs box can create a
        rise frontier anchored on the current small traversable cluster. SAM
        confirmation is still required before entering stair semantic mode.
        """
        if score < self._rise_stair_context_threshold:
            return None

        # Do not let this become a general DINO trigger. It is only allowed
        # during the explicit post-upstairs scan or when normal frontiers are
        # already exhausted on a small traversable patch.
        allow_context = self._post_upstairs_probe_state == "scan"
        if not allow_context and len(getattr(self, "frontier_clusters", [])) == 0:
            cluster = self._current_floor_traversable_cluster(agent_voxel)
            allow_context = self._is_small_traversable_cluster(cluster)
        else:
            cluster = self._current_floor_traversable_cluster(agent_voxel) if allow_context else set()
        if not allow_context:
            return None
        if not cluster:
            cluster = self._current_floor_traversable_cluster(agent_voxel)
        if not self._is_small_traversable_cluster(cluster):
            return None

        nav_vid = self._find_cluster_center_voxel(cluster, agent_voxel)
        if nav_vid is None:
            nav_vid = self.one_map._find_nearest_safe_traversable(agent_voxel, max_radius=4)
        if nav_vid is None:
            return None

        return {
            "nav_voxel_id": nav_vid,
            "box": tuple(float(v) for v in box),
            "confidence": float(score),
            "geometry_score": 0.20,
            "bottom_xy": None,
            "approach_xy": None,
            "contextual": True,
        }

    def _update_rise_stair_candidates(
        self,
        stair_dets: dict,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Tuple[int, int, int],
    ) -> None:
        """Update upstairs stair candidates from current-frame stair detections."""
        self._expire_rise_stair_candidates()
        self._expire_recent_down_stair_regions()
        self._purge_recent_down_stair_rise_candidates()
        if agent_voxel is None or self.stair_sem_mode or self._fake_frontier_pitched:
            return
        if not stair_dets.get("boxes"):
            return

        now = self.one_map._iters
        for box, cls, score in zip(
            stair_dets.get("boxes", []),
            stair_dets.get("classes", []),
            stair_dets.get("scores", []),
        ):
            validation = self._validate_rise_stair_candidate(
                np.asarray(box, dtype=np.float32), float(score), depth, odometry, agent_voxel
            )
            if validation is None:
                validation = self._validate_rise_stair_view_hint(
                    np.asarray(box, dtype=np.float32), float(score), depth, odometry, agent_voxel
                )
            if validation is None:
                validation = self._validate_contextual_rise_stair_candidate(
                    np.asarray(box, dtype=np.float32), float(score), agent_voxel
                )
                if validation is None:
                    continue
            nav_vid = validation["nav_voxel_id"]
            if self._is_in_recent_down_stair_region(nav_vid):
                print(
                    f"[Nav3D][RiseDebug] skip current-frame rise proposal in recent_down_region: "
                    f"nav={nav_vid}, score={float(score):.2f}, geom={validation['geometry_score']:.2f}"
                )
                continue
            matched_key = self._find_matching_rise_candidate(nav_vid)
            prev = self.rise_stair_candidates.get(matched_key, {}) if matched_key is not None else {}
            keep_active_nav = bool(prev.get("active", False) and matched_key is not None)
            if matched_key is not None and matched_key != nav_vid and not keep_active_nav:
                self.rise_stair_candidates.pop(matched_key, None)
            if keep_active_nav and matched_key != nav_vid:
                print(
                    f"[Nav3D][RiseDebug] keep active rise nav stable: "
                    f"key={matched_key}, new_obs_nav={nav_vid}, "
                    f"stable_nav={prev.get('nav_voxel_id', matched_key)}, "
                    f"prev_seen={prev.get('seen_count', 0)}"
                )
            key = tuple(matched_key) if keep_active_nav else nav_vid
            stable_nav_vid = tuple(prev.get("nav_voxel_id", key)) if keep_active_nav else nav_vid
            seen_count = prev.get("seen_count", 0) + 1
            active = (
                float(score) >= self._rise_stair_strong_threshold or
                seen_count >= self._rise_stair_required_seen
            )
            # Confidence, geometry, and seen_count decide whether the proposal
            # is valid/active; they no longer rank active ascent proposals.
            # Keep the legacy semantic_score field as a distance-priority
            # compatibility value. FrontierManager recomputes it from the
            # current agent position immediately before final selection.
            ax, ay, az = self.one_map.voxel_to_metric(*agent_voxel)
            gx, gy, gz = self.one_map.voxel_to_metric(*stable_nav_vid)
            proposal_distance_m = float(
                ((gx - ax) ** 2 + (gy - ay) ** 2 + (gz - az) ** 2) ** 0.5
            )
            semantic_score = 1.0 / (1.0 + proposal_distance_m)
            self.rise_stair_candidates[key] = {
                "nav_voxel_id": stable_nav_vid,
                "box": validation["box"],
                "confidence": max(float(score), prev.get("confidence", 0.0)),
                "geometry_score": max(
                    validation["geometry_score"],
                    prev.get("geometry_score", 0.0),
                ),
                "semantic_score": semantic_score,
                "seen_count": seen_count,
                "last_seen_step": now,
                "active": active,
                "contextual": validation.get("contextual", False),
                "view_hint": validation.get("view_hint", False),
                "bottom_xy": validation.get("bottom_xy"),
                "approach_xy": validation.get("approach_xy"),
            }
            if active:
                print(f"[Nav3D] Rise stair candidate active: nav={key}, "
                      f"seen={seen_count}, conf={float(score):.2f}, "
                      f"geom={validation['geometry_score']:.2f}, "
                      f"contextual={validation.get('contextual', False)}, "
                      f"view_hint={validation.get('view_hint', False)}")

    def _count_stair_near(
        self,
        center: Tuple[int, int, int],
        radius_xy: int = 12,
        min_z: Optional[int] = None,
    ) -> int:
        """Count semantic stair voxels near a candidate."""
        from mapping.voxel_map import LOCKED_TYPES

        count = 0
        for vid, vd in self.one_map.voxels.items():
            if vd.voxel_type not in LOCKED_TYPES:
                continue
            if min_z is not None and vid[2] < min_z:
                continue
            if abs(vid[0] - center[0]) <= radius_xy and abs(vid[1] - center[1]) <= radius_xy:
                count += 1
        return count

    def _confirm_rise_stair_frontier(
        self,
        frontier: RiseStairFrontier3D,
        stair_dets: dict,
        image: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Tuple[int, int, int],
    ) -> Optional[bool]:
        """Run SAM on a verified rise stair box and enter upward stair semantic mode."""
        best = None
        checked = 0
        if stair_dets.get("boxes"):
            for box, cls, score in zip(
                stair_dets.get("boxes", []),
                stair_dets.get("classes", []),
                stair_dets.get("scores", []),
            ):
                checked += 1
                validation = self._validate_rise_stair_candidate(
                    np.asarray(box, dtype=np.float32), float(score), depth, odometry, agent_voxel
                )
                if validation is None:
                    continue
                if best is None or validation["geometry_score"] > best["geometry_score"]:
                    best = validation

        if best is None:
            if (
                self._rise_confirm_scan_goal == frontier.nav_voxel_id and
                self._rise_confirm_scan_remaining <= 0
            ):
                print("[Nav3D] Rise stair SAM confirmation rejected after scan: "
                      "no current-frame stairs box passed geometry")
                self._reset_rise_confirm_scan()
                self.rise_stair_candidates.pop(frontier.nav_voxel_id, None)
                return False
            scores = [float(s) for s in stair_dets.get("scores", [])]
            print(
                f"[Nav3D] Rise stair confirmation waiting: goal={frontier.nav_voxel_id}, "
                f"checked={checked}, scores={[round(s, 2) for s in scores[:4]]}, "
                f"scan_goal={self._rise_confirm_scan_goal}, "
                f"scan_remaining={self._rise_confirm_scan_remaining}"
            )
            self._start_rise_confirm_scan(frontier, odometry)
            return None

        self._reset_rise_confirm_scan()
        print(
            f"[Nav3D] Rise stair confirmation geometry passed: "
            f"goal={frontier.nav_voxel_id}, conf={float(best['confidence']):.2f}, "
            f"geom={float(best['geometry_score']):.2f}"
        )
        mem_id = self._matching_stair_memory_id(frontier.nav_voxel_id, direction="up")
        if mem_id is None:
            mem_id = self._remember_stair_frontier(
                "up", frontier.nav_voxel_id, source_box=frontier.candidate_box
            )
        self._active_stair_memory_id = mem_id

        dets = {
            "boxes": [np.asarray(best["box"], dtype=np.float32)],
            "classes": ["stairs"],
            "scores": [float(best["confidence"])],
        }
        before = self._count_stair_near(frontier.nav_voxel_id, min_z=agent_voxel[2])
        marked = self._mark_stair_with_sam(dets, image, depth, odometry)
        expanded = self._expand_stair_26_neighbors()
        demoted = self._demote_short_stair_clusters()
        after = self._count_stair_near(frontier.nav_voxel_id, min_z=agent_voxel[2])
        if after < 3 and after - before < 3:
            print(f"[Nav3D] Rise stair SAM confirmation failed: stair_count {before}->{after}")
            if demoted and self._active_stair_memory_id is not None and self._active_stair_memory_id in self.stair_frontier_memory:
                self.stair_frontier_memory[self._active_stair_memory_id]["status"] = "blocked"
                self._active_stair_memory_id = None
            return False

        entered = self._enter_stair_sem_mode(
            dets,
            image,
            depth,
            odometry,
            agent_voxel,
            direction="up",
            verified_entry=True,
            initial_session_voxels=marked | expanded,
        )
        if not entered:
            return False
        self.rise_stair_candidates.pop(frontier.nav_voxel_id, None)
        return True

    def _post_upstairs_probe_active(self) -> bool:
        return self._post_upstairs_probe_state != "inactive" or self._post_upstairs_probe_pending
