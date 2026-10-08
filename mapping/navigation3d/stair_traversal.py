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

def stair_mode_label(direction, semantic: bool) -> str:
    value = StairDirection.normalize(direction).value
    return f"StairSem-{value}" if semantic else f"Explore-{value}"

class StairTraversalMixin:
    def _is_consumed_or_blocked_stair_region(
        self,
        nav_voxel: Tuple[int, int, int],
        direction: Optional[str] = None,
    ) -> bool:
        mem_id = self._matching_stair_memory_id(nav_voxel, direction=direction)
        if mem_id is None:
            return False
        return self.stair_frontier_memory[mem_id].get("status") == "blocked"

    def _reset_stair_sem_stability(self) -> None:
        """Reset per-entry stair semantic stabilization state."""
        self._stair_sem_entry_iter = 0
        self._stair_sem_down_floor_counter = 0
        self._stair_sem_up_floor_counter = 0
        self._stair_sem_locked_frontier = None
        self._stair_sem_descend_corridor = None
        self._stair_sem_xy_refined = False
        self._stair_sem_last_replan_iter = -999

    def _try_lock_descend_session_corridor(
        self,
        agent_voxel: Tuple[int, int, int],
        reference: Optional[StairSemanticFrontier3D],
    ) -> None:
        """Estimate and lock the lateral width of the current descend flight.

        Only SAM-confirmed STAIR voxels between the agent and the current
        target participate.  STAIR_EXPANDED and endpoint platforms must not
        widen flight A into an adjacent flight B.  Weak evidence leaves the
        corridor unset and is retried after later mapping frames.
        """
        if self._stair_sem_descend_corridor is not None or reference is None:
            return

        target_z = int(reference.nav_voxel_id[2])
        z_low = min(target_z, int(agent_voxel[2]))
        z_high = max(target_z, int(agent_voxel[2]))
        layers: Dict[int, Set[Tuple[int, int]]] = {}
        for vid in self._stair_sem_session_voxels:
            if not (z_low <= vid[2] <= z_high):
                continue
            vd = self.one_map.voxels.get(vid)
            if vd is None or vd.voxel_type != VoxelType.STAIR:
                continue
            layers.setdefault(int(vid[2]), set()).add((int(vid[0]), int(vid[1])))

        records = []
        for z_val, xy_values in layers.items():
            if len(xy_values) < 3:
                continue
            xy_array = np.asarray(sorted(xy_values), dtype=np.float64)
            records.append((z_val, xy_array, np.median(xy_array, axis=0)))
        records.sort(key=lambda item: item[0])
        if len(records) < 3:
            return

        layer_zs = np.asarray([item[0] for item in records], dtype=np.float64)
        layer_centers = np.stack([item[2] for item in records], axis=0)
        z_centered = layer_zs - float(np.mean(layer_zs))
        z_energy = float(np.dot(z_centered, z_centered))
        if z_energy <= 1e-6:
            return
        xy_centered = layer_centers - np.mean(layer_centers, axis=0, keepdims=True)
        slope_xy = (z_centered[:, None] * xy_centered).sum(axis=0) / z_energy
        slope_norm = float(np.linalg.norm(slope_xy))
        if slope_norm < 0.25:
            return

        travel_axis = slope_xy / slope_norm
        width_axis = np.asarray([-travel_axis[1], travel_axis[0]], dtype=np.float64)
        longitudinal = layer_centers @ travel_axis
        if float(np.std(longitudinal)) <= 1e-6:
            return
        direction_corr = float(abs(np.corrcoef(layer_zs, longitudinal)[0, 1]))
        travel_span = float(np.ptp(longitudinal))
        lateral_centers = layer_centers @ width_axis
        lateral_center = float(np.median(lateral_centers))
        lateral_residual = np.abs(lateral_centers - lateral_center)
        if not (
            direction_corr >= 0.60 and
            travel_span >= 2.0 and
            float(np.percentile(lateral_residual, 90)) <= 3.0
        ):
            return

        width_records = records[1:-1] if len(records) >= 5 else records
        lows = []
        highs = []
        for _, xy_array, _ in width_records:
            lateral = xy_array @ width_axis
            lows.append(float(np.percentile(lateral, 10)))
            highs.append(float(np.percentile(lateral, 90)))
        corridor_low = float(np.median(lows))
        corridor_high = float(np.median(highs))
        if corridor_high < corridor_low:
            corridor_low, corridor_high = corridor_high, corridor_low
        if corridor_high - corridor_low < 2.0:
            corridor_mid = 0.5 * (corridor_low + corridor_high)
            corridor_low = corridor_mid - 1.0
            corridor_high = corridor_mid + 1.0

        self._stair_sem_descend_corridor = {
            "width_axis": (float(width_axis[0]), float(width_axis[1])),
            "low": corridor_low,
            "high": corridor_high,
            "margin": 1.0,
        }
        print(
            f"[Nav3D] StairSem-descend corridor locked: "
            f"width_axis=({width_axis[0]:.3f},{width_axis[1]:.3f}), "
            f"lateral=[{corridor_low:.2f},{corridor_high:.2f}], margin=1.0, "
            f"layers={len(records)}, corr={direction_corr:.2f}"
        )

    def _is_valid_stair_sem_frontier_for_direction(
        self,
        frontier: Optional[StairSemanticFrontier3D],
        agent_voxel: Tuple[int, int, int],
    ) -> bool:
        if frontier is None:
            return False
        nav_vid = frontier.nav_voxel_id
        if self._stair_sem_direction == "up":
            # Upward stair semantic targets may be floor/frontier voxels at the
            # landing. Keep them valid briefly after the agent reaches nearly the
            # same z layer, so navigation can finish the XY approach before the
            # force-forward/platform probe handoff.
            return (nav_vid[2] + 4 > agent_voxel[2]) and frontier.z_difference < 0
        # Descend targets are locked for the current semantic session.  Keep a
        # locked target valid when the agent reaches the same z or passes it by
        # at most three layers, so the remaining XY approach can complete and
        # trigger the normal 3-D arrival check instead of dropping the goal.
        return (nav_vid[2] - 4 < agent_voxel[2]) and frontier.z_difference > 0

    def _stair_sem_target_is_pathable(self, frontier: Optional[StairSemanticFrontier3D]) -> bool:
        if frontier is None:
            return False
        from mapping.voxel_map import TARGETABLE_TYPES

        vd = self.one_map.voxels.get(frontier.nav_voxel_id)
        return vd is not None and vd.voxel_type in TARGETABLE_TYPES

    def _update_stair_sem_session_voxels(self, agent_voxel: Tuple[int, int, int]) -> None:
        """Incrementally keep the stair semantic frontier local to the current stair segment."""
        from mapping.voxel_map import LOCKED_TYPES

        anchors = set(self._stair_sem_session_voxels)
        anchors.add(agent_voxel)
        if self._current_down_stair_anchor is not None and self._stair_sem_direction == "down":
            anchors.add(self._current_down_stair_anchor)

        if not anchors:
            return

        radius_m = float(self._stair_sem_session_radius_m)
        radius_sq = radius_m * radius_m
        voxel_size_xy = float(self.one_map.voxel_size)
        voxel_size_z = float(self.one_map.voxel_size_z)
        for vid, vd in self.one_map.voxels.items():
            if vd.voxel_type not in LOCKED_TYPES:
                continue
            for anchor in anchors:
                dx = float(vid[0] - anchor[0]) * voxel_size_xy
                dy = float(vid[1] - anchor[1]) * voxel_size_xy
                dz = float(vid[2] - anchor[2]) * voxel_size_z
                if dx * dx + dy * dy + dz * dz <= radius_sq:
                    self._stair_sem_session_voxels.add(vid)
                    break

    def _refresh_stair_sem_frontier(self, agent_voxel: Tuple[int, int, int]) -> None:
        """Refresh stair evidence while keeping the session's semantic target stable."""
        self._update_stair_sem_session_voxels(agent_voxel)
        exclude_regions = None
        if self._stair_sem_direction == "down":
            self._expire_recent_down_stair_regions()
            exclude_regions = self._recent_down_stair_regions
        candidate = create_stair_semantic_frontier(
            self.one_map,
            agent_voxel,
            direction=self._stair_sem_direction,
            exclude_regions=exclude_regions,
            include_voxels=set(self._stair_sem_session_voxels),
            lateral_corridor=(
                self._stair_sem_descend_corridor
                if self._stair_sem_direction == "down"
                else None
            ),
        )
        if not self._is_valid_stair_sem_frontier_for_direction(candidate, agent_voxel):
            candidate = None

        if self._stair_sem_direction == "down":
            locked = self._stair_sem_locked_frontier
            corridor_was_unset = self._stair_sem_descend_corridor is None
            self._try_lock_descend_session_corridor(
                agent_voxel,
                locked if locked is not None else candidate,
            )
            if corridor_was_unset and self._stair_sem_descend_corridor is not None:
                # The first candidate was computed before the persistent
                # corridor existed.  Recompute it immediately so an adjacent
                # lower flight cannot take ownership during this one frame.
                candidate = create_stair_semantic_frontier(
                    self.one_map,
                    agent_voxel,
                    direction="down",
                    exclude_regions=exclude_regions,
                    include_voxels=set(self._stair_sem_session_voxels),
                    lateral_corridor=self._stair_sem_descend_corridor,
                )
                if not self._is_valid_stair_sem_frontier_for_direction(candidate, agent_voxel):
                    candidate = None
            locked_valid = (
                self._is_valid_stair_sem_frontier_for_direction(locked, agent_voxel) and
                self._stair_sem_target_is_pathable(locked)
            )
            if not locked_valid:
                previous_nav = None if locked is None else locked.nav_voxel_id
                locked = candidate
                if locked is not None and locked.nav_voxel_id != previous_nav:
                    reason = "initial" if previous_nav is None else "previous_invalid"
                    print(
                        f"[Nav3D] StairSem-descend target locked: "
                        f"nav={locked.nav_voxel_id}, reason={reason}"
                    )
            elif (
                candidate is not None and
                self._stair_sem_target_is_pathable(candidate) and
                candidate.nav_voxel_id[2] <= locked.nav_voxel_id[2] - 2 and
                self._stair_sem_force_forward == 0
            ):
                # Keep same/near-layer descend targets stable, but do not let
                # that hysteresis hide newly restored stair evidence farther
                # down the current flight.  Descend progress is monotonic:
                # only a candidate at least two z layers lower may replace a
                # still-valid lock.  Clear the old path so the new semantic
                # frontier takes ownership immediately.
                previous_nav = locked.nav_voxel_id
                locked = candidate
                self.path = None
                self.path_id = 0
                self.last_nav_goal = None
                self._stair_sem_last_replan_iter = -999
                print(
                    f"[Nav3D] StairSem-descend target advanced: "
                    f"{previous_nav}->{locked.nav_voxel_id}, "
                    f"z_drop={previous_nav[2] - locked.nav_voxel_id[2]}"
                )
            self._stair_sem_locked_frontier = locked
            self._stair_sem_frontier = locked
            return

        locked = self._stair_sem_locked_frontier
        locked_valid = (
            self._is_valid_stair_sem_frontier_for_direction(locked, agent_voxel) and
            self._stair_sem_target_is_pathable(locked)
        )

        if not locked_valid:
            locked = candidate
        elif candidate is not None:
            entry_age = max(0, self.one_map._iters - self._stair_sem_entry_iter)
            refine_ready = (
                not self._stair_sem_xy_refined and
                entry_age >= self._stair_sem_xy_refine_delay_frames and
                self._stair_sem_force_forward == 0
            )
            # Both the higher-layer update and the same/near-layer recenter
            # need the displacement between the old lock and the new
            # candidate.  Compute it before either branch can replace
            # ``locked``; otherwise the higher-layer branch may reference an
            # uninitialized value (or would measure zero after assignment).
            xy_shift = self._xy_distance(candidate.nav_voxel_id, locked.nav_voxel_id)
            if candidate.nav_voxel_id[2] >= locked.nav_voxel_id[2] + self._stair_sem_up_continue_margin:
                # 更高层 → 直接更新
                old_nav = locked.nav_voxel_id
                locked = candidate
                if refine_ready and xy_shift > 0:
                    self._stair_sem_xy_refined = True
                    print(
                        f"[Nav3D] StairSem-ascent delayed target refinement: "
                        f"{old_nav}->{candidate.nav_voxel_id}, age={entry_age}, reason=higher_layer"
                    )
            else:
                if refine_ready:
                    # The first SAM view commonly contains only one edge of the
                    # staircase.  Keep that initial goal briefly, then accept
                    # exactly one same/near-layer recenter even when the shift
                    # is below the normal 3-voxel hysteresis threshold.
                    old_nav = locked.nav_voxel_id
                    locked = candidate
                    self._stair_sem_xy_refined = True
                    print(
                        f"[Nav3D] StairSem-ascent delayed target refinement: "
                        f"{old_nav}->{candidate.nav_voxel_id}, age={entry_age}, "
                        f"xy_shift={xy_shift:.2f}"
                    )
                elif self._stair_sem_xy_refined and xy_shift >= 3:
                    # After the one-time refinement, restore the original
                    # hysteresis so the goal does not oscillate frame to frame.
                    locked = candidate

        self._stair_sem_locked_frontier = locked
        self._stair_sem_frontier = locked

    def _has_upward_stair_continuation(self, agent_voxel: Tuple[int, int, int]) -> bool:
        """Return True when an upward stair semantic target still exists ahead."""
        frontier = create_stair_semantic_frontier(self.one_map, agent_voxel, direction="up")
        return (
            frontier is not None and
            self._is_valid_stair_sem_frontier_for_direction(frontier, agent_voxel) and
            frontier.nav_voxel_id[2] >= agent_voxel[2] + self._stair_sem_up_continue_margin
        )

    def _fake_frontier_blacklist_region(self, cluster: FrontierCluster3D) -> Dict[str, Any]:
        aabb = cluster.fake_anchor_aabb or cluster.aabb
        center = self._anchor_center(cluster)
        return {
            "center": center,
            "aabb": aabb,
            "radius_xy": 8,
            "last_step": self.one_map._iters,
        }

    def _is_fake_frontier_region_blacklisted(self, cluster: FrontierCluster3D) -> bool:
        if not cluster.is_stair_frontier:
            return False
        if not getattr(self, "blacklisted_fake_frontier_regions", None):
            return False
        center = self._anchor_center(cluster)
        aabb = cluster.fake_anchor_aabb or cluster.aabb
        for region in self.blacklisted_fake_frontier_regions:
            rcenter = region["center"]
            if abs(center[2] - rcenter[2]) > 2:
                continue
            radius_xy = int(region.get("radius_xy", 8))
            if self._xy_distance(center, rcenter) <= radius_xy:
                return True
            raabb = region["aabb"]
            if (
                aabb[0][0] <= raabb[1][0] + 2 and aabb[1][0] >= raabb[0][0] - 2 and
                aabb[0][1] <= raabb[1][1] + 2 and aabb[1][1] >= raabb[0][1] - 2 and
                aabb[0][2] <= raabb[1][2] + 1 and aabb[1][2] >= raabb[0][2] - 1
            ):
                return True
        return False

    def _down_stair_geometry_score(
        self,
        anchor: Tuple[int, int, int],
        box: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Cross-check a downstairs stair proposal with depth and historical obstacle voxels.

        Stairs should form local, wide-enough, descending structure near the fake
        anchor. Vertical walls tend to have many z values in very few xy columns
        and little stepped forward progression.
        """
        depth_voxels = self._box_depth_voxels(box, depth, odometry, max_points=2500)
        local_depth = [
            v for v in depth_voxels
            if self._xy_distance(v, anchor) <= self._down_stair_anchor_radius_xy
        ]

        historical = []
        for vid, vd in self.one_map.voxels.items():
            if vd.voxel_type != VoxelType.OCCUPIED:
                continue
            if self._xy_distance(vid, anchor) > self._down_stair_anchor_radius_xy:
                continue
            if agent_voxel[2] - 24 <= vid[2] <= agent_voxel[2] + 4:
                historical.append(vid)

        pts = list(local_depth) + historical
        if len(pts) < 20:
            return {"passed": False, "geometry_score": 0.0, "reason": "too_few_local_points"}

        below = [v for v in pts if v[2] <= agent_voxel[2] - 2]
        if len(below) < 15:
            return {"passed": False, "geometry_score": 0.0, "reason": "too_few_below_points"}

        xs = np.array([v[0] for v in below], dtype=np.float32)
        ys = np.array([v[1] for v in below], dtype=np.float32)
        zs = np.array([v[2] for v in below], dtype=np.float32)
        width_x = float(xs.max() - xs.min() + 1.0)
        width_y = float(ys.max() - ys.min() + 1.0)
        width = max(width_x, width_y)
        z_span = float(zs.max() - zs.min() + 1.0)
        if width < 3.0:
            return {"passed": False, "geometry_score": 0.0, "reason": "too_narrow"}

        xy_columns: Dict[Tuple[int, int], List[int]] = {}
        for v in below:
            xy_columns.setdefault((v[0], v[1]), []).append(v[2])
        tall_columns = sum(1 for vals in xy_columns.values() if max(vals) - min(vals) >= 8)
        vertical_ratio = tall_columns / max(1, len(xy_columns))
        if vertical_ratio > 0.45 and len(xy_columns) < 18:
            return {"passed": False, "geometry_score": 0.05, "reason": "vertical_wall_columns"}

        def axis_layer_samples(axis: np.ndarray) -> List[Tuple[float, float, int]]:
            if float(axis.max() - axis.min()) < 2.0:
                return []
            bin_ids = np.floor((axis - axis.min()) / 2.0).astype(np.int32)
            samples = []
            for bin_id in sorted(set(bin_ids)):
                mask = bin_ids == bin_id
                if int(mask.sum()) < 4:
                    continue
                p_med = float(np.median(axis[mask]))
                # Low percentile captures the visible foothold/step surface and
                # is robust to vertical riser points in the same bin.
                z_low = float(np.percentile(zs[mask], 25))
                samples.append((p_med, z_low, int(mask.sum())))
            return samples

        axis_candidates = []
        for samples in (axis_layer_samples(xs), axis_layer_samples(ys)):
            if len(samples) < 2:
                continue
            samples.sort(key=lambda item: item[0])
            z_candidate = np.array([item[1] for item in samples], dtype=np.float32)
            diffs_candidate = np.diff(z_candidate)
            axis_score = float(abs(z_candidate[-1] - z_candidate[0]))
            axis_score += float(np.sum(np.abs(diffs_candidate) >= 1.0))
            axis_candidates.append((axis_score, samples))

        if not axis_candidates:
            return {"passed": False, "geometry_score": 0.1, "reason": "too_few_forward_layers"}

        axis_candidates.sort(key=lambda item: item[0], reverse=True)
        layer_samples = axis_candidates[0][1]
        layer_samples.sort(key=lambda item: item[0])
        z_layers = np.array([item[1] for item in layer_samples], dtype=np.float32)
        diffs = np.diff(z_layers)
        trend = float(abs(z_layers[-1] - z_layers[0]))
        descending_steps = int(np.sum(np.abs(diffs) >= 1.0))
        stable_steps = int(np.sum(np.abs(diffs) <= 5.0))
        mid_layers = [
            z for z in sorted(set(int(v[2]) for v in below))
            if agent_voxel[2] - 12 <= z <= agent_voxel[2] - 2
        ]

        if len(mid_layers) < self._down_stair_confirmation_min_mid_layers:
            return {"passed": False, "geometry_score": 0.1, "reason": "missing_mid_descent_layers"}
        if min(int(v[2]) for v in below) < agent_voxel[2] - 24 and len(mid_layers) < 4:
            return {"passed": False, "geometry_score": 0.1, "reason": "direct_lower_floor_leak"}
        if descending_steps < 1 and trend < 2.0:
            return {"passed": False, "geometry_score": 0.15, "reason": "no_descending_trend"}
        if stable_steps < 1:
            return {"passed": False, "geometry_score": 0.15, "reason": "height_jumps_too_large"}

        score = 0.0
        score += min(1.0, len(below) / 80.0) * 0.20
        score += min(1.0, width / 10.0) * 0.20
        score += min(1.0, len(layer_samples) / 4.0) * 0.20
        score += min(1.0, max(trend, float(len(mid_layers))) / 5.0) * 0.25
        score += max(0.0, 1.0 - vertical_ratio) * 0.15

        return {
            "passed": score >= self._down_stair_geometry_score_threshold,
            "geometry_score": float(score),
            "reason": "passed" if score >= self._down_stair_geometry_score_threshold else "score_below_threshold",
            "mid_layers": len(mid_layers),
            "vertical_ratio": float(vertical_ratio),
        }

    def _down_stair_mask_geometry_score(
        self,
        anchor: Tuple[int, int, int],
        mask_voxels: List[Tuple[int, int, int]],
    ) -> Dict[str, Any]:
        """Validate the actual SAM mask before converting its voxels to STAIR."""
        local = [
            v for v in mask_voxels
            if self._xy_distance(v, anchor) <= self._down_stair_anchor_radius_xy
        ]
        if len(local) < 15:
            return {"passed": False, "geometry_score": 0.0, "reason": "mask_too_far_or_small"}

        anchor_z = anchor[2]
        below = [v for v in local if v[2] <= anchor_z - 2]
        if len(below) < 10:
            return {"passed": False, "geometry_score": 0.0, "reason": "mask_not_below_anchor"}

        xs = np.array([v[0] for v in below], dtype=np.float32)
        ys = np.array([v[1] for v in below], dtype=np.float32)
        zs = np.array([v[2] for v in below], dtype=np.float32)
        width = max(float(xs.max() - xs.min() + 1.0), float(ys.max() - ys.min() + 1.0))
        if width < 3.0:
            return {"passed": False, "geometry_score": 0.05, "reason": "mask_too_narrow"}

        xy_columns: Dict[Tuple[int, int], List[int]] = {}
        for v in below:
            xy_columns.setdefault((v[0], v[1]), []).append(v[2])
        tall_columns = sum(1 for vals in xy_columns.values() if max(vals) - min(vals) >= 8)
        vertical_ratio = tall_columns / max(1, len(xy_columns))
        if vertical_ratio > 0.55 and len(xy_columns) < 18:
            return {"passed": False, "geometry_score": 0.05, "reason": "mask_vertical_wall"}

        mid_layers = [
            z for z in sorted(set(int(v[2]) for v in below))
            if anchor_z - 12 <= z <= anchor_z - 2
        ]
        if len(mid_layers) < self._down_stair_confirmation_min_mid_layers:
            return {"passed": False, "geometry_score": 0.10, "reason": "mask_missing_mid_layers"}

        score = 0.0
        score += min(1.0, len(below) / 80.0) * 0.25
        score += min(1.0, width / 10.0) * 0.25
        score += min(1.0, len(mid_layers) / 4.0) * 0.30
        score += max(0.0, 1.0 - vertical_ratio) * 0.20
        return {
            "passed": score >= 0.50,
            "geometry_score": float(score),
            "reason": "passed" if score >= 0.50 else "mask_score_below_threshold",
        }

    def _expire_down_stair_semantic_hints(self) -> None:
        now = self.one_map._iters
        self.down_stair_semantic_hints = [
            hint for hint in self.down_stair_semantic_hints
            if now - hint.get("last_seen_step", now) <= self._down_stair_hint_ttl
        ]

    def _update_down_stair_semantic_hints(
        self,
        stair_dets: dict,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Optional[Tuple[int, int, int]],
    ) -> None:
        self._expire_down_stair_semantic_hints()
        if agent_voxel is None or self.stair_sem_mode or self._fake_frontier_pitched:
            return
        if not stair_dets.get("boxes"):
            return

        for box, score in zip(stair_dets.get("boxes", []), stair_dets.get("scores", [])):
            if float(score) < self._down_stair_hint_threshold:
                continue
            voxels = self._box_depth_voxels(np.asarray(box, dtype=np.float32), depth, odometry, max_points=1200)
            below = [v for v in voxels if v[2] <= agent_voxel[2] - 2]
            if len(below) < 10:
                continue
            center = (
                int(round(np.mean([v[0] for v in below]))),
                int(round(np.mean([v[1] for v in below]))),
                int(round(np.mean([v[2] for v in below]))),
            )
            self.down_stair_semantic_hints.append({
                "voxel": center,
                "confidence": float(score),
                "last_seen_step": self.one_map._iters,
            })

    def _down_stair_hint_bonus(self, cluster: FrontierCluster3D) -> float:
        self._expire_down_stair_semantic_hints()
        if not cluster.is_stair_frontier or not self.down_stair_semantic_hints:
            return 0.0
        center = self._anchor_center(cluster)
        best = 0.0
        for hint in self.down_stair_semantic_hints:
            hv = hint["voxel"]
            dist_xy = self._xy_distance(center, hv)
            if dist_xy > self._down_stair_anchor_radius_xy:
                continue
            if hv[2] > center[2] + 2:
                continue
            best = max(best, 250.0 * float(hint.get("confidence", 0.0)) - 8.0 * dist_xy)
        return max(0.0, best)

    def _filter_down_stair_detections_for_anchor(
        self,
        stair_dets: dict,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Optional[Tuple[int, int, int]],
    ) -> dict:
        filtered = {"boxes": [], "classes": [], "scores": [], "masks": []}
        if agent_voxel is None or self._fake_frontier_target is None:
            return filtered
        anchor_cluster = self._fake_frontier_cluster
        anchor = self._anchor_center(anchor_cluster) if anchor_cluster is not None else self._fake_frontier_target
        radius_xy = self._down_stair_anchor_radius_xy

        stair_masks = stair_dets.get("masks", [])
        for det_index, (box, cls, score) in enumerate(zip(
            stair_dets.get("boxes", []),
            stair_dets.get("classes", []),
            stair_dets.get("scores", []),
        )):
            if float(score) < self._down_stair_pitched_conf_threshold:
                continue
            geometry = self._down_stair_geometry_score(
                anchor, np.asarray(box, dtype=np.float32), depth, odometry, agent_voxel
            )
            if not geometry["passed"]:
                print(f"[Nav3D] Down stair geometry rejected box conf={float(score):.2f}: "
                      f"{geometry['reason']} score={geometry['geometry_score']:.2f}")
                continue

            filtered["boxes"].append(box)
            filtered["classes"].append(cls)
            filtered["scores"].append(score)
            if det_index < len(stair_masks):
                filtered["masks"].append(stair_masks[det_index])
            filtered.setdefault("geometry_scores", []).append(float(geometry["geometry_score"]))

        return filtered

    def _selected_down_stair_goal(self) -> Optional[FrontierCluster3D]:
        """Return the currently selected down-stair frontier, if it really owns navigation."""
        if self.navigation_mode == "navigate" or self.object_detected:
            return None
        goal = self.last_nav_goal
        if not isinstance(goal, FrontierCluster3D) or not goal.is_stair_frontier:
            return None
        if goal.nav_voxel_id in set(tuple(b) if not isinstance(b, tuple) else b for b in self.blacklisted_goals):
            return None
        if self._is_fake_frontier_region_blacklisted(goal):
            return None

        # nav_goals is a priority-ordered candidate list, but committed-goal
        # persistence may intentionally select an entry other than index 0.
        # Require the actual last_nav_goal to remain in that list; checking only
        # nav_goals[0] clears a valid descend fallback before path planning can
        # confirm its committed owner.
        if not self.nav_goals:
            return None
        selected_is_current = any(
            isinstance(candidate, FrontierCluster3D) and
            candidate.is_stair_frontier and
            (
                candidate is goal or
                tuple(candidate.nav_voxel_id) == tuple(goal.nav_voxel_id)
            )
            for candidate in self.nav_goals
        )
        if not selected_is_current:
            return None
        return goal

    def _fake_frontier_tracking_is_allowed(self) -> bool:
        return self._selected_down_stair_goal() is not None

    def _selected_down_stair_target(self) -> Optional[Tuple[int, int, int]]:
        """Return the effective anchor owned by the selected descend goal.

        A cluster refresh may expose its formal representative again after A*
        selected a reachable alternative.  Accept that alternative only for
        the exact owning representative (or the already-mutated goal), so a
        nearby but independent descend cluster cannot inherit confirmation.
        """
        selected = self._selected_down_stair_goal()
        if selected is None:
            return None
        selected_id = tuple(int(v) for v in selected.nav_voxel_id)
        anchor = self._active_descend_fallback_anchor
        owner = self._active_descend_fallback_owner
        if anchor is not None and owner is not None:
            anchor = tuple(int(v) for v in anchor)
            owner = tuple(int(v) for v in owner)
            if selected_id == owner or selected_id == anchor:
                return anchor
        return selected_id

    def _clear_unselected_fake_frontier_state(self) -> None:
        if self._fake_frontier_target is None:
            return
        selected = self._selected_down_stair_goal()
        selected_target = self._selected_down_stair_target()
        if selected is not None and selected_target == self._fake_frontier_target:
            return
        print(
            f"[Nav3D] Fake frontier tracking cleared: target={self._fake_frontier_target} "
            f"is not the selected down-stair fallback goal"
        )
        self._reset_fake_frontier_state()

    def _should_block_fake_frontier_tracking(self, agent_voxel: Optional[Tuple[int, int, int]]) -> bool:
        if agent_voxel is None:
            return True
        if self._post_upstairs_probe_active():
            return True
        return not self._fake_frontier_tracking_is_allowed()

    def _should_block_fake_frontier_confirmation(self, agent_voxel: Optional[Tuple[int, int, int]]) -> bool:
        if self._fake_frontier_target is None:
            return False
        if self._post_upstairs_probe_active():
            return True
        selected = self._selected_down_stair_goal()
        selected_target = self._selected_down_stair_target()
        if selected is None or selected_target != self._fake_frontier_target:
            return True
        if agent_voxel is None:
            return True
        return (
            self._upstairs_transition_cooldown > 0 and
            self._fake_frontier_target[2] <= agent_voxel[2] + 1
        )

    def _update_fake_frontier_tracking(self, agent_voxel: Optional[Tuple[int, int, int]]) -> None:
        """
        In geometric mode: check if agent is near a fake (stair) frontier.
        If so, set desired_pitch_level=2 to trigger look-down.
        Once pitch reaches 60°, start the confirmation window.
        """
        if agent_voxel is None or self.initializing:
            return

        selected = self._selected_down_stair_goal()
        if selected is None:
            self._clear_unselected_fake_frontier_state()
            return
        selected_target = self._selected_down_stair_target()
        if selected_target is None:
            self._clear_unselected_fake_frontier_state()
            return

        if (self._upstairs_transition_cooldown > 0 and
                selected_target[2] <= agent_voxel[2] + 1):
            self._reset_fake_frontier_state()
            return

        dist_xy = self._xy_distance(agent_voxel, selected_target)
        if dist_xy <= 6:
            # Agent is near the selected fake frontier — request pitch down.
            if self._fake_frontier_target != selected_target:
                print(f"[Nav3D] Selected down-stair frontier reached: {selected_target}")
            self._fake_frontier_target = selected_target
            self._fake_frontier_cluster = selected
            # desired_pitch_level will be set in _update_desired_pitch
        else:
            # Not near any fake frontier — clear state if was tracking
            if self._fake_frontier_target is not None:
                self._reset_fake_frontier_state()

    def _notify_pitch_reached_for_fake_frontier(self) -> None:
        """
        Called when desired_pitch_level=2 is reached and we're tracking a fake frontier.
        Once pitched, stair detections can enter down-stair geometry verification.
        """
        if self._fake_frontier_target is not None and not self._fake_frontier_pitched:
            selected = self._selected_down_stair_goal()
            selected_target = self._selected_down_stair_target()
            if selected is None or selected_target != self._fake_frontier_target:
                self._reset_fake_frontier_state()
                return
            self._fake_frontier_pitched = True
            print(
                f"[Nav3D] Fake frontier pitch reached 60° — checking stair conf >= "
                f"{self._down_stair_pitched_conf_threshold} with local geometry verification"
            )

    def _stash_recent_fake_frontier_confirmation(self) -> None:
        """Keep the current fake anchor briefly for look-up/down semantic recheck."""
        if self._fake_frontier_target is None:
            return
        self._recent_fake_frontier_confirmation = {
            "target": self._fake_frontier_target,
            "cluster": self._fake_frontier_cluster,
            "expires_step": self.one_map._iters + self._recent_fake_frontier_confirmation_ttl,
        }

    def _try_recent_fake_frontier_semantic_confirmation(
        self,
        stair_dets: dict,
        image: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Optional[Tuple[int, int, int]],
    ) -> bool:
        """Allow a just-failed fake anchor to enter down sem during pitch recovery.

        This is deliberately not a global flat-view stair trigger: the detection
        must still pass the same local fake-anchor down-stair geometry verifier.
        """
        recent = self._recent_fake_frontier_confirmation
        if recent is None or agent_voxel is None or self.stair_sem_mode:
            return False
        if self.one_map._iters > int(recent.get("expires_step", -1)):
            self._recent_fake_frontier_confirmation = None
            return False
        # The anchor was already a formally selected descend frontier before
        # it entered this short recovery window.  The normal rejection path
        # blacklists/removes that goal before looking up, so requiring it to
        # remain nav_goals[0] makes the recovery recheck impossible.  Keep the
        # recheck local to the stored anchor instead.
        target = recent.get("target")
        if target is None:
            self._recent_fake_frontier_confirmation = None
            return False
        if (
            self._xy_distance(agent_voxel, target) > 8 or
            abs(agent_voxel[2] - target[2]) > 3
        ):
            self._recent_fake_frontier_confirmation = None
            return False
        if not stair_dets.get("boxes"):
            return False

        old_target = self._fake_frontier_target
        old_cluster = self._fake_frontier_cluster
        old_pitched = self._fake_frontier_pitched
        old_wait = self._fake_frontier_wait_counter
        self._fake_frontier_target = recent.get("target")
        self._fake_frontier_cluster = recent.get("cluster")
        self._fake_frontier_pitched = True
        self._fake_frontier_wait_counter = 0
        filtered = {"boxes": [], "classes": [], "scores": []}
        try:
            filtered = self._filter_down_stair_detections_for_anchor(
                stair_dets, depth, odometry, agent_voxel
            )
        finally:
            if not filtered.get("boxes"):
                self._fake_frontier_target = old_target
                self._fake_frontier_cluster = old_cluster
                self._fake_frontier_pitched = old_pitched
                self._fake_frontier_wait_counter = old_wait

        if not filtered.get("boxes"):
            return False

        cluster = recent.get("cluster")
        self._current_down_stair_anchor = (
            self._anchor_center(cluster)
            if cluster is not None
            else recent.get("target")
        )
        print(
            f"[Nav3D] === {stair_mode_label(StairDirection.DESCENT, True)} === "
            f"(recent fake anchor semantic recheck at {recent.get('target')})"
        )
        self._recent_fake_frontier_confirmation = None
        self._fake_frontier_target = recent.get("target")
        self._fake_frontier_cluster = cluster
        self._fake_frontier_wait_counter = 0
        entered = self._enter_stair_sem_mode(filtered, image, depth, odometry, agent_voxel, direction="down")
        if not entered:
            self._reset_fake_frontier_state()
            return False
        return True

    def _reset_fake_frontier_state(self) -> None:
        """Reset fake frontier confirmation state."""
        self._fake_frontier_target = None
        self._fake_frontier_cluster = None
        self._active_descend_fallback_anchor = None
        self._active_descend_fallback_owner = None
        self._fake_frontier_pitched = False
        self._fake_frontier_wait_counter = 0

    def _blacklist_fake_frontier(self, nav_voxel: Tuple[int, int, int]) -> None:
        """排除 fake frontier：加入局部 blacklist，从 frontier_clusters 中移除。"""
        self.blacklisted_goals.append(nav_voxel)
        cluster = self._fake_frontier_cluster
        if cluster is None:
            for c in self.frontier_clusters:
                if c.nav_voxel_id == nav_voxel and c.is_stair_frontier:
                    cluster = c
                    break
        if cluster is not None:
            self.blacklisted_fake_frontier_regions.append(self._fake_frontier_blacklist_region(cluster))
        self.frontier_clusters = [
            c for c in self.frontier_clusters
            if c.nav_voxel_id != nav_voxel and not (
                c.is_stair_frontier and self._is_fake_frontier_region_blacklisted(c)
            )
        ]
        self.desired_pitch_level = 0  # 抬头恢复水平

    def _valid_stair_sem_session_count(self) -> int:
        """Count remaining STAIR/STAIR_EXPANDED voxels in the current stair-sem session."""
        valid_types = {VoxelType.STAIR, VoxelType.STAIR_EXPANDED}
        return sum(
            1 for vid in self._stair_sem_session_voxels
            if self.one_map.voxels.get(vid) is not None and
            self.one_map.voxels[vid].voxel_type in valid_types
        )

    def _abort_stair_sem_as_fake_stair(self, reason: str) -> None:
        """Abort stair semantic mode after its session is proven to be fake stairs."""
        print(f"[Nav3D] Abort stair sem as FAKE_STAIR: {reason}")
        if self._active_stair_memory_id is not None and self._active_stair_memory_id in self.stair_frontier_memory:
            self.stair_frontier_memory[self._active_stair_memory_id]["status"] = "blocked"
        self._active_stair_memory_id = None
        self.stair_sem_mode = False
        self._stair_sem_frontier = None
        self._stair_sem_locked_frontier = None
        self._stair_sem_entry_z = None
        self._stair_sem_direction = "down"
        self._stair_sem_force_forward = 0
        self._stair_sem_session_voxels.clear()
        self._current_down_stair_anchor = None
        self.one_map.mapping_mode = "geometric"
        self.desired_pitch_level = 0
        self.path = None
        self.path_id = 0
        self.last_nav_goal = None
        self.nav_goals.clear()
        self._reset_fake_frontier_state()
        self._reset_rise_confirm_scan()

    def _enter_stair_sem_mode(
        self,
        stair_dets: dict,
        image: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
        agent_voxel: Tuple[int, int, int],
        direction: str = "down",
        verified_entry: bool = False,
        initial_session_voxels: Optional[Set[Tuple[int, int, int]]] = None,
    ) -> bool:
        """
        Enter stair semantic mode:
        1. Mark stair voxels with SAM
        2. Expand STAIR to 26-neighbors (assimilate OCCUPIED)
        3. Create semantic stair frontier (max z-diff cluster)
        4. Set flags
        """
        self.stair_sem_mode = True
        self.navigation_mode = "explore"
        self._navigate_goal = None
        self._reset_post_upstairs_probe()
        self._reset_stair_sem_stability()
        self._stair_sem_entry_z = agent_voxel[2]  # record entry z for exit guard
        self._stair_sem_entry_iter = self.one_map._iters
        self._stair_sem_direction = direction
        self._upstairs_transition_cooldown = 0
        self._fake_frontier_pitched = False
        self._stair_sem_force_forward = 0
        self._stair_sem_session_voxels.clear()
        if initial_session_voxels:
            self._stair_sem_session_voxels.update(
                vid for vid in initial_session_voxels
                if self.one_map.voxels.get(vid) is not None and
                self.one_map.voxels[vid].voxel_type in (VoxelType.STAIR, VoxelType.STAIR_EXPANDED)
            )
        # Descend proposals are represented directly by persistent geometry;
        # no proposal or connector memory is created on entry.

        # SAM segmentation for precise stair marking (use stricter threshold >= 0.8)
        if stair_dets["boxes"]:
            high_conf_dets = {"boxes": [], "classes": [], "scores": [], "masks": []}
            sam_threshold = (
                self._down_stair_pitched_sam_threshold
                if direction == "down" and stair_dets.get("geometry_scores")
                else self._stair_sam_threshold
            )
            stair_masks = stair_dets.get("masks", [])
            for det_index, (box, cls, score) in enumerate(zip(
                stair_dets["boxes"], stair_dets["classes"], stair_dets["scores"]
            )):
                if verified_entry or score >= sam_threshold:
                    high_conf_dets["boxes"].append(box)
                    high_conf_dets["classes"].append(cls)
                    high_conf_dets["scores"].append(score)
                    if det_index < len(stair_masks):
                        high_conf_dets["masks"].append(stair_masks[det_index])
            if high_conf_dets["boxes"]:
                down_anchor = self._current_down_stair_anchor if direction == "down" else None
                marked = self._mark_stair_with_sam(high_conf_dets, image, depth, odometry, down_anchor=down_anchor)
                self._stair_sem_session_voxels.update(marked)

        # 26-neighbor expansion
        expanded = self._expand_stair_26_neighbors()
        self._stair_sem_session_voxels.update(expanded)
        demoted = self._demote_short_stair_clusters()
        if demoted:
            self._stair_sem_session_voxels.difference_update(demoted)
        if self._valid_stair_sem_session_count() < 3:
            print("[Nav3D] Stair sem entry rejected: all detected stair voxels demoted to FAKE_STAIR")
            self._abort_stair_sem_as_fake_stair("entry stair voxels demoted or too few")
            return False

        # Force semantic mapping mode
        self.one_map.mapping_mode = "semantic"

        # Create semantic stair frontier
        self._refresh_stair_sem_frontier(agent_voxel)
        if self._stair_sem_frontier is not None:
            print(f"[Nav3D] Stair sem frontier created: nav={self._stair_sem_frontier.nav_voxel_id}, "
                  f"z_diff={self._stair_sem_frontier.z_difference:.1f}")
        else:
            print(f"[Nav3D] WARNING: Stair sem mode entered but no frontier created")

        # Clear current path to force replan
        self.path = None
        self.last_nav_goal = None
        return True

    def _exit_stair_sem_mode(self, agent_voxel: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Exit stair semantic mode:
        - Restore geometric mapping mode
        - Clear semantic frontier
        - Keep normal and fake frontiers
        - Trigger re-initialization rotation
        """
        was_upstairs = self._stair_sem_direction == "up"
        was_downstairs = self._stair_sem_direction == "down"
        # This method is reached from the confirmed floor/landing exit path.
        # Record direction only after successful traversal; proposals, failed
        # confirmations and aborted semantic sessions must not bias later
        # exploration ordering.
        if was_upstairs:
            self._explore_stair_direction_preference = "up"
        elif was_downstairs and agent_voxel is not None:
            self._explore_stair_direction_preference = "down"
        if was_downstairs and agent_voxel is not None:
            self._remember_recent_down_stair_region(agent_voxel)
        self._consume_active_stair_memory(agent_voxel)
        self.stair_sem_mode = False
        self.navigation_mode = "explore"
        self._navigate_goal = None
        self._stair_sem_frontier = None
        self._stair_sem_force_forward = 0
        self._stair_sem_entry_z = None
        self._reset_stair_sem_stability()
        self._stair_sem_direction = "down"
        self._fake_frontier_target = None
        self._fake_frontier_pitched = False
        self._current_down_stair_anchor = None
        if was_upstairs:
            self._upstairs_transition_cooldown = self._upstairs_transition_cooldown_steps
        self.one_map.mapping_mode = "geometric"
        self.desired_pitch_level = 0

        # Trigger re-initialization: agent will do a 360° scan
        self.initializing = True
        self._frontier_min_cluster_size = 2  # reinit 期间放宽前沿检测
        self.first_obs = True
        self.request_reinit = True
        self.path = None
        self.last_nav_goal = None
        # 换层后清空 blacklist（上层的 blacklist 对新楼层无意义）
        self.blacklisted_goals.clear()
        self.frontier_clusters.clear()
        self.rise_stair_candidates.clear()
        self.nav_goals.clear()

    def _check_stair_sem_exit(self, agent_voxel: Tuple[int, int, int]) -> bool:
        """
        Check exit condition for stair sem mode:
        1. Agent must have moved enough z-layers in the stair direction
           (prevents immediate exit when standing at the stair entry)
        2. Agent's horizontal 8-neighbors have TRAVERSABLE (floor-supported) voxels
           (means agent reached a new floor)
        """
        from mapping.voxel_map import VoxelType

        if self._stair_sem_direction == "up":
            if self._has_upward_stair_continuation(agent_voxel):
                self._stair_sem_up_floor_counter = 0
                return False

            if self._stair_sem_entry_z is None:
                return False
            z_progress = agent_voxel[2] - self._stair_sem_entry_z
            step_progress = self.one_map._iters - self._stair_sem_entry_iter
            if z_progress < self._stair_sem_up_min_ascent:
                self._stair_sem_up_floor_counter = 0
                return False
            if step_progress < self._stair_sem_up_min_steps:
                self._stair_sem_up_floor_counter = 0
                return False

            if self._count_floor_traversable_neighbors(agent_voxel) >= 3:
                self._stair_sem_up_floor_counter += 1
            else:
                self._stair_sem_up_floor_counter = 0
            return self._stair_sem_up_floor_counter >= self._stair_sem_up_floor_confirm_frames

        # Guard: don't exit if agent hasn't moved enough in the selected stair direction.
        if self._stair_sem_entry_z is not None:
            z_progress = self._stair_sem_entry_z - agent_voxel[2]
            if z_progress < self._stair_sem_min_descent:
                self._stair_sem_down_floor_counter = 0
                return False

        # A still-active semantic target substantially below the agent means
        # the staircase continues. Do not mistake step-side traversable voxels
        # for a landing while that lower target remains at least 0.3 m away.
        frontier = self._stair_sem_frontier
        if frontier is not None:
            remaining_descent = agent_voxel[2] - frontier.nav_voxel_id[2]
            if remaining_descent >= self._stair_sem_down_frontier_guard_layers:
                self._stair_sem_down_floor_counter = 0
                return False

        # Descend now uses the same temporal stabilization principle as ascent:
        # platform evidence must persist for consecutive frames.
        if self._count_floor_traversable_neighbors(agent_voxel) >= 3:
            self._stair_sem_down_floor_counter += 1
        else:
            self._stair_sem_down_floor_counter = 0
        return (
            self._stair_sem_down_floor_counter >=
            self._stair_sem_down_floor_confirm_frames
        )

    def _find_stair_intermediate_path(
        self,
        agent_voxel: Tuple[int, int, int],
        ultimate_goal: Tuple[int, int, int],
    ) -> Optional[List[Tuple[int, int, int]]]:
        """
        When A* to the ultimate stair sem goal fails (too far / disconnected),
        find an intermediate STAIR/STAIR_EXPANDED voxel that:
          1. Is reachable from agent via A*
          2. Has z moving toward the ultimate goal (descending or ascending)
          3. Among reachable candidates, pick the one with max z-progress

        Returns the A* path to the intermediate target, or None if no candidate works.
        """
        from mapping.voxel_map import VoxelType, LOCKED_TYPES

        voxels = self.one_map.voxels
        agent_z = agent_voxel[2]
        goal_z = ultimate_goal[2]
        descending = goal_z < agent_z  # True if going downstairs

        # Collect STAIR/STAIR_EXPANDED voxels with z-progress toward goal
        candidates = []
        session_filter = self._stair_sem_session_voxels
        for vid, vd in voxels.items():
            if vd.voxel_type not in LOCKED_TYPES:
                continue
            if session_filter and vid not in session_filter:
                continue
            if vid == agent_voxel:
                continue
            # Must be in correct z-direction
            if descending and vid[2] >= agent_z:
                continue
            if not descending and vid[2] <= agent_z:
                continue
            # Rank by z-progress (how far toward goal)
            z_progress = abs(vid[2] - agent_z)
            candidates.append((z_progress, vid))

        if not candidates:
            return None
        else:
            # Sort by z-progress descending (pick farthest reachable)
            candidates.sort(key=lambda x: -x[0])

        # Try A* to candidates (limit attempts to avoid slowdown)
        max_attempts = min(8, len(candidates))
        for _, candidate_vid in candidates[:max_attempts]:
            path = self.one_map.a_star_3d(agent_voxel, candidate_vid, max_iterations=40000, stair_mode=True)
            if path is not None and len(path) >= 2:
                return path

        return None

    def _expand_stair_26_neighbors(self) -> Set[Tuple[int, int, int]]:
        """
        Expand STAIR voxels: for each SAM-confirmed STAIR voxel,
        check neighbors. If a neighbor is OCCUPIED, assimilate it as STAIR_EXPANDED.
        STAIR_EXPANDED does NOT propagate further expansion, preventing recursive wall-eating.
        """
        from mapping.voxel_map import VoxelType, VoxelData, LOCKED_TYPES

        voxels = self.one_map.voxels
        to_convert = []

        # 8-connectivity (same z-layer) + z+1 center and its 4-neighbors
        neighbors_8_plus_above = [
            # Same z-layer: 8 horizontal/diagonal neighbors
            (-1, -1, 0), (-1, 0, 0), (-1, 1, 0),
            ( 0, -1, 0),             ( 0, 1, 0),
            ( 1, -1, 0), ( 1, 0, 0), ( 1, 1, 0),
            # z+1: center + 4-neighbors (above the voxel)
            ( 0,  0, 1),
            (-1,  0, 1), ( 1,  0, 1),
            ( 0, -1, 1), ( 0,  1, 1),
        ]

        for vid, vd in list(voxels.items()):
            # 仅 SAM-confirmed STAIR（非扩展）可以传播
            if vd.voxel_type != VoxelType.STAIR:
                continue
            for dx, dy, dz in neighbors_8_plus_above:
                nb = (vid[0] + dx, vid[1] + dy, vid[2] + dz)
                if nb in voxels:
                    nb_vd = voxels[nb]
                    # 只有 OCCUPIED 可以被扩展为 STAIR_EXPANDED
                    if nb_vd.voxel_type == VoxelType.OCCUPIED:
                        to_convert.append(nb)

        for nb in to_convert:
            vd = voxels[nb]
            vd.voxel_type = VoxelType.STAIR_EXPANDED
            vd.obstacle_likelihood = 0.0
            self.one_map._register_stair_voxel(nb)

        if to_convert:
            self.one_map.mark_voxels_changed(to_convert)
            print(f"[Nav3D] Stair expansion: {len(to_convert)} OCCUPIED → STAIR_EXPANDED")
        return set(to_convert)

    def _stair_cluster_has_directional_step_trend(
        self,
        cluster: List[Tuple[int, int, int]],
    ) -> Tuple[bool, str]:
        """
        Verify that a stair-like voxel component has a dominant XY direction
        whose low surface height changes in several small, ordered steps.
        """
        if len(cluster) < 12:
            return False, "too_few_voxels"

        columns: Dict[Tuple[int, int], List[int]] = {}
        for x, y, z in cluster:
            columns.setdefault((x, y), []).append(z)
        if len(columns) < 8:
            return False, "too_few_xy_columns"

        voxel_size = float(self.one_map.voxel_size)
        voxel_size_z = float(getattr(self.one_map, "voxel_size_z", self.one_map.voxel_size))

        pts_xy = np.array([[x, y] for (x, y) in columns.keys()], dtype=np.float32)
        low_z = np.array([min(vals) for vals in columns.values()], dtype=np.float32)
        pts_centered = pts_xy - pts_xy.mean(axis=0, keepdims=True)
        cov = (pts_centered.T @ pts_centered) / max(1, pts_centered.shape[0] - 1)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return False, "pca_failed"
        order = np.argsort(eigvals)[::-1]
        main_val = float(eigvals[order[0]])
        second_val = float(eigvals[order[1]]) if eigvals.size > 1 else 0.0
        if main_val <= 1e-6:
            return False, "no_xy_extent"
        linearity = main_val / max(second_val, 1e-6)

        main_dir = eigvecs[:, order[0]]
        forward = pts_centered @ main_dir
        forward_m = forward * voxel_size
        height_m = (low_z - float(low_z.min())) * voxel_size_z
        forward_span_m = float(forward_m.max() - forward_m.min())
        height_span_m = float(height_m.max() - height_m.min())
        if forward_span_m < 0.45:
            return False, f"short_forward={forward_span_m:.2f}"
        if height_span_m < 0.25:
            return False, f"small_height={height_span_m:.2f}"

        # True stairs normally have a dominant travel direction. Keep this
        # threshold modest so wide stairs are not rejected merely for being wide.
        if linearity < 1.35:
            return False, f"weak_main_direction={linearity:.2f}"

        bin_size_m = 0.25
        bin_ids = np.floor((forward_m - forward_m.min()) / bin_size_m).astype(np.int32)
        layer_samples: List[Tuple[float, float, int]] = []
        for bid in sorted(set(int(v) for v in bin_ids)):
            mask = bin_ids == bid
            if int(mask.sum()) < 3:
                continue
            f_med = float(np.median(forward_m[mask]))
            h_low = float(np.percentile(height_m[mask], 25))
            layer_samples.append((f_med, h_low, int(mask.sum())))
        if len(layer_samples) < 3:
            return False, f"few_forward_layers={len(layer_samples)}"

        fs = np.array([s[0] for s in layer_samples], dtype=np.float32)
        hs = np.array([s[1] for s in layer_samples], dtype=np.float32)
        if float(hs[-1] - hs[0]) < 0.0:
            fs = -fs
            order_idx = np.argsort(fs)
            fs = fs[order_idx]
            hs = hs[order_idx]

        diffs = np.diff(hs)
        nondecreasing = int(np.sum(diffs >= -0.08))
        step_like = (diffs >= 0.05) & (diffs <= 0.30)
        big_jumps = int(np.sum(diffs > 0.50))
        if int(step_like.sum()) < 2:
            return False, f"step_like={int(step_like.sum())}"
        if nondecreasing < max(1, len(diffs) - 1):
            return False, "not_monotonic"
        if big_jumps > 0:
            return False, "height_jump_too_large"

        corr = 0.0
        if fs.size >= 3 and float(np.std(fs)) > 1e-6 and float(np.std(hs)) > 1e-6:
            corr = float(np.corrcoef(fs, hs)[0, 1])
        if corr < 0.45:
            return False, f"low_forward_height_corr={corr:.2f}"

        return True, (
            f"layers={len(layer_samples)}, steps={int(step_like.sum())}, "
            f"corr={corr:.2f}, linearity={linearity:.2f}"
        )

    def _stair_cluster_demote_reason(
        self,
        cluster: List[Tuple[int, int, int]],
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Return whether a stair-like cluster should be demoted to FAKE_STAIR."""
        if not cluster:
            return True, "empty", {}

        voxel_size = float(self.one_map.voxel_size)
        voxel_size_z = float(getattr(self.one_map, "voxel_size_z", self.one_map.voxel_size))
        min_height_m = float(getattr(self, "_fake_stair_min_height_m", 2.0))

        z_values = [vid[2] for vid in cluster]
        x_values = [vid[0] for vid in cluster]
        y_values = [vid[1] for vid in cluster]
        height_span_m = (max(z_values) - min(z_values)) * voxel_size_z
        x_span_m = (max(x_values) - min(x_values)) * voxel_size
        y_span_m = (max(y_values) - min(y_values)) * voxel_size
        xy_span_m = float(np.hypot(x_span_m, y_span_m))

        columns: Dict[Tuple[int, int], List[int]] = {}
        for x, y, z in cluster:
            columns.setdefault((x, y), []).append(z)
        column_spans_m = [
            (max(vals) - min(vals)) * voxel_size_z
            for vals in columns.values()
            if vals
        ]
        vertical_min_m = float(getattr(self, "_fake_stair_vertical_column_min_height_m", 0.55))
        vertical_ratio_th = float(getattr(self, "_fake_stair_vertical_column_ratio", 0.30))
        wall_max_xy_m = float(getattr(self, "_fake_stair_wall_max_xy_span_m", 0.35))
        thin_wall_min_height_m = float(getattr(self, "_fake_stair_thin_wall_min_height_m", 0.45))
        thin_wall_max_thickness_m = float(getattr(self, "_fake_stair_thin_wall_max_thickness_m", 0.20))
        thin_wall_min_length_m = float(getattr(self, "_fake_stair_thin_wall_min_length_m", 0.50))
        vertical_columns = sum(1 for span in column_spans_m if span >= vertical_min_m)
        vertical_ratio = vertical_columns / max(1, len(column_spans_m))
        max_column_span_m = max(column_spans_m) if column_spans_m else 0.0
        xy_thickness_m = min(x_span_m, y_span_m)
        xy_length_m = max(x_span_m, y_span_m)

        metrics = {
            "height_span_m": height_span_m,
            "x_span_m": x_span_m,
            "y_span_m": y_span_m,
            "xy_span_m": xy_span_m,
            "xy_thickness_m": xy_thickness_m,
            "xy_length_m": xy_length_m,
            "vertical_ratio": vertical_ratio,
            "max_column_span_m": max_column_span_m,
            "min_height_m": min_height_m,
        }

        short_stair = height_span_m < min_height_m
        thin_wall_like = (
            height_span_m >= thin_wall_min_height_m and
            xy_thickness_m <= thin_wall_max_thickness_m and
            xy_length_m >= thin_wall_min_length_m
        )
        wall_like = (
            thin_wall_like or
            (
                height_span_m >= vertical_min_m and
                max_column_span_m >= vertical_min_m and
                (
                    xy_span_m <= wall_max_xy_m or
                    vertical_ratio >= vertical_ratio_th
                )
            )
        )
        if wall_like:
            return True, "wall", metrics
        if short_stair:
            return True, "short", metrics

        shape_like, shape_reason = self._stair_cluster_has_directional_step_trend(cluster)
        if not shape_like:
            metrics["shape_reason"] = shape_reason
            return True, "shape", metrics
        metrics["shape_reason"] = shape_reason
        return False, "real", metrics

    def _connected_fake_stair_cluster(
        self,
        seed: Tuple[int, int, int],
    ) -> Set[Tuple[int, int, int]]:
        """Return the 26-connected FAKE_STAIR component containing seed."""
        voxels = self.one_map.voxels
        seed_vd = voxels.get(seed)
        if seed_vd is None or seed_vd.voxel_type != VoxelType.FAKE_STAIR:
            return set()

        neighbors_26 = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        cluster: Set[Tuple[int, int, int]] = set()
        queue = deque([seed])
        cluster.add(seed)
        while queue:
            vid = queue.popleft()
            for dx, dy, dz in neighbors_26:
                nb = (vid[0] + dx, vid[1] + dy, vid[2] + dz)
                if nb in cluster:
                    continue
                nb_vd = voxels.get(nb)
                if nb_vd is None or nb_vd.voxel_type != VoxelType.FAKE_STAIR:
                    continue
                cluster.add(nb)
                queue.append(nb)
        return cluster

    def _split_fake_stair_cluster_for_restore(
        self,
        cluster: Set[Tuple[int, int, int]],
    ) -> List[Set[Tuple[int, int, int]]]:
        """
        Split a large FAKE_STAIR component into two spatial subclusters for
        conservative restore re-check.

        This is used only after the whole fake component fails the normal stair
        test and its vertical span is large enough to plausibly contain two
        connected stair flights. The split is deterministic and based on XY
        column positions, so vertical duplicate voxels do not bias the center.
        """
        if len(cluster) < 24:
            return []

        columns: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for vid in cluster:
            columns.setdefault((vid[0], vid[1]), []).append(vid)
        if len(columns) < 12:
            return []

        keys = list(columns.keys())
        pts = np.array([[x, y] for x, y in keys], dtype=np.float32)
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        i0, i1 = np.unravel_index(int(np.argmax(d2)), d2.shape)
        if i0 == i1:
            return []

        centers = np.stack([pts[i0], pts[i1]], axis=0)
        labels = np.zeros((pts.shape[0],), dtype=np.int32)
        for _ in range(8):
            dist = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dist, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for idx in (0, 1):
                mask = labels == idx
                if np.any(mask):
                    centers[idx] = pts[mask].mean(axis=0)

        groups: List[Set[Tuple[int, int, int]]] = [set(), set()]
        for idx, key in enumerate(keys):
            groups[int(labels[idx])].update(columns[key])

        if min(len(groups[0]), len(groups[1])) < 12:
            return []

        voxel_size = float(self.one_map.voxel_size)
        center_sep_m = float(np.linalg.norm((centers[0] - centers[1]) * voxel_size))
        if center_sep_m < 0.55:
            return []

        return groups

    def _trim_fake_stair_endpoint_platforms_for_restore(
        self,
        cluster: Set[Tuple[int, int, int]],
    ) -> Tuple[List[Set[Tuple[int, int, int]]], Set[Tuple[int, int, int]]]:
        """Remove only dominant horizontal layers near either vertical endpoint.

        Repeated partial SAM observations can accumulate into a broad landing
        that remains 26-connected to the actual stair flight.  This helper is
        intentionally conservative: the lowest/highest two layers are merely
        candidates, and are removed only when they are both much denser than
        the inner layers and broad in both horizontal axes.
        """
        if len(cluster) < 24:
            return [], set()

        layers: Dict[int, Set[Tuple[int, int, int]]] = {}
        for vid in cluster:
            layers.setdefault(vid[2], set()).add(vid)
        z_values = sorted(layers)
        if len(z_values) < 5:
            return [], set()

        inner_z = z_values[2:-2]
        baseline_z = inner_z if inner_z else z_values
        baseline = max(
            1.0,
            float(np.median([len(layers[z]) for z in baseline_z])),
        )
        min_layer_size = max(40, int(np.ceil(2.5 * baseline)))
        voxel_size = float(self.one_map.voxel_size)
        endpoint_z = list(dict.fromkeys(z_values[:2] + z_values[-2:]))

        platform_voxels: Set[Tuple[int, int, int]] = set()
        platform_details: List[str] = []
        for z in endpoint_z:
            layer = layers[z]
            if len(layer) < min_layer_size:
                continue
            xs = [vid[0] for vid in layer]
            ys = [vid[1] for vid in layer]
            x_span_m = (max(xs) - min(xs)) * voxel_size
            y_span_m = (max(ys) - min(ys)) * voxel_size
            if min(x_span_m, y_span_m) < 0.8:
                continue
            platform_voxels.update(layer)
            platform_details.append(
                f"z={z},n={len(layer)},span={x_span_m:.2f}x{y_span_m:.2f}m"
            )

        if not platform_voxels:
            return [], set()

        remaining = set(cluster) - platform_voxels
        neighbors_26 = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        groups: List[Set[Tuple[int, int, int]]] = []
        unseen = set(remaining)
        while unseen:
            seed = unseen.pop()
            group = {seed}
            queue = deque([seed])
            while queue:
                vid = queue.popleft()
                for dx, dy, dz in neighbors_26:
                    nb = (vid[0] + dx, vid[1] + dy, vid[2] + dz)
                    if nb not in unseen:
                        continue
                    unseen.remove(nb)
                    group.add(nb)
                    queue.append(nb)
            if len(group) >= 12:
                groups.append(group)

        groups.sort(key=len, reverse=True)
        print(
            f"[Nav3D] Fake stair endpoint platform trim: "
            f"platform_n={len(platform_voxels)}, components={len(groups)}, "
            f"baseline={baseline:.1f}, " + "; ".join(platform_details)
        )
        return groups, platform_voxels

    def _restore_completed_fake_stair_clusters(
        self,
        seeds: Set[Tuple[int, int, int]],
    ) -> Set[Tuple[int, int, int]]:
        """
        Re-check newly demoted fake stair components after they connect to older
        FAKE_STAIR voxels. If the completed component now satisfies the real
        stair shape test, restore the whole component to STAIR.
        """
        restored: Set[Tuple[int, int, int]] = set()
        processed: Set[Tuple[int, int, int]] = set()
        for seed in list(seeds):
            if seed in processed:
                continue
            cluster_set = self._connected_fake_stair_cluster(seed)
            if not cluster_set:
                continue
            processed.update(cluster_set)
            should_demote, reason, metrics = self._stair_cluster_demote_reason(list(cluster_set))
            restore_groups: List[Tuple[Set[Tuple[int, int, int]], str, Dict[str, float]]] = []
            if not should_demote:
                restore_groups = [(cluster_set, reason, metrics)]
            else:
                height_span_m = float(metrics.get("height_span_m", 0.0))
                if height_span_m >= 1.20:
                    trimmed_groups, platform_voxels = (
                        self._trim_fake_stair_endpoint_platforms_for_restore(cluster_set)
                    )
                    for sub_cluster in trimmed_groups:
                        sub_should_demote, sub_reason, sub_metrics = self._stair_cluster_demote_reason(
                            list(sub_cluster)
                        )
                        if not sub_should_demote:
                            restore_groups.append((sub_cluster, sub_reason, sub_metrics))
                            print(
                                f"[Nav3D] Fake stair platform-trim restore accepted: "
                                f"n={len(sub_cluster)}, height={sub_metrics.get('height_span_m', 0.0):.2f}m, "
                                f"xy_span={sub_metrics.get('xy_span_m', 0.0):.2f}m, "
                                f"reason={sub_metrics.get('shape_reason', sub_reason)}"
                            )
                    if platform_voxels and not restore_groups:
                        print(
                            f"[Nav3D] Fake stair platform-trim restore rejected: "
                            f"whole_n={len(cluster_set)}, height={height_span_m:.2f}m, "
                            f"reason={metrics.get('shape_reason', reason)}"
                        )

                    # If the platform-free remainder still combines multiple
                    # stair flights, apply the existing XY split to that clean
                    # geometry before falling back to the contaminated whole
                    # component.
                    if platform_voxels and not restore_groups:
                        for trimmed_group in trimmed_groups:
                            for sub_cluster in self._split_fake_stair_cluster_for_restore(
                                trimmed_group
                            ):
                                sub_should_demote, sub_reason, sub_metrics = (
                                    self._stair_cluster_demote_reason(list(sub_cluster))
                                )
                                if sub_should_demote:
                                    continue
                                restore_groups.append((sub_cluster, sub_reason, sub_metrics))
                                print(
                                    f"[Nav3D] Fake stair platform-trim split restore accepted: "
                                    f"n={len(sub_cluster)}, "
                                    f"height={sub_metrics.get('height_span_m', 0.0):.2f}m, "
                                    f"xy_span={sub_metrics.get('xy_span_m', 0.0):.2f}m, "
                                    f"reason={sub_metrics.get('shape_reason', sub_reason)}"
                                )

                    # Preserve the previous XY split as a fallback only when
                    # endpoint-platform trimming did not recover a stair body.
                    if not restore_groups:
                        split_groups = self._split_fake_stair_cluster_for_restore(cluster_set)
                        for sub_cluster in split_groups:
                            sub_should_demote, sub_reason, sub_metrics = self._stair_cluster_demote_reason(
                                list(sub_cluster)
                            )
                            if not sub_should_demote:
                                restore_groups.append((sub_cluster, sub_reason, sub_metrics))
                                print(
                                    f"[Nav3D] Fake stair split restore accepted: "
                                    f"n={len(sub_cluster)}, height={sub_metrics.get('height_span_m', 0.0):.2f}m, "
                                    f"xy_span={sub_metrics.get('xy_span_m', 0.0):.2f}m, "
                                    f"reason={sub_metrics.get('shape_reason', sub_reason)}"
                                )
                        if split_groups and not restore_groups:
                            print(
                                f"[Nav3D] Fake stair split restore rejected: "
                                f"whole_n={len(cluster_set)}, height={height_span_m:.2f}m, "
                                f"reason={metrics.get('shape_reason', reason)}"
                            )

            for group, group_reason, group_metrics in restore_groups:
                restored_now: Set[Tuple[int, int, int]] = set()
                for vid in group:
                    vd = self.one_map.voxels.get(vid)
                    if vd is None or vd.voxel_type != VoxelType.FAKE_STAIR:
                        continue
                    vd.voxel_type = VoxelType.STAIR
                    vd.obstacle_likelihood = 0.0
                    vd.inflated = False
                    restored.add(vid)
                    restored_now.add(vid)
                if restored_now:
                    self.one_map.mark_voxels_changed(restored_now)
                    print(
                        f"[Nav3D] Fake stair restored as STAIR: n={len(restored_now)}, "
                        f"height={group_metrics.get('height_span_m', 0.0):.2f}m, "
                        f"xy_span={group_metrics.get('xy_span_m', 0.0):.2f}m, "
                        f"reason={group_metrics.get('shape_reason', group_reason)}"
                    )
        return restored

    def _demote_short_stair_clusters(self) -> Set[Tuple[int, int, int]]:
        """
        Convert real stair components with insufficient vertical span to FAKE_STAIR.

        FAKE_STAIR is a locked, non-targetable type. It blocks ray generation,
        remains available to A* with a high type cost, and is excluded from
        stair semantic frontiers.
        """
        from mapping.voxel_map import LOCKED_TYPES, VoxelType

        voxels = self.one_map.voxels
        stair_voxels = {
            vid for vid, vd in voxels.items()
            if vd.voxel_type in LOCKED_TYPES
        }
        if not stair_voxels:
            return set()

        voxel_size_z = float(getattr(self.one_map, "voxel_size_z", self.one_map.voxel_size))
        min_height_m = float(getattr(self, "_fake_stair_min_height_m", 2.0))
        demoted: Set[Tuple[int, int, int]] = set()
        demoted_short_clusters = 0
        demoted_wall_clusters = 0
        demoted_shape_clusters = 0
        max_demoted_span = 0.0

        clusters = cluster_stair_voxels_without_expanded_bridges(stair_voxels, voxels)
        for cluster_set in clusters:
            cluster = list(cluster_set)
            if not cluster:
                continue

            should_demote, demote_reason, metrics = self._stair_cluster_demote_reason(cluster)
            if not should_demote:
                continue

            if demote_reason == "wall":
                demoted_wall_clusters += 1
            elif demote_reason == "short":
                demoted_short_clusters += 1
            else:
                demoted_shape_clusters += 1
                print(
                    f"[Nav3D] Fake stair shape rejection: n={len(cluster)}, "
                    f"height={metrics.get('height_span_m', 0.0):.2f}m, "
                    f"xy_span={metrics.get('xy_span_m', 0.0):.2f}m, "
                    f"thin={metrics.get('xy_thickness_m', 0.0):.2f}m, "
                    f"reason={metrics.get('shape_reason', demote_reason)}"
                )
            max_demoted_span = max(max_demoted_span, metrics.get("height_span_m", 0.0))
            for vid in cluster:
                vd = voxels.get(vid)
                if vd is None or vd.voxel_type not in LOCKED_TYPES:
                    continue
                vd.voxel_type = VoxelType.FAKE_STAIR
                vd.obstacle_likelihood = max(float(vd.obstacle_likelihood), 1.0)
                vd.inflated = False
                demoted.add(vid)
        if demoted:
            self.one_map.mark_voxels_changed(demoted)

        if demoted:
            restored = self._restore_completed_fake_stair_clusters(demoted)
            if restored:
                demoted.difference_update(restored)
                self._stair_sem_session_voxels.update(restored)
            self._stair_sem_session_voxels.difference_update(demoted)
            print(
                f"[Nav3D] Fake stair demotion: {len(demoted)} stair voxels -> FAKE_STAIR "
                f"short_clusters={demoted_short_clusters} wall_clusters={demoted_wall_clusters} "
                f"shape_clusters={demoted_shape_clusters} "
                f"max_height_span={max_demoted_span:.2f}m "
                f"threshold={min_height_m:.2f}m"
            )
        return demoted

    def _mark_stair_with_sam(
        self,
        stair_dets: dict,
        image: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
        down_anchor: Optional[Tuple[int, int, int]] = None,
    ) -> Set[Tuple[int, int, int]]:
        """
        Use SegFormer masks directly (or SAM for a legacy box-only source), then
        back-project masked depth points to 3D and mark as STAIR voxels.

        This replaces the coarse mark_stair_voxels (bbox center only).
        """
        marked: Set[Tuple[int, int, int]] = set()
        if not self.one_map.camera_initialized or self.one_map.agent_height_0 is None:
            return marked

        R_mat = odometry[:3, :3]
        t_vec = odometry[:3, 3]

        detection_masks = stair_dets.get("masks", [])
        if len(detection_masks) < len(stair_dets["boxes"]):
            self.sam_predictor.set_image(image.transpose(1, 2, 0))

        for det_index, box in enumerate(stair_dets["boxes"]):
            if det_index < len(detection_masks):
                mask = np.asarray(detection_masks[det_index], dtype=bool)
                if mask.shape != depth.shape:
                    mask = np.asarray(
                        torch.nn.functional.interpolate(
                            torch.from_numpy(mask.astype(np.float32))[None, None],
                            size=depth.shape,
                            mode="nearest",
                        )[0, 0].numpy(),
                        dtype=bool,
                    )
            else:
                try:
                    masks, _, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array(box)[None, :],
                        multimask_output=False,
                    )
                    mask = masks[0]
                except Exception:
                    mask = np.zeros(depth.shape, dtype=bool)
                    x1, y1, x2, y2 = map(int, box)
                    mask[y1:y2, x1:x2] = True

            # Get valid depth pixels within SAM mask
            mask_ids = np.argwhere(mask & (depth != 0) & (depth != float("inf")))
            if mask_ids.shape[0] == 0:
                continue

            depths = depth[mask_ids[:, 0], mask_ids[:, 1]]

            # Back-project to world coordinates
            u = mask_ids[:, 1].astype(np.float32)
            v = mask_ids[:, 0].astype(np.float32)
            x_cam = (u - self.one_map.cx) * depths / self.one_map.fx
            y_cam = (v - self.one_map.cy) * depths / self.one_map.fy
            pts_cam = np.stack([depths, -x_cam, -y_cam], axis=-1)  # (N, 3)
            pts_world = (R_mat @ pts_cam.T).T + t_vec  # (N, 3)

            # Subsample if too many points (perf)
            max_pts = 2000
            if pts_world.shape[0] > max_pts:
                indices = np.random.choice(pts_world.shape[0], max_pts, replace=False)
                pts_world = pts_world[indices]

            if down_anchor is not None:
                mask_voxels = [
                    self.one_map.metric_to_voxel(float(pt[0]), float(pt[1]), float(pt[2]))
                    for pt in pts_world
                ]
                mask_geometry = self._down_stair_mask_geometry_score(
                    down_anchor, mask_voxels
                )
                if not mask_geometry["passed"]:
                    print(f"[Nav3D] Down stair SAM mask rejected: {mask_geometry['reason']} "
                          f"score={mask_geometry['geometry_score']:.2f}")
                    continue

            # Convert to voxel IDs and mark as STAIR
            for pt in pts_world:
                vid = self.one_map.metric_to_voxel(pt[0], pt[1], pt[2])
                if vid in self.one_map.voxels:
                    vd = self.one_map.voxels[vid]
                    if vd.voxel_type == VoxelType.OCCUPIED:
                        vd.voxel_type = VoxelType.STAIR
                        vd.obstacle_likelihood = 0.0
                        self.one_map._register_stair_voxel(vid)
                        marked.add(vid)
                    elif vd.voxel_type == VoxelType.STAIR:
                        marked.add(vid)
                    # 其他类型（TRAVERSABLE系列、已有STAIR）不变
                else:
                    self.one_map.voxels[vid] = VoxelData(
                        voxel_type=VoxelType.STAIR,
                        feature=np.zeros(self.one_map.feature_dim, dtype=np.float32),
                        confidence=0.5,
                        obstacle_likelihood=0.0,
                    )
                    self.one_map._register_stair_voxel(vid)
                    marked.add(vid)

        if marked:
            self.one_map.mark_voxels_changed(marked)
        return marked
