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

def history_poi_enabled(navigator) -> bool:
    return bool(navigator.allow_poi and navigator.enable_history_poi)

class POIManagerMixin:
    def _plan_history_poi_path_bidirectional(
        self,
        agent_voxel: Tuple[int, int, int],
        goal_voxel: Tuple[int, int, int],
        max_iterations: int,
        log_reverse: bool = False,
    ) -> Tuple[Optional[List[Tuple[int, int, int]]], str]:
        """Plan to a history POI, reversing a validated down-path if needed.

        The voxel graph contains directional drop connectors, so a known
        staircase can be reachable from the upper floor to the lower floor but
        not in the opposite search direction.  A successful reverse search is
        therefore recorded in reverse order and used as the forward route.
        """
        common_kwargs = dict(
            max_iterations=max_iterations,
            stair_mode=False,
            prefer_traversable=False,
            allow_inflated_traversable=True,
            allow_inflated_stair=False,
            allow_inflated_traversable_stair=True,
            connect_traversable_stair_drop=True,
        )
        path = self.one_map.a_star_3d(
            agent_voxel,
            goal_voxel,
            **common_kwargs,
        )
        if path is not None:
            return list(path), "forward"

        reverse_path = self.one_map.a_star_3d(
            goal_voxel,
            agent_voxel,
            **common_kwargs,
        )
        if reverse_path is None:
            return None, "no_path"

        path = list(reversed(reverse_path))
        if log_reverse:
            print(
                f"[Nav3D] History POI reverse A* fallback: "
                f"agent={agent_voxel}, goal={goal_voxel}, path_len={len(path)}"
            )
        return path, "reverse"

    def _start_poi_confirm_scan(
        self,
        goal: Cluster3D,
        odometry: Optional[np.ndarray] = None,
    ) -> None:
        if self._poi_confirm_scan_remaining > 0:
            return

        # History POI centers are semantic object locations, while nav_voxel_id
        # is only a nearby traversable viewpoint.  Use the camera-to-center
        # elevation angle to choose a single 30-degree pitch adjustment before
        # the confirmation rotation.  Do not use agent_voxel here: it lies on
        # the lower ray-casting plane rather than at camera height.
        self._poi_confirm_pitch_level = 0
        elevation_deg: Optional[float] = None
        if self._is_history_poi(goal) and odometry is not None:
            camera_xyz = np.asarray(odometry[:3, 3], dtype=np.float64)
            center_xyz = np.asarray(
                self.one_map.voxel_to_metric(*goal.center_voxel),
                dtype=np.float64,
            )
            horizontal_m = float(np.linalg.norm(center_xyz[:2] - camera_xyz[:2]))
            elevation_deg = float(np.degrees(np.arctan2(
                center_xyz[2] - camera_xyz[2],
                horizontal_m,
            )))
            threshold_deg = float(
                getattr(self, "_poi_confirm_pitch_angle_threshold_deg", 45.0)
            )
            if elevation_deg > threshold_deg:
                self._poi_confirm_pitch_level = -1  # 30 degrees up
            elif elevation_deg < -threshold_deg:
                self._poi_confirm_pitch_level = 1   # 30 degrees down

        self.desired_pitch_level = self._poi_confirm_pitch_level
        self._poi_confirm_scan_remaining = int(getattr(self, "_poi_confirm_scan_steps", 12))
        self.path = None
        pitch_name = {
            -1: "up-30deg",
            0: "horizontal",
            1: "down-30deg",
        }[self._poi_confirm_pitch_level]
        angle_text = "n/a" if elevation_deg is None else f"{elevation_deg:.1f}deg"
        print(
            f"[Nav3D] POI reached {goal.nav_voxel_id}; scanning for target "
            f"({self._poi_confirm_scan_remaining} frames), "
            f"center={goal.center_voxel}, elevation={angle_text}, pitch={pitch_name}"
        )

    def _all_pois(self) -> List[Cluster3D]:
        return list(self.live_pois.values()) + self._visible_history_pois()

    def _visible_history_pois(self) -> List[Cluster3D]:
        return [poi for poi in self.history_pois if not self._is_poi_cluster_blacklisted(poi)]

    def _is_poi_blacklisted(self, nav_voxel_id: Tuple[int, int, int]) -> bool:
        """Return True if a POI is inside a locally blacklisted 3-D cylinder."""
        vid = tuple(int(v) for v in nav_voxel_id)
        z_radius_m = float(getattr(self, "_poi_blacklist_radius_z_m", 0.6))
        voxel_size_z = float(self.one_map.voxel_size_z)
        for blocked in self._poi_blacklisted_goals:
            z_distance_m = abs(vid[2] - blocked[2]) * voxel_size_z
            if (
                self._xy_distance(vid, blocked) <= self._poi_blacklist_radius_xy and
                z_distance_m <= z_radius_m
            ):
                return True
        return False

    @staticmethod
    def _xy_distance_to_bbox(
        vid: Tuple[int, int, int],
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
    ) -> float:
        x, y = int(vid[0]), int(vid[1])
        min_x, min_y = int(bbox_min[0]), int(bbox_min[1])
        max_x, max_y = int(bbox_max[0]), int(bbox_max[1])
        dx = max(min_x - x, 0, x - max_x)
        dy = max(min_y - y, 0, y - max_y)
        return float(np.hypot(dx, dy))

    def _xyz_distance_to_bbox_m(
        self,
        vid: Tuple[int, int, int],
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
    ) -> float:
        x, y, z = int(vid[0]), int(vid[1]), int(vid[2])
        min_x, min_y, min_z = int(bbox_min[0]), int(bbox_min[1]), int(bbox_min[2])
        max_x, max_y, max_z = int(bbox_max[0]), int(bbox_max[1]), int(bbox_max[2])
        dx = max(min_x - x, 0, x - max_x) * float(self.one_map.voxel_size)
        dy = max(min_y - y, 0, y - max_y) * float(self.one_map.voxel_size)
        dz = max(min_z - z, 0, z - max_z) * float(self.one_map.voxel_size_z)
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _z_distance_to_bbox_m(
        self,
        vid: Tuple[int, int, int],
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
    ) -> float:
        """Vertical distance from a voxel to a 3-D bbox, in metres."""
        z = int(vid[2])
        min_z = int(bbox_min[2])
        max_z = int(bbox_max[2])
        dz = max(min_z - z, 0, z - max_z)
        return float(dz) * float(self.one_map.voxel_size_z)

    def _voxel_xyz_distance_m(
        self,
        a: Tuple[int, int, int],
        b: Tuple[int, int, int],
    ) -> float:
        dx = (int(a[0]) - int(b[0])) * float(self.one_map.voxel_size)
        dy = (int(a[1]) - int(b[1])) * float(self.one_map.voxel_size)
        dz = (int(a[2]) - int(b[2])) * float(self.one_map.voxel_size_z)
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _is_history_poi(self, cluster: Cluster3D) -> bool:
        return "history" in str(getattr(cluster, "poi_type", "history"))

    @staticmethod
    def _history_poi_blacklist_key(
        cluster: Cluster3D,
    ) -> Tuple[
        Tuple[int, int, int],
        Tuple[int, int, int],
        Tuple[int, int, int],
    ]:
        """Return the stable semantic identity of one generated history POI.

        Nearby history hypotheses may share the same traversable observation
        point but require different viewing directions.  Their nav voxels must
        therefore never be used as blacklist identities.
        """
        center = tuple(int(v) for v in cluster.center_voxel)
        bbox_min = tuple(int(v) for v in getattr(cluster, "bbox_min", center))
        bbox_max = tuple(int(v) for v in getattr(cluster, "bbox_max", center))
        return center, bbox_min, bbox_max

    def _is_history_poi_navigate_active(self) -> bool:
        return (
            self.navigation_mode == "navigate" and
            isinstance(self._navigate_goal, Cluster3D) and
            self._is_history_poi(self._navigate_goal)
        )

    def _detection_matches_history_poi(
        self,
        det_voxel: Tuple[int, int, int],
        history_poi: Cluster3D,
    ) -> Tuple[bool, float, float]:
        """Check whether a live detection belongs to the active history POI.

        The POI navigation voxel is an observation pose and can be displaced
        from the object surface, so takeover is gated against the semantic
        feature bbox instead of the snapped navigation target.
        """
        bbox_min = getattr(history_poi, "bbox_min", None)
        bbox_max = getattr(history_poi, "bbox_max", None)
        if bbox_min is None or bbox_max is None:
            bbox_min = history_poi.center_voxel
            bbox_max = history_poi.center_voxel

        xy_distance_m = (
            self._xy_distance_to_bbox(det_voxel, bbox_min, bbox_max) *
            float(self.one_map.voxel_size)
        )
        z_distance_m = self._z_distance_to_bbox_m(det_voxel, bbox_min, bbox_max)
        xy_limit_m = float(getattr(self, "_history_poi_objectnav_match_xy_m", 1.0))
        z_limit_m = float(getattr(self, "_history_poi_objectnav_match_z_m", 1.0))
        return (
            xy_distance_m <= xy_limit_m and z_distance_m <= z_limit_m,
            xy_distance_m,
            z_distance_m,
        )

    def _poi_cluster_xy_distance(self, agent_voxel: Tuple[int, int, int], cluster: Cluster3D) -> float:
        distances = [
            self._xy_distance(agent_voxel, cluster.nav_voxel_id),
            self._xy_distance(agent_voxel, cluster.center_voxel),
        ]
        bmin = getattr(cluster, "bbox_min", None)
        bmax = getattr(cluster, "bbox_max", None)
        if bmin is not None and bmax is not None:
            distances.append(self._xy_distance_to_bbox(agent_voxel, bmin, bmax))
        return float(min(distances))

    def _history_poi_arrival_metrics(
        self,
        agent_voxel: Tuple[int, int, int],
        cluster: Cluster3D,
    ) -> Tuple[float, float]:
        # History POI arrival is based on the snapped navigation target only.
        # The feature bbox/center may be large or inside an object/wall; they are
        # useful for visualization, overlap, and blacklist, but not for deciding
        # that the agent reached the observation point.
        xy_dist_vox = self._xy_distance(agent_voxel, cluster.nav_voxel_id)
        xyz_dist_m = self._voxel_xyz_distance_m(agent_voxel, cluster.nav_voxel_id)
        return xy_dist_vox, xyz_dist_m

    def _history_poi_arrived(
        self,
        agent_voxel: Tuple[int, int, int],
        cluster: Cluster3D,
    ) -> Tuple[bool, float, float]:
        xy_dist_vox, xyz_dist_m = self._history_poi_arrival_metrics(agent_voxel, cluster)
        return (
            xy_dist_vox <= self._history_poi_arrival_radius_xy and
            xyz_dist_m <= self._history_poi_arrival_radius_xyz_m,
            xy_dist_vox * float(self.one_map.voxel_size),
            xyz_dist_m,
        )

    def _is_poi_cluster_blacklisted(self, cluster: Cluster3D) -> bool:
        if self._is_history_poi(cluster):
            return (
                self._history_poi_blacklist_key(cluster) in
                self._history_poi_blacklisted_keys
            )

        # Live POIs can jitter by several voxels between frames.  Keep the
        # existing spatial cylinder for them so a rejected detection does not
        # immediately reappear under a slightly shifted center.
        if self._is_poi_blacklisted(cluster.nav_voxel_id):
            return True
        if self._is_poi_blacklisted(cluster.center_voxel):
            return True
        bmin = getattr(cluster, "bbox_min", None)
        bmax = getattr(cluster, "bbox_max", None)
        if bmin is None or bmax is None:
            return False
        z_radius_m = float(getattr(self, "_poi_blacklist_radius_z_m", 0.6))
        for blocked in self._poi_blacklisted_goals:
            if (
                self._xy_distance_to_bbox(blocked, bmin, bmax) <= self._poi_blacklist_radius_xy and
                self._z_distance_to_bbox_m(blocked, bmin, bmax) <= z_radius_m
            ):
                return True
        return False

    def _blacklist_poi_cluster(self, cluster: Cluster3D) -> None:
        if self._is_history_poi(cluster):
            key = self._history_poi_blacklist_key(cluster)
            self._history_poi_blacklisted_keys.add(key)
            print(
                f"[Nav3D] History POI blacklisted individually: "
                f"center={key[0]}, bbox={key[1]}->{key[2]}"
            )
            return

        self._poi_blacklisted_goals.add(tuple(int(v) for v in cluster.nav_voxel_id))
        self._poi_blacklisted_goals.add(tuple(int(v) for v in cluster.center_voxel))

    def _poi_cluster_key(self, cluster: Cluster3D) -> Tuple[int, int, int]:
        return tuple(int(v) for v in cluster.center_voxel)

    def _merge_live_poi(self, cluster: Cluster3D) -> None:
        if self._is_poi_cluster_blacklisted(cluster):
            return
        best_key = None
        best_dist = float("inf")
        center = np.array(cluster.center_voxel, dtype=np.float32)
        for key, existing in self.live_pois.items():
            other = np.array(existing.center_voxel, dtype=np.float32)
            dist_xy = float(np.linalg.norm(center[:2] - other[:2]))
            if dist_xy <= 5.0 and abs(float(center[2] - other[2])) <= 4.0 and dist_xy < best_dist:
                best_key = key
                best_dist = dist_xy
        if best_key is None:
            self.live_pois[self._poi_cluster_key(cluster)] = cluster
            return
        old = self.live_pois[best_key]
        if cluster.evidence_score >= old.evidence_score:
            self.live_pois[best_key] = cluster
        else:
            old.last_seen_iter = self.one_map._iters

    def _expire_live_pois(self) -> None:
        ttl = int(getattr(self, "_live_poi_ttl", 20))
        now = self.one_map._iters
        self.live_pois = {
            key: poi for key, poi in self.live_pois.items()
            if now - int(getattr(poi, "last_seen_iter", now)) <= ttl
            and not self._is_poi_cluster_blacklisted(poi)
        }

    def _generate_history_pois(self, agent_voxel: Tuple[int, int, int]) -> None:
        self.history_pois = []
        if not self.allow_poi:
            self._history_poi_pending = False
            return
        clusters = cluster_high_similarity_3d(
            self.one_map,
            agent_voxel,
            candidate_percentile=90.0,
            candidate_min_similarity=0.04,
            seed_percentile=95.0,
            seed_min_similarity=0.06,
            neighborhood_radius=3,
            min_cluster_size=3,
            relative_threshold=0.65,
            max_clusters=12,
            search_radius=self.one_map.n_cells,
            poi_type="history",
        )
        self.history_pois = sorted(
            [c for c in clusters if not self._is_poi_cluster_blacklisted(c)],
            key=lambda c: c.evidence_score,
            reverse=True,
        )[:4]
        self._history_poi_pending = False
        if self.history_pois:
            best = self.history_pois[0]
            print(
                f"[Nav3D] History POI generated: n={len(self.history_pois)}, "
                f"best={best.nav_voxel_id}, evidence={best.evidence_score:.3f}, "
                f"max={best.max_similarity:.3f}, topk={best.topk_mean_similarity:.3f}"
            )
        else:
            print("[Nav3D] History POI generated: n=0")

    def _update_feature_live_pois(
        self,
        agent_voxel: Tuple[int, int, int],
        updated_voxels: Set[Tuple[int, int, int]],
    ) -> None:
        # Live POI is an approach target for current GroundingDINO proposals.
        # Feature-only live POIs were too eager and caused navigate mode to
        # chase small similarity blobs along walls/edges.
        self._expire_live_pois()

    def _upsert_dino_live_poi(
        self,
        det_voxel: Tuple[int, int, int],
        agent_voxel: Tuple[int, int, int],
        confidence: float,
        region_sims: List[float],
        close_approach: bool = False,
    ) -> None:
        if self._is_history_poi_navigate_active() or self._objectnav_source == "history_poi":
            return
        nav_vid = self._find_object_approach_voxel(
            det_voxel,
            agent_voxel,
            preferred_radius_m=0.5 if close_approach else 0.9,
        )
        if nav_vid is None:
            return
        max_sim = float(max(region_sims)) if region_sims else float(
            self.one_map.similarity_cache.get(det_voxel, 0.0)
        )
        mean_sim = float(np.mean(region_sims)) if region_sims else max_sim
        evidence = max_sim + 0.05 * float(confidence)
        cluster = Cluster3D(
            nav_voxel_id=nav_vid,
            center_voxel=det_voxel,
            cluster_score=evidence,
            floor_iz=det_voxel[2],
            cluster_size=max(1, len(region_sims)),
            max_similarity=max_sim,
            topk_mean_similarity=max_sim,
            mean_similarity=mean_sim,
            confidence_weighted_similarity=mean_sim,
            bbox_min=tuple(int(det_voxel[i] - 2) for i in range(3)),
            bbox_max=tuple(int(det_voxel[i] + 2) for i in range(3)),
            evidence_score=evidence,
            poi_type="dino_live",
            last_seen_iter=self.one_map._iters,
        )
        self._merge_live_poi(cluster)

    def _poi_overlap_evidence(
        self,
        det_voxel: Tuple[int, int, int],
    ) -> Tuple[bool, Optional[Cluster3D]]:
        det = np.array(det_voxel, dtype=np.float32)
        radius = float(getattr(self, "_poi_overlap_radius", 8))
        min_evidence = float(getattr(self, "_poi_overlap_min_evidence", 0.08))
        best = None
        best_score = -float("inf")
        for poi in self._all_pois():
            if getattr(poi, "poi_type", "") != "history":
                continue
            if poi.evidence_score < min_evidence:
                continue
            bmin = np.array(poi.bbox_min, dtype=np.float32) - radius
            bmax = np.array(poi.bbox_max, dtype=np.float32) + radius
            in_bbox = bool(np.all(det >= bmin) and np.all(det <= bmax))
            center_dist = float(np.linalg.norm(det[:2] - np.array(poi.center_voxel[:2], dtype=np.float32)))
            if in_bbox or center_dist <= radius:
                if poi.evidence_score > best_score:
                    best = poi
                    best_score = poi.evidence_score
        return best is not None, best
