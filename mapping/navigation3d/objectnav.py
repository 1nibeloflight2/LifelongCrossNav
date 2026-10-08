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
from .semantic_confirmation import masked_sed_top20_mean

try:
    import rerun as rr
    _RERUN = True
except ImportError:
    _RERUN = False

def confirmed_detection_available(navigator) -> bool:
    return bool(navigator.object_detected and navigator.chosen_detection_3d is not None)

class ObjectNavMixin:
    def _reset_sed_found_state(self) -> None:
        self._sed_rejected_detections = []
        self._current_confirmed_sed_input = None

    def _sed_query_key(self) -> Tuple[str, ...]:
        return tuple(str(value) for value in self.query_text)

    def _is_sed_detection_cooling(self, det_voxel: Tuple[int, int, int]) -> bool:
        """Query a local cooldown only; never evaluate SED during approach."""
        if not self._enable_sed_found_confirmation:
            return False
        now = int(self.one_map._iters)
        self._sed_rejected_detections = [
            item for item in self._sed_rejected_detections if now < item["expires"]
        ]
        query = self._sed_query_key()
        for item in self._sed_rejected_detections:
            if item["query"] != query:
                continue
            blocked = item["voxel"]
            xy_m = self._xy_distance(det_voxel, blocked) * self.one_map.voxel_size
            z_m = abs(det_voxel[2] - blocked[2]) * self.one_map.voxel_size_z
            if xy_m <= self._sed_reject_radius_xy_m and z_m <= self._sed_reject_radius_z_m:
                return True
        return False

    def _retain_confirmed_sed_input(self, dense_features, mask: np.ndarray) -> None:
        """Keep the confirmed view's inputs, without scoring or filtering it."""
        if not self._enable_sed_found_confirmation:
            self._current_confirmed_sed_input = None
            return
        self._current_confirmed_sed_input = {
            "dense": dense_features.detach() if dense_features is not None else None,
            "mask": np.asarray(mask, dtype=bool).copy(),
            "query": self._sed_query_key(),
            "query_features": self.query_text_features[0].detach().clone(),
            "frame": int(self.one_map._iters),
        }

    def _finalize_object_found(self, agent_voxel: Tuple[int, int, int],
                               source: str, clear_cache: bool = True) -> bool:
        """The sole final gate for all existing ObjectNav found branches."""
        # Actor already suppresses found during its forced initialization scan.
        # Do not score or cool down a candidate on those non-found frames.
        if self.initializing:
            self._reset_objectnav_correction(clear_cache=clear_cache)
            return False
        if self._enable_sed_found_confirmation:
            inputs = self._current_confirmed_sed_input
            score = None
            reason = "missing_confirmed_frame_inputs"
            if inputs is not None:
                if inputs["query"] != self._sed_query_key():
                    reason = "stale_query"
                else:
                    try:
                        score = masked_sed_top20_mean(
                            inputs["dense"], inputs["query_features"], inputs["mask"]
                        )
                        reason = (
                            "nonfinite_score" if not np.isfinite(score) else
                            "low_score" if score < self._sed_found_threshold else "passed"
                        )
                    except (ValueError, RuntimeError, TypeError) as error:
                        reason = f"invalid_inputs:{error}"
            if score is None or not np.isfinite(score) or score < self._sed_found_threshold:
                self._reject_sed_found(agent_voxel, source, score, reason)
                return False
            print(
                f"[Nav3D][SED] found accepted: source={source}, "
                f"score={score:.4f} >= {self._sed_found_threshold:.4f}, "
                f"frame={inputs['frame']}"
            )
        self._reset_objectnav_correction(clear_cache=clear_cache)
        return True

    def _reject_sed_found(self, agent_voxel: Tuple[int, int, int],
                          source: str, score: Optional[float], reason: str) -> None:
        det_voxel = self.current_confirmed_detection_voxel or self.chosen_detection_3d
        if det_voxel is not None:
            det_voxel = tuple(int(value) for value in det_voxel)
            self._sed_rejected_detections.append({
                "query": self._sed_query_key(), "voxel": det_voxel,
                "expires": int(self.one_map._iters) + self._sed_reject_cooldown_frames,
            })
        value = f"{score:.4f}" if score is not None else "unavailable"
        print(
            f"[Nav3D][SED] found rejected: source={source}, score={value}, "
            f"threshold={self._sed_found_threshold:.4f}, reason={reason}, "
            f"voxel={det_voxel}, cooldown={self._sed_reject_cooldown_frames} frames"
        )
        # Live hypotheses alone are removed. History POI identity, selection
        # and blacklist rules are left to the existing return/scan workflow.
        self.live_pois = {
            key: poi for key, poi in self.live_pois.items()
            if not self._is_sed_detection_cooling(poi.center_voxel)
        }
        self._clear_object_detection("SED veto at found output")
        self.desired_pitch_level = 0
        self._compute_frontiers_and_POIs(agent_voxel)
        self._compute_best_path(agent_voxel)
        self._update_desired_pitch(agent_voxel)

    def _agent_to_chosen_distance_m(
        self,
        agent_voxel: Tuple[int, int, int],
    ) -> float:
        """Horizontal distance used by the internal ObjectNav found check."""
        if self.chosen_detection_3d is None:
            return float("inf")
        distance_voxels = np.hypot(
            float(agent_voxel[0] - self.chosen_detection_3d[0]),
            float(agent_voxel[1] - self.chosen_detection_3d[1]),
        )
        return float(distance_voxels * self.one_map.cell_size)

    def _reset_current_confirmed_detection(self) -> None:
        self.current_confirmed_detection_box = None
        self.current_confirmed_detection_score = None
        self.current_confirmed_detection_class = None
        self.current_confirmed_detection_voxel = None
        self.current_confirmed_detection_box_area_ratio = None
        self.current_confirmed_detection_center_offset = None
        self.current_confirmed_detection_edge_margin_px = None
        self.current_confirmed_detection_mask_voxels = None
        self.current_confirmed_detection_image_shape = None
        self._current_confirmed_sed_input = None

    def _set_current_confirmed_detection(
        self,
        box: List[float],
        confidence: float,
        det_class: str,
        det_voxel: Tuple[int, int, int],
        quality: dict,
    ) -> None:
        self.current_confirmed_detection_box = [float(v) for v in box]
        self.current_confirmed_detection_score = float(confidence)
        self.current_confirmed_detection_class = str(det_class)
        self.current_confirmed_detection_voxel = det_voxel
        self.current_confirmed_detection_box_area_ratio = float(quality.get("box_area_ratio", 0.0))
        self.current_confirmed_detection_center_offset = float(quality.get("center_offset", 1.0))
        self.current_confirmed_detection_edge_margin_px = float(quality.get("edge_margin_px", -1.0))
        self.current_confirmed_detection_mask_voxels = int(quality.get("mask_voxels", 0))
        image_shape = quality.get("image_shape")
        self.current_confirmed_detection_image_shape = (
            tuple(int(v) for v in image_shape) if image_shape is not None else None
        )

    def _cache_current_confirmed_detection(self, image_rgb_hwc: Optional[np.ndarray] = None) -> None:
        if self.current_confirmed_detection_box is None or self.current_confirmed_detection_voxel is None:
            return
        self._objectnav_cached_detection = {
            "box": list(self.current_confirmed_detection_box),
            "score": self.current_confirmed_detection_score,
            "class": self.current_confirmed_detection_class,
            "voxel": tuple(int(v) for v in self.current_confirmed_detection_voxel),
            "box_area_ratio": self.current_confirmed_detection_box_area_ratio,
            "center_offset": self.current_confirmed_detection_center_offset,
            "edge_margin_px": self.current_confirmed_detection_edge_margin_px,
            "mask_voxels": self.current_confirmed_detection_mask_voxels,
            "image_shape": self.current_confirmed_detection_image_shape,
            "sed_input": self._current_confirmed_sed_input,
            "chosen_detection_3d": (
                tuple(int(v) for v in self.chosen_detection_3d)
                if self.chosen_detection_3d is not None else None
            ),
            "chosen_detection": self.chosen_detection,
            "image": np.asarray(image_rgb_hwc).copy() if image_rgb_hwc is not None else None,
        }

    def _restore_cached_confirmed_detection(self, agent_voxel: Tuple[int, int, int]) -> bool:
        cached = self._objectnav_cached_detection
        if not cached:
            return False
        det_voxel = cached.get("chosen_detection_3d") or cached.get("voxel")
        if det_voxel is None:
            return False
        det_voxel = tuple(int(v) for v in det_voxel)
        if self._xy_distance(agent_voxel, det_voxel) > self.objectnav_found_distance:
            return False

        self.current_confirmed_detection_box = list(cached["box"])
        self.current_confirmed_detection_score = cached["score"]
        self.current_confirmed_detection_class = cached["class"]
        self.current_confirmed_detection_voxel = tuple(int(v) for v in cached["voxel"])
        self.current_confirmed_detection_box_area_ratio = cached["box_area_ratio"]
        self.current_confirmed_detection_center_offset = cached["center_offset"]
        self.current_confirmed_detection_edge_margin_px = cached["edge_margin_px"]
        self.current_confirmed_detection_mask_voxels = cached["mask_voxels"]
        self.current_confirmed_detection_image_shape = cached.get("image_shape")
        self._current_confirmed_sed_input = cached.get("sed_input")
        self.chosen_detection_3d = det_voxel
        self.chosen_detection = cached.get("chosen_detection")
        self.object_detected = True
        self.navigation_mode = "objectnav"
        return True

    def _object_detection_quality_ok_for_found(self) -> Tuple[bool, str]:
        score = self.current_confirmed_detection_score
        area = self.current_confirmed_detection_box_area_ratio
        center_offset = self.current_confirmed_detection_center_offset
        edge_margin = self.current_confirmed_detection_edge_margin_px
        mask_voxels = self.current_confirmed_detection_mask_voxels
        if score is None or area is None or center_offset is None or edge_margin is None or mask_voxels is None:
            return False, "missing_quality"
        if score < self._object_found_min_confidence:
            return False, f"conf={score:.2f}<min={self._object_found_min_confidence:.2f}"
        if area < self._object_found_min_box_area_ratio:
            return False, f"area={area:.4f}<min={self._object_found_min_box_area_ratio:.4f}"
        if int(mask_voxels) < self._object_found_min_mask_voxels:
            return False, f"mask_voxels={int(mask_voxels)}<min={self._object_found_min_mask_voxels}"
        if edge_margin < self._object_found_edge_margin_px:
            return False, f"edge_margin={edge_margin:.1f}<min={self._object_found_edge_margin_px:.1f}"
        if center_offset > self._object_found_max_center_offset:
            return False, f"center_offset={center_offset:.2f}>max={self._object_found_max_center_offset:.2f}"
        return True, (
            f"conf={score:.2f}, area={area:.4f}, voxels={int(mask_voxels)}, "
            f"edge={edge_margin:.1f}, center={center_offset:.2f}"
        )

    def _detection_quality_from_box_and_points(
        self,
        box: List[float],
        pts_world: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> dict:
        h, w = image_shape
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        x1c = max(0.0, min(float(w), x1))
        x2c = max(0.0, min(float(w), x2))
        y1c = max(0.0, min(float(h), y1))
        y2c = max(0.0, min(float(h), y2))
        box_w = max(0.0, x2c - x1c)
        box_h = max(0.0, y2c - y1c)
        area_ratio = (box_w * box_h) / max(1.0, float(w * h))
        center_offset = max(
            abs(cx - 0.5 * w) / max(1.0, 0.5 * w),
            abs(cy - 0.5 * h) / max(1.0, 0.5 * h),
        )
        edge_margin = min(x1c, y1c, max(0.0, float(w) - x2c), max(0.0, float(h) - y2c))

        mask_voxels = 0
        if pts_world.size > 0:
            z_origin = self.one_map.z_origin if self.one_map.z_origin is not None else 0.0
            ix = np.floor(pts_world[:, 0] / self.one_map.voxel_size).astype(np.int32) + self.one_map.map_center
            iy = np.floor(pts_world[:, 1] / self.one_map.voxel_size).astype(np.int32) + self.one_map.map_center
            iz = (
                np.floor((pts_world[:, 2] - z_origin) / self.one_map.voxel_size_z).astype(np.int32)
                + self.one_map.max_z_voxels
            )
            valid = (
                (ix >= 0) & (ix < self.one_map.n_cells) &
                (iy >= 0) & (iy < self.one_map.n_cells) &
                (iz >= 0) & (iz < 2 * self.one_map.max_z_voxels)
            )
            if np.any(valid):
                voxel_arr = np.stack([ix[valid], iy[valid], iz[valid]], axis=1)
                mask_voxels = int(np.unique(voxel_arr, axis=0).shape[0])

        return {
            "box_area_ratio": float(area_ratio),
            "center_offset": float(center_offset),
            "edge_margin_px": float(edge_margin),
            "mask_voxels": int(mask_voxels),
            "image_shape": (int(h), int(w)),
        }

    def _reset_objectnav_pitch_scan(self) -> None:
        self._objectnav_pitch_scan_index = 0
        self._objectnav_pitch_scan_goal = None
        self._objectnav_pitch_scan_active = False

    def _reset_objectnav_correction(self, clear_cache: bool = False) -> None:
        self._objectnav_correction_action = None
        self._objectnav_align_action = None
        self._objectnav_restore_action = None
        self._objectnav_restore_pending = False
        self._objectnav_force_found_after_restore = False
        self._reset_objectnav_edge_oscillation()
        if clear_cache:
            self._objectnav_cached_detection = None

    def _reset_objectnav_edge_oscillation(self) -> None:
        self._objectnav_edge_oscillation_active = False
        self._objectnav_edge_oscillation_last_yaw = None
        self._objectnav_edge_oscillation_last_xy = None
        self._objectnav_edge_oscillation_last_sign = None
        self._objectnav_edge_oscillation_turns = 0

    def _arm_objectnav_edge_oscillation(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> None:
        """Start an ObjectNav-only left/right oscillation audit after edge failure."""
        if self.navigation_mode != "objectnav":
            return
        if self._objectnav_edge_oscillation_active:
            return
        self._objectnav_edge_oscillation_active = True
        self._objectnav_edge_oscillation_last_xy = (float(x), float(y))
        self._objectnav_edge_oscillation_last_yaw = float(yaw)
        self._objectnav_edge_oscillation_last_sign = None
        self._objectnav_edge_oscillation_turns = 0
        print("[Nav3D] ObjectNav edge oscillation audit armed")

    def _update_objectnav_edge_oscillation(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> bool:
        """Count strictly alternating in-place yaw actions; return True at the limit."""
        if not self._objectnav_edge_oscillation_active:
            return False
        if (
            self.navigation_mode != "objectnav" or
            not self.object_detected or
            self.chosen_detection_3d is None
        ):
            self._reset_objectnav_edge_oscillation()
            return False

        previous_xy = self._objectnav_edge_oscillation_last_xy
        previous_yaw = self._objectnav_edge_oscillation_last_yaw
        self._objectnav_edge_oscillation_last_xy = (float(x), float(y))
        self._objectnav_edge_oscillation_last_yaw = float(yaw)
        if previous_xy is None or previous_yaw is None:
            return False

        translation = float(np.hypot(float(x) - previous_xy[0], float(y) - previous_xy[1]))
        if translation > 0.05:
            # Actual navigation resumed: this is not a visual left/right loop.
            self._objectnav_edge_oscillation_last_sign = None
            self._objectnav_edge_oscillation_turns = 0
            return False

        yaw_delta = self._wrap_angle(float(yaw) - float(previous_yaw))
        if abs(yaw_delta) < np.deg2rad(15.0):
            return False
        turn_sign = 1 if yaw_delta > 0.0 else -1
        if self._objectnav_edge_oscillation_last_sign is None:
            self._objectnav_edge_oscillation_turns = 1
        elif turn_sign != self._objectnav_edge_oscillation_last_sign:
            self._objectnav_edge_oscillation_turns += 1
        else:
            # Require a strict L/R/L/R sequence; repeated turns in one
            # direction are legitimate visual search, not oscillation.
            self._objectnav_edge_oscillation_turns = 1
        self._objectnav_edge_oscillation_last_sign = turn_sign

        completed_cycles = self._objectnav_edge_oscillation_turns // 2
        if self._objectnav_edge_oscillation_turns % 2 == 0:
            print(
                f"[Nav3D] ObjectNav edge oscillation: "
                f"cycle={completed_cycles}/{self._objectnav_edge_oscillation_found_cycles}"
            )
        return completed_cycles >= self._objectnav_edge_oscillation_found_cycles

    def _queue_objectnav_yaw_alignment(
        self,
        odometry: np.ndarray,
        yaw: float,
    ) -> bool:
        if (
            self._objectnav_correction_action is not None or
            self._objectnav_restore_pending or
            self._objectnav_restore_action is not None or
            self._objectnav_force_found_after_restore
        ):
            return False
        if self._objectnav_align_action is not None:
            return True
        if self.chosen_detection_3d is None:
            return False

        tx, ty, _ = self.one_map.voxel_to_metric(*self.chosen_detection_3d)
        dx = float(tx) - float(odometry[0, 3])
        dy = float(ty) - float(odometry[1, 3])
        if dx * dx + dy * dy < 0.05 * 0.05:
            return False

        target_yaw = float(np.arctan2(dy, dx))
        yaw_error = self._wrap_angle(target_yaw - float(yaw))
        if abs(yaw_error) <= float(self._objectnav_yaw_align_threshold):
            return False

        self._objectnav_align_action = "L" if yaw_error >= 0.0 else "R"
        self._objectnav_miss_counter = 0
        self.path = None
        print(
            f"[Nav3D] Objectnav yaw alignment queued: action={self._objectnav_align_action}, "
            f"yaw_error={np.rad2deg(yaw_error):.1f}deg, target={self.chosen_detection_3d}"
        )
        return True

    def _clear_object_detection(self, reason: str = "") -> None:
        if reason:
            print(f"[Nav3D] Exit objectnav: {reason}")
        return_goal = self._objectnav_return_goal
        return_source = self._objectnav_source
        self.object_detected = False
        self.chosen_detection = None
        self.chosen_detection_3d = None
        self.objectnav_goal_voxel = None
        self._reset_current_confirmed_detection()
        self._objectnav_miss_counter = 0
        self._reset_objectnav_pitch_scan()
        self._reset_objectnav_correction(clear_cache=True)
        if return_source == "history_poi" and isinstance(return_goal, Cluster3D) and not self._is_poi_cluster_blacklisted(return_goal):
            self.navigation_mode = "navigate"
            self._navigate_goal = return_goal
            self.last_nav_goal = return_goal
            self.nav_goals = [return_goal]
            self._objectnav_return_goal = return_goal
            self._objectnav_source = return_source
        elif self.navigation_mode == "objectnav":
            self.navigation_mode = "explore"
            self._objectnav_return_goal = None
            self._objectnav_source = None
        self.path = None
        self.path_id = 0

    def _objectnav_quality_needs_view_correction(self, reason: str) -> bool:
        return reason.startswith("edge_margin=") or reason.startswith("center_offset=")

    def _objectnav_correction_from_box(self) -> Optional[str]:
        box = self.current_confirmed_detection_box
        image_shape = self.current_confirmed_detection_image_shape
        if box is None or image_shape is None:
            return None
        h, w = image_shape
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        margins = {
            "left": x1,
            "right": float(w) - x2,
            "top": y1,
            "bottom": float(h) - y2,
        }
        # Yaw alignment handles left/right. Box correction is reserved for
        # vertical framing so a single 30° turn does not fight pitch recovery.
        if margins["top"] < self._object_found_edge_margin_px:
            return "U"
        if margins["bottom"] < self._object_found_edge_margin_px:
            return "D"

        dx = (cx - 0.5 * float(w)) / max(1.0, 0.5 * float(w))
        dy = (cy - 0.5 * float(h)) / max(1.0, 0.5 * float(h))
        if abs(dy) <= abs(dx):
            return None
        if abs(dy) <= self._object_found_max_center_offset:
            return None
        return "U" if dy < 0.0 else "D"

    def _queue_objectnav_view_correction(
        self,
        reason: str,
        image_rgb_hwc: Optional[np.ndarray] = None,
    ) -> bool:
        if self._objectnav_correction_action or self._objectnav_restore_pending:
            return False
        if not self._objectnav_quality_needs_view_correction(reason):
            return False
        action = self._objectnav_correction_from_box()
        restore = self._inverse_objectnav_action(action) if action is not None else None
        if action is None or restore is None:
            return False
        self._cache_current_confirmed_detection(image_rgb_hwc)
        if not self._objectnav_cached_detection:
            return False
        self._objectnav_correction_action = action
        self._objectnav_restore_action = restore
        self._objectnav_miss_counter = 0
        self.path = None
        print(
            f"[Nav3D] Object found view correction queued: action={action}, "
            f"restore={restore}, reason={reason}"
        )
        return True

    def _maybe_force_found_after_objectnav_restore(
        self, agent_voxel: Tuple[int, int, int]
    ) -> Optional[bool]:
        """None means no cached found attempt; False means SED vetoed one."""
        if not self._objectnav_force_found_after_restore:
            return None
        self._objectnav_force_found_after_restore = False
        if self._restore_cached_confirmed_detection(agent_voxel):
            print(
                "[Nav3D] Object found after view restore using cached confirmed box: "
                f"agent_to_chosen={self._agent_to_chosen_distance_m(agent_voxel):.3f}m"
            )
            return self._finalize_object_found(
                agent_voxel, "cached_view_restore", clear_cache=False
            )
        self._reset_objectnav_correction(clear_cache=True)
        self._clear_object_detection("view correction restore had no valid cached target")
        return None

    def _find_object_approach_voxel(
        self,
        det_voxel: Tuple[int, int, int],
        agent_voxel: Tuple[int, int, int],
        preferred_radius_m: float = 0.9,
    ) -> Optional[Tuple[int, int, int]]:
        targetable = {VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR}
        min_radius = max(4, round(0.45 / self.one_map.cell_size))
        preferred_radius = max(min_radius, round(preferred_radius_m / self.one_map.cell_size))
        max_radius = max(16, round(1.8 / self.one_map.cell_size))
        agent_xy = np.array(agent_voxel[:2], dtype=np.float32)
        det_xy = np.array(det_voxel[:2], dtype=np.float32)
        view_vec = agent_xy - det_xy
        view_norm = float(np.linalg.norm(view_vec))
        if view_norm > 1e-3:
            view_dir = view_vec / view_norm
        else:
            view_dir = np.zeros(2, dtype=np.float32)

        best_vid = None
        best_score = float("inf")
        for vid, vd in self.one_map.voxels.items():
            if vd.voxel_type not in targetable or vd.inflated:
                continue
            if abs(vid[2] - agent_voxel[2]) > 2:
                continue
            offset = np.array([vid[0] - det_voxel[0], vid[1] - det_voxel[1]], dtype=np.float32)
            obj_dist = float(np.linalg.norm(offset))
            if obj_dist < min_radius or obj_dist > max_radius:
                continue
            clearance = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nb = (vid[0] + dx, vid[1] + dy, vid[2])
                    nb_vd = self.one_map.voxels.get(nb)
                    if nb_vd is not None and nb_vd.voxel_type in targetable and not nb_vd.inflated:
                        clearance += 1
            if clearance < 8:
                continue
            agent_dist = float(np.linalg.norm(np.array(vid[:2], dtype=np.float32) - agent_xy))
            radial_penalty = abs(obj_dist - preferred_radius)
            if view_norm > 1e-3:
                offset_dir = offset / max(obj_dist, 1e-6)
                angle_penalty = 1.0 - float(np.clip(np.dot(offset_dir, view_dir), -1.0, 1.0))
            else:
                angle_penalty = 0.0
            score = radial_penalty * 2.0 + angle_penalty * 6.0 + agent_dist * 0.05 - clearance * 0.02
            if score < best_score:
                best_score = score
                best_vid = vid

        if best_vid is not None:
            return best_vid
        return self.one_map._find_nearest_safe_traversable(det_voxel, max_radius=max_radius)

    def _process_target_detections(
        self,
        detections: dict,
        image: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
        yaw: float,
        px: int,
        py: int,
        agent_voxel: Tuple[int, int, int],
        dense_features: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Project target detections to 3D, validate with CLIP similarity,
        update chosen_detection_3d.

        Mirrors OneMap's add_data detection block:
          - SAM mask for precise back-projection
          - Depth filter (< 2.5 m for close-range reliability)
          - Consensus filtering: only accept if projected point is in top-K% similarity
          - Update chosen_detection only if new point has higher similarity
        """
        R_mat = odometry[:3, :3]
        t_vec = odometry[:3, 3]
        object_valid_any = False
        classes = detections.get("classes", [])

        for det_idx, (box, confidence) in enumerate(zip(detections["boxes"], detections["scores"])):
            # Skip low-confidence detections
            if confidence < self._obj_confidence:
                continue
            det_class = classes[det_idx] if det_idx < len(classes) else self.query_text[0]

            if self.log and _RERUN:
                try:
                    rr.log("camera/detection",
                           rr.Boxes2D(array_format=rr.Box2DFormat.XYXY, array=box))
                except Exception:
                    pass

            # SAM segmentation
            try:
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=np.array(box)[None, :],
                    multimask_output=False,
                )
                mask = masks[0]
            except Exception:
                continue

            mask_ids = np.argwhere(mask & (depth != 0) & (depth != float("inf")))
            if mask_ids.shape[0] == 0:
                continue

            depths = depth[mask_ids[:, 0], mask_ids[:, 1]]

            # Depth filter
            center_depth = depth[
                int((box[1] + box[3]) / 2), int((box[0] + box[2]) / 2)
            ]
            if self.filter_detections_depth and center_depth > 2.5:
                continue

            # Back-project to world
            u = mask_ids[:, 1].astype(np.float32)
            v = mask_ids[:, 0].astype(np.float32)
            x_cam = (u - self.one_map.cx) * depths / self.one_map.fx
            y_cam = (v - self.one_map.cy) * depths / self.one_map.fy
            pts_cam = np.stack([depths, -x_cam, -y_cam], axis=-1)
            pts_world = (R_mat @ pts_cam.T).T + t_vec
            if image.ndim == 3:
                image_shape = (int(image.shape[1]), int(image.shape[2]))
            else:
                image_shape = (int(image.shape[0]), int(image.shape[1]))
            detection_quality = self._detection_quality_from_box_and_points(
                box, pts_world, image_shape
            )

            # Use a robust representative depth for the object point. Taking
            # argmin(depths) is too sensitive to nearby walls/doorframes inside
            # the mask and can project a far target onto a close obstacle.
            median_depth = float(np.median(depths))
            depth_band = np.abs(depths - median_depth)
            min_band = float(np.percentile(depth_band, 25)) if depth_band.size else 0.0
            near_median = depth_band <= max(0.10, min_band)
            if not np.any(near_median):
                near_median = depth_band <= max(0.25, float(np.percentile(depth_band, 50)))
            candidate_ids = np.flatnonzero(near_median)
            if candidate_ids.size == 0:
                best_idx = int(np.argmin(depth_band))
            else:
                box_cx = 0.5 * (float(box[0]) + float(box[2]))
                box_cy = 0.5 * (float(box[1]) + float(box[3]))
                du = u[candidate_ids] - box_cx
                dv = v[candidate_ids] - box_cy
                best_idx = int(candidate_ids[int(np.argmin(du * du + dv * dv))])
            det_world = pts_world[best_idx]
            det_voxel = self.one_map.metric_to_voxel(
                det_world[0], det_world[1], det_world[2]
            )

            if self._is_sed_detection_cooling(det_voxel):
                continue

            # Consensus filtering (mirrors OneMap)
            object_valid = True
            region_sims: List[float] = []
            region_max_sim = 0.0
            weak_feature_support = False
            if self.consensus_filtering:
                sim_values = list(self.one_map.similarity_cache.values())
                if sim_values:
                    threshold = np.percentile(sim_values, self.percentile_exploitation)
                    weak_threshold = np.percentile(sim_values, 90.0)
                    # Check similarity in small region around detected voxel
                    for ddx in range(-2, 3):
                        for ddy in range(-2, 3):
                            nb = (det_voxel[0] + ddx, det_voxel[1] + ddy, det_voxel[2])
                            sim = self.one_map.similarity_cache.get(nb, 0.0)
                            region_sims.append(sim)
                    region_max_sim = float(max(region_sims)) if region_sims else 0.0
                    weak_feature_support = bool(region_sims and region_max_sim >= weak_threshold)
                    poi_overlap, poi = self._poi_overlap_evidence(det_voxel)
                    if region_sims and region_max_sim < threshold:
                        if poi_overlap:
                            box_area = float(detection_quality.get("box_area_ratio", 0.0))
                            if box_area >= self._object_found_min_box_area_ratio:
                                object_valid = True
                            else:
                                object_valid = False
                                if poi is not None:
                                    print(
                                        f"[Nav3D] Detection near history POI kept as hint: "
                                        f"det={det_voxel}, poi={poi.nav_voxel_id}, "
                                        f"area={box_area:.4f}<min={self._object_found_min_box_area_ratio:.4f}, "
                                        f"evidence={poi.evidence_score:.3f}, sim={region_max_sim:.3f}"
                                    )
                            if poi is not None:
                                if object_valid:
                                    print(
                                        f"[Nav3D] Detection confirmed by POI overlap: "
                                        f"det={det_voxel}, poi={poi.nav_voxel_id}, "
                                        f"area={box_area:.4f}, evidence={poi.evidence_score:.3f}, "
                                        f"sim={region_max_sim:.3f}"
                                    )
                        else:
                            object_valid = False

            small_box = (
                float(detection_quality.get("box_area_ratio", 0.0)) <
                self._object_found_min_box_area_ratio
            )

            if not object_valid:
                self._upsert_dino_live_poi(
                    det_voxel,
                    agent_voxel,
                    float(confidence),
                    region_sims,
                    close_approach=small_box,
                )
                continue

            # A valid but visually small detection is still only a POI hint.
            # Use a closer live observation pose when no history POI currently
            # owns navigation; ObjectNav takeover is gated separately on area.
            if small_box or weak_feature_support:
                self._upsert_dino_live_poi(
                    det_voxel,
                    agent_voxel,
                    float(confidence),
                    region_sims,
                    close_approach=small_box,
                )

            # Update chosen detection only if new point is strictly better
            if self.object_detected and self.chosen_detection_3d is not None:
                old_sim = self.one_map.similarity_cache.get(self.chosen_detection_3d, -1.0)
                new_sim = self.one_map.similarity_cache.get(det_voxel, -1.0)
                if new_sim < old_sim * 1.1:
                    dist_xy = np.sqrt(
                        (det_voxel[0] - self.chosen_detection_3d[0]) ** 2 +
                        (det_voxel[1] - self.chosen_detection_3d[1]) ** 2
                    )
                    if dist_xy <= self.max_detect_distance and abs(det_voxel[2] - self.chosen_detection_3d[2]) <= 5:
                        object_valid_any = True
                        self._set_current_confirmed_detection(
                            box, float(confidence), str(det_class), det_voxel, detection_quality
                        )
                        self._retain_confirmed_sed_input(dense_features, mask)
                    continue

            self.chosen_detection_3d = det_voxel
            self.chosen_detection = self.one_map.metric_to_px(det_world[0], det_world[1])
            self.object_detected = True
            self._set_current_confirmed_detection(
                box, float(confidence), str(det_class), det_voxel, detection_quality
            )
            self._retain_confirmed_sed_input(dense_features, mask)
            object_valid_any = True

        return object_valid_any
