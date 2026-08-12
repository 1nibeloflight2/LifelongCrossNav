from __future__ import annotations

import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from config import Conf, SpotControllerConf
from mapping.voxel_map import TARGETABLE_TYPES, VoxelMap, VoxelType, VoxelData
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

def euclidean_heuristic(a, b) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5

class PathManagerMixin:
    def get_path(self) -> Union[np.ndarray, str, None]:
        """Return current path for the controller."""
        if self._objectnav_align_action is not None:
            action = self._objectnav_align_action
            self._objectnav_align_action = None
            self.path = None
            return action
        if self._objectnav_correction_action is not None:
            action = self._objectnav_correction_action
            self._objectnav_correction_action = None
            self._objectnav_restore_pending = True
            self.path = None
            return action
        if self._objectnav_restore_pending and self._objectnav_restore_action is not None:
            action = self._objectnav_restore_action
            self._objectnav_restore_pending = False
            self._objectnav_force_found_after_restore = True
            self.path = None
            return action
        if self.navigation_mode == "navigate" and self._poi_confirm_scan_remaining > 0:
            self._poi_confirm_scan_remaining -= 1
            if self.object_detected:
                self._capture_objectnav_return_goal()
                self._exit_navigate_mode(blacklist_goal=False)
                self.navigation_mode = "objectnav"
                return None
            if self._poi_confirm_scan_remaining == 0:
                print("[Nav3D] POI scan finished without target confirmation; blacklisting POI")
                # A failed confirmation scan must leave the camera horizontal
                # before normal exploration resumes.  The actor applies this
                # pitch request ahead of the next navigation action.
                self._poi_confirm_pitch_level = 0
                self.desired_pitch_level = 0
                self._exit_navigate_mode(blacklist_goal=True)
                return None
            return "L"
        if self._rise_confirm_scan_goal is not None and self._rise_confirm_scan_remaining > 0:
            self._rise_confirm_scan_remaining -= 1
            return self._rise_confirm_scan_turn
        if self._post_upstairs_probe_state == "pre_scan":
            if self.stair_sem_mode:
                self._reset_post_upstairs_probe()
            else:
                if self._post_upstairs_probe_scan_remaining > 0:
                    self._post_upstairs_probe_scan_remaining -= 1
                    return "L"
                direction = getattr(self, "_post_stair_probe_direction", None) or "up"
                print(
                    f"[Nav3D] Post-{'upstairs' if direction == 'up' else 'downstairs'} "
                    f"pre-center scan complete; computing platform center"
                )
                self._post_upstairs_probe_state = "compute_center"
                self.path = None
                self.last_nav_goal = None
                return None
        if self._post_upstairs_probe_state == "scan":
            if self.stair_sem_mode:
                self._reset_post_upstairs_probe()
            else:
                if self._post_upstairs_probe_scan_remaining > 0:
                    self._post_upstairs_probe_scan_remaining -= 1
                    return "L"
                probe_direction = getattr(self, "_post_stair_probe_direction", None) or "up"
                has_next_rise = (
                    probe_direction == "up" and self._has_active_rise_stair_candidate()
                )
                all_rise = [
                    (
                        tuple(int(v) for v in key),
                        tuple(int(v) for v in cand.get("nav_voxel_id", key)),
                        bool(cand.get("active", False)),
                        int(cand.get("seen_count", 0)),
                        int(cand.get("last_seen_step", -1)),
                        round(float(cand.get("confidence", 0.0)), 3),
                        round(float(cand.get("geometry_score", 0.0)), 3),
                    )
                    for key, cand in self.rise_stair_candidates.items()
                ]
                active_rise = [
                    (tuple(cand.get("nav_voxel_id", key)),
                     int(cand.get("seen_count", 0)),
                     float(cand.get("confidence", 0.0)),
                     float(cand.get("geometry_score", 0.0)))
                    for key, cand in self.rise_stair_candidates.items()
                    if cand.get("active", False)
                ]
                if probe_direction == "up":
                    print(
                        f"[Nav3D][RiseDebug] post-upstairs scan end: "
                        f"has_next={has_next_rise}, all={all_rise[:8]}"
                    )
                    if active_rise:
                        print(f"[Nav3D] Post-upstairs active rise candidates at scan end: {active_rise[:4]}")
                if has_next_rise:
                    print("[Nav3D] Post-upstairs probe scan complete; continuing to next rise stair candidate")
                elif probe_direction == "down":
                    print("[Nav3D] Post-downstairs probe scan complete; rebuilding current-floor frontiers")
                else:
                    print("[Nav3D] Post-upstairs probe scan complete; returning to normal exploration")
                self._reset_post_upstairs_probe()
                self.path = None
                self.last_nav_goal = None
                self.nav_goals.clear()
                if has_next_rise:
                    self._post_upstairs_prioritize_rise = True
                else:
                    if hasattr(self, "frontier_clusters"):
                        self.frontier_clusters.clear()
                    self.first_obs = True
                    self._frontier_min_cluster_size = 2
                    if probe_direction == "down":
                        self._force_current_floor_frontier_check = True
                return None
        if not self.path:
            return None
        if not self.object_detected:
            if self.saw_left:
                return "L"
            if self.saw_right:
                return "R"
        remaining = self.path[min(self.path_id, len(self.path)):]
        if not remaining:
            return None
        if self.navigation_mode == "navigate" and not self.stair_sem_mode:
            remaining = self._actor_followable_path(remaining)
            if not remaining:
                return None
        return self.one_map.path_3d_to_metric_2d(remaining)

    def _actor_followable_path(
        self,
        path_3d: List[Tuple[int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """
        Keep A* connector voxels out of the actor waypoint stream.

        Cross-floor navigate may use STAIR voxels to keep the graph connected,
        but the Habitat agent only occupies traversable floor slices. If STAIR
        waypoints are emitted directly, the actor can turn toward an unreachable
        stair voxel forever. The planner may still route through STAIR; the
        controller only follows TRAVERSABLE/TRAVERSABLE_STAIR waypoints.
        """
        followable_types = {VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR}
        filtered: List[Tuple[int, int, int]] = []
        last_xy: Optional[Tuple[int, int]] = None
        for vid in path_3d:
            vd = self.one_map.voxels.get(vid)
            if vd is None or vd.voxel_type not in followable_types:
                continue
            xy = (vid[0], vid[1])
            if xy == last_xy:
                continue
            filtered.append(vid)
            last_xy = xy
        return filtered

    def _update_desired_pitch(self, agent_voxel: Tuple[int, int, int]) -> None:
        """
        Compute desired_pitch_level based on priority:

        Priority 0 (highest): Stair sem mode → always 60° down
        Priority 1: Fake frontier proximity → 60° down (triggers confirmation)
        Priority 2: Path z-trend (Trajectory-Driven Gaze, +/-30 degrees)
        Priority 3: Default horizontal
        """
        # --- Priority 0: Stair sem mode ---
        if self.stair_sem_mode:
            self.desired_pitch_level = 2 if self._stair_sem_direction == "down" else 0
            return

        if self._post_upstairs_probe_active():
            self.desired_pitch_level = 0
            return

        # Keep the history-POI confirmation pitch stable throughout its full
        # rotation.  ObjectNav takes ownership as soon as a target is detected,
        # because that transition clears the confirmation counter.
        if self.navigation_mode == "navigate" and self._poi_confirm_scan_remaining > 0:
            self.desired_pitch_level = int(self._poi_confirm_pitch_level)
            return

        if self._objectnav_pitch_scan_active:
            idx = min(
                self._objectnav_pitch_scan_index,
                len(self._objectnav_pitch_scan_levels) - 1,
            )
            self.desired_pitch_level = int(self._objectnav_pitch_scan_levels[idx])
            return

        if (
            self._objectnav_correction_action is not None or
            self._objectnav_restore_pending or
            self._objectnav_restore_action is not None or
            self._objectnav_force_found_after_restore
        ):
            return

        # --- Priority 1: Fake frontier proximity ---
        # If tracking a fake frontier and agent is near it, look down
        if self._fake_frontier_target is not None:
            selected = self._selected_down_stair_goal()
            selected_target = self._selected_down_stair_target()
            if selected is None or selected_target != self._fake_frontier_target:
                self._reset_fake_frontier_state()
                self.desired_pitch_level = 0
                return
            # Check if still near the target
            dist_xy = self._xy_distance(agent_voxel, self._fake_frontier_target)
            if dist_xy <= 6:
                self.desired_pitch_level = 2
                # Notify that pitch target is set — when actor reaches 60°,
                # the confirmation window starts
                if not self._fake_frontier_pitched and self.desired_pitch_level == 2:
                    # Will be activated when actor actually reaches pitch=2
                    pass
                return
            else:
                # Moved away from fake frontier — reset
                self._reset_fake_frontier_state()

        # --- Priority 2: Path z-trend ---
        if self.path and len(self.path) > self.path_id:
            remaining_path = self.path[self.path_id:]
            lookahead = min(8, len(remaining_path))
            if lookahead >= 3:
                z_values = [remaining_path[i][2] for i in range(lookahead)]
                z_start = z_values[0]
                z_end = z_values[-1]
                delta_z = z_end - z_start

                descent_threshold = -2
                ascent_threshold = 2
                flat_threshold = 1

                if delta_z <= descent_threshold:
                    # Cross-floor A* following uses a single 30-degree pitch
                    # step.  A 60-degree navigation gaze frequently removes
                    # the forward scene and object evidence from the image.
                    # StairSem/fake-frontier confirmation retains its explicit
                    # 60-degree behavior in the higher-priority branches.
                    self.desired_pitch_level = 1
                    return
                elif delta_z >= ascent_threshold:
                    self.desired_pitch_level = -1
                    return
                elif abs(delta_z) <= flat_threshold:
                    self.desired_pitch_level = 0
                    return

        # --- Priority 3: Default horizontal ---
        self.desired_pitch_level = 0

    def _try_descend_cluster_alternative_path(
        self,
        agent_voxel: Tuple[int, int, int],
        goal: FrontierCluster3D,
        failed_goal: Tuple[int, int, int],
        max_iterations: int,
        allow_inflated_from_inflated_start: bool,
    ) -> Tuple[Optional[List[Tuple[int, int, int]]], Optional[Tuple[int, int, int]]]:
        """Try local anchor alternatives before blacklisting a descend region.

        Incremental rebuilding may move a fake-frontier anchor by a few cells.
        The shifted cell can be disconnected even though the previously locked
        anchor, or another safe cell in the same local cluster, is reachable.
        """
        if not goal.is_stair_frontier:
            return None, None

        blacklist_set = {
            tuple(int(v) for v in item)
            for item in self.blacklisted_goals
        }
        candidates: List[Tuple[int, int, int]] = []
        seen: Set[Tuple[int, int, int]] = {failed_goal}

        def add_candidate(vid: Tuple[int, int, int]) -> None:
            vid = tuple(int(v) for v in vid)
            if vid in seen or vid in blacklist_set:
                return
            vd = self.one_map.voxels.get(vid)
            if vd is None or vd.voxel_type not in TARGETABLE_TYPES or vd.inflated:
                return
            seen.add(vid)
            candidates.append(vid)

        # Prefer the previously committed anchor when it still belongs to this
        # local edge. This is the stable target that was already being followed.
        previous = self.last_nav_goal
        if isinstance(previous, FrontierCluster3D) and previous.is_stair_frontier:
            previous_vid = tuple(int(v) for v in previous.nav_voxel_id)
            aabb = goal.fake_anchor_aabb or goal.aabb
            in_local_aabb = (
                aabb[0][0] - 2 <= previous_vid[0] <= aabb[1][0] + 2 and
                aabb[0][1] - 2 <= previous_vid[1] <= aabb[1][1] + 2 and
                aabb[0][2] - 1 <= previous_vid[2] <= aabb[1][2] + 1
            )
            close_to_shifted_goal = (
                abs(previous_vid[2] - failed_goal[2]) <= 1 and
                self._xy_distance(previous_vid, failed_goal) <=
                max(2, round(0.5 / self.one_map.cell_size))
            )
            if in_local_aabb or close_to_shifted_goal:
                add_candidate(previous_vid)

        # Then exhaust the current local cluster. Fake-frontier clusters are
        # spatially tiled, so this search remains local and bounded.
        cluster_voxels = set(getattr(goal, "cluster_voxels", set()) or set())
        cluster_voxels.update(
            getattr(goal, "fake_triggered_voxels", set()) or set()
        )
        ordered_cluster_voxels = sorted(
            (tuple(int(v) for v in vid) for vid in cluster_voxels),
            key=lambda vid: (
                (vid[0] - failed_goal[0]) ** 2 +
                (vid[1] - failed_goal[1]) ** 2 +
                (vid[2] - failed_goal[2]) ** 2,
                vid,
            ),
        )
        for vid in ordered_cluster_voxels:
            add_candidate(vid)

        # Long fake edges are split into 8x8 local tiles by frontier detection.
        # Include adjacent descend tiles that would be covered by the same
        # radius-based region blacklist; geometrically they are alternative
        # anchors on the same physical edge, not unrelated stair proposals.
        current_center = self._anchor_center(goal)
        sibling_voxels: Set[Tuple[int, int, int]] = set()
        sibling_goals: List[Tuple[int, int, int]] = []
        for sibling in self.frontier_clusters:
            if sibling is goal or not sibling.is_stair_frontier:
                continue
            sibling_center = self._anchor_center(sibling)
            if abs(sibling_center[2] - current_center[2]) > 2:
                continue
            if self._xy_distance(sibling_center, current_center) > 8:
                continue
            sibling_goals.append(tuple(int(v) for v in sibling.nav_voxel_id))
            sibling_voxels.update(
                tuple(int(v) for v in vid)
                for vid in (getattr(sibling, "cluster_voxels", set()) or set())
            )
            sibling_voxels.update(
                tuple(int(v) for v in vid)
                for vid in (getattr(sibling, "fake_triggered_voxels", set()) or set())
            )

        for vid in sorted(
            sibling_goals,
            key=lambda item: (
                (item[0] - failed_goal[0]) ** 2 +
                (item[1] - failed_goal[1]) ** 2 +
                (item[2] - failed_goal[2]) ** 2,
                item,
            ),
        ):
            add_candidate(vid)
        for vid in sorted(
            sibling_voxels,
            key=lambda item: (
                (item[0] - failed_goal[0]) ** 2 +
                (item[1] - failed_goal[1]) ** 2 +
                (item[2] - failed_goal[2]) ** 2,
                item,
            ),
        ):
            add_candidate(vid)

        for attempt, alternative in enumerate(candidates, start=1):
            path = self.one_map.a_star_3d(
                agent_voxel,
                alternative,
                max_iterations=max_iterations,
                stair_mode=False,
                prefer_traversable=True,
                allow_inflated_from_inflated_start=allow_inflated_from_inflated_start,
            )
            if path is None:
                path = self.one_map.a_star_3d(
                    agent_voxel,
                    alternative,
                    max_iterations=max_iterations,
                    stair_mode=False,
                    prefer_traversable=False,
                    allow_inflated_from_inflated_start=allow_inflated_from_inflated_start,
                )
            if path is None:
                continue

            print(
                f"[Nav3D] Descend anchor fallback: failed={failed_goal}, "
                f"using={alternative}, attempt={attempt}/{len(candidates)}, "
                f"path_len={len(path)}"
            )
            return path, alternative

        if candidates:
            print(
                f"[Nav3D] Descend anchor fallback exhausted: failed={failed_goal}, "
                f"alternatives={len(candidates)}"
            )
        return None, None

    def _compute_best_path(self, agent_voxel: Tuple[int, int, int]) -> None:
        """
        Compute path to best nav goal (Frontier3D or Cluster3D).

        Implements the full OneMap frontier-persistence + cyclic-detection logic:
          1. If last_nav_goal still exists in nav_goals with competitive score → reuse it
          2. Otherwise: iterate nav_goals, skip cyclic/blacklisted, pick first plannable
          3. Stuck at nav_goal > threshold → blacklist it
          4. If no path found → clear nav_goals so they're recomputed next frame
        """
        self.path_id = 0

        # Actor3D owns the 29-action initialization scan.  During that scan the
        # mapper may keep building voxels/frontiers, but it must not commit a
        # path or interpret the deliberate in-place rotations as a stuck goal.
        if self.initializing:
            self.path = None
            self.last_nav_goal = None
            self.stuck_at_nav_goal_counter = 0
            self.stuck_at_cell_counter = 0
            self._goal_missing_counter = 0
            self._no_goal_available = False
            return

        if self._post_upstairs_probe_state != "inactive":
            if self._compute_post_upstairs_probe_path(agent_voxel):
                return

        if (self.stair_sem_mode and self._stair_sem_frontier is not None and
                self.path is not None and isinstance(self.last_nav_goal, StairSemanticFrontier3D)):
            same_target = np.array_equal(
                self.last_nav_goal.get_descr_point(),
                self._stair_sem_frontier.get_descr_point(),
            )
            replan_age = self.one_map._iters - self._stair_sem_last_replan_iter
            if same_target and replan_age < self._stair_sem_replan_period:
                return

        if self.object_detected and not self.stair_sem_mode:
            self.navigation_mode = "objectnav"
            self._navigate_goal = None
            self._poi_confirm_scan_remaining = 0
            self._compute_path_to_object(agent_voxel)
            if self.path is not None:
                return
            # Path to detected object failed — fall through to frontier exploration
            # Keep object_detected=True so we retry next frame after moving closer
            print(f"[Nav3D] Path to detected object failed, falling back to frontier exploration")

        if self.navigation_mode == "navigate" and self._navigate_goal is not None:
            self.nav_goals = [self._navigate_goal]
        elif self.navigation_mode == "navigate":
            self.navigation_mode = "explore"

        # Exploration mode
        if not self.nav_goals:
            missing_basic = (
                isinstance(self.last_nav_goal, FrontierCluster3D) and
                not self.last_nav_goal.is_stair_frontier
            )
            if (
                missing_basic and
                self.path is not None and
                self._goal_missing_counter < self._goal_missing_grace
            ):
                # Do not run a full scan on every grace frame.  Preserve the
                # old path for exactly N frames, then perform the current-floor
                # rescan once before falling through to stair categories.
                self._goal_missing_counter += 1
                print(
                    f"[Nav3D] Basic frontier disappeared: keeping old path "
                    f"for grace {self._goal_missing_counter}/"
                    f"{self._goal_missing_grace}"
                )
                return

            if not self.initializing:
                # Force full re-detection
                current_floor_rescan = (
                    self._force_current_floor_frontier_check or missing_basic
                )
                z_range = (
                    self._frontier_full_check_z_range
                    if current_floor_rescan
                    else self._frontier_z_range
                )
                self.frontier_clusters = detect_frontiers_3d_full(
                    self.one_map, agent_voxel,
                    # An explicit/missing-basic current-floor rescan covers all
                    # known XY positions in that z band.  Other generic full
                    # fallbacks retain the normal agent-local window.
                    search_radius=None if current_floor_rescan else 60,
                    frontier_sim_depth=self.frontier_sim_depth,
                    sim_weight=self.sim_weight,
                    min_cluster_size=self._frontier_min_cluster_size,
                    z_range=z_range,
                    max_clusters=None,
                )
                self._force_current_floor_frontier_check = False
                self._compute_frontiers_and_POIs(
                    agent_voxel,
                    # The scan above is already the required current-floor
                    # fallback for an expired basic goal.
                    allow_current_floor_rescan=not missing_basic,
                )
            if not self.nav_goals:
                # Grace has expired (or there was no path to preserve).  Never
                # expose NO NAV GOALS while an unrelated stale path remains,
                # otherwise Actor3D follows it and the evaluator cannot count
                # consecutive no-goal frames.
                self.path = None
                if missing_basic:
                    self.last_nav_goal = None
                    self._goal_missing_counter = 0
                self._debug_no_nav_goals(agent_voxel)
                print(f"[Nav3D] NO NAV GOALS: frontier_clusters={len(self.frontier_clusters)}, "
                      f"blacklisted={len(self.blacklisted_goals)}, voxels={len(self.one_map.voxels)}")
                if self.log and _RERUN:
                    try:
                        rr.log("path_updates", rr.TextLog(
                            f"NO NAV GOALS: frontier_clusters={len(self.frontier_clusters)}, "
                            f"blacklisted={len(self.blacklisted_goals)}, "
                            f"total_voxels={len(self.one_map.voxels)}"
                        ))
                    except Exception:
                        pass
                return

        self.is_goal_path = False

        # Top-2 scores for cyclic detection context (matches OneMap's top_two_vals)
        if len(self.nav_goals) >= 2:
            top_two_vals = (self.nav_goals[0].get_score(), self.nav_goals[1].get_score())
        else:
            top_two_vals = (self.nav_goals[0].get_score(), self.nav_goals[0].get_score())

        agent_2d = np.array([agent_voxel[0], agent_voxel[1]], dtype=np.float32)

        # ---- Frontier persistence: try to reuse last_nav_goal ----
        best_idx: Optional[int] = None
        committed_frontier_types = (FrontierCluster3D, RiseStairFrontier3D)

        if self.navigation_mode == "navigate" and self._navigate_goal is not None:
            best_idx = 0
        elif self.last_nav_goal is not None:
            last_descr = self.last_nav_goal.get_descr_point()
            # Search for the same or close goal in current list
            curr_index = None
            for nav_id, goal in enumerate(self.nav_goals):
                if np.array_equal(goal.get_descr_point(), last_descr):
                    curr_index = nav_id
                    break
            if curr_index is None:
                # Look for a close neighbour (within threshold voxels)
                threshold = max(2, round(0.5 / self.one_map.cell_size))
                same_kind_close_index = None
                for nav_id, goal in enumerate(self.nav_goals):
                    if np.linalg.norm(
                        goal.get_descr_point()[:2] - last_descr[:2]
                    ) <= threshold:
                        if (
                            isinstance(self.last_nav_goal, FrontierCluster3D) and
                            isinstance(goal, FrontierCluster3D)
                        ) or (
                            isinstance(self.last_nav_goal, RiseStairFrontier3D) and
                            isinstance(goal, RiseStairFrontier3D)
                        ):
                            same_kind_close_index = nav_id
                            break
                        if curr_index is None:
                            curr_index = nav_id
                if same_kind_close_index is not None:
                    curr_index = same_kind_close_index

            if curr_index is None:
                # Goal disappeared from nav_goals — grace period before switching
                self._goal_missing_counter += 1
                if self._goal_missing_counter <= self._goal_missing_grace and self.path is not None:
                    # Keep old path and goal, skip replanning this frame
                    print(
                        f"[Nav3D] Basic frontier disappeared: keeping old path "
                        f"for grace {self._goal_missing_counter}/"
                        f"{self._goal_missing_grace}"
                    )
                    return
                # Grace period expired: discard only the stale committed goal
                # and its path.  Keep the freshly generated nav_goals so this
                # frame can select the next basic, then descend, then ascent.
                if (
                    isinstance(self.last_nav_goal, FrontierCluster3D) and
                    not self.last_nav_goal.is_stair_frontier
                ):
                    self.path = None
                    self.last_nav_goal = None
                    self._goal_missing_counter = 0
                    print("[Nav3D] Basic frontier grace expired: selecting next available category")
            elif curr_index is not None:
                # Goal found — reset grace counter
                self._goal_missing_counter = 0

            if curr_index is not None:
                curr_goal = self.nav_goals[curr_index]
                if (
                    isinstance(self.last_nav_goal, committed_frontier_types) and
                    isinstance(curr_goal, committed_frontier_types) and
                    (
                        (
                            isinstance(self.last_nav_goal, FrontierCluster3D) and
                            isinstance(curr_goal, FrontierCluster3D)
                        ) or (
                            isinstance(self.last_nav_goal, RiseStairFrontier3D) and
                            isinstance(curr_goal, RiseStairFrontier3D)
                        )
                    )
                ):
                    # Frontier exploration is committed: keep the selected
                    # normal/down/rise frontier until it disappears, invalidates,
                    # fails planning, or navigation/objectnav interrupts it.
                    best_idx = curr_index
                    curr_goal = None

            if curr_index is not None and best_idx is None:
                curr_goal = self.nav_goals[curr_index]
                # Re-score current frontier with fresh similarity
                if isinstance(curr_goal, Frontier3D):
                    curr_goal = rescore_frontier(
                        curr_goal, self.one_map, agent_voxel,
                        frontier_sim_depth=self.frontier_sim_depth,
                        sim_weight=self.sim_weight,
                    )
                    self.nav_goals[curr_index] = curr_goal

                curr_score = curr_goal.get_score()
                last_score = self.last_nav_goal.get_score()

                # Prefer current goal if score within tolerance of best
                best_score_overall = self.nav_goals[0].get_score() if self.nav_goals else 0.0
                if (curr_score + curr_score * self.persistence_tolerance >= last_score and
                        curr_score >= best_score_overall * (1 - self.persistence_tolerance)):
                    best_idx = curr_index

        # ---- If not persisting, search for best non-cyclic goal ----
        if best_idx is None:
            blacklist_set = set(
                tuple(b) if not isinstance(b, tuple) else b
                for b in self.blacklisted_goals
            )
            for nav_id, goal in enumerate(self.nav_goals):
                goal_descr = goal.get_descr_point()
                goal_voxel_id = tuple(int(v) for v in goal_descr)

                if goal_voxel_id in blacklist_set:
                    continue

                goal_2d = goal_descr[:2].astype(np.float32)
                if self.cyclic_checker.check_cyclic(agent_2d, goal_2d, top_two_vals):
                    continue

                best_idx = nav_id
                break

        if best_idx is None:
            # All goals are cyclic or blacklisted — reset
            print(f"[Nav3D] All {len(self.nav_goals)} goals cyclic/blacklisted, resetting")
            if self.log and _RERUN:
                try:
                    rr.log("path_updates", rr.TextLog(
                        "All goals cyclic/blacklisted, resetting frontiers"))
                except Exception:
                    pass
            self.frontier_clusters.clear()
            self.nav_goals.clear()
            return

        # ---- Plan path to chosen goal ----
        self.path = None  # clear old path before replanning
        self._goal_missing_counter = 0  # successfully picked a goal, reset grace counter
        chosen_goal = self.nav_goals[best_idx]
        goal_voxel = tuple(int(v) for v in chosen_goal.get_descr_point())
        goal_vd = self.one_map.voxels.get(goal_voxel)
        if goal_vd is not None and goal_vd.voxel_type == VoxelType.FAKE_STAIR:
            print(f"[Nav3D] Rejecting FAKE_STAIR nav goal {goal_voxel}")
            self.nav_goals.pop(best_idx)
            self.rise_stair_candidates.pop(goal_voxel, None)
            self.path = None
            return
        if self._rise_confirm_scan_goal is not None:
            if (
                not isinstance(chosen_goal, RiseStairFrontier3D) or
                chosen_goal.nav_voxel_id != self._rise_confirm_scan_goal
            ):
                self._reset_rise_confirm_scan()

        # Register cyclic state. Navigate mode owns a fixed goal, so cyclic
        # frontier replacement should not mutate it mid-route.
        if self.navigation_mode != "navigate":
            self.cyclic_checker.add_state_action(
                agent_2d, chosen_goal.get_descr_point()[:2].astype(np.float32), top_two_vals
            )

        # A* iteration budget depends on goal type
        if isinstance(chosen_goal, StairSemanticFrontier3D):
            max_iter = 80000  # stair paths can be long (multi-floor)
        elif isinstance(chosen_goal, RiseStairFrontier3D):
            max_iter = 30000
        elif isinstance(chosen_goal, Frontier3D):
            max_iter = 30000
        else:
            max_iter = 20000
        # Use stair_mode for StairSemanticFrontier3D to bridge empty layers
        use_stair_mode = isinstance(chosen_goal, StairSemanticFrontier3D)
        use_navigate_a_star = self.navigation_mode == "navigate" or isinstance(chosen_goal, Cluster3D)
        is_history_poi_goal = (
            isinstance(chosen_goal, Cluster3D) and
            "history" in str(getattr(chosen_goal, "poi_type", "")).lower()
        )
        allow_inflated_explore = self.navigation_mode == "explore" and not self.stair_sem_mode
        validated_path = getattr(chosen_goal, "_validated_path", None)
        validated_start = getattr(chosen_goal, "_validated_path_start", None)
        reuse_validated_goal_path = (
            (
                isinstance(chosen_goal, FrontierCluster3D) or
                is_history_poi_goal
            ) and
            validated_path is not None and
            validated_start == tuple(int(v) for v in agent_voxel)
        )
        # Explore模式：非楼梯前沿优先走 TRAVERSABLE/TRAVERSABLE_STAIR，
        # 避免路径穿过 STAIR 体素（可能是已确认的楼梯区域）
        if reuse_validated_goal_path:
            self.path = list(validated_path)
            delattr(chosen_goal, "_validated_path")
            delattr(chosen_goal, "_validated_path_start")
        elif use_navigate_a_star:
            if is_history_poi_goal:
                self.path, _ = self._plan_history_poi_path_bidirectional(
                    agent_voxel,
                    goal_voxel,
                    max_iterations=max_iter,
                    log_reverse=True,
                )
            else:
                self.path = self.one_map.a_star_3d(
                    agent_voxel, goal_voxel, max_iterations=max_iter,
                    stair_mode=False,
                    prefer_traversable=False,
                    allow_inflated_traversable=False,
                    allow_inflated_stair=True,
                    allow_inflated_traversable_stair=True,
                    connect_traversable_stair_drop=True,
                )
        elif not use_stair_mode and not isinstance(chosen_goal, RiseStairFrontier3D):
            self.path = self.one_map.a_star_3d(
                agent_voxel, goal_voxel, max_iterations=max_iter,
                stair_mode=False, prefer_traversable=True,
                allow_inflated_from_inflated_start=allow_inflated_explore,
            )
            if self.path is None:
                # fallback: 加入 STAIR/STAIR_EXPANDED 重新规划
                self.path = self.one_map.a_star_3d(
                    agent_voxel, goal_voxel, max_iterations=max_iter,
                    stair_mode=False, prefer_traversable=False,
                    allow_inflated_from_inflated_start=allow_inflated_explore,
                )
        else:
            if use_stair_mode:
                # StairSem uses a single permissive 1 m physical-radius
                # neighbourhood; there is no separate strict-26 / sparse-cube
                # planning pass.
                self.path = self.one_map.a_star_3d(
                    agent_voxel, goal_voxel, max_iterations=max_iter,
                    stair_mode=True,
                    allow_inflated_from_inflated_start=allow_inflated_explore,
                )
            else:
                self.path = self.one_map.a_star_3d(
                    agent_voxel, goal_voxel, max_iterations=max_iter,
                    stair_mode=False,
                    allow_inflated_from_inflated_start=allow_inflated_explore,
                )

        # Do not retain a fallback identity after the formal representative
        # itself becomes reachable (or navigation changes category).  Without
        # this guard a stale anchor could keep owning look-down confirmation
        # even though the current A* path ends somewhere else.
        if self.path is not None:
            if isinstance(chosen_goal, FrontierCluster3D) and chosen_goal.is_stair_frontier:
                active_anchor = self._active_descend_fallback_anchor
                if active_anchor is not None and tuple(active_anchor) != tuple(goal_voxel):
                    self._active_descend_fallback_anchor = None
                    self._active_descend_fallback_owner = None
            else:
                self._active_descend_fallback_anchor = None
                self._active_descend_fallback_owner = None

        # A descend anchor can drift slightly when its incremental local
        # cluster is rebuilt. Do not blacklist the whole fake-frontier region
        # merely because that one shifted landing cell is disconnected.
        if (
            self.path is None and
            isinstance(chosen_goal, FrontierCluster3D) and
            chosen_goal.is_stair_frontier
        ):
            alternative_path, alternative_goal = self._try_descend_cluster_alternative_path(
                agent_voxel,
                chosen_goal,
                goal_voxel,
                max_iter,
                allow_inflated_explore,
            )
            if alternative_path is not None and alternative_goal is not None:
                self.path = alternative_path
                # Preserve both identities.  Incremental frontier rebuilding
                # may restore chosen_goal.nav_voxel_id to goal_voxel on the
                # next frame; stair confirmation must still own and track the
                # reachable anchor actually used by A*.
                self._active_descend_fallback_owner = goal_voxel
                self._active_descend_fallback_anchor = alternative_goal
                chosen_goal.nav_voxel_id = alternative_goal
                goal_voxel = alternative_goal

        if self.path is None:
            # --- Stair Semantic Frontier fallback: try intermediate STAIR target ---
            if isinstance(chosen_goal, StairSemanticFrontier3D):
                fallback_path = self._find_stair_intermediate_path(agent_voxel, goal_voxel)
                if fallback_path is not None:
                    self.path = fallback_path
                    print(f"[Nav3D] A* to stair sem goal {goal_voxel} failed, "
                          f"using intermediate target {fallback_path[-1]} (len={len(fallback_path)})")
                else:
                    # No intermediate reachable — remove and let next frame retry
                    # (SAM may mark more STAIR on next observation)
                    print(f"[Nav3D] A* FAILED (stair sem): agent={agent_voxel} -> goal={goal_voxel}, "
                          f"no intermediate STAIR reachable, removed")
                    self.nav_goals.pop(best_idx)
                    if self.log and _RERUN:
                        try:
                            rr.log("path_updates", rr.TextLog(f"No stair path to {goal_voxel}"))
                        except Exception:
                            pass
                    return
            else:
                # Normal goal: remove unplannable
                print(f"[Nav3D] A* FAILED: agent={agent_voxel} -> goal={goal_voxel}, removed")
                if self.navigation_mode == "navigate":
                    self._exit_navigate_mode(blacklist_goal=True)
                    return
                if isinstance(chosen_goal, FrontierCluster3D) and chosen_goal.is_stair_frontier:
                    self._fake_frontier_cluster = chosen_goal
                    self._blacklist_fake_frontier(goal_voxel)
                    force_current_floor_check = False
                else:
                    force_current_floor_check = False
                    if isinstance(chosen_goal, FrontierCluster3D):
                        if not chosen_goal.is_stair_frontier:
                            added = self._blacklist_basic_frontier_region(chosen_goal)
                            print(
                                f"[Nav3D] Basic frontier local blacklist: "
                                f"goal={goal_voxel}, added={added}"
                            )
                    self.frontier_clusters = [
                        c for c in self.frontier_clusters
                        if c.nav_voxel_id != goal_voxel
                    ]
                self.nav_goals.pop(best_idx)
                if force_current_floor_check:
                    self.nav_goals.clear()
                self.rise_stair_candidates.pop(goal_voxel, None)
                if self.log and _RERUN:
                    try:
                        rr.log("path_updates", rr.TextLog(f"No path to {goal_voxel}, removed"))
                    except Exception:
                        pass
                return

        # ---- Stuck-at-nav-goal detection ----
        if (self.last_nav_goal is not None and
                np.array_equal(self.last_nav_goal.get_descr_point(),
                               chosen_goal.get_descr_point())):
            if self.path is not None and len(self.path) < 5 and self.last_pose is not None:
                self.stuck_at_nav_goal_counter += 1
        else:
            self.stuck_at_nav_goal_counter = 0

        if self.stuck_at_nav_goal_counter > 3:
            self.blacklisted_goals.append(goal_voxel)
            if self.navigation_mode == "navigate":
                self._exit_navigate_mode(blacklist_goal=True)
                self.stuck_at_nav_goal_counter = 0
                return
            if isinstance(chosen_goal, FrontierCluster3D) and chosen_goal.is_stair_frontier:
                self._fake_frontier_cluster = chosen_goal
                self._blacklist_fake_frontier(goal_voxel)
                force_current_floor_check = False
            else:
                force_current_floor_check = False
                if isinstance(chosen_goal, FrontierCluster3D):
                    if not chosen_goal.is_stair_frontier:
                        added = self._blacklist_basic_frontier_region(chosen_goal)
                        print(
                            f"[Nav3D] Basic frontier local blacklist: "
                            f"goal={goal_voxel}, added={added}"
                        )
                self.frontier_clusters = [
                    c for c in self.frontier_clusters
                    if c.nav_voxel_id != goal_voxel
                ]
            self.rise_stair_candidates.pop(goal_voxel, None)
            self.nav_goals.pop(best_idx)
            if force_current_floor_check:
                self.nav_goals.clear()
            self.path = None
            if self.log and _RERUN:
                try:
                    rr.log("path_updates", rr.TextLog(f"Blacklisted unreachable goal {goal_voxel}"))
                except Exception:
                    pass
            self.stuck_at_nav_goal_counter = 0
            return

        self.last_nav_goal = chosen_goal
        if isinstance(chosen_goal, Cluster3D):
            self.navigation_mode = "navigate"
            self._navigate_goal = chosen_goal
        elif not isinstance(chosen_goal, StairSemanticFrontier3D):
            self.navigation_mode = "explore"
        if isinstance(chosen_goal, StairSemanticFrontier3D):
            self._stair_sem_last_replan_iter = self.one_map._iters

        if self.path and self.log and _RERUN:
            try:
                path_metric = self.one_map.path_3d_to_metric_2d(self.path)
                rr.log("map/path", rr.LineStrips2D(
                    path_metric[:, ::-1],
                    colors=np.repeat(np.array([[0, 0, 255]]), len(path_metric), axis=0)
                ))
                if isinstance(chosen_goal, StairSemanticFrontier3D):
                    goal_type = "StairSemFrontier"
                elif isinstance(chosen_goal, RiseStairFrontier3D):
                    goal_type = "RiseStairFrontier"
                elif isinstance(chosen_goal, Frontier3D):
                    goal_type = "Frontier"
                else:
                    goal_type = "POI-Cluster"
                rr.log("path_updates", rr.TextLog(
                    f"[{goal_type}] score={chosen_goal.get_score():.1f} "
                    f"path_len={len(self.path)} goal={goal_voxel}"
                ))
            except Exception:
                pass

    def _compute_path_to_object(self, agent_voxel: Tuple[int, int, int]) -> None:
        """Plan 3D A* path to an observation pose for the detected object."""
        if self.chosen_detection_3d is None:
            return
        goal_voxel = self._find_object_approach_voxel(self.chosen_detection_3d, agent_voxel)
        if goal_voxel is None:
            goal_voxel = self.one_map._find_nearest_traversable(
                self.chosen_detection_3d, max_radius=15
            )
        if goal_voxel is None:
            self.path = None
            self.objectnav_goal_voxel = None
            return
        self.objectnav_goal_voxel = goal_voxel
        self.path = self.one_map.a_star_3d(agent_voxel, goal_voxel, max_iterations=50000)
        self.path_id = 0
        self.is_goal_path = self.path is not None

        if self.path and self.log and _RERUN:
            try:
                path_metric = self.one_map.path_3d_to_metric_2d(self.path)
                rr.log("map/path", rr.LineStrips2D(
                    path_metric[:, ::-1],
                    colors=np.repeat(np.array([[0, 255, 0]]), len(path_metric), axis=0)
                ))
                rr.log("path_updates", rr.TextLog(
                    f"Path to {self.query_text[0]}: {len(self.path)} waypoints"
                ))
            except Exception:
                pass
