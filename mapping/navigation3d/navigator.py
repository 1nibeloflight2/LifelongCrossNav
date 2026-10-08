"""
Navigator3D: 3D voxel-based navigator for cross-floor multi-object navigation.

Heavily inspired by the original Navigator (mapping/navigator.py) but adapted for
3D voxel-hash storage.  Core improvements over the first draft:

Similarity-guided frontier scoring
    Each frontier voxel is scored by BFS into unknown space accumulating CLIP
    similarity values from adjacent OCCUPIED voxels — mirrors OneMap's C++
    compute_reachable_area_score().

POI exploitation channel
    High-similarity OCCUPIED voxels are clustered in 3D (region-growing on
    similarity_cache) and the cluster centres map to the nearest TRAVERSABLE voxel.
    POI goals are inserted alongside frontiers in the unified nav_goals list.

Unified scoring and frontier persistence
    Frontiers and POIs are both NavGoal subclasses with get_score().
    The current best nav_goal is re-evaluated before being replaced (persistence).
    Only switch if a new goal scores significantly better than the current one.

Cyclic detection and blacklist
    Same CyclicChecker pattern as the original Navigator.
    Goals that repeatedly fail (stuck_at_nav_goal_counter > threshold) are
    blacklisted for the episode.

Incremental frontier detection
    VoxelMap.new_traversable_voxels tracks voxels added this frame.
    Only those + their 6-neighbours are re-evaluated → O(new) not O(all).

Stair awareness
    STAIR-adjacent frontiers get a large bonus, encouraging cross-floor moves.
    STAIR voxels are traversable for A*, so cross-floor paths plan naturally.
"""

import time
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from config import Conf, SpotControllerConf
from mapping.voxel_map import VoxelMap, VoxelType, VoxelData
from mapping.nav_goals.frontier_3d import (
    Frontier3D,
    FrontierCluster3D,
    Cluster3D,
    StairSemanticFrontier3D,
    RiseStairFrontier3D,
    detect_frontiers_3d_incremental,
    detect_frontiers_3d_full,
    cluster_high_similarity_3d,
    rescore_frontier,
    check_fake_frontier_has_stair_below,
    create_stair_semantic_frontier,
    cluster_stair_voxels_without_expanded_bridges,
)
from mapping.nav_goals.navigation_goals import NavGoal
from mobile_sam import sam_model_registry, SamPredictor
from planning import Planning, Controllers
from vision_models.base_model import BaseModel
from mapping.navigation3d.frontier_manager import FrontierManagerMixin, rank_frontier_candidates
from mapping.navigation3d.poi_manager import POIManagerMixin, history_poi_enabled
from mapping.navigation3d.stair_proposals import StairProposalMixin, proposal_is_persistent
from mapping.navigation3d.stair_traversal import StairTraversalMixin, stair_mode_label
from mapping.navigation3d.objectnav import ObjectNavMixin
from mapping.navigation3d.path_manager import PathManagerMixin

try:
    import rerun as rr
    _RERUN = True
except ImportError:
    _RERUN = False


# ---------------------------------------------------------------------------
# Cyclic detection helpers (identical to original navigator.py)
# ---------------------------------------------------------------------------

class HistoricDetectData:
    def __init__(self, position: np.ndarray, action: str, other: Any = None):
        self.position = position
        self.other = other
        self.action = action

    def __hash__(self) -> int:
        return hash(f"{self.position}_{self.action}_{self.other}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HistoricDetectData):
            return NotImplemented
        return (np.array_equal(self.position, other.position) and
                self.action == other.action and self.other == other.other)


class CyclicDetectChecker:
    def __init__(self):
        self.history: Set[HistoricDetectData] = set()

    def check_cyclic(self, position: np.ndarray, action: str, other: Any = None) -> bool:
        return HistoricDetectData(position, action, other) in self.history

    def add_state_action(self, position: np.ndarray, action: str, other: Any = None) -> None:
        self.history.add(HistoricDetectData(position, action, other))


class HistoricData:
    def __init__(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None):
        self.position = position
        self.frontier_pt = frontier_pt
        self.other = other

    def __hash__(self) -> int:
        return hash(f"{self.position}_{self.frontier_pt}_{self.other}")


class CyclicChecker:
    def __init__(self):
        self.history: Set[HistoricData] = set()

    def check_cyclic(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> bool:
        return HistoricData(position, frontier_pt, other) in self.history

    def add_state_action(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> None:
        self.history.add(HistoricData(position, frontier_pt, other))


# ---------------------------------------------------------------------------
# Navigator3D
# ---------------------------------------------------------------------------

class Navigator3D(FrontierManagerMixin, POIManagerMixin, StairProposalMixin, StairTraversalMixin, ObjectNavMixin, PathManagerMixin):
    """
    3D voxel-based navigator.

    Public interface (compatible with MONActor / evaluator):
      - set_camera_matrix(K)
      - set_query(txt)
      - add_data(image, depth, odometry) -> bool
      - get_path() -> np.ndarray | str | None
      - get_map(return_map) -> np.ndarray | None
      - reset()

    Important internal state:
      - one_map: VoxelMap          — 3D voxel storage + similarity cache
      - frontier_clusters: List    — persistent FrontierCluster3D (TravExplorer-style)
      - nav_goals: List[NavGoal]   — sorted FrontierCluster3D + Cluster3D goals
      - last_nav_goal: NavGoal     — the goal followed last step (persistence)
      - blacklisted_goals: List    — voxel IDs that are permanently blacklisted
    """

    def __init__(
        self,
        model: BaseModel,
        target_detector: Any,
        stair_detector: Any,
        config: Conf,
    ) -> None:

        self.config = config
        self.model = model
        self.target_detector = target_detector
        self.stair_detector = stair_detector

        # SAM for detection mask refinement
        sam_model_t = "vit_t"
        sam_checkpoint = "weights/mobile_sam.pt"
        self.sam = sam_model_registry[sam_model_t](checkpoint=sam_checkpoint)
        self.sam.to(device="cuda")
        self.sam.eval()
        self.sam_predictor = SamPredictor(self.sam)

        # 3D Voxel Map
        self.one_map = VoxelMap(self.model.feature_dim, config.mapping, map_device="cuda")
        # Frontier detector is map-centric; attach the configured normalized
        # scoring parameters once so full/incremental paths share them.
        self.one_map.frontier_dirty_margin = int(
            getattr(config.planner, "frontier_dirty_margin", 5)
        )

        # Query
        self.query_text: List[str] = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to("cuda")

        # Frontier and POI state
        self.frontier_clusters: List[FrontierCluster3D] = []  # persistent frontier clusters (TravExplorer)
        self.nav_goals: List[NavGoal] = []                     # sorted FrontierCluster3D + Cluster3D
        self.blacklisted_goals: List[Tuple[int, int, int]] = []  # permanently removed nav targets
        self.artificial_obstacles: List[Tuple[int, int]] = []

        # Navigation state
        self.last_nav_goal: Optional[NavGoal] = None
        self.last_pose: Optional[Tuple[int, int, float]] = None
        self.last_executed_action: Optional[str] = None
        self.saw_left = False
        self.saw_right = False
        self.first_obs = True
        self.allow_poi = True
        self.navigation_mode = "explore"  # explore | navigate | objectnav
        self._navigate_goal: Optional[NavGoal] = None
        self._poi_confirm_scan_remaining = 0
        self._poi_confirm_scan_steps = 12
        self._poi_confirm_pitch_level = 0
        self._poi_confirm_pitch_angle_threshold_deg = 45.0
        self._poi_blacklisted_goals: Set[Tuple[int, int, int]] = set()
        self._history_poi_blacklisted_keys: Set[
            Tuple[
                Tuple[int, int, int],
                Tuple[int, int, int],
                Tuple[int, int, int],
            ]
        ] = set()
        self._poi_blacklist_radius_xy = max(1, round(0.8 / self.one_map.voxel_size))
        self._poi_blacklist_radius_z_m = 0.6
        self._history_poi_arrival_radius_xy = max(1, round(1.0 / self.one_map.voxel_size))
        self._history_poi_arrival_radius_xyz_m = 1.5
        self.history_pois: List[Cluster3D] = []
        self.live_pois: Dict[Tuple[int, int, int], Cluster3D] = {}
        self._history_poi_pending = False
        self._live_poi_ttl = 20
        self._poi_overlap_radius = 8
        self._poi_overlap_min_evidence = 0.08
        self._history_poi_objectnav_match_xy_m = float(
            getattr(config.planner, "history_poi_objectnav_match_xy_m", 1.0)
        )
        self._history_poi_objectnav_match_z_m = float(
            getattr(config.planner, "history_poi_objectnav_match_z_m", 1.0)
        )
        self._history_poi_provisional_detection: Optional[Dict[str, Any]] = None
        self._objectnav_miss_counter = 0
        self._objectnav_miss_limit = int(getattr(config.planner, "objectnav_miss_limit", 5))
        self._objectnav_edge_oscillation_found_cycles = int(
            getattr(config.planner, "objectnav_edge_oscillation_found_cycles", 5)
        )
        self._objectnav_edge_oscillation_active = False
        self._objectnav_edge_oscillation_last_yaw: Optional[float] = None
        self._objectnav_edge_oscillation_last_xy: Optional[Tuple[float, float]] = None
        self._objectnav_edge_oscillation_last_sign: Optional[int] = None
        self._objectnav_edge_oscillation_turns = 0
        self._objectnav_pitch_scan_levels = [0, 1, -1, -1, 1, 0]
        self._objectnav_pitch_scan_index = 0
        self._objectnav_pitch_scan_goal: Optional[Tuple[int, int, int]] = None
        self._objectnav_pitch_scan_active = False
        self._objectnav_correction_action: Optional[str] = None
        self._objectnav_align_action: Optional[str] = None
        self._objectnav_restore_action: Optional[str] = None
        self._objectnav_restore_pending = False
        self._objectnav_force_found_after_restore = False
        self._objectnav_cached_detection: Optional[Dict[str, Any]] = None
        self._enable_sed_found_confirmation = bool(
            getattr(config.planner, "enable_sed_found_confirmation", True)
        )
        self._sed_found_threshold = float(getattr(config.planner, "sed_found_threshold", 0.15))
        self._sed_reject_cooldown_frames = int(
            getattr(config.planner, "sed_reject_cooldown_frames", 50)
        )
        self._sed_reject_radius_xy_m = float(
            getattr(config.planner, "sed_reject_radius_xy_m", 0.5)
        )
        self._sed_reject_radius_z_m = float(
            getattr(config.planner, "sed_reject_radius_z_m", 0.6)
        )
        if (
            not np.isfinite(self._sed_found_threshold) or
            not -1.0 <= self._sed_found_threshold <= 1.0 or
            self._sed_reject_cooldown_frames < 1 or
            not np.isfinite(self._sed_reject_radius_xy_m) or
            not np.isfinite(self._sed_reject_radius_z_m) or
            self._sed_reject_radius_xy_m < 0 or self._sed_reject_radius_z_m < 0
        ):
            raise ValueError("Invalid SED found threshold/cooldown configuration")
        self._reset_sed_found_state()
        self._objectnav_return_goal: Optional[NavGoal] = None
        self._objectnav_source: Optional[str] = None
        self._no_goal_available = False
        self._frontier_full_check_z_range = 1
        self._force_current_floor_frontier_check = False

        # Episode-local known stair entrances. Used only after local exploration
        # is exhausted, and only as a fixed navigation target to a known entry.
        self.stair_frontier_memory: Dict[int, Dict[str, Any]] = {}
        self._stair_memory_next_id = 1
        self._active_stair_memory_id: Optional[int] = None

        # Stair mode state
        # Pitch control (Trajectory-Driven Gaze)
        # desired_pitch_level: target pitch for actor (positive=down, negative=up)
        # Actor moves one 30° step per frame toward this target.
        self.desired_pitch_level = 0          # 0=horizontal, 2=60° down, -2=60° up
        self.look_down_requested = False      # legacy compat flag

        # Stair detection confidence threshold
        # Stair thresholds by stage:
        # - stair_sem mode uses a stricter visual threshold for continuous SAM updates.
        # - pitched down-stair confirmation can use lower visual confidence because
        #   it is gated by the local fake anchor and the geometry verifier.
        self._stair_sam_threshold = float(getattr(config.planner, "stair_sam_threshold", 0.689))
        self._down_stair_pitched_conf_threshold = float(getattr(config.planner, "descend_pitched_conf_threshold", 0.48))
        self._down_stair_pitched_sam_threshold = float(getattr(config.planner, "descend_pitched_sam_threshold", 0.48))
        self._down_stair_geometry_score_threshold = float(getattr(config.planner, "descend_geometry_threshold", 0.65))
        self._fake_frontier_wait_limit = int(getattr(config.planner, "descend_wait_frames", 5))
        self._fake_frontier_wait_counter = 0  # pitched 后等待计数器
        # === Stair Semantic Mode State ===
        self.stair_sem_mode = False           # True when in stair semantic navigation mode
        self._stair_sem_frontier: Optional[StairSemanticFrontier3D] = None  # current semantic stair goal
        self._stair_sem_force_forward = 0     # frames of forced forward movement on arrival
        self._stair_sem_entry_z: Optional[int] = None  # agent z when entering stair sem (exit guard)
        self._stair_sem_entry_iter = 0        # map step when entering stair sem
        self._stair_sem_direction = "down"    # "down" for fake frontier, "up" for rise stair frontier
        # Normal exploration stair ordering.  None keeps the historical
        # basic -> descend -> ascent order.  A successful StairSem traversal
        # updates this episode-local preference so exploration continues in
        # the same vertical direction after the current floor's basic
        # frontiers are exhausted.  The post-upstairs probe has its own fixed
        # priority flow and does not use this flag.
        self._explore_stair_direction_preference: Optional[str] = None
        self._stair_sem_force_forward_steps = int(getattr(config.planner, "stair_force_forward_steps", 3))
        self._stair_sem_force_forward_steps_up = self._stair_sem_force_forward_steps
        self._stair_sem_min_descent = int(getattr(config.planner, "stair_min_descent_layers", 3))
        self._stair_sem_down_floor_confirm_frames = int(
            getattr(config.planner, "stair_descent_floor_confirm_frames", 3)
        )
        descent_guard_distance = float(
            getattr(config.planner, "stair_descent_frontier_guard_distance", 0.30)
        )
        self._stair_sem_down_frontier_guard_layers = max(
            1, round(descent_guard_distance / self.one_map.voxel_size_z)
        )
        self._stair_sem_down_floor_counter = 0
        self._stair_sem_up_min_steps = 6
        self._stair_sem_up_min_ascent = int(getattr(config.planner, "stair_min_ascent_layers", 6))
        self._stair_sem_up_floor_confirm_frames = int(getattr(config.planner, "stair_ascent_floor_confirm_frames", 3))
        self._stair_sem_up_floor_counter = 0
        self._stair_sem_up_continue_margin = 4
        self._stair_sem_locked_frontier: Optional[StairSemanticFrontier3D] = None
        # Lateral width corridor of the active descend flight.  This is reset
        # with each StairSem session so the next flight across a landing can
        # establish its own corridor.
        self._stair_sem_descend_corridor: Optional[Dict[str, Any]] = None
        self._stair_sem_xy_refine_delay_frames = int(
            getattr(config.planner, "stair_sem_xy_refine_delay_frames", 3)
        )
        self._stair_sem_xy_refined = False
        self._stair_sem_replan_period = int(getattr(config.planner, "stair_replan_period", 3))
        self._stair_sem_last_replan_iter = -999
        self._stair_sem_session_voxels: Set[Tuple[int, int, int]] = set()
        self._stair_sem_session_radius_m = float(
            getattr(config.planner, "stair_sem_session_radius_m", 2.5)
        )
        self._fake_stair_min_height_m = 0.649
        self._fake_stair_vertical_column_min_height_m = 0.55
        self._fake_stair_vertical_column_ratio = 0.30
        self._fake_stair_wall_max_xy_span_m = 0.35
        self._fake_stair_thin_wall_min_height_m = 0.45
        self._fake_stair_thin_wall_max_thickness_m = 0.20
        self._fake_stair_thin_wall_min_length_m = 0.50
        # Fake frontier confirmation state
        self._fake_frontier_target: Optional[Tuple[int, int, int]] = None  # nav_voxel of current fake frontier being confirmed
        self._fake_frontier_cluster: Optional[FrontierCluster3D] = None
        # The formal descend-cluster representative can be disconnected while
        # another anchor in that same cluster is reachable.  Keep the actual
        # A* anchor separate so look-down confirmation follows the point the
        # agent is really approaching instead of the refreshed representative.
        self._active_descend_fallback_anchor: Optional[Tuple[int, int, int]] = None
        self._active_descend_fallback_owner: Optional[Tuple[int, int, int]] = None
        self._fake_frontier_pitched = False   # True once pitch reached 60° at fake frontier
        self._recent_fake_frontier_confirmation: Optional[Dict[str, Any]] = None
        self._recent_fake_frontier_confirmation_ttl = 4
        self.blacklisted_fake_frontier_regions: List[Dict[str, Any]] = []
        self.down_stair_semantic_hints: List[Dict[str, Any]] = []
        self._down_stair_hint_ttl = 12
        self._down_stair_hint_threshold = 0.70
        self._down_stair_anchor_radius_xy = 12
        self._down_stair_confirmation_min_mid_layers = 1
        self._current_down_stair_anchor: Optional[Tuple[int, int, int]] = None
        self._upstairs_transition_cooldown = 0
        self._upstairs_transition_cooldown_steps = 20
        self._recent_down_stair_regions: List[Dict[str, Any]] = []
        self._recent_down_stair_region_ttl = 90
        self._recent_down_stair_radius_xy = 24
        # Upstairs stair proposal state (semantic proposal + depth geometry)
        self.rise_stair_candidates: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._rise_stair_proposal_threshold = float(getattr(config.planner, "ascent_proposal_threshold", 0.55))
        self._rise_stair_strong_threshold = float(getattr(config.planner, "ascent_strong_threshold", 0.80))
        self._rise_stair_view_hint_threshold = 0.80
        self._rise_stair_context_threshold = 0.80
        self._rise_stair_required_seen = int(getattr(config.planner, "ascent_required_seen", 2))
        self._rise_stair_candidate_ttl = int(getattr(config.planner, "ascent_candidate_ttl", 10))
        self._rise_stair_approach_distance_m = float(getattr(config.planner, "ascent_approach_distance", 0.60))
        self._rise_confirm_scan_steps = 4
        self._rise_confirm_scan_remaining = 0
        self._rise_confirm_scan_goal: Optional[Tuple[int, int, int]] = None
        self._rise_confirm_scan_turn = "L"
        # After upstairs semantic stair traversal, small landings get one
        # recenter+scan pass before falling back to normal exploration.
        self._post_upstairs_probe_state = "inactive"  # inactive | pre_scan | compute_center | recenter | scan
        self._post_upstairs_probe_target: Optional[Tuple[int, int, int]] = None
        self._post_upstairs_probe_scan_remaining = 0
        self._post_upstairs_probe_pending = False
        # Direction of the stair traversal that started the landing probe.
        # Keep the established field names for compatibility, but the probe
        # is shared by ascent and descent landings.
        self._post_stair_probe_direction: Optional[str] = None
        self._post_upstairs_probe_scan_steps = 12
        self._post_upstairs_probe_max_cluster_size = 180  # legacy; probe now gates by local 2D span
        self._post_upstairs_probe_max_span = 26
        self._post_upstairs_probe_local_radius_xy = 16
        self._post_upstairs_probe_min_center_cluster_size = 50
        self._post_upstairs_probe_settle_remaining = 0
        self._post_upstairs_probe_settle_attempts = 0
        self._post_upstairs_probe_max_settle_attempts = 1
        self._post_upstairs_prioritize_rise = False

        # Detection state
        self.object_detected = False
        self.chosen_detection: Optional[Tuple[int, int]] = None        # 2D px for evaluator
        self.chosen_detection_3d: Optional[Tuple[int, int, int]] = None  # 3D voxel
        self.objectnav_goal_voxel: Optional[Tuple[int, int, int]] = None  # snapped traversable objectnav target
        self.current_confirmed_detection_box: Optional[List[float]] = None
        self.current_confirmed_detection_score: Optional[float] = None
        self.current_confirmed_detection_class: Optional[str] = None
        self.current_confirmed_detection_voxel: Optional[Tuple[int, int, int]] = None
        self.current_confirmed_detection_box_area_ratio: Optional[float] = None
        self.current_confirmed_detection_center_offset: Optional[float] = None
        self.current_confirmed_detection_edge_margin_px: Optional[float] = None
        self.current_confirmed_detection_mask_voxels: Optional[int] = None
        self.last_detections: dict = {"target": {"boxes": [], "classes": [], "scores": []},
                                      "stair": {"boxes": [], "classes": [], "scores": []}}  # last frame detections

        # Path state
        self.is_goal_path = False
        self.path: Optional[List[Tuple[int, int, int]]] = None
        self.path_id = 0

        # Counters
        self.initializing = True
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0
        self._goal_missing_counter = 0  # grace period: frames since last_nav_goal disappeared from nav_goals
        self._goal_missing_grace = int(getattr(config.planner, "frontier_missing_grace", 3))

        # 前沿最小簇大小: reinit 期间放宽到2, 正常模式恢复5
        self._frontier_min_cluster_size = int(getattr(config.planner, "frontier_init_min_cluster_size", 2))
        self._frontier_z_range = 2          # 前沿检测 z 范围: agent_iz ± 2 层

        # Re-initialization request (set by stair sem exit, consumed by actor)
        self.request_reinit = False

        # Cyclic detection
        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()

        # Config-derived thresholds (in voxel units)
        cell_size = self.one_map.cell_size
        self.percentile_exploitation = config.planner.percentile_exploitation
        self.frontier_depth = round(config.planner.frontier_depth / cell_size)
        self.no_nav_radius = round(config.planner.no_nav_radius / cell_size)
        self.max_detect_distance = round(config.planner.max_detect_distance / cell_size)
        # Opt in to the OneMap-v1 stopping radius only for an explicitly
        # single-target evaluation.  Configurations without ``multi_object``
        # retain the established (multi-target/default) behavior.
        if getattr(config, "multi_object", None) is False:
            objectnav_found_distance_m = getattr(
                config.planner,
                "single_objectnav_found_distance",
                config.planner.max_detect_distance,
            )
        else:
            objectnav_found_distance_m = getattr(
                config.planner, "objectnav_found_distance", config.planner.max_detect_distance
            )
        self.objectnav_found_distance = round(objectnav_found_distance_m / cell_size)
        self.obstcl_kernel_size = round(config.planner.obstcl_kernel_size / cell_size)
        self.min_goal_dist = round(config.planner.min_goal_dist / cell_size)

        self.filter_detections_depth = config.planner.filter_detections_depth
        self.consensus_filtering = config.planner.consensus_filtering
        self.log = config.log_rerun
        self.allow_replan = config.planner.allow_replan
        self.use_frontiers = config.planner.use_frontiers
        self.allow_far_plan = config.planner.allow_far_plan
        self._objectnav_yaw_align_threshold = np.deg2rad(30.01)

        # Frontier scoring hyper-parameters
        self.frontier_sim_depth = self.frontier_depth   # BFS depth for similarity
        self.sim_weight = 30.0                          # similarity score weight
        self.enable_history_poi = bool(getattr(config.planner, "enable_history_poi", True))
        print(
            "[Navigator3D] History POI ablation: "
            f"{'enabled' if self.enable_history_poi else 'disabled'}"
        )
        self.persistence_tolerance = 0.05              # keep current goal if within 5% of best

        # Class name remapping to the closed COCO target labels.
        self.class_map = {
            "chair": "chair", "tv_monitor": "tv", "tv": "tv",
            "plant": "potted plant", "potted plant": "potted plant",
            "sofa": "couch", "couch": "couch", "bed": "bed", "toilet": "toilet",
        }
        self._closed_target_classes = {
            "chair", "toilet", "bed", "couch", "tv", "potted plant"
        }
        self._target_detector_class: Optional[str] = None
        self._target_detection_backend = "disabled"
        self._stair_classes = ["stairs", "staircase", "stairway"]
        self._obj_confidence = getattr(config.planner, 'obj_confidence', 0.8)
        self._object_found_min_confidence = max(
            float(self._obj_confidence),
            float(getattr(config.planner, "object_found_min_confidence", 0.7)),
        )
        self._object_found_min_box_area_ratio = float(
            getattr(config.planner, "object_found_min_box_area_ratio", 1.0 / 64.0)
        )
        self._object_found_min_mask_voxels = int(
            getattr(config.planner, "object_found_min_mask_voxels", 10)
        )
        self._object_found_edge_margin_px = float(
            getattr(config.planner, "object_found_edge_margin_px", 8.0)
        )
        self._object_found_max_center_offset = float(
            getattr(config.planner, "object_found_max_center_offset", 0.75)
        )

        self.is_spot = type(config.controller) == SpotControllerConf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:
        self.one_map.set_camera_matrix(camera_matrix)

    def _configure_target_detector(self) -> None:
        query = self.query_text[0] if self.query_text else ""
        if query in self._closed_target_classes:
            self._target_detector_class = query
            self._target_detection_backend = "dfine"
            if hasattr(self.target_detector, "set_classes"):
                self.target_detector.set_classes([query])
            return
        self._target_detector_class = None
        # The benchmark uses six closed COCO classes through D-FINE-X. GroundingDINO is
        # not instantiated while SegFormer supplies stair semantics.
        self._target_detection_backend = "disabled"

    def _detect_target_objects(self, image_rgb_hwc: np.ndarray) -> dict:
        if self._target_detection_backend == "dfine" and self._target_detector_class:
            detections = self.target_detector.detect(image_rgb_hwc)
        else:
            detections = {"boxes": [], "classes": [], "scores": []}
        if detections is None:
            detections = {}
        detections.setdefault("boxes", [])
        detections.setdefault("scores", [])
        detections.setdefault("classes", [])
        if len(detections["classes"]) < len(detections["boxes"]):
            detections["classes"] = [self.query_text[0]] * len(detections["boxes"])
        return detections

    @staticmethod
    def _xy_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Euclidean XY distance in voxel units for arrival/proximity checks."""
        return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))







    def set_query(self, txt: List[str], allow_poi: bool = True) -> None:
        self._reset_sed_found_state()
        self.allow_poi = allow_poi
        for i, t in enumerate(txt):
            if t in self.class_map:
                txt[i] = self.class_map[t]
        if txt == self.query_text:
            self.object_detected = False
            self.chosen_detection = None
            self.chosen_detection_3d = None
            self.objectnav_goal_voxel = None
            self._objectnav_return_goal = None
            self._objectnav_source = None
            self._history_poi_provisional_detection = None
            self._reset_current_confirmed_detection()
            self._objectnav_miss_counter = 0
            self._reset_objectnav_pitch_scan()
            self._reset_objectnav_correction(clear_cache=True)
            self.nav_goals.clear()
            self.path = None
            self.path_id = 0
            self.last_nav_goal = None
            self.navigation_mode = "explore"
            self._navigate_goal = None
            self.history_pois.clear()
            self.live_pois.clear()
            self._poi_blacklisted_goals.clear()
            self._history_poi_blacklisted_keys.clear()
            self._poi_confirm_scan_remaining = 0
            self._poi_confirm_pitch_level = 0
            self._history_poi_pending = bool(allow_poi and self.enable_history_poi)
            self.allow_poi = allow_poi
            return
        if txt != self.query_text:
            print(f"[Navigator3D] Setting query to {txt}")
            self.query_text = txt
            self.query_text_features = self.model.get_text_features(
                ["a " + self.query_text[0]]
            ).to("cuda")

            # Reset similarity cache — all voxels need recomputing
            self.one_map.similarity_cache.clear()
            self.one_map.updated_voxel_ids = {
                vid for vid, vd in self.one_map.voxels.items()
                if vd.voxel_type in (VoxelType.OCCUPIED, VoxelType.STAIR, VoxelType.STAIR_EXPANDED)
            }

            # Reset detection / frontier state for new query
            self.object_detected = False
            self.chosen_detection = None
            self.chosen_detection_3d = None
            self.objectnav_goal_voxel = None
            self._objectnav_return_goal = None
            self._objectnav_source = None
            self._history_poi_provisional_detection = None
            self._reset_current_confirmed_detection()
            self._objectnav_miss_counter = 0
            self._reset_objectnav_pitch_scan()
            self._reset_objectnav_correction(clear_cache=True)
            self.frontier_clusters.clear()
            self.nav_goals.clear()
            self.rise_stair_candidates.clear()
            self.blacklisted_fake_frontier_regions.clear()
            self.down_stair_semantic_hints.clear()
            self._recent_down_stair_regions.clear()
            self._current_down_stair_anchor = None
            self._reset_stair_sem_stability()
            self.stair_sem_mode = False
            self._stair_sem_frontier = None
            self._stair_sem_entry_z = None
            self._stair_sem_direction = "down"
            self._stair_sem_force_forward = 0
            self._stair_sem_session_voxels.clear()
            self._reset_fake_frontier_state()
            self._upstairs_transition_cooldown = 0
            self._reset_post_upstairs_probe()
            self._reset_rise_confirm_scan()
            self.path = None
            self.path_id = 0
            self.last_nav_goal = None
            self.navigation_mode = "explore"
            self._navigate_goal = None
            self._poi_confirm_scan_remaining = 0
            self._poi_confirm_pitch_level = 0
            self.history_pois.clear()
            self.live_pois.clear()
            self._poi_blacklisted_goals.clear()
            self._history_poi_blacklisted_keys.clear()
            self._history_poi_pending = bool(allow_poi and self.enable_history_poi)
            self._no_goal_available = False
            self._force_current_floor_frontier_check = False

            # Update closed-set target routing; stairs use SegFormer-B2.
            self._configure_target_detector()
            print(
                f"[Navigator3D] detector target prompt: {self.query_text[0]} "
                f"via {self._target_detection_backend}"
            )

    def reset(self) -> None:
        self._reset_sed_found_state()
        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to("cuda")
        self.object_detected = False
        self.chosen_detection = None
        self.chosen_detection_3d = None
        self.objectnav_goal_voxel = None
        self._objectnav_return_goal = None
        self._objectnav_source = None
        self._history_poi_provisional_detection = None
        self._reset_current_confirmed_detection()
        self._objectnav_miss_counter = 0
        self._reset_objectnav_pitch_scan()
        self._reset_objectnav_correction(clear_cache=True)
        self.last_detections = {"target": {"boxes": [], "classes": [], "scores": []},
                                "stair": {"boxes": [], "classes": [], "scores": []}}
        self.last_pose = None
        self.last_executed_action = None
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0
        self._goal_missing_counter = 0
        self.is_goal_path = False
        self.path = None
        self.path_id = 0
        self.initializing = True
        self._frontier_min_cluster_size = int(getattr(self.config.planner, "frontier_init_min_cluster_size", 2))
        self.first_obs = True
        self.allow_poi = True
        self.navigation_mode = "explore"
        self._navigate_goal = None
        self.history_pois.clear()
        self.live_pois.clear()
        self._poi_blacklisted_goals.clear()
        self._history_poi_blacklisted_keys.clear()
        self._history_poi_pending = False
        self._poi_confirm_scan_remaining = 0
        self._poi_confirm_pitch_level = 0
        self._poi_blacklisted_goals.clear()
        self._no_goal_available = False
        self._force_current_floor_frontier_check = False
        self.stair_frontier_memory.clear()
        self._stair_memory_next_id = 1
        self._active_stair_memory_id = None
        self.frontier_clusters.clear()
        self.nav_goals.clear()
        self.blacklisted_goals.clear()
        self.artificial_obstacles.clear()
        self.last_nav_goal = None
        self.saw_left = False
        self.saw_right = False
        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()
        self.desired_pitch_level = 0
        self.look_down_requested = False
        # Stair sem mode reset
        self.stair_sem_mode = False
        self.navigation_mode = "explore"
        self._navigate_goal = None
        self._stair_sem_frontier = None
        self._stair_sem_force_forward = 0
        self._stair_sem_entry_z = None
        self._stair_sem_entry_iter = 0
        self._stair_sem_direction = "down"
        self._explore_stair_direction_preference = None
        self._stair_sem_down_floor_counter = 0
        self._stair_sem_up_floor_counter = 0
        self._stair_sem_locked_frontier = None
        self._stair_sem_last_replan_iter = -999
        self._stair_sem_session_voxels.clear()
        self._fake_frontier_target = None
        self._fake_frontier_cluster = None
        self._active_descend_fallback_anchor = None
        self._active_descend_fallback_owner = None
        self._fake_frontier_pitched = False
        self._fake_frontier_wait_counter = 0
        self._current_down_stair_anchor = None
        self._upstairs_transition_cooldown = 0
        self._recent_down_stair_regions.clear()
        self.rise_stair_candidates.clear()
        self.blacklisted_fake_frontier_regions.clear()
        self.down_stair_semantic_hints.clear()
        self._reset_post_upstairs_probe()
        self._reset_rise_confirm_scan()
        self.request_reinit = False
        self.one_map.reset()

    def set_last_executed_action(self, action: Optional[str]) -> None:
        """Receive the discrete action actually issued on the previous frame."""
        self.last_executed_action = action

    def set_initializing(self, active: bool) -> None:
        """Synchronize mapper planning with Actor3D's initialization scan.

        Mapping and frontier extraction continue while ``active`` is true, but
        path commitment, arrival/stuck handling and terminal no-goal state are
        deferred until Actor3D has executed the complete camera scan.
        """
        active = bool(active)
        if self.initializing == active:
            if active:
                self._no_goal_available = False
            return

        self.initializing = active
        self._no_goal_available = False
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0
        self._goal_missing_counter = 0
        self.path = None
        self.last_nav_goal = None

        if active:
            self._frontier_min_cluster_size = int(
                getattr(self.config.planner, "frontier_init_min_cluster_size", 2)
            )
            return

        # Do not carry a relaxed initialization candidate into normal
        # navigation.  The first post-scan frame rebuilds the current floor
        # using the normal cluster threshold and the complete initialized map.
        self.nav_goals.clear()
        self._frontier_min_cluster_size = int(
            getattr(self.config.planner, "frontier_normal_min_cluster_size", 5)
        )
        self._force_current_floor_frontier_check = True





    def _exit_navigate_mode(self, blacklist_goal: bool = False) -> None:
        if blacklist_goal and self._navigate_goal is not None:
            if isinstance(self._navigate_goal, Cluster3D):
                self._blacklist_poi_cluster(self._navigate_goal)
            else:
                goal = tuple(int(v) for v in self._navigate_goal.get_descr_point())
                self.blacklisted_goals.append(goal)
        self.navigation_mode = "explore"
        self._navigate_goal = None
        self._poi_confirm_scan_remaining = 0
        self.path = None
        self.last_nav_goal = None


    def _update_objectnav_pitch_scan(
        self,
        agent_voxel: Tuple[int, int, int],
        object_confirmed_this_frame: bool,
    ) -> None:
        """Legacy fixed pitch scan is disabled; box-aware correction handles near targets."""
        self._reset_objectnav_pitch_scan()



    @staticmethod
    def _inverse_objectnav_action(action: str) -> Optional[str]:
        return {"L": "R", "R": "L", "U": "D", "D": "U"}.get(action)












    def _capture_objectnav_return_goal(self) -> None:
        if self._is_history_poi_navigate_active():
            self._objectnav_return_goal = self._navigate_goal
            self._objectnav_source = "history_poi"
        elif self.navigation_mode != "objectnav":
            self._objectnav_return_goal = None
            self._objectnav_source = None
















    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi




    def get_map(self, return_map: bool = True) -> Optional[np.ndarray]:
        """Return 2D similarity projection for visualization."""
        self.one_map.compute_similarity(self.query_text_features)
        if not return_map:
            return None
        sim_2d = np.zeros((self.one_map.n_cells, self.one_map.n_cells), dtype=np.float32)
        for vid, score in self.one_map.similarity_cache.items():
            ix, iy = vid[0], vid[1]
            if 0 <= ix < self.one_map.n_cells and 0 <= iy < self.one_map.n_cells:
                sim_2d[ix, iy] = max(sim_2d[ix, iy], score)
        return sim_2d[np.newaxis, ...]  # (1, H, W)

    # ------------------------------------------------------------------
    # Main per-frame entry
    # ------------------------------------------------------------------

    def add_data(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
    ) -> bool:
        """
        Per-frame update: process image/depth → update map → plan path.

        Returns True if the target object is found within detection distance.
        """
        odometry = odometry.astype(np.float32)
        x = odometry[0, 3]
        y = odometry[1, 3]
        yaw = np.arctan2(odometry[1, 0], odometry[0, 0])

        px, py = self.one_map.metric_to_px(x, y)
        # NOTE: agent_voxel computed later, after update(), so agent_height_0 is set
        agent_voxel = None  # placeholder

        # --- Collision/stuck detection ---
        # Only a failed forward command is evidence of a physical obstacle.
        # Look/turn actions deliberately keep the agent at the same XY position
        # and must never create artificial OCCUPIED voxels.
        if (
            self.last_executed_action == "move_forward"
            and self.last_pose is not None
            and self.path is not None
            and px == self.last_pose[0]
            and py == self.last_pose[1]
        ):
            self.stuck_at_cell_counter += 1
        else:
            self.stuck_at_cell_counter = 0
        if self.stuck_at_cell_counter > 5:
            dx_fwd = np.cos(yaw)
            dy_fwd = np.sin(yaw)
            self.artificial_obstacles.append((px + round(dx_fwd), py + round(dy_fwd)))

        if not self.one_map.camera_initialized:
            warnings.warn("[Navigator3D] Camera matrix not set")
            return False

        # --- Target and stair detections ---
        image_rgb_hwc = image.transpose(1, 2, 0)  # (H, W, 3)
        # Target objects use D-FINE-X for the six HM3D benchmark classes.
        target_detections = self._detect_target_objects(image_rgb_hwc)
        stair_dets = self.stair_detector.detect(image_rgb_hwc)

        # Save for rerun visualization
        self.last_detections = {
            "target": target_detections,
            "stair": stair_dets,
        }

        # --- CLIP feature extraction ---
        image_features = self.model.get_image_features(image[np.newaxis, ...]).squeeze(0)

        # --- Update VoxelMap (new voxels added → new_traversable_voxels updated) ---
        self.one_map.update(image_features, depth, odometry, self.artificial_obstacles)

        # Now agent_height_0 is guaranteed to be set — compute agent voxel at raycasting plane
        agent_voxel = self.one_map.get_agent_voxel(odometry)
        # Count the completed left/right cycle now, but do not return before
        # this frame's raw detector output has been converted into a confirmed
        # 3D detection.  Returning here used to produce called_found=True with
        # current_confirmed_detection_voxel=None (and therefore detect_dist=inf
        # in the evaluator), even when the current RGB clearly contained the
        # requested object.
        edge_oscillation_force_found = self._update_objectnav_edge_oscillation(
            float(x), float(y), float(yaw)
        )
        # New OCCUPIED voxels can appear after a STAIR voxel was already marked.
        # Re-assimilate immediate neighbors every frame so late wall/step hits do
        # not remain red obstacles next to confirmed stairs.
        if any(vd.voxel_type == VoxelType.STAIR for vd in self.one_map.voxels.values()):
            expanded_late = self._expand_stair_26_neighbors()
            if expanded_late and self.stair_sem_mode:
                self._stair_sem_session_voxels.update(expanded_late)
            demoted_late = self._demote_short_stair_clusters()
            if demoted_late and self.stair_sem_mode and self._valid_stair_sem_session_count() < 3:
                self._abort_stair_sem_as_fake_stair(
                    "session stair voxels demoted during late stair cleanup"
                )
        if self._upstairs_transition_cooldown > 0 and not self.stair_sem_mode:
            self._upstairs_transition_cooldown -= 1

        # --- Stair Detection State Machine ---
        # Three modes:
        #   1. Geometric (default): stair detections not trusted unless near fake frontier
        #   2. Fake frontier confirmation: agent at fake frontier, pitched 60° down, counting stair detections
        #   3. Stair sem mode: actively navigating stairs with SAM segmentation

        suppress_stair_semantics = self.navigation_mode == "navigate" and not self.stair_sem_mode

        if self.stair_sem_mode:
            # === STAIR SEM MODE: apply high-confidence SegFormer stair masks ===
            if stair_dets["boxes"]:
                high_conf_dets = {"boxes": [], "classes": [], "scores": [], "masks": []}
                stair_masks = stair_dets.get("masks", [])
                for det_index, (box, cls, score) in enumerate(zip(
                    stair_dets["boxes"], stair_dets["classes"], stair_dets["scores"]
                )):
                    if score >= self._stair_sam_threshold:
                        high_conf_dets["boxes"].append(box)
                        high_conf_dets["classes"].append(cls)
                        high_conf_dets["scores"].append(score)
                        if det_index < len(stair_masks):
                            high_conf_dets["masks"].append(stair_masks[det_index])
                if high_conf_dets["boxes"]:
                    marked = self._mark_stair_with_sam(high_conf_dets, image, depth, odometry)
                    self._stair_sem_session_voxels.update(marked)
                    # 26-neighbor expansion: assimilate adjacent OCCUPIED as STAIR
                    expanded = self._expand_stair_26_neighbors()
                    self._stair_sem_session_voxels.update(expanded)
                    demoted = self._demote_short_stair_clusters()
                    if demoted:
                        if self._valid_stair_sem_session_count() < 3:
                            self._abort_stair_sem_as_fake_stair(
                                "session stair voxels demoted during semantic mode"
                            )

            # Exit condition: agent's horizontal 8-neighbors have non-STAIR TRAVERSABLE
            if self.stair_sem_mode and agent_voxel is not None:
                _exit_stair_sem = self._post_upstairs_probe_pending or self._check_stair_sem_exit(agent_voxel)
                if _exit_stair_sem:
                    if self._stair_sem_direction == "up":
                        self._post_upstairs_probe_pending = False
                        print("[Nav3D] StairSem-ascent floor/landing detected; checking platform probe")
                        self._handle_upstairs_stair_sem_exit(agent_voxel)
                    else:
                        self._post_upstairs_probe_pending = False
                        print("[Nav3D] StairSem-descend floor/landing detected; checking opposite-side platform probe")
                        self._handle_downstairs_stair_sem_exit(agent_voxel)

            # Force semantic mapping mode while still in stair sem. Upstairs
            # post-probe may have just left stair_sem_mode in this branch.
            if self.stair_sem_mode:
                self.one_map.update_stair_mode(stair_dets, agent_voxel=agent_voxel, force_semantic=True)

        elif suppress_stair_semantics:
            self._reset_fake_frontier_state()
            self._reset_rise_confirm_scan()

        elif self._fake_frontier_pitched:
            # === FAKE FRONTIER CONFIRMATION: pitched 60° down ===
            # Low visual threshold is allowed only here; entry still requires
            # local fake-anchor consistency and down-stair geometry verification.
            if self._should_block_fake_frontier_confirmation(agent_voxel):
                self._reset_fake_frontier_state()
                self._update_rise_stair_candidates(stair_dets, depth, odometry, agent_voxel)
            else:
                max_conf = max(stair_dets["scores"]) if stair_dets.get("scores") else 0.0
                if max_conf >= self._down_stair_pitched_conf_threshold:
                    filtered_stair_dets = self._filter_down_stair_detections_for_anchor(
                        stair_dets, depth, odometry, agent_voxel
                    )
                    if not filtered_stair_dets.get("boxes"):
                        self._fake_frontier_wait_counter += 1
                        print(f"[Nav3D] Fake frontier {self._fake_frontier_target} stair conf={max_conf:.2f} "
                              f"failed local down-stair geometry check "
                              f"({self._fake_frontier_wait_counter}/{self._fake_frontier_wait_limit})")
                        if self._fake_frontier_wait_counter >= self._fake_frontier_wait_limit:
                            self._stash_recent_fake_frontier_confirmation()
                            self._blacklist_fake_frontier(self._fake_frontier_target)
                            self._reset_fake_frontier_state()
                        return
                    print(f"[Nav3D] === {stair_mode_label(self._stair_sem_direction, True)} === "
                          f"(pitched + stair detected conf={max_conf:.2f} >= {self._down_stair_pitched_conf_threshold} "
                          f"at fake frontier {self._fake_frontier_target})")
                    self._fake_frontier_wait_counter = 0
                    if self._fake_frontier_cluster is not None:
                        self._current_down_stair_anchor = self._anchor_center(self._fake_frontier_cluster)
                    else:
                        self._current_down_stair_anchor = self._fake_frontier_target
                    entered = self._enter_stair_sem_mode(
                        filtered_stair_dets, image, depth, odometry, agent_voxel, direction="down"
                    )
                    if not entered:
                        self._reset_fake_frontier_state()
                    return  # Skip further processing this frame
                else:
                    # conf 不足，计数器 +1
                    self._fake_frontier_wait_counter += 1
                    if self._fake_frontier_wait_counter >= self._fake_frontier_wait_limit:
                        # 连续 N 帧未确认楼梯 → 排除该 fake frontier，恢复几何模式
                        print(f"[Nav3D] Fake frontier {self._fake_frontier_target} rejected "
                              f"(no stair conf >= {self._down_stair_pitched_conf_threshold} in {self._fake_frontier_wait_limit} frames)")
                        self._stash_recent_fake_frontier_confirmation()
                        self._blacklist_fake_frontier(self._fake_frontier_target)
                        self._reset_fake_frontier_state()

        else:
            # === GEOMETRIC MODE: check if agent is near a fake frontier ===
            # Only trust stair detections in geometric mode if near a fake frontier
            # (prevents false positives from furniture being classified as stairs)
            if self._try_recent_fake_frontier_semantic_confirmation(
                stair_dets, image, depth, odometry, agent_voxel
            ):
                return
            self._update_down_stair_semantic_hints(stair_dets, depth, odometry, agent_voxel)
            if self._should_block_fake_frontier_tracking(agent_voxel):
                self._reset_fake_frontier_state()
            else:
                self._update_fake_frontier_tracking(agent_voxel)
            if self._fake_frontier_target is None:
                self._update_rise_stair_candidates(stair_dets, depth, odometry, agent_voxel)

        # Stair promotion/demotion happens after VoxelMap.update().  Flush its
        # local reverse dependencies before similarity/frontier selection so
        # support types and inflation are consistent in this same frame.
        self.one_map.refresh_dirty_derived()

        # --- Compute similarity (lazy, only updated_voxel_ids) ---
        updated_for_similarity = set(self.one_map.updated_voxel_ids)
        history_poi_pending = self._history_poi_pending
        self.one_map.compute_similarity(self.query_text_features)
        if history_poi_pending:
            self._generate_history_pois(agent_voxel)
            self._expire_live_pois()
        else:
            self._update_feature_live_pois(agent_voxel, updated_for_similarity)

        # --- First observation: do full frontier detection ---
        if self.first_obs:
            self.first_obs = False
            self.frontier_clusters = detect_frontiers_3d_full(
                self.one_map, agent_voxel,
                # The first-frame map already contains only observed sparse
                # voxels.  Do not discard valid observed frontiers merely
                # because their depth/raycast endpoint is over 6 m away.
                search_radius=None,
                frontier_sim_depth=self.frontier_sim_depth,
                sim_weight=self.sim_weight,
                min_cluster_size=self._frontier_min_cluster_size,
                z_range=self._frontier_z_range,
                max_clusters=None,
            )
            print(f"[Nav3D] FIRST full frontier detection: {len(self.frontier_clusters)} clusters found "
                  f"(min_size={self._frontier_min_cluster_size})")
        else:
            # --- Incremental frontier update (AABB-based) ---
            n_new = len(self.one_map.new_traversable_voxels)
            self.frontier_clusters = detect_frontiers_3d_incremental(
                self.one_map,
                agent_voxel,
                self.frontier_clusters,
                frontier_sim_depth=self.frontier_sim_depth,
                sim_weight=self.sim_weight,
                min_cluster_size=self._frontier_min_cluster_size,
                z_range=self._frontier_z_range,
                max_clusters=None,
            )
            if n_new > 0 or len(self.frontier_clusters) == 0:
                print(f"[Nav3D] Incremental: new_trav={n_new}, clusters={len(self.frontier_clusters)}")
        # Clear incremental tracking set for next frame
        self.one_map.clear_new_traversable()

        # --- Object detection processing ---
        self.saw_left = False
        self.saw_right = False
        object_confirmed_this_frame = False
        self._reset_current_confirmed_detection()
        objectnav_was_active = self.navigation_mode == "objectnav"
        active_history_poi = (
            self._navigate_goal
            if self._is_history_poi_navigate_active() else None
        )

        if target_detections["boxes"]:
            self.sam_predictor.set_image(image_rgb_hwc)
            obj_found = self._process_target_detections(
                target_detections, image, depth, odometry, yaw, px, py, agent_voxel,
                dense_features=image_features,
            )
            if obj_found and self.object_detected:
                box_area = float(self.current_confirmed_detection_box_area_ratio or 0.0)
                small_box = box_area < self._object_found_min_box_area_ratio
                if small_box and not objectnav_was_active:
                    # Keep a small detection as a POI hint.  If navigation is
                    # already owned by a history/live POI, preserve that goal;
                    # otherwise the live POI inserted during detection
                    # processing will be selected by the normal POI ranking.
                    print(
                        f"[Nav3D] Small detection kept as POI hint: "
                        f"area={box_area:.4f}<min={self._object_found_min_box_area_ratio:.4f}"
                    )
                    self.object_detected = False
                    self.chosen_detection = None
                    self.chosen_detection_3d = None
                    self.objectnav_goal_voxel = None
                    self._objectnav_miss_counter = 0
                    self._reset_objectnav_correction(clear_cache=True)
                elif isinstance(active_history_poi, Cluster3D):
                    det_voxel = self.current_confirmed_detection_voxel
                    if det_voxel is None:
                        det_voxel = self.chosen_detection_3d
                    matches_history = False
                    xy_distance_m = float("inf")
                    z_distance_m = float("inf")
                    if det_voxel is not None:
                        matches_history, xy_distance_m, z_distance_m = (
                            self._detection_matches_history_poi(det_voxel, active_history_poi)
                        )
                    if not matches_history:
                        self._history_poi_provisional_detection = {
                            "voxel": tuple(int(v) for v in det_voxel) if det_voxel is not None else None,
                            "score": self.current_confirmed_detection_score,
                            "box": (
                                list(self.current_confirmed_detection_box)
                                if self.current_confirmed_detection_box is not None else None
                            ),
                            "history_poi": tuple(int(v) for v in active_history_poi.center_voxel),
                            "xy_distance_m": xy_distance_m,
                            "z_distance_m": z_distance_m,
                            "iter": int(self.one_map._iters),
                        }
                        print(
                            f"[Nav3D] Detection kept provisional during history POI navigation: "
                            f"det={det_voxel}, history={active_history_poi.center_voxel}, "
                            f"bbox_xy={xy_distance_m:.2f}m, bbox_z={z_distance_m:.2f}m, "
                            f"limits=({self._history_poi_objectnav_match_xy_m:.2f}m, "
                            f"{self._history_poi_objectnav_match_z_m:.2f}m); "
                            f"continuing history goal={active_history_poi.nav_voxel_id}"
                        )
                        self.object_detected = False
                        self.chosen_detection = None
                        self.chosen_detection_3d = None
                        self.objectnav_goal_voxel = None
                        self._reset_current_confirmed_detection()
                        self._objectnav_miss_counter = 0
                    else:
                        self._history_poi_provisional_detection = None
                        object_confirmed_this_frame = True
                        self._objectnav_miss_counter = 0
                        self._capture_objectnav_return_goal()
                        self.navigation_mode = "objectnav"
                        self._navigate_goal = None
                        self._poi_confirm_scan_remaining = 0
                        self._compute_path_to_object(agent_voxel)
                        if not self.path:
                            self._clear_object_detection("confirmed target has no path")
                else:
                    self._history_poi_provisional_detection = None
                    object_confirmed_this_frame = True
                    self._objectnav_miss_counter = 0
                    self._capture_objectnav_return_goal()
                    self.navigation_mode = "objectnav"
                    self._navigate_goal = None
                    self._poi_confirm_scan_remaining = 0
                    self._compute_path_to_object(agent_voxel)
                    if not self.path:
                        self._clear_object_detection("confirmed target has no path")
            elif self.navigation_mode == "objectnav" or self.object_detected:
                self._objectnav_miss_counter += 1
        elif self.navigation_mode == "objectnav" or self.object_detected:
            self._objectnav_miss_counter += 1
        else:
            self.chosen_detection = None
            self.chosen_detection_3d = None

        if object_confirmed_this_frame and (
            self._objectnav_correction_action is not None or
            self._objectnav_restore_pending or
            self._objectnav_force_found_after_restore
        ):
            # During box-aware correction, keep the latest confirmed frame.
            # If the correction view sees the target clearly but the restore
            # view loses it, found uses this most recent valid frame instead
            # of the older frame that originally queued the correction.
            self._cache_current_confirmed_detection(image_rgb_hwc)

        if edge_oscillation_force_found:
            cycles = self._objectnav_edge_oscillation_found_cycles
            confirmation_source = None
            if (
                object_confirmed_this_frame and
                self.current_confirmed_detection_voxel is not None
            ):
                # Preserve the exact frame/box/3D point that caused the final
                # alternating turn.  The evaluator consumes the current
                # confirmed voxel after add_data() returns.
                self._cache_current_confirmed_detection(image_rgb_hwc)
                confirmation_source = "current-frame confirmed detection"
            elif self._restore_cached_confirmed_detection(agent_voxel):
                # A turn can move an edge box completely outside the image.
                # In that case reuse the most recent confirmed edge view rather
                # than emitting found without any spatial detection evidence.
                confirmation_source = "cached confirmed detection"

            if confirmation_source is not None:
                print(
                    f"[Nav3D] ObjectNav edge oscillation reached {cycles} left/right cycles; "
                    f"forcing found with {confirmation_source} and using the normal "
                    "evaluation classification"
                )
                # Keep the cache alive until the caller has saved the found
                # frame and evaluated current_confirmed_detection_voxel.
                return self._finalize_object_found(
                    agent_voxel, "edge_oscillation", clear_cache=False
                )

            print(
                f"[Nav3D] ObjectNav edge oscillation reached {cycles} left/right cycles, "
                "but no confirmed detection is available; continuing ObjectNav"
            )
            self._reset_objectnav_edge_oscillation()

        self._update_objectnav_pitch_scan(agent_voxel, object_confirmed_this_frame)

        if self._objectnav_force_found_after_restore:
            if object_confirmed_this_frame:
                quality_ok, quality_reason = self._object_detection_quality_ok_for_found()
                if quality_ok:
                    print(
                        f"[Nav3D] Object found quality OK after view restore: {quality_reason}, "
                        f"agent_to_chosen={self._agent_to_chosen_distance_m(agent_voxel):.3f}m"
                    )
                    return self._finalize_object_found(agent_voxel, "current_view_restore")
            restored_found = self._maybe_force_found_after_objectnav_restore(agent_voxel)
            if restored_found is not None:
                return restored_found

        if (self.navigation_mode == "objectnav" or self.object_detected) and not object_confirmed_this_frame:
            if self._objectnav_correction_action or self._objectnav_restore_pending:
                self._objectnav_miss_counter = 0
            elif self._objectnav_miss_counter >= self._objectnav_miss_limit:
                self._clear_object_detection(
                    f"target not confirmed for {self._objectnav_miss_counter} frames"
                )

        # --- Arrival check: invalidate reached/stale frontiers immediately ---
        self._check_frontier_arrival(agent_voxel, stair_dets, image, depth, odometry)

        # --- Frontier + POI goal list ---
        self._compute_frontiers_and_POIs(agent_voxel)

        # --- Path planning ---
        objectnav_dist_voxels = None
        if self.object_detected and self.chosen_detection_3d is not None and not self.stair_sem_mode:
            self.navigation_mode = "objectnav"
            objectnav_dist_voxels = np.sqrt(
                (agent_voxel[0] - self.chosen_detection_3d[0]) ** 2 +
                (agent_voxel[1] - self.chosen_detection_3d[1]) ** 2
            )
            if (object_confirmed_this_frame and
                    objectnav_dist_voxels <= self.objectnav_found_distance):
                quality_ok, quality_reason = self._object_detection_quality_ok_for_found()
                if quality_ok:
                    if self._queue_objectnav_yaw_alignment(odometry, yaw):
                        return False
                    print(
                        f"[Nav3D] Object found quality OK: {quality_reason}, "
                        f"agent_to_chosen={objectnav_dist_voxels * self.one_map.cell_size:.3f}m"
                    )
                    return self._finalize_object_found(agent_voxel, "normal")
                print(f"[Nav3D] Object found delayed: {quality_reason}")
                if quality_reason.startswith("edge_margin="):
                    self._cache_current_confirmed_detection(image_rgb_hwc)
                    self._arm_objectnav_edge_oscillation(float(x), float(y), float(yaw))
                else:
                    self._reset_objectnav_edge_oscillation()
                if not self._queue_objectnav_view_correction(quality_reason, image_rgb_hwc):
                    self._queue_objectnav_yaw_alignment(odometry, yaw)
                # Once the target is confirmed and within found distance, do
                # not let path following insert an opposite turn. The remaining
                # work is visual alignment / fallback-found, not navigation.
                self.path = None
                return False

        if self.allow_replan or not self.path:
            self._compute_best_path(agent_voxel)

        # Frontier refresh only builds the candidate list; _compute_best_path()
        # is what finally resolves committed-goal persistence and any reachable
        # descend anchor fallback.  Validate/look down against that final owner,
        # not against nav_goals[0] before planning has made its choice.
        if not self.stair_sem_mode:
            if self._should_block_fake_frontier_tracking(agent_voxel):
                self._reset_fake_frontier_state()
            else:
                self._update_fake_frontier_tracking(agent_voxel)
            self._clear_unselected_fake_frontier_state()

        if (object_confirmed_this_frame and self.object_detected and self.path and
                len(self.path) < 3 and not self.stair_sem_mode and
                objectnav_dist_voxels is not None and
                objectnav_dist_voxels <= self.objectnav_found_distance):
            quality_ok, quality_reason = self._object_detection_quality_ok_for_found()
            if quality_ok:
                if self._queue_objectnav_yaw_alignment(odometry, yaw):
                    return False
                print(
                    f"[Nav3D] Object found quality OK: {quality_reason}, "
                    f"agent_to_chosen={objectnav_dist_voxels * self.one_map.cell_size:.3f}m"
                )
                return self._finalize_object_found(agent_voxel, "short_path")
            print(f"[Nav3D] Object found delayed: {quality_reason}")
            if quality_reason.startswith("edge_margin="):
                self._cache_current_confirmed_detection(image_rgb_hwc)
                self._arm_objectnav_edge_oscillation(float(x), float(y), float(yaw))
            else:
                self._reset_objectnav_edge_oscillation()
            if not self._queue_objectnav_view_correction(quality_reason, image_rgb_hwc):
                self._queue_objectnav_yaw_alignment(odometry, yaw)
            self.path = None
            return False

        # --- Trajectory-Driven Gaze: update desired pitch from path geometry ---
        self._update_desired_pitch(agent_voxel)

        self.last_pose = (px, py, yaw)
        return False

    # ------------------------------------------------------------------
    # Trajectory-Driven Gaze: pitch control from path geometry
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Arrival check — immediately discard frontiers that are no longer valid
    # ------------------------------------------------------------------


        # Check 3: If this is a stair (fake) frontier, the fake frontier tracking
        # in _update_fake_frontier_tracking will handle it via proximity detection.
        # The pitch will be set to 60° down, and confirmation window will start.
        # No need to invalidate the goal here — agent stays near it during confirmation.

    def _invalidate_current_goal(self, reason: str) -> None:
        """Remove the current frontier goal and force re-plan."""
        if self.last_nav_goal is not None and isinstance(self.last_nav_goal, FrontierCluster3D):
            nav_vid = self.last_nav_goal.nav_voxel_id
            self.frontier_clusters = [
                c for c in self.frontier_clusters
                if c.nav_voxel_id != nav_vid
            ]
        elif self.last_nav_goal is not None and isinstance(self.last_nav_goal, RiseStairFrontier3D):
            self.rise_stair_candidates.pop(self.last_nav_goal.nav_voxel_id, None)
        self.last_nav_goal = None
        self.path = None
        self.stuck_at_nav_goal_counter = 0
        self._goal_missing_counter = 0
        if self.log and _RERUN:
            try:
                rr.log("path_updates", rr.TextLog(
                    f"Frontier invalidated on arrival: {reason}"
                ))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Frontier + POI goal computation (mirrors OneMap's compute_frontiers_and_POIs)
    # ------------------------------------------------------------------














    # ------------------------------------------------------------------
    # Best path computation (mirrors OneMap's compute_best_path)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Path to detected object
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Post-upstairs probe helpers
    # ------------------------------------------------------------------












    # ------------------------------------------------------------------
    # Rise stair candidate helpers
    # ------------------------------------------------------------------







    def _find_floor_nav_voxel_near_metric(
        self,
        xy_metric: Tuple[float, float],
        agent_voxel: Tuple[int, int, int],
        radius: int = 8,
    ) -> Optional[Tuple[int, int, int]]:
        """Find a current-floor non-inflated TRAVERSABLE voxel near a metric xy point."""
        cx = int(np.floor(xy_metric[0] / self.one_map.voxel_size)) + self.one_map.map_center
        cy = int(np.floor(xy_metric[1] / self.one_map.voxel_size)) + self.one_map.map_center
        iz = agent_voxel[2]
        best = None
        best_cost = float("inf")
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                vid = (cx + dx, cy + dy, iz)
                vd = self.one_map.voxels.get(vid)
                if vd is None or vd.voxel_type != VoxelType.TRAVERSABLE or vd.inflated:
                    continue
                cost = self._xy_distance(vid, agent_voxel)
                cost += 0.5 * (dx * dx + dy * dy)
                if cost < best_cost:
                    best_cost = cost
                    best = vid
        return best








    # ------------------------------------------------------------------
    # Stair Semantic Mode helpers
    # ------------------------------------------------------------------


    def _count_floor_traversable_neighbors(self, agent_voxel: Tuple[int, int, int]) -> int:
        """Count real-floor TRAVERSABLE neighbours on the current z layer.

        ``TRAVERSABLE`` can be supported by either OCCUPIED or FAKE_STAIR in
        the global map.  Only the former is valid evidence that StairSem has
        reached a real landing.  Mirror VoxelMap's nearest-support scan here so
        air above a demoted fake stair cannot terminate ascent or descent.
        """
        voxels = self.one_map.voxels
        ix, iy, iz = agent_voxel
        count = 0

        def has_occupied_floor_support(vid: Tuple[int, int, int]) -> bool:
            for dz in range(1, self.one_map.support_check_depth + 1):
                below_vd = voxels.get((vid[0], vid[1], vid[2] - dz))
                if below_vd is None:
                    continue
                if below_vd.voxel_type == VoxelType.OCCUPIED:
                    return True
                if below_vd.voxel_type in (
                    VoxelType.FAKE_STAIR,
                    VoxelType.STAIR,
                    VoxelType.STAIR_EXPANDED,
                ):
                    return False
            return False

        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            neighbor = (ix + dx, iy + dy, iz)
            vd = voxels.get(neighbor)
            if (
                vd is not None and
                vd.voxel_type == VoxelType.TRAVERSABLE and
                not vd.inflated and
                has_occupied_floor_support(neighbor)
            ):
                count += 1
        return count







    def _anchor_center(self, cluster: FrontierCluster3D) -> Tuple[int, int, int]:
        anchor = cluster.fake_triggered_voxels or cluster.cluster_voxels
        if not anchor:
            return cluster.nav_voxel_id
        return (
            int(round(np.mean([v[0] for v in anchor]))),
            int(round(np.mean([v[1] for v in anchor]))),
            int(round(np.mean([v[2] for v in anchor]))),
        )



    def _box_depth_voxels(
        self,
        box: np.ndarray,
        depth: np.ndarray,
        odometry: np.ndarray,
        max_points: int = 2500,
    ) -> List[Tuple[int, int, int]]:
        if self.one_map.fx is None or self.one_map.agent_height_0 is None:
            return []
        H, W = depth.shape
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
        y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            return []

        ys, xs = np.mgrid[y1:y2 + 1, x1:x2 + 1]
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        depths = depth[ys, xs]
        valid = np.isfinite(depths) & (depths > 0.05)
        if valid.sum() < 20:
            return []
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        depths = depths[valid].astype(np.float32)
        if xs.size > max_points:
            idx = np.linspace(0, xs.size - 1, max_points).astype(np.int64)
            xs = xs[idx]
            ys = ys[idx]
            depths = depths[idx]

        x_cam = (xs - self.one_map.cx) * depths / self.one_map.fx
        y_cam = (ys - self.one_map.cy) * depths / self.one_map.fy
        pts_cam = np.stack([depths, -x_cam, -y_cam], axis=-1)
        R_mat = odometry[:3, :3]
        t_vec = odometry[:3, 3]
        pts_world = (R_mat @ pts_cam.T).T + t_vec
        return [self.one_map.metric_to_voxel(float(pt[0]), float(pt[1]), float(pt[2])) for pt in pts_world]
































    # ------------------------------------------------------------------
    # Detection filtering and processing
    # ------------------------------------------------------------------






# Alias for backward compatibility
Navigator = Navigator3D
