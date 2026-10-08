"""
VoxelMap: 3D Hash Voxel Map for cross-floor multi-object navigation.

Pure 3D sparse voxel storage. No 2D grid maps.
- OCCUPIED voxels: depth back-projection, carry CLIP features
- TRAVERSABLE voxels: raycasting from agent to obstacles (air with support below)
- STAIR voxels: YOLO-detected stairs, treated as traversable for path planning

Path planning: 3D A* on TRAVERSABLE + STAIR voxels.
Target finding: similarity query on OCCUPIED voxels, then map to nearest TRAVERSABLE.
"""

from __future__ import annotations

import heapq
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

from config import MappingConf


# ---------------------------------------------------------------------------
# Voxel types
# ---------------------------------------------------------------------------

class VoxelType(Enum):
    OCCUPIED          = 1   # 深度点云障碍物，携带CLIP特征
    TRAVERSABLE       = 2   # 地板可通行：下方有OCCUPIED支撑
    TRAVERSABLE_STAIR = 3   # 楼梯可通行：下方有STAIR/STAIR_EXPANDED支撑
    TRAVERSABLE_FAKE  = 4   # 无支撑悬空：标记落差，不可通行
    STAIR             = 5   # SAM确认的楼梯体素
    STAIR_EXPANDED    = 6   # 楼梯邻域扩展体素
    FAKE_STAIR        = 7   # 楼梯误检体素：A*可经过，但不可作为目标/前沿


# ---------------------------------------------------------------------------
# Fast lookup sets
# ---------------------------------------------------------------------------

# A*可通行集合（排除 TRAVERSABLE_FAKE）
PATHABLE_TYPES = frozenset({
    VoxelType.TRAVERSABLE,
    VoxelType.TRAVERSABLE_STAIR,
    VoxelType.STAIR,
    VoxelType.STAIR_EXPANDED,
    VoxelType.FAKE_STAIR,
})

# 导航目标可落点：允许真实楼梯/楼梯可通行层，但不允许 FAKE_STAIR
TARGETABLE_TYPES = PATHABLE_TYPES - frozenset({VoxelType.FAKE_STAIR})

# 前沿候选类型（仅地板可通行）
FRONTIER_ELIGIBLE = frozenset({VoxelType.TRAVERSABLE})

# 普通障碍类型：阻挡射线；FAKE_STAIR 同时可供 A* 通行，但不生成前沿
OBSTACLE_TYPES = frozenset({VoxelType.OCCUPIED, VoxelType.FAKE_STAIR})

# 支撑提供者
SUPPORT_FLOOR = frozenset({VoxelType.OCCUPIED, VoxelType.FAKE_STAIR})
SUPPORT_STAIR = frozenset({VoxelType.STAIR, VoxelType.STAIR_EXPANDED})

# 真实楼梯语义类型（可通行、可生成楼梯语义前沿）
LOCKED_TYPES = frozenset({VoxelType.STAIR, VoxelType.STAIR_EXPANDED})

# Head-clearance blockers for pathable voxels. STAIR remains pathable when used
# as the floor/step surface, but it blocks navigation if it sits overhead.
HEAD_BLOCKING_TYPES = OBSTACLE_TYPES | LOCKED_TYPES

# 不可被深度覆盖的固定类型
IMMUTABLE_TYPES = frozenset({
    VoxelType.STAIR,
    VoxelType.STAIR_EXPANDED,
    VoxelType.FAKE_STAIR,
})


def is_occluded_unknown_column(
    neighbor: Tuple[int, int, int],
    voxels: Dict[Tuple[int, int, int], "VoxelData"],
    occlusion_layers: int,
) -> bool:
    """Return whether a missing ray-layer voxel is hidden by a higher blocker."""
    if neighbor in voxels:
        return False
    return any(
        (
            (column_vd := voxels.get((neighbor[0], neighbor[1], neighbor[2] + dz)))
            is not None
            and column_vd.voxel_type in HEAD_BLOCKING_TYPES
        )
        for dz in range(1, max(0, int(occlusion_layers)) + 1)
    )


# ---------------------------------------------------------------------------
# Per-voxel data
# ---------------------------------------------------------------------------

@dataclass
class VoxelData:
    voxel_type: VoxelType
    feature: np.ndarray          # shape (feature_dim,) for OCCUPIED; zeros for TRAVERSABLE
    confidence: float = 0.0
    obstacle_likelihood: float = 0.0
    inflated: bool = False       # True if within inflation_radius of an OCCUPIED voxel
                                 # Used for frontier detection (counts as explored) but
                                 # excluded from A* path planning (too close to wall)
                                 # 这是唯一的派生属性，按 dirty 反向依赖局部重算

    def merge(self, new_feat: np.ndarray, new_conf: float) -> None:
        """Cumulative Weighted Average using accumulated observation confidence."""
        total = self.confidence + new_conf
        if total < 1e-9:
            return
        w_old = self.confidence / total
        w_new = new_conf / total
        self.feature = self.feature * w_old + new_feat * w_new
        self.confidence = total


# ---------------------------------------------------------------------------
# VoxelMap
# ---------------------------------------------------------------------------

class VoxelMap:
    """
    3D hash voxel map for cross-floor navigation.

    Core storage: self.voxels: Dict[(ix, iy, iz), VoxelData]

    Public API:
      - set_camera_matrix(K)
      - update(values, depth, tf, artificial_obstacles)
      - compute_similarity(query_features) -> Dict[voxel_id, score]
      - find_nav_target(query_features) -> Optional[voxel_id]
      - a_star_3d(start, goal) -> Optional[List[voxel_id]]
      - get_traversable_set() -> Set[voxel_id]
      - mark_stair_voxels(detections, depth, tf)
      - update_stair_mode(detections)
      - metric_to_px(x, y), px_to_metric(px, py)
      - voxel_to_metric(ix, iy, iz) -> (x, y, z)
      - metric_to_voxel(x, y, z) -> (ix, iy, iz)
      - get_agent_voxel(tf) -> (ix, iy, iz)
      - mark_voxels_changed(voxel_ids)
      - refresh_dirty_derived()
      - reset()
    """

    def __init__(
        self,
        feature_dim: int,
        config: MappingConf,
        map_device: str = "cuda",
    ) -> None:

        # ---- Map geometry ----
        self.n_cells       = config.n_points
        self.size          = config.size
        self.cell_size     = self.size / self.n_cells
        self.feature_dim   = feature_dim
        self.map_device    = map_device

        # ---- 3D voxel parameters ----
        self.voxel_size    = self.cell_size          # xy resolution默认0.1m
        self.voxel_size_z  = self.cell_size                    # z resolution (# xy resolution cm，默认与xy相同，后续可调节)
        # At the default 0.1 m resolution this is about +/-30 m around
        # z_origin, i.e. roughly 60 m total indexed height.
        self.max_z_voxels  = 300
        self.z_origin      = None                    # agent birth z (set on first update)

        # Center offset for xy (voxel ix=n_cells//2 corresponds to world x=0)
        self.map_center    = self.n_cells // 2

        # ---- Agent / obstacle params ----
        self.agent_radius          = config.agent_radius
        self.obstacle_min          = config.obstacle_min
        self.obstacle_max          = config.obstacle_max
        self.inflation_radius_voxels = max(1, round(self.agent_radius / self.voxel_size))

        # ---- Observation scoring params ----
        self.gradient_factor         = config.gradient_factor
        self.optimal_object_distance = config.optimal_object_distance
        self.optimal_object_factor   = config.optimal_object_factor

        # ---- Traversability params ----
        self.support_check_depth = max(1, round(0.4 / self.voxel_size_z))  # 0.4m physical depth, auto-adapts to voxel_size_z

        # ---- Camera / agent geometry ----
        # Habitat: camera is at y=+0.88m relative to agent origin (foot/ground).
        # Our world frame uses camera as origin (0,0,0), so agent foot is at z=-0.88.
        self.camera_height = 0.88       # camera above agent origin (m)
        # Raycasting slice at 1/3 agent height above foot (inside agent cylinder)
        self.raycast_slice_z = -self.camera_height * (2.0 / 3.0)  # -0.587m
        # Ordinary-frontier occlusion uses the same vertical interval as the
        # 2D obstacle-column aggregation: foot+camera_height/5 through
        # foot+camera_height/0.8.  At 0.1 m resolution this is about 10 layers.
        self.frontier_occlusion_layers = max(
            1,
            int(np.ceil(
                (self.camera_height / 0.8 - self.camera_height / 5.0)
                / self.voxel_size_z
            )),
        )

        # ---- Camera ----
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_initialized = False
        self.agent_height_0: Optional[float] = None
        self.current_agent_z: float = 0.0

        # ---- 3D voxel storage ----
        self.voxels: Dict[Tuple[int, int, int], VoxelData] = {}
        self.updated_voxel_ids: Set[Tuple[int, int, int]] = set()

        # ---- Incremental frontier tracking ----
        # Voxels that became TRAVERSABLE this frame — used by detect_frontiers_3d_incremental
        self.new_traversable_voxels: Set[Tuple[int, int, int]] = set()
        # Local reverse dependencies for derived map attributes.  The map is
        # persistent globally, but only these candidates are recomputed after
        # geometry/type changes.
        self.support_dirty_voxels: Set[Tuple[int, int, int]] = set()
        self.lateral_inflation_dirty_voxels: Set[Tuple[int, int, int]] = set()
        self.head_clearance_dirty_voxels: Set[Tuple[int, int, int]] = set()
        self.frontier_dirty_voxels: Set[Tuple[int, int, int]] = set()

        # ---- Similarity cache ----
        self.similarity_cache: Dict[Tuple[int, int, int], float] = {}

        # ---- Stair detection state ----
        self.mapping_mode         = "geometric"
        self.stair_detect_counter = 0
        self.stair_detect_frames  = 5

        self._iters = 0

        print(
            f"VoxelMap initialized: voxel_xy={self.voxel_size:.3f}m, "
            f"voxel_z={self.voxel_size_z:.3f}m, feature_dim={feature_dim}"
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:
        self.camera_initialized = True
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.voxels.clear()
        self.updated_voxel_ids.clear()
        self.similarity_cache.clear()
        self.new_traversable_voxels.clear()
        self.support_dirty_voxels.clear()
        self.lateral_inflation_dirty_voxels.clear()
        self.head_clearance_dirty_voxels.clear()
        self.frontier_dirty_voxels.clear()
        self.z_origin = None
        self.agent_height_0 = None
        self.current_agent_z = 0.0
        self.mapping_mode = "geometric"
        self.stair_detect_counter = 0
        self._iters = 0

    def mark_voxels_changed(
        self,
        voxel_ids,
    ) -> None:
        """Propagate local reverse dependencies after voxel geometry changes.

        Support classification depends on providers up to 0.4 m below an air
        voxel; lateral inflation depends on same-layer 8-neighbors; head
        clearance depends on blockers 1--3 layers above a pathable voxel.
        Frontier classification additionally depends on immediate horizontal
        and nearby vertical context, while the detector adds its own AABB
        margin when rebuilding clusters.
        """
        changed = {
            (int(vid[0]), int(vid[1]), int(vid[2]))
            for vid in voxel_ids
            if vid is not None and len(vid) >= 3
        }
        if not changed:
            return

        self.updated_voxel_ids.update(changed)
        for x, y, z in changed:
            # A changed support provider can reclassify air voxels above it.
            for dz in range(0, self.support_check_depth + 1):
                self.support_dirty_voxels.add((x, y, z + dz))

            # A changed collision/pathable voxel affects itself and same-layer
            # floor-like neighbors.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.lateral_inflation_dirty_voxels.add((x + dx, y + dy, z))

            # A blocker in an unknown XY column makes horizontally adjacent
            # lower TRAVERSABLE boundary voxels unsafe: the apparent free edge
            # was produced by vertical projection occlusion, not visible floor.
            # Propagate that reverse dependency so adding/removing the blocker
            # sets or clears their derived inflated flag locally.
            for dz in range(1, self.frontier_occlusion_layers + 1):
                lower_z = z - dz
                occlusion_dependents = {
                    (x + 1, y, lower_z),
                    (x - 1, y, lower_z),
                    (x, y + 1, lower_z),
                    (x, y - 1, lower_z),
                }
                for dependent in occlusion_dependents:
                    dependent_vd = self.voxels.get(dependent)
                    if (
                        dependent_vd is not None and
                        dependent_vd.voxel_type == VoxelType.TRAVERSABLE
                    ):
                        self.lateral_inflation_dirty_voxels.add(dependent)

            # A changed overhead blocker affects pathable voxels below it; the
            # voxel itself is included in case its pathable type changed.
            self.head_clearance_dirty_voxels.add((x, y, z))
            for dz in range(1, 4):
                self.head_clearance_dirty_voxels.add((x, y, z - dz))

            # Candidate rules inspect horizontal N8 and vertical shadow layers.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.frontier_dirty_voxels.add((x + dx, y + dy, z))
            for dz in range(-self.frontier_occlusion_layers, 4):
                self.frontier_dirty_voxels.add((x, y, z + dz))

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def metric_to_px(self, x: float, y: float) -> Tuple[int, int]:
        """World metric (x, y) -> 2D pixel index (for controller compatibility)."""
        return (
            int(x / self.voxel_size + self.map_center + 1e-9),
            int(y / self.voxel_size + self.map_center + 1e-9),
        )

    def px_to_metric(self, px: int, py: int) -> Tuple[float, float]:
        """2D pixel index -> world metric (x, y)."""
        return (
            (px - self.map_center) * self.voxel_size,
            (py - self.map_center) * self.voxel_size,
        )

    def metric_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """World metric (x, y, z) -> voxel index (ix, iy, iz).
        z is relative to z_origin (agent birth z), so birth position maps to iz=max_z_voxels.
        """
        z_origin = self.z_origin if self.z_origin is not None else 0.0
        ix = int(np.floor(x / self.voxel_size)) + self.map_center
        iy = int(np.floor(y / self.voxel_size)) + self.map_center
        iz = int(np.floor((z - z_origin) / self.voxel_size_z)) + self.max_z_voxels
        return ix, iy, iz

    def voxel_to_metric(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Voxel index -> world metric center of voxel."""
        z_origin = self.z_origin if self.z_origin is not None else 0.0
        x = (ix - self.map_center + 0.5) * self.voxel_size
        y = (iy - self.map_center + 0.5) * self.voxel_size
        z = (iz - self.max_z_voxels + 0.5) * self.voxel_size_z + z_origin
        return x, y, z

    def get_agent_voxel(self, tf: np.ndarray) -> Tuple[int, int, int]:
        """
        Get agent's current voxel from transform matrix.

        IMPORTANT: Returns the voxel at the raycasting plane height
        (agent_height_0 + camera_height/5), NOT camera height.
        All TRAVERSABLE voxels are generated at this z-level by
        _compute_traversability(), so the agent voxel must match
        for A* to find connected paths.
        """
        x = tf[0, 3]
        y = tf[1, 3]
        z = tf[2, 3]
        if tf.shape[0] == 4:
            x /= tf[3, 3]
            y /= tf[3, 3]
            z /= tf[3, 3]
        # Use raycasting plane z instead of camera z
        if self.agent_height_0 is not None:
            ray_z = self.agent_height_0 + self.camera_height / 5.0
            return self.metric_to_voxel(x, y, ray_z)
        return self.metric_to_voxel(x, y, z)

    # ------------------------------------------------------------------
    # Main update entry point
    # ------------------------------------------------------------------

    def update(
        self,
        values: torch.Tensor,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        artificial_obstacles: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Main per-frame update.
        Args:
            values: (feature_dim, H, W) CLIP features
            depth: (H, W) depth image
            tf_camera_to_episodic: (4, 4) camera-to-world transform
                                   Camera is at origin; agent foot is at z=-camera_height.
            artificial_obstacles: list of (px, py) pixel coords for stuck handling
        """
        assert values.shape[0] == self.feature_dim
        if not self.camera_initialized:
            warnings.warn("Camera matrix must be set before updating the map")
            return

        tf = tf_camera_to_episodic.astype(np.float32)
        cam_z = tf[2, 3] / tf[3, 3]

        # Record birth z on first frame (used as z-axis origin for voxel indexing)
        # Snap to voxel_size_z grid so voxel world-z stays at clean decimals
        if self.z_origin is None:

            self.z_origin = np.floor(cam_z / self.voxel_size_z) * self.voxel_size_z

        # agent_height_0 is the SUPPORT PLANE (foot/ground), not camera height.
        # Camera is at z=cam_z; foot is at z=cam_z - camera_height.
        foot_z = cam_z - self.camera_height
        # Update every frame to track floor changes (stairs, slopes, etc.)
        self.agent_height_0 = foot_z
        self.current_agent_z = cam_z

        # Reshape values: (F, H, W) -> (H, W, F)
        if len(values.shape) == 3:
            vals_hwf = values.permute(1, 2, 0).to("cuda")
        else:
            warnings.warn("VoxelMap.update: unsupported values shape")
            return

        depth_t = torch.as_tensor(depth, dtype=torch.float32, device="cuda")

        # Align feature resolution with depth
        if vals_hwf.shape[:2] != depth_t.shape:
            vals_hwf = torch.nn.functional.interpolate(
                vals_hwf.permute(2, 0, 1).unsqueeze(0),
                size=tuple(depth_t.shape),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        self._update_3d(vals_hwf, depth_t, tf, artificial_obstacles)
        self._iters += 1

    # ------------------------------------------------------------------
    # Core 3D pipeline
    # ------------------------------------------------------------------

    def _update_3d(
        self,
        vals_hwf: torch.Tensor,   # (H, W, F) on cuda
        depth_t: torch.Tensor,    # (H, W) on cuda
        tf: np.ndarray,           # (4, 4)
        artificial_obstacles,
    ) -> None:
        H, W = depth_t.shape
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        # --- 1. Back-project to 3D point cloud (camera frame) ---
        xx = torch.arange(W, device="cuda", dtype=torch.float32)
        yy = torch.arange(H, device="cuda", dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing="xy")
        zz = depth_t
        x_cam = (grid_x - cx) * zz / fx
        y_cam = (grid_y - cy) * zz / fy
        # Camera frame: forward=z, right=x, down=y
        # World frame: x=forward, y=left, z=up
        pts_cam = torch.stack([zz, -x_cam, -y_cam], dim=-1).reshape(-1, 3)

        # --- 2. 6DOF transform to world ---
        tf_t = torch.as_tensor(tf[:3, :], dtype=torch.float32, device="cuda")
        R_mat = tf_t[:, :3]
        t_vec = tf_t[:, 3]
        pts_world = (R_mat @ pts_cam.T).T + t_vec

        # --- 3. Observation scores (depth gradient + optimal distance) ---
        scores_flat = self._compute_scores(depth_t, fx, fy, H, W)

        # --- 4. Valid mask ---
        depth_flat = depth_t.flatten()
        valid = (depth_flat != 0) & (depth_flat != float("inf")) & (scores_flat > 1e-5)

        # --- 5. Classify: ALL valid depth points -> OCCUPIED voxels ---
        # No height restriction: every valid back-projected point is an obstacle
        # carrying its CLIP feature.  Height bands are only used for raycasting
        # (deciding which obstacles block the traversability slice).
        occupied_mask = valid

        # --- 6. Voxelize occupied points ---
        pts_occ = pts_world[occupied_mask]
        feat_occ = vals_hwf.reshape(-1, self.feature_dim)[occupied_mask]
        scr_occ = scores_flat[occupied_mask]

        if pts_occ.shape[0] > 0:
            z_origin = self.z_origin if self.z_origin is not None else 0.0
            # Subtract z_origin before voxelizing z-axis
            pts_occ_shifted = pts_occ.clone()
            pts_occ_shifted[:, 2] -= z_origin
            vids_float = pts_occ_shifted / torch.tensor(
                [self.voxel_size, self.voxel_size, self.voxel_size_z], device="cuda"
            )
            vids_int = torch.floor(vids_float).to(torch.int32)
            vids_int[:, 0] += self.map_center
            vids_int[:, 1] += self.map_center
            vids_int[:, 2] += self.max_z_voxels

            # Bounds check
            inbounds = (
                (vids_int[:, 0] >= 0) & (vids_int[:, 0] < self.n_cells)
                & (vids_int[:, 1] >= 0) & (vids_int[:, 1] < self.n_cells)
                & (vids_int[:, 2] >= 0) & (vids_int[:, 2] < 2 * self.max_z_voxels)
            )
            vids_int = vids_int[inbounds]
            feat_occ = feat_occ[inbounds]
            scr_occ = scr_occ[inbounds]

            self._insert_occupied_voxels(vids_int, feat_occ, scr_occ)

        # --- 7. Raycasting: generate TRAVERSABLE voxels ---
        # Only use CURRENT FRAME obstacle projections (not global history).
        # This avoids ray leaking through unseen walls.
        cam_world = t_vec.cpu().numpy()
        current_frame_obs_xy = self._extract_current_frame_obstacles(
            pts_world, valid, h_min_iz=None, h_max_iz=None,
        )
        self._compute_traversability(cam_world, artificial_obstacles, current_frame_obs_xy)

        # --- 8. Validate traversability (support check) ---
        self._validate_traversability()

    def _compute_scores(
        self, depth_t: torch.Tensor, fx: float, fy: float, H: int, W: int
    ) -> torch.Tensor:
        """Compute per-pixel observation quality scores."""
        depth_sm = depth_t.clone()
        inf_mask = depth_sm == float("inf")
        if inf_mask.any() and (~inf_mask).any():
            depth_sm[inf_mask] = depth_sm[~inf_mask].max()
        ksize = 11
        pad = ksize // 2
        depth_sm = -torch.nn.functional.max_pool2d(
            -depth_sm.unsqueeze(0), ksize, padding=pad, stride=1
        ).squeeze(0)
        dx = torch.gradient(depth_sm, dim=1)[0] / (fx / W)
        dy = torch.gradient(depth_sm, dim=0)[0] / (fy / H)
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2)
        grad_mag = torch.nn.functional.max_pool2d(
            grad_mag.unsqueeze(0), 11, stride=1, padding=5
        ).squeeze(0)
        scores = (
            (1 - torch.tanh(grad_mag * self.gradient_factor))
            * torch.exp(
                -((self.optimal_object_distance - depth_t) / self.optimal_object_factor) ** 2 / 3.0
            )
        )
        return scores.reshape(-1)

    # ------------------------------------------------------------------
    # Voxel insertion (OCCUPIED) with EMA
    # ------------------------------------------------------------------

    def _insert_occupied_voxels(
        self,
        vids_int: torch.Tensor,   # (M, 3) int32
        feat_occ: torch.Tensor,   # (M, F)
        scr_occ: torch.Tensor,    # (M,)
    ) -> None:
        """Aggregate and EMA-merge occupied voxels."""
        # Encode voxel IDs to unique keys
        keys = (
            vids_int[:, 0].long() * self.n_cells * (2 * self.max_z_voxels)
            + vids_int[:, 1].long() * (2 * self.max_z_voxels)
            + vids_int[:, 2].long()
        )
        unique_keys, inv = keys.unique(return_inverse=True)
        n_u = unique_keys.shape[0]

        # Aggregate features weighted by score
        agg_feat = torch.zeros((n_u, self.feature_dim), device="cuda")
        agg_score = torch.zeros(n_u, device="cuda")
        count = torch.zeros(n_u, device="cuda")

        agg_feat.index_add_(0, inv, feat_occ * scr_occ.unsqueeze(1))
        agg_score.index_add_(0, inv, scr_occ)
        count.index_add_(0, inv, torch.ones(vids_int.shape[0], device="cuda"))

        agg_feat = agg_feat / (agg_score.unsqueeze(1) + 1e-9)
        agg_score = agg_score / (count + 1e-9)

        # Decode and merge into voxel dict
        ks = unique_keys.cpu().numpy()
        feat_np = agg_feat.cpu().numpy()
        score_np = agg_score.cpu().numpy()

        geometry_changed_ids: Set[Tuple[int, int, int]] = set()
        for i in range(n_u):
            k = int(ks[i])
            iz = k % (2 * self.max_z_voxels)
            k //= (2 * self.max_z_voxels)
            iy = k % self.n_cells
            k //= self.n_cells
            ix = k
            vid = (ix, iy, iz)

            new_feat = feat_np[i]
            new_conf = float(score_np[i])

            self.updated_voxel_ids.add(vid)
            geometry_changed = False

            if vid in self.voxels:
                vd = self.voxels[vid]
                # 固定类型（真实楼梯、虚假楼梯）不可被深度覆盖
                if vd.voxel_type in IMMUTABLE_TYPES:
                    continue
                if vd.voxel_type in (VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR, VoxelType.TRAVERSABLE_FAKE):
                    # 原为空气，现在有深度命中 -> OCCUPIED
                    self.voxels[vid] = VoxelData(
                        voxel_type=VoxelType.OCCUPIED,
                        feature=new_feat.copy(),
                        confidence=new_conf,
                        obstacle_likelihood=new_conf,
                    )
                    geometry_changed = True
                else:
                    vd.merge(new_feat, new_conf)
            else:
                self.voxels[vid] = VoxelData(
                    voxel_type=VoxelType.OCCUPIED,
                    feature=new_feat.copy(),
                    confidence=new_conf,
                    obstacle_likelihood=new_conf,
                )
                geometry_changed = True

            if geometry_changed:
                geometry_changed_ids.add(vid)

        self.mark_voxels_changed(geometry_changed_ids)

    # ------------------------------------------------------------------
    # Stair voxel registration (legacy stub — callers not yet cleaned up)
    # ------------------------------------------------------------------

    def _register_stair_voxel(self, vid: Tuple[int, int, int]) -> None:
        """No-op stub. Kept to avoid breaking callers; will be removed later."""
        pass

    # ------------------------------------------------------------------
    # Current-frame obstacle extraction
    # ------------------------------------------------------------------

    def _extract_current_frame_obstacles(
        self,
        pts_world: torch.Tensor,    # (N, 3) all back-projected points in world frame
        valid: torch.Tensor,        # (N,) bool mask for valid depth
        h_min_iz=None,
        h_max_iz=None,
    ) -> Set[Tuple[int, int]]:
        """
        Extract obstacle (ix, iy) projections from CURRENT FRAME depth only.

        Only points within the agent collision height band are considered.
        Returns a set of (ix, iy) tuples on the raycasting plane.
        """
        if self.agent_height_0 is None:
            return set()

        band_min_z = self.agent_height_0 + self.camera_height / 5.0
        band_max_z = self.agent_height_0 + self.camera_height / 0.8

        # Filter valid points in height band
        pts_valid = pts_world[valid]
        if pts_valid.shape[0] == 0:
            return set()

        z_vals = pts_valid[:, 2]
        in_band = (z_vals >= band_min_z) & (z_vals <= band_max_z)
        pts_band = pts_valid[in_band]

        if pts_band.shape[0] == 0:
            return set()

        # Convert to voxel (ix, iy)
        ix_arr = torch.floor(pts_band[:, 0] / self.voxel_size).to(torch.int32) + self.map_center
        iy_arr = torch.floor(pts_band[:, 1] / self.voxel_size).to(torch.int32) + self.map_center

        # Bounds check
        in_bounds = (
            (ix_arr >= 0) & (ix_arr < self.n_cells) &
            (iy_arr >= 0) & (iy_arr < self.n_cells)
        )
        ix_arr = ix_arr[in_bounds].cpu().numpy()
        iy_arr = iy_arr[in_bounds].cpu().numpy()

        obs_xy: Set[Tuple[int, int]] = set()
        for i in range(len(ix_arr)):
            obs_xy.add((int(ix_arr[i]), int(iy_arr[i])))

        return obs_xy

    # ------------------------------------------------------------------
    # Traversability via 2D raycasting
    # ------------------------------------------------------------------

    def _compute_traversability(
        self,
        cam_world: np.ndarray,
        artificial_obstacles,
        current_frame_obs_xy: Set[Tuple[int, int]] = None,
    ) -> None:
        """
        2D raycasting on a single horizontal slice for traversability.

        LOCAL UPDATE: Only uses obstacles from the CURRENT FRAME depth image
        (passed via current_frame_obs_xy). This ensures:
          - Rays only target obstacles the agent can actually see right now
          - No ray leaks through unseen walls from historical observations
          - Each frame incrementally grows the traversable region

        Design:
          - current_frame_obs_xy: (ix, iy) columns with depth hits in height band
            from THIS frame's depth image only.
          - Cast rays from agent to each current-frame obstacle.
          - Ray stops at the FIRST obstacle column it encounters (using the
            GLOBAL obs_xy set for occlusion checking — accumulated walls block rays).
          - Mark intermediate voxels as TRAVERSABLE.
          - After all rays: inflate near obstacles.

        """
        if self.agent_height_0 is None:
            return

        if not current_frame_obs_xy:
            return

        # Raycasting plane: 1/5 agent height above foot
        ray_z = self.agent_height_0 + self.camera_height / 5.0
        z_origin = self.z_origin if self.z_origin is not None else 0.0
        ray_iz = int(np.floor((ray_z - z_origin) / self.voxel_size_z)) + self.max_z_voxels

        # Agent voxel position (xy projection on the raycasting slice)
        agent_ix = int(np.floor(cam_world[0] / self.voxel_size)) + self.map_center
        agent_iy = int(np.floor(cam_world[1] / self.voxel_size)) + self.map_center

        # Build GLOBAL obstacle set for occlusion checking.
        # This includes all historically accumulated OCCUPIED columns in height band.
        # Used ONLY for blocking rays mid-path (not as ray targets).
        band_min_z = self.agent_height_0 + self.camera_height / 5.0
        band_max_z = self.agent_height_0 + self.camera_height / 0.8
        h_min_iz = int(np.floor((band_min_z - z_origin) / self.voxel_size_z)) + self.max_z_voxels
        h_max_iz = int(np.floor((band_max_z - z_origin) / self.voxel_size_z)) + self.max_z_voxels

        global_obs_xy: Set[Tuple[int, int]] = set()
        for vid, vd in self.voxels.items():
            if vd.voxel_type in OBSTACLE_TYPES:
                if h_min_iz <= vid[2] <= h_max_iz:
                    global_obs_xy.add((vid[0], vid[1]))

        # Raycast from agent to each CURRENT FRAME obstacle.
        # Ray stops when hitting any global obstacle column (occlusion).
        new_traversable_this_frame: Set[Tuple[int, int]] = set()
        for (obs_ix, obs_iy) in current_frame_obs_xy:
            ray = self._bresenham_2d(agent_ix, agent_iy, obs_ix, obs_iy)
            for (rx, ry) in ray[:-1]:  # exclude endpoint (the target obstacle itself)
                if not (0 <= rx < self.n_cells and 0 <= ry < self.n_cells):
                    break
                # Occlusion check: if this column has ANY known obstacle in height band
                # (from global history), ray cannot pass through.
                if (rx, ry) in global_obs_xy and (rx, ry) != (agent_ix, agent_iy):
                    break
                # Also check if ray_iz layer itself is occupied/stair
                vid = (rx, ry, ray_iz)
                if vid in self.voxels and self.voxels[vid].voxel_type in (
                    VoxelType.OCCUPIED,
                    VoxelType.FAKE_STAIR,
                    VoxelType.STAIR,
                    VoxelType.STAIR_EXPANDED,
                ):
                    break
                # Mark as traversable (初始为TRAVERSABLE，后续_validate_traversability会细分类型)
                if vid not in self.voxels:
                    self.voxels[vid] = VoxelData(
                        voxel_type=VoxelType.TRAVERSABLE,
                        feature=np.zeros(self.feature_dim, dtype=np.float32),
                        confidence=0.0,
                        obstacle_likelihood=0.0,
                    )
                    self.new_traversable_voxels.add(vid)
                    new_traversable_this_frame.add((rx, ry))
                elif self.voxels[vid].voxel_type in (VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR, VoxelType.TRAVERSABLE_FAKE):
                    # 射线确认该体素可通行，标记为待验证
                    new_traversable_this_frame.add((rx, ry))

        # Lateral inflation is recomputed in _validate_traversability() from
        # the pathable voxel side using same-layer 8-neighbor collision checks.
        # The original obstacle-radius expansion is intentionally not applied
        # here because it over-prunes narrow stair/landing connectors.
        radius = self.inflation_radius_voxels

        # Artificial obstacles (stuck handling) — mark as OCCUPIED
        if artificial_obstacles:
            r2 = radius * radius
            artificial_changed: Set[Tuple[int, int, int]] = set()
            for obs in artificial_obstacles:
                oix, oiy = int(obs[0]), int(obs[1])
                for dx in range(-radius, radius + 1):
                    dx2 = dx * dx
                    for dy in range(-radius, radius + 1):
                        if dx2 + dy * dy <= r2:
                            vid = (oix + dx, oiy + dy, ray_iz)
                            if vid not in self.voxels or self.voxels[vid].voxel_type in (
                                VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR, VoxelType.TRAVERSABLE_FAKE
                            ):
                                self.voxels[vid] = VoxelData(
                                    voxel_type=VoxelType.OCCUPIED,
                                    feature=np.zeros(self.feature_dim, dtype=np.float32),
                                    confidence=1.0,
                                    obstacle_likelihood=1.0,
                                )
                                artificial_changed.add(vid)
            self.mark_voxels_changed(artificial_changed)

        # --- Always mark agent's current position as TRAVERSABLE (non-inflated) ---
        # The agent is physically standing here, so it must be traversable.
        # This fills the "blind spot" gap under the camera FOV and ensures
        # A* can always start from the agent's position.
        agent_vid = (agent_ix, agent_iy, ray_iz)
        if agent_vid not in self.voxels or self.voxels[agent_vid].voxel_type in (
            VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR, VoxelType.TRAVERSABLE_FAKE
        ):
            self.voxels[agent_vid] = VoxelData(
                voxel_type=VoxelType.TRAVERSABLE,
                feature=np.zeros(self.feature_dim, dtype=np.float32),
                confidence=0.0,
                obstacle_likelihood=0.0,
                inflated=False,
            )
            self.new_traversable_voxels.add(agent_vid)
        # Also mark a small neighborhood (agent has physical extent)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nb_vid = (agent_ix + dx, agent_iy + dy, ray_iz)
                if nb_vid not in self.voxels:
                    self.voxels[nb_vid] = VoxelData(
                        voxel_type=VoxelType.TRAVERSABLE,
                        feature=np.zeros(self.feature_dim, dtype=np.float32),
                        confidence=0.0,
                        obstacle_likelihood=0.0,
                        inflated=False,
                    )
                    self.new_traversable_voxels.add(nb_vid)

    def _validate_traversability(self) -> None:
        """
        Locally reclassify dirty air voxels from support, then recompute
        inflation only for affected reverse dependencies.

        Two-phase support update: first compute all proposed types from the
        same map snapshot, then commit them together so iteration order cannot
        affect the result.

        支撑分类：
          - 下方有 OCCUPIED → TRAVERSABLE（地板可通行）
          - 下方有 FAKE_STAIR → TRAVERSABLE（撤销的楼梯按普通支撑）
          - 下方有 STAIR/STAIR_EXPANDED → TRAVERSABLE_STAIR（楼梯可通行）
          - 无支撑 → TRAVERSABLE_FAKE（悬空，不可通行）
        """
        air_types = {
            VoxelType.TRAVERSABLE,
            VoxelType.TRAVERSABLE_STAIR,
            VoxelType.TRAVERSABLE_FAKE,
        }
        support_candidates = set(self.new_traversable_voxels)
        support_candidates.update(self.support_dirty_voxels)
        self.support_dirty_voxels.clear()

        proposed_types: Dict[Tuple[int, int, int], VoxelType] = {}
        for vid in support_candidates:
            vd = self.voxels.get(vid)
            if vd is None or vd.voxel_type not in air_types:
                continue

            proposed = VoxelType.TRAVERSABLE_FAKE
            for dz in range(1, self.support_check_depth + 1):
                below = (vid[0], vid[1], vid[2] - dz)
                below_vd = self.voxels.get(below)
                if below_vd is None:
                    continue
                if below_vd.voxel_type in SUPPORT_FLOOR:
                    proposed = VoxelType.TRAVERSABLE
                    break
                if below_vd.voxel_type in SUPPORT_STAIR:
                    proposed = VoxelType.TRAVERSABLE_STAIR
                    break
            proposed_types[vid] = proposed

        type_changed: Set[Tuple[int, int, int]] = set()
        for vid, proposed in proposed_types.items():
            vd = self.voxels.get(vid)
            if vd is not None and vd.voxel_type != proposed:
                vd.voxel_type = proposed
                type_changed.add(vid)

        # Every support candidate needs its own derived inflation refreshed.
        inflation_candidates = set(support_candidates)
        inflation_candidates.update(self.lateral_inflation_dirty_voxels)
        inflation_candidates.update(self.head_clearance_dirty_voxels)
        self.lateral_inflation_dirty_voxels.clear()
        self.head_clearance_dirty_voxels.clear()

        if type_changed:
            inflation_candidates.update(type_changed)
            for x, y, z in type_changed:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        inflation_candidates.add((x + dx, y + dy, z))
                        self.frontier_dirty_voxels.add((x + dx, y + dy, z))
                for dz in range(-3, 4):
                    self.frontier_dirty_voxels.add((x, y, z + dz))

        self._recompute_inflation(inflation_candidates)

    def refresh_dirty_derived(self) -> None:
        """Apply pending support/inflation dependencies after external edits.

        Navigator stair routines mutate voxel types after ``update()``.  They
        call ``mark_voxels_changed()`` and then use this hook before frontier
        detection so derived attributes are consistent in the same frame.
        """
        if (
            self.support_dirty_voxels
            or self.lateral_inflation_dirty_voxels
            or self.head_clearance_dirty_voxels
        ):
            self._validate_traversability()

    def _recompute_inflation(
        self,
        candidates: Set[Tuple[int, int, int]],
    ) -> None:
        """Recompute contact, head clearance, and occluded-boundary inflation."""
        # Same-layer contact inflation protects floor-like navigation cells
        # from true geometric obstacles only.  STAIR/STAIR_EXPANDED are valid
        # walking surfaces in StairSem and must not cut the strict 26-connected
        # floor-to-stair graph merely by touching TRAVERSABLE(_STAIR).  Their
        # wall-like/overhead cases remain covered by the head-clearance check
        # below.  FAKE_STAIR is likewise pathable and is not a lateral blocker.
        lateral_collision_types = {VoxelType.OCCUPIED}
        floor_like = {VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR}
        for vid in candidates:
            vd = self.voxels.get(vid)
            if vd is None or vd.voxel_type not in PATHABLE_TYPES:
                continue
            inflated = False
            if vd.voxel_type in floor_like:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nb_vd = self.voxels.get((vid[0] + dx, vid[1] + dy, vid[2]))
                        if nb_vd is not None and nb_vd.voxel_type in lateral_collision_types:
                            inflated = True
                            break
                    if inflated:
                        break
            if vd.voxel_type == VoxelType.TRAVERSABLE and not inflated:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (vid[0] + dx, vid[1] + dy, vid[2])
                    if is_occluded_unknown_column(
                        neighbor,
                        self.voxels,
                        self.frontier_occlusion_layers,
                    ):
                        inflated = True
                        break
            for dz in range(1, 4):
                above_vd = self.voxels.get((vid[0], vid[1], vid[2] + dz))
                if above_vd is not None and above_vd.voxel_type in HEAD_BLOCKING_TYPES:
                    inflated = True
                    break
            vd.inflated = inflated

    def get_fake_traversable_near(self, agent_voxel: Tuple[int, int, int], radius: int = 5) -> List[Tuple[int, int, int]]:
        """Return TRAVERSABLE_FAKE voxels within radius of agent (for stair detection)."""
        result = []
        ax, ay, az = agent_voxel
        for vid, vd in self.voxels.items():
            if vd.voxel_type != VoxelType.TRAVERSABLE_FAKE:
                continue
            if abs(vid[0] - ax) <= radius and abs(vid[1] - ay) <= radius:
                result.append(vid)
        return result

    def has_stair_nearby(self, agent_voxel: Tuple[int, int, int], radius: int = 5) -> bool:
        """Return True if there are STAIR/STAIR_EXPANDED voxels within radius of agent."""
        ax, ay, az = agent_voxel
        for vid, vd in self.voxels.items():
            if vd.voxel_type not in LOCKED_TYPES:
                continue
            if abs(vid[0] - ax) <= radius and abs(vid[1] - ay) <= radius:
                return True
        return False

    # ------------------------------------------------------------------
    # Stair detection
    # ------------------------------------------------------------------

    def mark_stair_voxels(
        self,
        detections: dict,
        depth: np.ndarray,
        tf: np.ndarray,
    ) -> None:
        """
        YOLO detects stairs -> mark surrounding OCCUPIED voxels as STAIR.
        STAIR voxels are treated as TRAVERSABLE during path planning.
        """
        if self.fx is None or self.agent_height_0 is None:
            return
        changed: Set[Tuple[int, int, int]] = set()
        for box, cls in zip(detections.get("boxes", []), detections.get("classes", [])):
            if "stair" not in cls.lower():
                continue
            cx_box = int((box[0] + box[2]) / 2)
            cy_box = int((box[1] + box[3]) / 2)
            if cy_box >= depth.shape[0] or cx_box >= depth.shape[1]:
                continue
            d = depth[cy_box, cx_box]
            if d == 0 or d == float("inf"):
                continue
            # Back-project to world
            x_cam = (cx_box - self.cx) * d / self.fx
            y_cam = (cy_box - self.cy) * d / self.fy
            pt_cam = np.array([d, -x_cam, -y_cam], dtype=np.float32)
            R_mat = tf[:3, :3]
            t_vec = tf[:3, 3]
            pt_w = R_mat @ pt_cam + t_vec

            vid = self.metric_to_voxel(pt_w[0], pt_w[1], pt_w[2])
            # Inflate around detected stair center
            for dix in range(-3, 4):
                for diy in range(-3, 4):
                    for diz in range(-4, 5):
                        nv = (vid[0] + dix, vid[1] + diy, vid[2] + diz)
                        if nv in self.voxels:
                            if self.voxels[nv].voxel_type == VoxelType.OCCUPIED:
                                self.voxels[nv].voxel_type = VoxelType.STAIR
                                self._register_stair_voxel(nv)
                                changed.add(nv)
                        else:
                            # Also create STAIR_EXPANDED voxels in empty space
                            self.voxels[nv] = VoxelData(
                                voxel_type=VoxelType.STAIR_EXPANDED,
                                feature=np.zeros(self.feature_dim, dtype=np.float32),
                                confidence=0.5,
                                obstacle_likelihood=0.0,
                            )
                            self._register_stair_voxel(nv)
                            changed.add(nv)
        self.mark_voxels_changed(changed)

    def update_stair_mode(self, detections: dict, agent_voxel: Optional[Tuple[int, int, int]] = None,
                          force_semantic: bool = False) -> None:  # noqa: C901
        """
        Switch geometric <-> semantic mode.

        Enters semantic mode when:
          - YOLO detects stairs for consecutive frames (going upstairs), OR
          - force_semantic=True (triggered by arriving at a stair frontier / look-down)

        Exits semantic mode when:
          - No stairs detected by YOLO AND
          - No STAIR voxels near agent (agent has left the stair area)
          - This means agent has returned to flat ground
        """
        stair_seen = any(
            "stair" in cls.lower()
            for cls in detections.get("classes", [])
        )

        if self.mapping_mode == "geometric":
            if force_semantic:
                # Directly enter semantic mode (stair frontier arrival / look-down)
                self.mapping_mode = "semantic"
                self.stair_detect_counter = 0
            elif stair_seen:
                self.stair_detect_counter += 1
                if self.stair_detect_counter >= self.stair_detect_frames:
                    self.mapping_mode = "semantic"
                    self.stair_detect_counter = 0
            else:
                self.stair_detect_counter = 0

        else:  # semantic mode
            # Exit condition: agent is far from ALL stair voxels (>2m)
            # This ensures the agent fully descends/ascends before switching back.
            stair_exit_radius = max(5, round(2.0 / self.voxel_size))
            if agent_voxel is not None:
                has_stair_nearby = self.has_stair_nearby(agent_voxel, radius=stair_exit_radius)
                if not has_stair_nearby:
                    self.mapping_mode = "geometric"
                    self.stair_detect_counter = 0

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def compute_similarity(
        self, query_features: torch.Tensor
    ) -> Dict[Tuple[int, int, int], float]:
        """
        Compute cosine similarity between query and all OCCUPIED/STAIR voxel features.
        Only recomputes for updated voxels (lazy cache).

        Args:
            query_features: (1, feature_dim) or (feature_dim,) tensor
        Returns:
            Dict mapping voxel_id -> similarity score
        """
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)

        # Gather voxels to compute
        to_compute = []
        vids_to_compute = []
        for vid in self.updated_voxel_ids:
            if vid in self.voxels and self.voxels[vid].voxel_type in (
                VoxelType.OCCUPIED, VoxelType.STAIR, VoxelType.STAIR_EXPANDED
            ):
                to_compute.append(self.voxels[vid].feature)
                vids_to_compute.append(vid)

        if to_compute:
            feat_tensor = torch.tensor(
                np.stack(to_compute), dtype=torch.float32, device="cuda"
            )
            query_norm = query_features / (query_features.norm(dim=-1, keepdim=True) + 1e-8)
            feat_norm = feat_tensor / (feat_tensor.norm(dim=-1, keepdim=True) + 1e-8)
            sims = (feat_norm @ query_norm.T).squeeze(-1)

            for i, vid in enumerate(vids_to_compute):
                self.similarity_cache[vid] = sims[i].item()

        self.updated_voxel_ids.clear()
        return self.similarity_cache

    def get_top_similarities(self, k: int = 10) -> List[Tuple[Tuple[int, int, int], float]]:
        """Return top-K voxels by similarity score."""
        if not self.similarity_cache:
            return []
        sorted_items = sorted(self.similarity_cache.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    # ------------------------------------------------------------------
    # Navigation target finding
    # ------------------------------------------------------------------

    def find_nav_target(
        self, query_features: torch.Tensor
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find the best navigation target:
        1. Get highest similarity OCCUPIED voxel
        2. Find nearest TRAVERSABLE/STAIR voxel to navigate to
        """
        self.compute_similarity(query_features)

        if not self.similarity_cache:
            return None

        # Sort by similarity descending
        sorted_vids = sorted(
            self.similarity_cache.items(), key=lambda x: x[1], reverse=True
        )

        for best_vid, score in sorted_vids[:20]:  # Try top 20 candidates
            # Find nearest traversable voxel near this target
            nav_vid = self._find_nearest_traversable(best_vid)
            if nav_vid is not None:
                return nav_vid

        return None

    def _find_nearest_traversable(
        self, target_vid: Tuple[int, int, int], max_radius: int = 20
    ) -> Optional[Tuple[int, int, int]]:
        """Find nearest non-FAKE targetable voxel, searching in expanding shells."""
        # Search below first (object is usually above walkable area)
        for dz in range(0, max_radius):
            check = (target_vid[0], target_vid[1], target_vid[2] - dz)
            if check in self.voxels and self.voxels[check].voxel_type in TARGETABLE_TYPES:
                return check

        # Search in xy neighborhood at each z level
        for r in range(1, max_radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # only shell
                    for dz in range(-r, r + 1):
                        check = (target_vid[0] + dx, target_vid[1] + dy, target_vid[2] + dz)
                        if check in self.voxels and self.voxels[check].voxel_type in TARGETABLE_TYPES:
                            return check
        return None

    def _find_nearest_safe_traversable(
        self,
        target_vid: Tuple[int, int, int],
        max_radius: int = 20,
        allow_inflated_stair: bool = False,
        allow_inflated_traversable_stair: bool = False,
        allow_inflated_traversable: bool = False,
    ) -> Optional[Tuple[int, int, int]]:
        """Find a nearby non-inflated, non-FAKE targetable voxel.

        Search the target height's immediate 4- and 8-neighbours first.  In
        this 3D map, a same-floor lateral neighbour is normally a better
        representation of the agent's real position than a voxel several
        layers below in the same column.  The legacy downward and full 3D
        searches remain as fallbacks for object-to-floor projection.
        """
        def is_safe_target(vd: VoxelData) -> bool:
            if vd.voxel_type not in TARGETABLE_TYPES:
                return False
            if not vd.inflated:
                return True
            return (
                (allow_inflated_traversable and vd.voxel_type == VoxelType.TRAVERSABLE) or
                (allow_inflated_stair and vd.voxel_type == VoxelType.STAIR) or
                (allow_inflated_traversable_stair and vd.voxel_type == VoxelType.TRAVERSABLE_STAIR)
            )

        # Exact position first.
        if target_vid in self.voxels and is_safe_target(self.voxels[target_vid]):
            return target_vid

        # Prefer immediate same-z neighbours: 4-neighbour before diagonals.
        # This prevents an inflated platform/stair start from snapping several
        # layers down while a safe lateral exit is only 0.1-0.14 m away.
        cardinal_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for dx, dy in cardinal_offsets:
            check = (target_vid[0] + dx, target_vid[1] + dy, target_vid[2])
            if check in self.voxels and is_safe_target(self.voxels[check]):
                return check

        diagonal_offsets = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for dx, dy in diagonal_offsets:
            check = (target_vid[0] + dx, target_vid[1] + dy, target_vid[2])
            if check in self.voxels and is_safe_target(self.voxels[check]):
                return check

        # Fall back to the same xy column below the target.  This is useful for
        # projecting object/map positions onto a supporting navigation layer,
        # but must not take priority over a closer same-floor neighbour.
        for dz in range(1, max_radius):
            check = (target_vid[0], target_vid[1], target_vid[2] - dz)
            if check in self.voxels and is_safe_target(self.voxels[check]):
                return check

        # Finally search the surrounding 3D shells.
        for r in range(1, max_radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # only shell
                    for dz in range(-r, r + 1):
                        # Immediate same-z candidates were already checked.
                        if dz == 0 and r == 1:
                            continue
                        check = (target_vid[0] + dx, target_vid[1] + dy, target_vid[2] + dz)
                        if check in self.voxels and is_safe_target(self.voxels[check]):
                            return check
        return None

    # ------------------------------------------------------------------
    # 3D A* path planning
    # ------------------------------------------------------------------

    def a_star_3d(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        max_iterations: int = 50000,
        z_margin: int = 2,
        stair_mode: bool = False,
        prefer_traversable: bool = False,
        prefer_traversable_cost: bool = True,
        allow_inflated_stair: bool = False,
        allow_inflated_traversable_stair: bool = False,
        connect_traversable_stair_drop: bool = False,
        allow_inflated_from_inflated_start: bool = False,
        allow_inflated_traversable: bool = False,
        inflated_cost_multiplier: float = 2.0,
    ) -> Optional[List[Tuple[int, int, int]]]:
        """
        Pure 3D A* on TRAVERSABLE + STAIR voxels.

        Design invariant: obstacle avoidance is handled by _compute_traversability
        (raycasting stops at inflated obstacles). Therefore A* only needs to:
          1. Check that the voxel is TRAVERSABLE or STAIR
          2. Respect a height-band constraint (z_margin) to keep search local
          3. Use a closed_set to prevent redundant expansion

        When stair_mode=True, expansion uses a 1 m physical-radius sphere.
        Candidate endpoints are checked using StairSem's permissive exploration
        policy; intermediate unknown/obstacle samples on a long edge are not
        inspected.
        """
        if start == goal:
            return [start]

        def is_allowed_inflated(vd: VoxelData) -> bool:
            if stair_mode:
                return vd.voxel_type in (
                    VoxelType.TRAVERSABLE,
                    VoxelType.TRAVERSABLE_STAIR,
                )
            return (
                (allow_inflated_traversable and vd.voxel_type == VoxelType.TRAVERSABLE) or
                (allow_inflated_stair and vd.voxel_type == VoxelType.STAIR) or
                (allow_inflated_traversable_stair and vd.voxel_type == VoxelType.TRAVERSABLE_STAIR)
            )

        # Explore-only fallback selected by the caller: if the real agent is
        # currently inside an inflated pathable voxel, this planning pass may
        # traverse inflated pathable voxels with the requested cost multiplier. A
        # later planning pass automatically becomes strict again as soon as
        # its real start voxel is non-inflated.
        start_is_inflated = bool(
            start in self.voxels and
            self.voxels[start].voxel_type in PATHABLE_TYPES and
            self.voxels[start].inflated
        )
        allow_inflated_path = bool(
            allow_inflated_from_inflated_start and start_is_inflated
        )

        def is_navigable_neighbor(vd: VoxelData) -> bool:
            if prefer_traversable:
                if vd.voxel_type not in (VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR):
                    return False
            elif vd.voxel_type not in PATHABLE_TYPES:
                return False
            if vd.inflated and not is_allowed_inflated(vd) and not allow_inflated_path:
                return False
            return True


        # Snap only a missing/non-pathable start. If the real agent already
        # occupies a PATHABLE-but-inflated voxel, retain its true position.
        # Normal calls still require safe neighbours; Explore may explicitly
        # enable the penalized inflated-path fallback above. Snapping the start
        # in shell traversal order creates a fictitious position and can make
        # the discrete controller oscillate between real and snapped cells.
        if (
            start not in self.voxels or
            self.voxels[start].voxel_type not in PATHABLE_TYPES
        ):
            start = self._find_nearest_safe_traversable(
                start,
                max_radius=15,
                allow_inflated_traversable=allow_inflated_traversable,
                allow_inflated_stair=allow_inflated_stair,
                allow_inflated_traversable_stair=allow_inflated_traversable_stair,
            )
            if start is None:
                return None

        if (goal not in self.voxels or
            self.voxels[goal].voxel_type not in TARGETABLE_TYPES or
            (self.voxels[goal].inflated and not is_allowed_inflated(self.voxels[goal]))):
            goal = self._find_nearest_safe_traversable(
                goal,
                max_radius=15,
                allow_inflated_traversable=allow_inflated_traversable,
                allow_inflated_stair=allow_inflated_stair,
                allow_inflated_traversable_stair=allow_inflated_traversable_stair,
            )
            if goal is None:
                return None

        # Height band: keep search within [min_z - margin, max_z + margin]
        z_min = min(start[2], goal[2]) - z_margin
        z_max = max(start[2], goal[2]) + z_margin

        # 26-connected neighbourhood (allows diagonal / stair moves)
        neighbors_26 = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        # StairSem uses one physical neighbourhood for every pathable type.
        # This replaces the old type-dependent STAIR ±4 / TRAVERSABLE ±6
        # cubes.  Generate it once per A* call rather than once per expanded
        # node; with the default 0.1 m grid the sphere contains ~4k offsets.
        stair_sphere_offsets = neighbors_26
        if stair_mode:
            radius_m = 1.0
            radius_x = max(1, int(np.floor(radius_m / self.voxel_size)))
            radius_z = max(1, int(np.floor(radius_m / self.voxel_size_z)))
            radius_sq = radius_m * radius_m + 1e-9
            stair_sphere_offsets = [
                (dx, dy, dz)
                for dx in range(-radius_x, radius_x + 1)
                for dy in range(-radius_x, radius_x + 1)
                for dz in range(-radius_z, radius_z + 1)
                if not (dx == 0 and dy == 0 and dz == 0)
                if (
                    (dx * self.voxel_size) ** 2 +
                    (dy * self.voxel_size) ** 2 +
                    (dz * self.voxel_size_z) ** 2
                ) <= radius_sq
            ]

        def heuristic(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            dz = a[2] - b[2]
            return (dx * dx + dy * dy + dz * dz) ** 0.5

        def voxel_type_penalty(vd: VoxelData) -> float:
            if not prefer_traversable_cost:
                return 0.0
            if vd.voxel_type in (VoxelType.TRAVERSABLE, VoxelType.TRAVERSABLE_STAIR):
                return 0.0
            if vd.voxel_type == VoxelType.STAIR:
                return 4.0
            if vd.voxel_type == VoxelType.STAIR_EXPANDED:
                return 5.0
            if vd.voxel_type == VoxelType.FAKE_STAIR:
                return 10.0
            return 20.0

        open_set = [(0.0, start)]
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start: 0.0}
        closed_set: Set[Tuple[int, int, int]] = set()
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # StairSem uses the same 1 m physical sphere for every pathable
            # voxel type. Other modes retain strict 26-connectivity.
            vd_current = self.voxels[current]
            neighbors_iter = stair_sphere_offsets if stair_mode else neighbors_26

            if connect_traversable_stair_drop and vd_current.voxel_type in (
                VoxelType.TRAVERSABLE,
                VoxelType.TRAVERSABLE_STAIR,
            ):
                # Navigate-only local bridge: if floor/stair traversable layers
                # are vertically sparse, connect to the first lower stair/floor
                # layer in the same column. This avoids routing backward through
                # unrelated connector voxels while keeping the graph strict-26
                # everywhere else.
                for dz_drop in range(1, self.support_check_depth + 1):
                    drop_neighbor = (current[0], current[1], current[2] - dz_drop)
                    if drop_neighbor[2] < z_min:
                        break
                    drop_vd = self.voxels.get(drop_neighbor)
                    if drop_vd is None:
                        continue
                    if drop_vd.voxel_type not in (
                        VoxelType.TRAVERSABLE,
                        VoxelType.TRAVERSABLE_STAIR,
                        VoxelType.STAIR,
                        VoxelType.STAIR_EXPANDED,
                        VoxelType.FAKE_STAIR,
                    ):
                        continue
                    if drop_vd.inflated and not is_allowed_inflated(drop_vd):
                        continue
                    neighbors_iter = list(neighbors_iter)
                    neighbors_iter.append((0, 0, -dz_drop))

                    # A same-column stair landing can be isolated from the
                    # lower floor by one inflated boundary cell.  Also expose
                    # a small set of nearby, non-inflated floor landings so A*
                    # can descend diagonally onto safe free space instead of
                    # entering that inflated cell.  Keep this local to a
                    # stair-like landing and no wider than the vertical gap.
                    if drop_vd.voxel_type in (
                        VoxelType.TRAVERSABLE_STAIR,
                        VoxelType.STAIR,
                        VoxelType.STAIR_EXPANDED,
                        VoxelType.FAKE_STAIR,
                    ):
                        landing_directions = (
                            (-1, 0), (1, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1),
                        )
                        for radius in range(1, dz_drop + 1):
                            for dir_x, dir_y in landing_directions:
                                landing = (
                                    current[0] + dir_x * radius,
                                    current[1] + dir_y * radius,
                                    current[2] - dz_drop,
                                )
                                landing_vd = self.voxels.get(landing)
                                if (
                                    landing_vd is not None and
                                    landing_vd.voxel_type in (
                                        VoxelType.TRAVERSABLE,
                                        VoxelType.TRAVERSABLE_STAIR,
                                    ) and
                                    not landing_vd.inflated
                                ):
                                    neighbors_iter.append((
                                        dir_x * radius,
                                        dir_y * radius,
                                        -dz_drop,
                                    ))
                    break

            for dx, dy, dz in neighbors_iter:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                # Height constraint
                if neighbor[2] < z_min or neighbor[2] > z_max:
                    continue

                # Already expanded — skip
                if neighbor in closed_set:
                    continue

                # Traversability check: must be in PATHABLE_TYPES
                if neighbor not in self.voxels:
                    continue
                vd_nb = self.voxels[neighbor]
                if not is_navigable_neighbor(vd_nb):
                    continue
                move_cost = (dx * dx + dy * dy + dz * dz) ** 0.5 + voxel_type_penalty(vd_nb)
                if vd_nb.inflated:
                    move_cost *= float(inflated_cost_multiplier)
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None  # No path found

    def path_3d_to_2d(
        self, path_3d: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int]]:
        """Project 3D voxel path to 2D pixel path (for controller)."""
        return [(v[0], v[1]) for v in path_3d]

    def path_3d_to_metric_2d(
        self, path_3d: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Convert 3D voxel path to 2D metric coordinates for controller."""
        points = []
        for v in path_3d:
            x, y, z = self.voxel_to_metric(v[0], v[1], v[2])
            points.append([x, y])
        return np.array(points, dtype=np.float32)

    # ------------------------------------------------------------------
    # Traversable set queries
    # ------------------------------------------------------------------

    def get_traversable_set(self) -> Set[Tuple[int, int, int]]:
        """Return all PATHABLE voxel IDs."""
        return {
            vid for vid, vd in self.voxels.items()
            if vd.voxel_type in PATHABLE_TYPES
        }

    def get_traversable_at_z(self, iz: int) -> Set[Tuple[int, int]]:
        """Return (ix, iy) of PATHABLE voxels at a specific z level."""
        return {
            (vid[0], vid[1]) for vid, vd in self.voxels.items()
            if vid[2] == iz and vd.voxel_type in PATHABLE_TYPES
        }

    def get_occupied_set(self) -> Set[Tuple[int, int, int]]:
        """Return all occupied voxel IDs."""
        return {
            vid for vid, vd in self.voxels.items()
            if vd.voxel_type in OBSTACLE_TYPES
        }

    # ------------------------------------------------------------------
    # Frontier detection (3D) — legacy full-scan fallback
    # ------------------------------------------------------------------

    def detect_frontiers_3d(
        self, agent_voxel: Tuple[int, int, int], search_radius: int = 60
    ) -> List[Tuple[int, int, int]]:
        """
        Full-scan 3D frontier detection (O(N) fallback when frontier_cache is empty).

        Prefer using detect_frontiers_3d_incremental() via Navigator3D which maintains
        a persistent frontier_cache and only rescans newly added TRAVERSABLE voxels.

        Returns frontier voxel IDs sorted by simple openness score (no similarity).
        """
        agent_ix, agent_iy, agent_iz = agent_voxel
        candidates = []

        neighbors_6 = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]

        for vid, vd in self.voxels.items():
            if vd.voxel_type not in FRONTIER_ELIGIBLE:
                continue
            if abs(vid[0] - agent_ix) > search_radius or abs(vid[1] - agent_iy) > search_radius:
                continue

            unexplored_count = 0
            for dx, dy, dz in neighbors_6:
                nb = (vid[0] + dx, vid[1] + dy, vid[2] + dz)
                if nb not in self.voxels and (
                    0 <= nb[0] < self.n_cells and
                    0 <= nb[1] < self.n_cells and
                    0 <= nb[2] < 2 * self.max_z_voxels
                ):
                    unexplored_count += 1

            if unexplored_count == 0:
                continue

            dist = abs(vid[0] - agent_ix) + abs(vid[1] - agent_iy)
            score = unexplored_count * 10.0 - dist * 0.5
            candidates.append((vid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    def clear_new_traversable(self) -> None:
        """
        Clear the incremental traversable tracking set.
        Called by Navigator3D after processing new voxels each frame.
        """
        self.new_traversable_voxels.clear()
        self.frontier_dirty_voxels.clear()

    # ------------------------------------------------------------------
    # Bresenham helper
    # ------------------------------------------------------------------

    @staticmethod
    def _bresenham_2d(
        x0: int, y0: int, x1: int, y1: int
    ) -> List[Tuple[int, int]]:
        """Bresenham 2D line from (x0,y0) to (x1,y1)."""
        pts = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            pts.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    # ------------------------------------------------------------------
    # 2D compatibility properties (for evaluator/logger access)
    # ------------------------------------------------------------------

    @property
    def navigable_map(self) -> np.ndarray:
        """2D projection of PATHABLE voxels at agent's current z-level."""
        nav = np.zeros((self.n_cells, self.n_cells), dtype=np.uint8)
        if self.agent_height_0 is None:
            return nav
        z_origin = self.z_origin if self.z_origin is not None else 0.0
        agent_iz = int(np.floor((self.current_agent_z - z_origin) / self.voxel_size_z)) + self.max_z_voxels
        for vid, vd in self.voxels.items():
            if vid[2] == agent_iz and vd.voxel_type in PATHABLE_TYPES:
                if 0 <= vid[0] < self.n_cells and 0 <= vid[1] < self.n_cells:
                    nav[vid[0], vid[1]] = 1
        return nav

    @property
    def confidence_map(self) -> torch.Tensor:
        """2D projection: max confidence per (ix, iy) across all z levels."""
        conf = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        for vid, vd in self.voxels.items():
            if vd.voxel_type == VoxelType.OCCUPIED and vd.confidence > 0:
                if 0 <= vid[0] < self.n_cells and 0 <= vid[1] < self.n_cells:
                    conf[vid[0], vid[1]] = max(conf[vid[0], vid[1]].item(), vd.confidence)
        return conf

    def reset_checked_map(self) -> None:
        """Compatibility stub for Navigator reset cycle."""
        pass

    def reset_updated_mask(self) -> None:
        """Compatibility stub."""
        pass

    # ------------------------------------------------------------------
    # Stats / debug
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return map statistics."""
        n_occupied = sum(1 for v in self.voxels.values() if v.voxel_type == VoxelType.OCCUPIED)
        n_traversable = sum(1 for v in self.voxels.values() if v.voxel_type == VoxelType.TRAVERSABLE)
        n_traversable_stair = sum(1 for v in self.voxels.values() if v.voxel_type == VoxelType.TRAVERSABLE_STAIR)
        n_traversable_fake = sum(1 for v in self.voxels.values() if v.voxel_type == VoxelType.TRAVERSABLE_FAKE)
        n_stair = sum(1 for v in self.voxels.values() if v.voxel_type == VoxelType.STAIR)
        n_stair_expanded = sum(1 for v in self.voxels.values() if v.voxel_type == VoxelType.STAIR_EXPANDED)
        n_fake_stair = sum(1 for v in self.voxels.values() if v.voxel_type == VoxelType.FAKE_STAIR)
        n_inflated = sum(1 for v in self.voxels.values() if v.inflated)
        return {
            "total_voxels": len(self.voxels),
            "occupied": n_occupied,
            "traversable": n_traversable,
            "traversable_stair": n_traversable_stair,
            "traversable_fake": n_traversable_fake,
            "stair": n_stair,
            "stair_expanded": n_stair_expanded,
            "fake_stair": n_fake_stair,
            "inflated": n_inflated,
            "similarity_cached": len(self.similarity_cache),
            "mapping_mode": self.mapping_mode,
        }
