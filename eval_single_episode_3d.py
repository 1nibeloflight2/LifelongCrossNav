"""
Single-episode evaluation with 3D VoxelMap navigation.

Usage:
  conda activate lcn
  python eval_single_episode_3d.py \
    --episode_id 170 \
    --suffix rerun \
    --config config/mon/eval_multi_conf_floor.yaml \
    --rerun

Outputs:
  - result_3d_ep{id}/voxels_{ep}_{seq}.csv       全局体素数据+当前目标similarity
  - result_3d_ep{id}/trajectory_{ep}_{seq}.csv    轨迹: map_x,map_y,map_z,yaw,hab_x,hab_y,hab_z
  - result_3d_ep{id}/stats_{ep}_{seq}.txt         统计
  - result_3d_ep{id}/state_{ep}.txt               结果状态

Rerun可视化 (--rerun):
  体素6类:
  - world/voxels/occupied           OCCUPIED 障碍物 (红色)
  - world/voxels/traversable        TRAVERSABLE 地板可通行 (绿色)
  - world/voxels/traversable_stair  TRAVERSABLE_STAIR 楼梯可通行 (浅绿色)
  - world/voxels/unsupported        TRAVERSABLE_FAKE 无支撑悬空 (灰色)
  - world/voxels/stair              STAIR SAM确认楼梯 (蓝色)
  - world/voxels/stair_expanded     STAIR_EXPANDED 邻域扩展 (浅蓝色)
  - world/voxels/fake_stair         FAKE_STAIR 误检楼梯固定障碍 (紫红色)
  - world/similarity/high_voxels    当前query高相似度OCCUPIED体素热力点云
  - world/similarity/top_voxels     当前query最高相似度Top体素高亮
  前沿3类:
  - world/frontiers/basic_clusters         基本前沿簇 (绿色系)
  - world/frontiers/basic_nav_targets      基本前沿导航目标 (白色)
  - world/frontiers/stair_geo_clusters     楼梯几何前沿簇 (橙色系)
  - world/frontiers/stair_geo_nav_targets  楼梯几何前沿导航目标 (橙色)
  - world/frontiers/rise_stair_nav_targets 上楼疑似语义前沿/观察点 (紫色系)
  - world/frontiers/rise_stair_bottom      上楼候选估计楼梯底部点 (粉色)
  - world/frontiers/rise_stair_approach    上楼候选观察/对准点 (紫色)
  - world/frontiers/memory_rise_nav_targets  保留的上楼入口memory前沿 (深紫色)
  - world/frontiers/memory_down_nav_targets  保留的下楼入口memory前沿 (深橙色)
  - world/frontiers/stair_semantic         楼梯语义前沿 (红色大球)
  - world/frontiers/live_poi_nav_targets   当前帧目标检测的live approach POI (黄色)
  - world/frontiers/history_poi_nav_targets 历史feature POI (青色)
  其他:
  - world/agent               agent位置
  - world/trajectory          轨迹线
  - world/path                当前规划路径
  - camera/rgb                相机RGB
  - camera/depth              深度图
"""
import argparse
import os
import sys
import io
import cv2
import numpy as np
import rerun as rr

from config import load_eval_config
from eval.actor_3d import MONActor3D
from eval import get_closest_dist
from eval.habitat_multi_evaluator import HabitatMultiEvaluator, Metrics, Result
from mapping.voxel_map import VoxelType
from scipy.spatial.transform import Rotation as R
import habitat_sim


# ---------------------------------------------------------------------------
# stdout 重定向: 将关键终端输出同步写入日志文件
# ---------------------------------------------------------------------------

# 匹配这些前缀的行会被写入日志文件
_LOG_PREFIXES = (
    "[Step",
    "[Rerun]",
    "[Nav3D]",
    "[Navigator3D]",
    "[Actor3D]",
    "  ✓",
    "  ✗",
    "  Seq ",
    "  called_found",
    "  体素CSV",
)


class TeeLogger(io.TextIOBase):
    """将 stdout 同时输出到终端和日志文件，日志文件只保留匹配前缀的行。"""

    def __init__(self, log_path: str, original_stdout):
        self._log_path = log_path
        self._stdout = original_stdout
        self._buf = ""
        self._log_file = open(log_path, "a", encoding="utf-8")

    def write(self, s: str):
        self._stdout.write(s)
        # 缓冲行，遇到换行符时判断是否写入日志
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.lstrip()
            if any(stripped.startswith(p) for p in _LOG_PREFIXES):
                self._log_file.write(line + "\n")
                self._log_file.flush()

    def flush(self):
        self._stdout.flush()
        self._log_file.flush()

    def close(self):
        # flush 残余
        if self._buf.strip():
            stripped = self._buf.lstrip()
            if any(stripped.startswith(p) for p in _LOG_PREFIXES):
                self._log_file.write(self._buf + "\n")
        self._log_file.close()

    def fileno(self):
        return self._stdout.fileno()

    @property
    def encoding(self):
        return self._stdout.encoding


def log_line(msg: str, log_path: str) -> None:
    print(msg)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _ours_metric_to_habitat(pos_ours) -> np.ndarray:
    """Convert Navigator metric xyz to Habitat xyz."""
    x, y, z = pos_ours
    return np.array([-y, z, -x], dtype=np.float32)


def _current_detection_habitat_pos(mapper) -> np.ndarray | None:
    voxel = getattr(mapper, "current_confirmed_detection_voxel", None)
    if voxel is None:
        return None
    return _ours_metric_to_habitat(mapper.one_map.voxel_to_metric(*voxel))


def save_object_found_frame(observations: dict, mapper, out_path: str) -> bool:
    """Save the called_found RGB frame with the current confirmed target box."""
    cached = getattr(mapper, "_objectnav_cached_detection", None)
    cached_image = cached.get("image") if isinstance(cached, dict) else None
    if cached_image is not None:
        image = np.asarray(cached_image[:, :, :3]).copy()
    else:
        rgb = observations.get("rgb")
        if rgb is None:
            return False
        image = np.asarray(rgb[:, :, :3]).copy()
    box = getattr(mapper, "current_confirmed_detection_box", None)
    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = getattr(mapper, "current_confirmed_detection_class", None) or "target"
        score = getattr(mapper, "current_confirmed_detection_score", None)
        text = f"{label} {score:.2f}" if score is not None else str(label)
        cv2.putText(
            image, text, (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        area = getattr(mapper, "current_confirmed_detection_box_area_ratio", None)
        voxels = getattr(mapper, "current_confirmed_detection_mask_voxels", None)
        center = getattr(mapper, "current_confirmed_detection_center_offset", None)
        if area is not None and voxels is not None and center is not None:
            quality_text = f"area={area:.3f} vox={int(voxels)} center={center:.2f}"
            cv2.putText(
                image, quality_text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA,
            )
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return True


def find_episode(evaluator: HabitatMultiEvaluator, episode_id):
    for ep in evaluator.episodes:
        if ep.episode_id == episode_id or str(ep.episode_id) == str(episode_id):
            return ep
    return None


# ---------------------------------------------------------------------------
# Rerun 3D 可视化
# ---------------------------------------------------------------------------

def log_voxels_to_rerun(voxel_map, step: int) -> None:
    """将体素地图以3D点云形式发送到rerun，按类型着色。
    - OCCUPIED: 红色
    - TRAVERSABLE: 绿色（地板可通行）
    - TRAVERSABLE_STAIR: 浅绿色（楼梯可通行）
    - TRAVERSABLE_FAKE: 灰色（无支撑悬空）
    - STAIR: 蓝色（SAM确认）
    - STAIR_EXPANDED: 浅蓝色（邻域扩展）
    - FAKE_STAIR: 紫红色（楼梯误检，固定障碍）
    """
    occ_pts = []
    trav_pts = []
    trav_stair_pts = []
    fake_trav_pts = []
    stair_pts = []
    stair_expanded_pts = []
    fake_stair_pts = []

    for vid, vd in voxel_map.voxels.items():
        x, y, z = voxel_map.voxel_to_metric(vid[0], vid[1], vid[2])
        pt = [x, y, z]
        if vd.voxel_type == VoxelType.OCCUPIED:
            occ_pts.append(pt)
        elif vd.voxel_type == VoxelType.TRAVERSABLE:
                trav_pts.append(pt)
        elif vd.voxel_type == VoxelType.TRAVERSABLE_STAIR:
            trav_stair_pts.append(pt)
        elif vd.voxel_type == VoxelType.TRAVERSABLE_FAKE:
            fake_trav_pts.append(pt)
        elif vd.voxel_type == VoxelType.STAIR:
                stair_pts.append(pt)
        elif vd.voxel_type == VoxelType.STAIR_EXPANDED:
            stair_expanded_pts.append(pt)
        elif vd.voxel_type == VoxelType.FAKE_STAIR:
            fake_stair_pts.append(pt)

    rr.set_time_sequence("step", step)

    if occ_pts:
        rr.log("world/voxels/occupied", rr.Points3D(
            np.array(occ_pts),
            colors=np.full((len(occ_pts), 4), [220, 60, 60, 180], dtype=np.uint8),
            radii=np.full(len(occ_pts), 0.03),
        ))
    if trav_pts:
        rr.log("world/voxels/traversable", rr.Points3D(
            np.array(trav_pts),
            colors=np.full((len(trav_pts), 4), [60, 200, 60, 150], dtype=np.uint8),
            radii=np.full(len(trav_pts), 0.025),
        ))
    if trav_stair_pts:
        rr.log("world/voxels/traversable_stair", rr.Points3D(
            np.array(trav_stair_pts),
            colors=np.full((len(trav_stair_pts), 4), [120, 230, 120, 150], dtype=np.uint8),
            radii=np.full(len(trav_stair_pts), 0.025),
        ))
    else:
        rr.log("world/voxels/traversable_stair", rr.Clear(recursive=False))
    if fake_trav_pts:
        rr.log("world/voxels/unsupported", rr.Points3D(
            np.array(fake_trav_pts),
            colors=np.full((len(fake_trav_pts), 4), [160, 160, 160, 150], dtype=np.uint8),
            radii=np.full(len(fake_trav_pts), 0.025),
        ))
    else:
        rr.log("world/voxels/unsupported", rr.Clear(recursive=False))
    if stair_pts:
        rr.log("world/voxels/stair", rr.Points3D(
            np.array(stair_pts),
            colors=np.full((len(stair_pts), 4), [60, 100, 255, 200], dtype=np.uint8),
            radii=np.full(len(stair_pts), 0.04),
        ))
    else:
        rr.log("world/voxels/stair", rr.Clear(recursive=False))
    if stair_expanded_pts:
        rr.log("world/voxels/stair_expanded", rr.Points3D(
            np.array(stair_expanded_pts),
            colors=np.full((len(stair_expanded_pts), 4), [150, 180, 255, 160], dtype=np.uint8),
            radii=np.full(len(stair_expanded_pts), 0.035),
        ))
    else:
        rr.log("world/voxels/stair_expanded", rr.Clear(recursive=False))
    if fake_stair_pts:
        rr.log("world/voxels/fake_stair", rr.Points3D(
            np.array(fake_stair_pts),
            colors=np.full((len(fake_stair_pts), 4), [190, 70, 190, 180], dtype=np.uint8),
            radii=np.full(len(fake_stair_pts), 0.035),
        ))
    else:
        rr.log("world/voxels/fake_stair", rr.Clear(recursive=False))


def _similarity_color(value: float) -> list:
    """Blue -> cyan -> yellow -> red heat color for value in [0, 1]."""
    v = float(np.clip(value, 0.0, 1.0))
    if v < 0.33:
        t = v / 0.33
        return [0, int(80 + 175 * t), 255, 230]
    if v < 0.66:
        t = (v - 0.33) / 0.33
        return [int(255 * t), 255, int(255 * (1.0 - t)), 235]
    t = (v - 0.66) / 0.34
    return [255, int(255 * (1.0 - t)), 0, 245]


def log_similarity_to_rerun(voxel_map, step: int, max_points: int = 2500) -> None:
    """Log high-similarity OCCUPIED voxels as a 3D heat point cloud."""
    rr.set_time_sequence("step", step)
    sim_items = []
    for vid, score in voxel_map.similarity_cache.items():
        vd = voxel_map.voxels.get(vid)
        if vd is None or vd.voxel_type != VoxelType.OCCUPIED:
            continue
        sim_items.append((vid, float(score)))

    if not sim_items:
        rr.log("world/similarity/high_voxels", rr.Clear(recursive=False))
        rr.log("world/similarity/top_voxels", rr.Clear(recursive=False))
        return

    scores = np.array([s for _, s in sim_items], dtype=np.float32)
    p90 = float(np.percentile(scores, 90.0))
    p95 = float(np.percentile(scores, 95.0))
    p99 = float(np.percentile(scores, 99.0))
    threshold = max(p90, 0.0)
    selected = [(vid, s) for vid, s in sim_items if s >= threshold]
    selected.sort(key=lambda x: x[1], reverse=True)
    if len(selected) > max_points:
        selected = selected[:max_points]

    denom = max(p99 - threshold, 1e-6)
    pts = []
    colors = []
    radii = []
    for vid, score in selected:
        x, y, z = voxel_map.voxel_to_metric(vid[0], vid[1], vid[2])
        pts.append([x, y, z])
        norm = (score - threshold) / denom
        colors.append(_similarity_color(norm))
        radii.append(0.035 + 0.055 * float(np.clip(norm, 0.0, 1.0)))

    if pts:
        rr.log("world/similarity/high_voxels", rr.Points3D(
            np.array(pts),
            colors=np.array(colors, dtype=np.uint8),
            radii=np.array(radii, dtype=np.float32),
        ))
    else:
        rr.log("world/similarity/high_voxels", rr.Clear(recursive=False))

    top = selected[:50]
    if top:
        top_pts = [voxel_map.voxel_to_metric(*vid) for vid, _ in top]
        rr.log("world/similarity/top_voxels", rr.Points3D(
            np.array(top_pts),
            colors=np.full((len(top_pts), 4), [255, 40, 0, 255], dtype=np.uint8),
            radii=np.full(len(top_pts), 0.11),
        ))
    else:
        rr.log("world/similarity/top_voxels", rr.Clear(recursive=False))

    rr.log("status/similarity", rr.TextLog(
        f"similarity high_voxels={len(selected)} p90={p90:.4f} p95={p95:.4f} "
        f"p99={p99:.4f} max={float(scores.max()):.4f}"
    ))


def log_agent_to_rerun(pose: np.ndarray, step: int, voxel_map=None) -> None:
    """记录agent位置和朝向。显示相机位置(黄色)和agent_voxel位置(橙色)。"""
    rr.set_time_sequence("step", step)
    pos_3d = np.array([[pose[0], pose[1], pose[2]]])
    rr.log("world/agent", rr.Points3D(
        pos_3d,
        colors=np.array([[255, 255, 0, 255]], dtype=np.uint8),
        radii=[0.08],
    ))
    # 显示 agent_voxel 的世界坐标位置（用于2D射线/A*计算的位置）
    if voxel_map is not None:
        z_offset = voxel_map.camera_height * 4.0 / 5.0
        voxel_pos = np.array([[pose[0], pose[1], pose[2] - z_offset]])
        rr.log("world/agent_voxel", rr.Points3D(
            voxel_pos,
            colors=np.array([[255, 140, 0, 255]], dtype=np.uint8),  # 橙色
            radii=[0.06],
        ))


def log_trajectory_to_rerun(poses: list, step: int, voxel_map=None) -> None:
    """记录完整轨迹线（使用agent_voxel高度，与A*一致）。"""
    if len(poses) < 2:
        return
    rr.set_time_sequence("step", step)
    # camera_z - ray_z = camera_height * 4/5 (固定偏移)
    if voxel_map is not None:
        z_offset = voxel_map.camera_height * 4.0 / 5.0
    else:
        z_offset = 0.0
    pts = np.array([[p[0], p[1], p[2] - z_offset] for p in poses])
    rr.log("world/trajectory", rr.LineStrips3D(
        [pts],
        colors=np.array([[255, 200, 0, 200]], dtype=np.uint8),
    ))


def log_path_to_rerun(voxel_map, path_3d, step: int) -> None:
    """记录当前规划路径。"""
    if not path_3d or len(path_3d) < 2:
        rr.set_time_sequence("step", step)
        rr.log("world/path", rr.Clear(recursive=False))
        return
    rr.set_time_sequence("step", step)
    pts = []
    for vid in path_3d:
        x, y, z = voxel_map.voxel_to_metric(vid[0], vid[1], vid[2])
        pts.append([x, y, z])
    rr.log("world/path", rr.LineStrips3D(
        [np.array(pts)],
        colors=np.array([[0, 255, 255, 255]], dtype=np.uint8),
    ))


def log_frontier_clusters_to_rerun(navigator, step: int) -> None:
    """
    可视化3类前沿:
      1. 基本前沿（绿色系）: TRAVERSABLE 邻接未知 → 探索边界
      2. 楼梯几何前沿（橙色系）: TRAVERSABLE 邻接 TRAVERSABLE_FAKE → 下楼梯候选
      3. 上楼疑似语义前沿（紫色系）: RiseStairFrontier3D/view_hint → 上楼观察点
      4. 楼梯语义前沿（红色大球）: StairSemanticFrontier3D → 确认后的楼梯目标
    
    每类前沿用不同颜色，nav_target 用对应色大球标记。
    """
    rr.set_time_sequence("step", step)

    clusters = getattr(navigator, 'frontier_clusters', [])
    voxel_map = navigator.one_map

    # --- 基本前沿（非 is_stair_frontier）---
    basic_pts = []
    basic_colors = []
    basic_nav_pts = []
    # --- 楼梯几何前沿（is_stair_frontier=True）---
    stair_geo_pts = []
    stair_geo_colors = []
    stair_geo_nav_pts = []

    # Color palettes
    basic_palette = [
        [100, 255, 100, 200],  # lime
        [0, 255, 200, 200],    # cyan
        [200, 255, 100, 200],  # yellow-green
        [100, 200, 255, 200],  # sky blue
        [150, 255, 150, 200],  # light green
    ]
    stair_geo_palette = [
        [255, 180, 0, 200],    # orange
        [255, 128, 50, 200],   # dark orange
        [255, 200, 80, 200],   # gold
        [255, 150, 100, 200],  # salmon
    ]

    basic_ci = 0
    stair_ci = 0
    for cluster in clusters[:20]:
        if cluster.is_stair_frontier:
            color = stair_geo_palette[stair_ci % len(stair_geo_palette)]
            stair_ci += 1
            for vid in list(cluster.cluster_voxels)[:150]:
                x, y, z = voxel_map.voxel_to_metric(vid[0], vid[1], vid[2])
                stair_geo_pts.append([x, y, z])
                stair_geo_colors.append(color)
            nx, ny, nz = voxel_map.voxel_to_metric(
                cluster.nav_voxel_id[0], cluster.nav_voxel_id[1], cluster.nav_voxel_id[2]
            )
            stair_geo_nav_pts.append([nx, ny, nz])
        else:
            color = basic_palette[basic_ci % len(basic_palette)]
            basic_ci += 1
            for vid in list(cluster.cluster_voxels)[:150]:
                x, y, z = voxel_map.voxel_to_metric(vid[0], vid[1], vid[2])
                basic_pts.append([x, y, z])
                basic_colors.append(color)
            nx, ny, nz = voxel_map.voxel_to_metric(
                cluster.nav_voxel_id[0], cluster.nav_voxel_id[1], cluster.nav_voxel_id[2]
            )
            basic_nav_pts.append([nx, ny, nz])

    # Log basic frontiers
    if basic_pts:
        rr.log("world/frontiers/basic_clusters", rr.Points3D(
            np.array(basic_pts),
            colors=np.array(basic_colors, dtype=np.uint8),
            radii=np.full(len(basic_pts), 0.04),
        ))
    else:
        rr.log("world/frontiers/basic_clusters", rr.Clear(recursive=False))
    if basic_nav_pts:
        rr.log("world/frontiers/basic_nav_targets", rr.Points3D(
            np.array(basic_nav_pts),
            colors=np.full((len(basic_nav_pts), 4), [255, 255, 255, 255], dtype=np.uint8),
            radii=np.full(len(basic_nav_pts), 0.12),
        ))
    else:
        rr.log("world/frontiers/basic_nav_targets", rr.Clear(recursive=False))

    # Log stair geometric frontiers
    if stair_geo_pts:
        rr.log("world/frontiers/stair_geo_clusters", rr.Points3D(
            np.array(stair_geo_pts),
            colors=np.array(stair_geo_colors, dtype=np.uint8),
            radii=np.full(len(stair_geo_pts), 0.045),
        ))
    else:
        rr.log("world/frontiers/stair_geo_clusters", rr.Clear(recursive=False))
    if stair_geo_nav_pts:
        rr.log("world/frontiers/stair_geo_nav_targets", rr.Points3D(
            np.array(stair_geo_nav_pts),
            colors=np.full((len(stair_geo_nav_pts), 4), [255, 160, 0, 255], dtype=np.uint8),
            radii=np.full(len(stair_geo_nav_pts), 0.14),
        ))
    else:
        rr.log("world/frontiers/stair_geo_nav_targets", rr.Clear(recursive=False))

    # Log upstairs semantic proposal / view-hint frontiers.
    rise_nav_pts = []
    rise_nav_colors = []
    rise_bottom_pts = []
    rise_approach_pts = []
    for nav_vid, cand in list(getattr(navigator, "rise_stair_candidates", {}).items()):
        if not cand.get("active", False):
            continue
        goal_vid = tuple(cand.get("nav_voxel_id", nav_vid))
        try:
            nx, ny, nz = voxel_map.voxel_to_metric(goal_vid[0], goal_vid[1], goal_vid[2])
        except Exception:
            continue
        is_view_hint = bool(cand.get("view_hint", False))
        rise_nav_pts.append([nx, ny, nz])
        rise_nav_colors.append([210, 70, 255, 255] if is_view_hint else [90, 80, 255, 255])

        bottom_xy = cand.get("bottom_xy")
        if bottom_xy is not None:
            rise_bottom_pts.append([float(bottom_xy[0]), float(bottom_xy[1]), nz])
        approach_xy = cand.get("approach_xy")
        if approach_xy is not None:
            rise_approach_pts.append([float(approach_xy[0]), float(approach_xy[1]), nz])

    if rise_nav_pts:
        rr.log("world/frontiers/rise_stair_nav_targets", rr.Points3D(
            np.array(rise_nav_pts),
            colors=np.array(rise_nav_colors, dtype=np.uint8),
            radii=np.full(len(rise_nav_pts), 0.18),
        ))
    else:
        rr.log("world/frontiers/rise_stair_nav_targets", rr.Clear(recursive=False))

    if rise_bottom_pts:
        rr.log("world/frontiers/rise_stair_bottom", rr.Points3D(
            np.array(rise_bottom_pts),
            colors=np.full((len(rise_bottom_pts), 4), [255, 80, 220, 220], dtype=np.uint8),
            radii=np.full(len(rise_bottom_pts), 0.10),
        ))
    else:
        rr.log("world/frontiers/rise_stair_bottom", rr.Clear(recursive=False))

    if rise_approach_pts:
        rr.log("world/frontiers/rise_stair_approach", rr.Points3D(
            np.array(rise_approach_pts),
            colors=np.full((len(rise_approach_pts), 4), [120, 80, 255, 220], dtype=np.uint8),
            radii=np.full(len(rise_approach_pts), 0.10),
        ))
    else:
        rr.log("world/frontiers/rise_stair_approach", rr.Clear(recursive=False))

    # Log persisted active stair frontier memories. These are saved upstairs /
    # downstairs entrances that can be restored as explore-mode stair frontiers
    # after ordinary frontiers are exhausted.
    memory_rise_pts = []
    memory_down_pts = []
    for mem in getattr(navigator, "stair_frontier_memory", {}).values():
        if mem.get("status") != "active":
            continue
        entry = mem.get("entry_voxel")
        if entry is None:
            continue
        try:
            ex, ey, ez = voxel_map.voxel_to_metric(int(entry[0]), int(entry[1]), int(entry[2]))
        except Exception:
            continue
        if mem.get("direction") == "up":
            memory_rise_pts.append([ex, ey, ez])
        elif mem.get("direction") == "down":
            memory_down_pts.append([ex, ey, ez])

    if memory_rise_pts:
        rr.log("world/frontiers/memory_rise_nav_targets", rr.Points3D(
            np.array(memory_rise_pts),
            colors=np.full((len(memory_rise_pts), 4), [170, 40, 255, 255], dtype=np.uint8),
            radii=np.full(len(memory_rise_pts), 0.22),
        ))
    else:
        rr.log("world/frontiers/memory_rise_nav_targets", rr.Clear(recursive=False))

    if memory_down_pts:
        rr.log("world/frontiers/memory_down_nav_targets", rr.Points3D(
            np.array(memory_down_pts),
            colors=np.full((len(memory_down_pts), 4), [255, 110, 0, 255], dtype=np.uint8),
            radii=np.full(len(memory_down_pts), 0.22),
        ))
    else:
        rr.log("world/frontiers/memory_down_nav_targets", rr.Clear(recursive=False))

    # Log stair semantic frontier (highest priority, single target)
    stair_sem_frontier = getattr(navigator, '_stair_sem_frontier', None)
    if stair_sem_frontier is not None:
        sf_vid = stair_sem_frontier.nav_voxel_id
        sx, sy, sz = voxel_map.voxel_to_metric(sf_vid[0], sf_vid[1], sf_vid[2])
        rr.log("world/frontiers/stair_semantic", rr.Points3D(
            np.array([[sx, sy, sz]]),
            colors=np.array([[255, 50, 50, 255]], dtype=np.uint8),  # bright red
            radii=np.full(1, 0.20),  # largest marker
        ))
    else:
        rr.log("world/frontiers/stair_semantic", rr.Clear(recursive=False))

    objectnav_goal = getattr(navigator, "objectnav_goal_voxel", None)
    if objectnav_goal is not None:
        ox, oy, oz = voxel_map.voxel_to_metric(*objectnav_goal)
        rr.log("world/frontiers/objectnav_goal", rr.Points3D(
            np.array([[ox, oy, oz]]),
            colors=np.array([[40, 255, 80, 255]], dtype=np.uint8),
            radii=np.full(1, 0.24),
        ))
    else:
        rr.log("world/frontiers/objectnav_goal", rr.Clear(recursive=False))

    # Log POI targets and evidence centers in the same 3D world view.
    history_pois = [
        poi for poi in getattr(navigator, "history_pois", [])
        if not getattr(navigator, "_is_poi_cluster_blacklisted", lambda p: False)(poi)
    ]
    for poi_name, poi_list, nav_color, center_color in (
        (
            "live_poi",
            list(getattr(navigator, "live_pois", {}).values()),
            [255, 210, 40, 255],
            [255, 120, 20, 230],
        ),
        (
            "history_poi",
            history_pois,
            [60, 220, 255, 255],
            [40, 120, 255, 230],
        ),
    ):
        rr.log(f"world/frontiers/{poi_name}_nav_targets", rr.Clear(recursive=False))
        rr.log(f"world/frontiers/{poi_name}_centers", rr.Clear(recursive=False))
        rr.log(f"world/frontiers/{poi_name}_bbox_corners", rr.Clear(recursive=False))

        nav_pts = []
        center_pts = []
        bbox_pts = []
        nav_radii = []
        center_radii = []
        for poi in poi_list:
            try:
                nav_pts.append(voxel_map.voxel_to_metric(*poi.nav_voxel_id))
                center_pts.append(voxel_map.voxel_to_metric(*poi.center_voxel))
                score = float(getattr(poi, "evidence_score", poi.get_score()))
                nav_radii.append(0.13 + min(0.12, max(0.0, score)))
                center_radii.append(0.07 + min(0.08, max(0.0, score) * 0.5))
                bmin = getattr(poi, "bbox_min", None)
                bmax = getattr(poi, "bbox_max", None)
                if bmin is not None and bmax is not None:
                    corners = [
                        (bmin[0], bmin[1], bmin[2]), (bmin[0], bmin[1], bmax[2]),
                        (bmin[0], bmax[1], bmin[2]), (bmin[0], bmax[1], bmax[2]),
                        (bmax[0], bmin[1], bmin[2]), (bmax[0], bmin[1], bmax[2]),
                        (bmax[0], bmax[1], bmin[2]), (bmax[0], bmax[1], bmax[2]),
                    ]
                    bbox_pts.extend([voxel_map.voxel_to_metric(*c) for c in corners])
            except Exception:
                continue

        if nav_pts:
            rr.log(f"world/frontiers/{poi_name}_nav_targets", rr.Points3D(
                np.array(nav_pts),
                colors=np.full((len(nav_pts), 4), nav_color, dtype=np.uint8),
                radii=np.array(nav_radii, dtype=np.float32),
            ))

        if center_pts:
            rr.log(f"world/frontiers/{poi_name}_centers", rr.Points3D(
                np.array(center_pts),
                colors=np.full((len(center_pts), 4), center_color, dtype=np.uint8),
                radii=np.array(center_radii, dtype=np.float32),
            ))

        if bbox_pts:
            rr.log(f"world/frontiers/{poi_name}_bbox_corners", rr.Points3D(
                np.array(bbox_pts),
                colors=np.full((len(bbox_pts), 4), center_color, dtype=np.uint8),
                radii=np.full(len(bbox_pts), 0.035),
            ))


def log_camera_to_rerun(observations: dict, step: int) -> None:
    """记录相机RGB和深度。"""
    rr.set_time_sequence("step", step)
    try:
        rr.log("camera/rgb", rr.Image(observations["rgb"][:, :, :3]).compress(jpeg_quality=60))
    except Exception:
        pass
    try:
        depth = observations.get("depth")
        if depth is not None:
            dmin, dmax = float(depth.min()), float(depth.max())
            if dmax > dmin:
                depth_norm = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                rr.log("camera/depth", rr.Image(depth_norm))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 体素数据导出
# ---------------------------------------------------------------------------

def save_voxel_csv(voxel_map, out_path: str) -> None:
    """
    导出全部体素为CSV。
    列: ix,iy,iz,x,y,z,type,inflated,confidence,similarity
    type: OCCUPIED=1, TRAVERSABLE=2, TRAVERSABLE_STAIR=3, TRAVERSABLE_FAKE=4, STAIR=5, STAIR_EXPANDED=6, FAKE_STAIR=7
    """
    lines = ["ix,iy,iz,x,y,z,type,inflated,confidence,similarity"]
    for vid, vd in voxel_map.voxels.items():
        x, y, z = voxel_map.voxel_to_metric(vid[0], vid[1], vid[2])
        sim = voxel_map.similarity_cache.get(vid, 0.0)
        inflated = int(getattr(vd, 'inflated', False))
        lines.append(
            f"{vid[0]},{vid[1]},{vid[2]},{x:.4f},{y:.4f},{z:.4f},"
            f"{vd.voxel_type.value},{inflated},{vd.confidence:.4f},{sim:.4f}"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  体素CSV已保存: {out_path} ({len(lines)-1} voxels)")


def save_stats(voxel_map, poses, result, out_path: str) -> None:
    """导出统计摘要。"""
    stats = voxel_map.stats()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== VoxelMap Statistics ===\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\n=== Navigation ===\n")
        f.write(f"total_steps: {len(poses)}\n")
        f.write(f"result: {result.name}\n")
        if len(poses) > 0:
            start = poses[0][:3]
            end = poses[-1][:3]
            travel_dist = sum(
                np.linalg.norm(np.array(poses[i][:3]) - np.array(poses[i-1][:3]))
                for i in range(1, len(poses))
            )
            f.write(f"start_map_pos: [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]\n")
            f.write(f"end_map_pos: [{end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f}]\n")
            if len(poses[0]) >= 7:
                start_hab = poses[0][4:7]
                end_hab = poses[-1][4:7]
                f.write(f"start_hab_pos: [{start_hab[0]:.3f}, {start_hab[1]:.3f}, {start_hab[2]:.3f}]\n")
                f.write(f"end_hab_pos: [{end_hab[0]:.3f}, {end_hab[1]:.3f}, {end_hab[2]:.3f}]\n")
            f.write(f"travel_distance: {travel_dist:.3f}m\n")
            # Z range (楼层范围)
            z_vals = [p[2] for p in poses]
            f.write(f"z_range: [{min(z_vals):.3f}, {max(z_vals):.3f}]\n")


# ---------------------------------------------------------------------------
# 主评估循环
# ---------------------------------------------------------------------------

def run_single_episode(evaluator: HabitatMultiEvaluator, episode_id: int, suffix: str,
                       log_path: str, out_dir: str, use_rerun: bool,
                       rerun_voxel_interval: int = 20,
                       episode_label: str | None = None,
                       save_stats_file: bool = True) -> Metrics:
    episode = find_episode(evaluator, episode_id)
    if episode is None:
        raise SystemExit(f"Episode {episode_id} not found in dataset.")
    file_ep = episode_label or str(episode_id)

    # Batch mode reuses one Python process, whereas the single-episode entry
    # starts a fresh process. Reset the algorithmic RNG per episode so random
    # point subsampling (currently used by stair processing) is independent of
    # which episodes happened to run before it.
    try:
        episode_seed = int(episode.episode_id) & 0xFFFFFFFF
    except (TypeError, ValueError):
        episode_seed = sum(ord(ch) for ch in str(episode.episode_id)) & 0xFFFFFFFF
    np.random.seed(episode_seed)

    results = Metrics(episode.episode_id)

    if evaluator.sim is None or not evaluator.sim.curr_scene_name in episode.scene_id:
        evaluator.load_scene(episode.scene_id)

    evaluator.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
    evaluator.actor.reset()
    log_line(
        "[Eval3D] Cross-target memory reset: "
        f"{'enabled' if evaluator.reset_map_between_targets else 'disabled'}",
        log_path,
    )

    sequence_id = 0
    current_obj = episode.obj_sequence[sequence_id]
    evaluator.actor.set_query(current_obj, allow_poi=False)

    log_line(f"Start episode {episode.episode_id}, scene={episode.scene_id}, "
             f"goals={episode.obj_sequence}", log_path)

    poses = []
    not_failed = True
    episode_step = 0
    while not_failed and sequence_id < len(episode.obj_sequence):
        if episode_step >= evaluator.max_steps:
            result = Result.FAILURE_OOT
            results.add_sequence(np.array(poses), result, episode.obj_sequence[sequence_id])
            log_line(
                f"  Seq {sequence_id} ({episode.obj_sequence[sequence_id]}): "
                f"{result.name}, episode_steps={episode_step}",
                log_path,
            )
            break
        steps = 0
        running = True
        no_goal_frames = 0
        while episode_step < evaluator.max_steps and running:
            rerun_step = episode_step
            observations = evaluator.sim.get_sensor_observations()
            observations["state"] = evaluator.sim.get_agent(0).get_state()
            agent_pos = observations["state"].position

            # 计算pose: Habitat (x_right, y_up, z_backward) -> 我们的 (x_fwd, y_left, z_up)
            # Camera is at y=+0.88m above agent origin, so add camera height to z.
            CAMERA_HEIGHT = 0.88
            sensor_state = observations["state"].sensor_states.get("depth")
            if sensor_state is not None:
                agent_cam_pos_hab = np.asarray(sensor_state.position, dtype=np.float32)
            else:
                agent_cam_pos_hab = np.array(agent_pos, dtype=np.float32).copy()
                agent_cam_pos_hab[1] += CAMERA_HEIGHT
            pose = np.zeros((7,))
            pose[0] = -agent_pos[2]                   # x = -z_hab (forward)
            pose[1] = -agent_pos[0]                   # y = -x_hab (left)
            pose[2] = agent_cam_pos_hab[1]            # z = camera y_hab
            pose[4] = agent_pos[0]                    # Habitat x
            pose[5] = agent_pos[1]                    # Habitat y
            pose[6] = agent_pos[2]                    # Habitat z

            orientation = observations["state"].rotation
            r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
            yaw, _, _ = r.as_euler("yxz")
            pose[3] = yaw
            poses.append(pose)

            # === Rerun 实时可视化 ===
            if use_rerun:
                log_camera_to_rerun(observations, rerun_step)
                log_agent_to_rerun(pose, rerun_step, voxel_map=evaluator.actor.mapper.one_map)

                if steps % 10 == 0:
                    log_trajectory_to_rerun(poses, rerun_step, voxel_map=evaluator.actor.mapper.one_map)

                # 每 rerun_voxel_interval 帧刷新一次体素点云
                if steps % rerun_voxel_interval == 0:
                    log_voxels_to_rerun(evaluator.actor.mapper.one_map, rerun_step)
                    log_similarity_to_rerun(evaluator.actor.mapper.one_map, rerun_step)

                # 记录当前路径
                if steps % 5 == 0:
                    log_path_to_rerun(
                        evaluator.actor.mapper.one_map,
                        evaluator.actor.mapper.path,
                        rerun_step,
                    )

                # 前沿簇可视化
                if steps % rerun_voxel_interval == 0:
                    log_frontier_clusters_to_rerun(evaluator.actor.mapper, rerun_step)

                # 文字状态
                if steps % 50 == 0:
                    stats = evaluator.actor.mapper.one_map.stats()
                    rr.set_time_sequence("step", rerun_step)
                    rr.log("status/voxels", rr.TextLog(
                        f"episode_step={rerun_step} target_step={steps} "
                        f"seq={sequence_id} total={stats['total_voxels']} "
                        f"OCC={stats['occupied']} TRAV={stats['traversable']} "
                        f"T_STAIR={stats['traversable_stair']} T_FAKE={stats['traversable_fake']} "
                        f"STAIR={stats['stair']} S_EXP={stats['stair_expanded']} "
                        f"F_STAIR={stats.get('fake_stair', 0)} "
                        f"inflated={stats['inflated']} mode={stats['mapping_mode']}"
                    ))
                    # 前沿分类统计
                    n_basic = sum(1 for c in evaluator.actor.mapper.frontier_clusters if not c.is_stair_frontier)
                    n_stair_geo = sum(1 for c in evaluator.actor.mapper.frontier_clusters if c.is_stair_frontier)
                    rise_candidates = getattr(evaluator.actor.mapper, "rise_stair_candidates", {})
                    n_rise = sum(1 for c in rise_candidates.values() if c.get("active", False))
                    n_rise_hint = sum(
                        1 for c in rise_candidates.values()
                        if c.get("active", False) and c.get("view_hint", False)
                    )
                    has_stair_sem = evaluator.actor.mapper._stair_sem_frontier is not None
                    rr.log("status/target", rr.TextLog(
                        f"target={current_obj} detected={evaluator.actor.mapper.object_detected} "
                        f"nav_goals={len(evaluator.actor.mapper.nav_goals)} "
                        f"frontiers: basic={n_basic} stair_geo={n_stair_geo} "
                        f"rise={n_rise} rise_hint={n_rise_hint} "
                        f"stair_sem={'Y' if has_stair_sem else 'N'}"
                    ))

            # === 执行动作 ===
            initialization_action = evaluator.actor._is_initializing()
            action, called_found = evaluator.actor.act(observations)
            initialization_action = (
                initialization_action or
                getattr(evaluator.actor.mapper, "initializing", False)
            )

            # 判断当前模式
            is_obj_detected = evaluator.actor.mapper.object_detected
            is_stair_sem = evaluator.actor.mapper.stair_sem_mode
            if initialization_action:
                init_kind = getattr(evaluator.actor, "initialization_kind", "init")
                mode_str = {
                    "reinit": "Reinit",
                    "target_init": "TargetInit",
                }.get(init_kind, "Init")
            elif is_stair_sem:
                direction = getattr(evaluator.actor.mapper, "_stair_sem_direction", "down")
                direction = "ascent" if direction in ("up", "rise", "ascent") else "descend"
                mode_str = f"StairSem-{direction}"
            elif getattr(evaluator.actor.mapper, "navigation_mode", "explore") == "navigate":
                nav_goal = getattr(evaluator.actor.mapper, "_navigate_goal", None)
                poi_type = str(getattr(nav_goal, "poi_type", ""))
                if "history" in poi_type:
                    mode_str = "Navigate-historypoi"
                elif "live" in poi_type:
                    mode_str = "Navigate-livepoi"
                else:
                    mode_str = "Navigate"
            elif is_obj_detected:
                mode_str = "ObjNav"
            else:
                mode_str = "Explore"
                goal = evaluator.actor.mapper.last_nav_goal
                goal_kind = goal.__class__.__name__ if goal is not None else ""
                if goal_kind == "RiseStairFrontier3D":
                    mode_str = "Explore-ascent"
                elif getattr(goal, "is_stair_frontier", False):
                    mode_str = "Explore-descend"
                elif goal_kind == "FrontierCluster3D":
                    mode_str = "Explore-basic"
            action_str = action.get('discrete', 'unknown')

            # 获取检测置信度
            last_dets = evaluator.actor.mapper.last_detections
            target_scores = last_dets.get("target", {}).get("scores", [])
            stair_scores = last_dets.get("stair", {}).get("scores", [])
            target_conf_str = f"T={max(target_scores):.2f}" if target_scores else "T=---"
            stair_conf_str = f"S={max(stair_scores):.2f}" if stair_scores else "S=---"

            # 额外状态信息
            extra_info = ""
            if evaluator.actor.mapper._fake_frontier_target is not None:
                extra_info += " [FakeFrontier]"
            if evaluator.actor.mapper._fake_frontier_pitched:
                extra_info += " [Pitched-WaitingStair]"
            if evaluator.actor.mapper._stair_sem_force_forward > 0:
                extra_info += f" [ForceF={evaluator.actor.mapper._stair_sem_force_forward}]"

            # 获取 agent 世界坐标和当前 goal 坐标 (使用 ray_z 高度，与A*一致)
            _one_map = evaluator.actor.mapper.one_map
            _z_offset = _one_map.camera_height * 4.0 / 5.0
            _agent_z = pose[2] - _z_offset
            agent_xyz_str = f"A=[{pose[0]:.2f},{pose[1]:.2f},{_agent_z:.2f}]"
            goal_str = ""
            if evaluator.actor.mapper.last_nav_goal is not None:
                gv = evaluator.actor.mapper.last_nav_goal.nav_voxel_id
                gw = evaluator.actor.mapper.one_map.voxel_to_metric(*gv)
                goal_str = f" G=[{gw[0]:.2f},{gw[1]:.2f},{gw[2]:.2f}]"

            # 终端打印每步动作详情 + 模式 + 置信度 + 额外状态
            print(f"[Step {steps:>4d}/{episode_step:>4d}] {mode_str:<10s} | action={action_str:<14s} | {target_conf_str} {stair_conf_str} {agent_xyz_str}{goal_str}{extra_info}")

            # Rerun: 在RGB图像内叠加文字显示动作和模式 + 检测框
            if use_rerun:
                rgb_overlay = observations["rgb"][:, :, :3].copy()

                # 画检测框：目标物体(绿色) + 楼梯(蓝色)
                last_dets = evaluator.actor.mapper.last_detections
                n_target = len(last_dets.get("target", {}).get("boxes", []))
                n_stair = len(last_dets.get("stair", {}).get("boxes", []))
                print(f"[Rerun] Drawing boxes: target={n_target}, stair={n_stair}")

                # 目标物体 - 绿色 BGR=(0, 255, 0)
                for box, cls, score in zip(
                    last_dets.get("target", {}).get("boxes", []),
                    last_dets.get("target", {}).get("classes", []),
                    last_dets.get("target", {}).get("scores", []),
                ):
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(rgb_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"[T] {cls} {score:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(rgb_overlay, (x1, y1 - lh - 6), (x1 + lw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(rgb_overlay, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # 楼梯 - SegFormer mask 半透明蓝色 + 组件框
                for mask in last_dets.get("stair", {}).get("masks", []):
                    mask_arr = np.asarray(mask, dtype=bool)
                    if mask_arr.shape != rgb_overlay.shape[:2]:
                        continue
                    stair_color = np.array([255, 100, 0], dtype=np.float32)
                    rgb_overlay[mask_arr] = (
                        0.55 * rgb_overlay[mask_arr].astype(np.float32) +
                        0.45 * stair_color
                    ).astype(np.uint8)
                for box, cls, score in zip(
                    last_dets.get("stair", {}).get("boxes", []),
                    last_dets.get("stair", {}).get("classes", []),
                    last_dets.get("stair", {}).get("scores", []),
                ):
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(rgb_overlay, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    label = f"[S] {cls} {score:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(rgb_overlay, (x1, y1 - lh - 6), (x1 + lw + 4, y1), (255, 100, 0), -1)
                    cv2.putText(rgb_overlay, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # 叠加文字: 左上角 Action, 右上角 Mode
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness_bg = 3
                thickness_fg = 1
                img_w = rgb_overlay.shape[1]
                # 左上角: Action
                action_text = f"Action: {action_str}"
                (_, th), _ = cv2.getTextSize(action_text, font, font_scale, thickness_bg)
                action_y = th + 15
                cv2.putText(rgb_overlay, action_text, (10, action_y),
                            font, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
                cv2.putText(rgb_overlay, action_text, (10, action_y),
                            font, font_scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
                # 右上角: Mode
                mode_text = f"Mode: {mode_str}"
                (mw, _), _ = cv2.getTextSize(mode_text, font, font_scale, thickness_bg)
                mode_x = img_w - mw - 10
                cv2.putText(rgb_overlay, mode_text, (mode_x, action_y),
                            font, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
                cv2.putText(rgb_overlay, mode_text, (mode_x, action_y),
                            font, font_scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
                rr.set_time_sequence("step", rerun_step)
                rr.log("camera/rgb", rr.Image(rgb_overlay).compress(jpeg_quality=60))

            evaluator.execute_action(action)

            if steps % 100 == 0:
                log_line(
                    f"Step {steps}, obj={current_obj}, ep={episode.episode_id}, "
                    f"agent=[{agent_pos[0]:.3f}, {agent_pos[1]:.3f}, {agent_pos[2]:.3f}], "
                    f"voxels={len(evaluator.actor.mapper.one_map.voxels)}",
                    log_path,
                )

            steps += 1
            episode_step += 1
            stuck_now = False
            if not initialization_action and len(poses) >= 30:
                pos_delta = np.linalg.norm(np.array(poses[-1][:2]) - np.array(poses[-30][:2]))
                recent_yaws = np.array([p[3] for p in poses[-30:]], dtype=np.float32)
                yaw_steps = (np.diff(recent_yaws) + np.pi) % (2.0 * np.pi) - np.pi
                yaw_motion = float(np.sum(np.abs(yaw_steps)))
                stuck_now = pos_delta < 0.05 and yaw_motion < np.deg2rad(10.0)
            if not initialization_action and not stuck_now and len(poses) >= 100:
                recent_xy = np.array([p[:2] for p in poses[-100:]], dtype=np.float32)
                xy_span = float(np.linalg.norm(recent_xy.max(axis=0) - recent_xy.min(axis=0)))
                stuck_now = xy_span < 0.05
            no_goal_now = (
                not initialization_action and
                getattr(evaluator.actor.mapper, "_no_goal_available", False) and
                len(evaluator.actor.mapper.nav_goals) == 0 and
                evaluator.actor.mapper.path is None and
                getattr(evaluator.actor.mapper, "navigation_mode", "explore") != "navigate" and
                not evaluator.actor.mapper.object_detected and
                not evaluator.actor.mapper.stair_sem_mode
            )
            no_goal_frames = no_goal_frames + 1 if no_goal_now else 0

            if called_found or stuck_now or no_goal_frames >= 5 or episode_step >= evaluator.max_steps:
                running = False
                result = Result.FAILURE_OOT

                if called_found:
                    suffix_text = f"_{suffix}" if suffix else ""
                    found_frame_path = os.path.join(
                        out_dir, f"object_found_{file_ep}_{sequence_id}{suffix_text}.png"
                    )
                    if save_object_found_frame(
                        observations, evaluator.actor.mapper, found_frame_path
                    ):
                        log_line(f"  saved_found_frame={found_frame_path}", log_path)

                    use_height = evaluator.multi_object or (
                        evaluator.single_target_height_aware and not evaluator.is_gibson
                    )
                    agent_eval_pos = (
                        agent_cam_pos_hab
                        if use_height
                        else np.asarray(agent_pos)[[0, 2]]
                    )
                    dist = get_closest_dist(
                        agent_eval_pos,
                        evaluator.scene_data[episode.scene_id].object_locations[current_obj],
                        evaluator.is_gibson,
                    )
                    det_hab_pos = _current_detection_habitat_pos(evaluator.actor.mapper)
                    dist_detect = float("inf")
                    if det_hab_pos is not None:
                        detection_eval_pos = (
                            det_hab_pos
                            if use_height
                            else np.asarray(det_hab_pos)[[0, 2]]
                        )
                        dist_detect = get_closest_dist(
                            detection_eval_pos,
                            evaluator.scene_data[episode.scene_id].object_locations[current_obj],
                            evaluator.is_gibson,
                        )

                    # 高度约束检查
                    closest_obj_pos = None
                    closest_obj_label = ""
                    if not evaluator.is_gibson:
                        min_dist_val = float('inf')
                        for aabb in evaluator.scene_data[episode.scene_id].object_locations[current_obj]:
                            h = abs(agent_cam_pos_hab[1] - aabb.bbox.center[1])
                            if h > 1.1:
                                continue
                            d = get_closest_dist(agent_cam_pos_hab, [aabb], evaluator.is_gibson)
                            if d < min_dist_val:
                                min_dist_val = d
                                closest_obj_pos = aabb.bbox.center
                                closest_obj_label = f"{aabb.object_category}(sid={aabb.semantic_id})"

                    if closest_obj_pos is not None:
                        log_line(
                            f"  called_found=True "
                            f"agent_body=[{agent_pos[0]:.3f}, {agent_pos[1]:.3f}, {agent_pos[2]:.3f}] "
                            f"agent_camera=[{agent_cam_pos_hab[0]:.3f}, {agent_cam_pos_hab[1]:.3f}, {agent_cam_pos_hab[2]:.3f}] "
                            f"target=[{closest_obj_pos[0]:.3f}, {closest_obj_pos[1]:.3f}, {closest_obj_pos[2]:.3f}] "
                            f"instance={closest_obj_label} dist={dist:.4f} detect_dist={dist_detect:.4f}",
                            log_path,
                        )

                    # Single-target success uses agent-to-target distance,
                    # optionally with the multi-target height gate.
                    # Multi-target also verifies the projected detection.
                    success = dist < evaluator.max_dist
                    if evaluator.multi_object:
                        success = success and dist_detect < evaluator.max_dist
                    if success:
                        result = Result.SUCCESS
                        log_line("  ✓ Object found!", log_path)
                    else:
                        if dist_detect < evaluator.max_dist:
                            result = Result.FAILURE_NOT_REACHED
                        else:
                            result = Result.FAILURE_MISDETECT
                        not_failed = False
                        log_line(
                            f"  ✗ Object not found. dist={dist:.3f}, "
                            f"detect_dist={dist_detect:.3f}",
                            log_path,
                        )
                else:
                    not_failed = False
                    if stuck_now:
                        result = Result.FAILURE_STUCK
                    if no_goal_frames >= 5:
                        result = Result.FAILURE_ALL_EXPLORED

                log_line(
                    f"  Seq {sequence_id} ({current_obj}): {result.name}, "
                    f"target_steps={steps}, episode_steps={episode_step}",
                    log_path,
                )

                results.add_sequence(np.array(poses), result, current_obj)

                # --- 保存体素CSV和统计 ---
                suffix_text = f"_{suffix}" if suffix else ""
                save_voxel_csv(
                    evaluator.actor.mapper.one_map,
                    os.path.join(out_dir, f"voxels_{file_ep}_{sequence_id}{suffix_text}.csv"),
                )
                if save_stats_file:
                    save_stats(
                        evaluator.actor.mapper.one_map, poses, result,
                        os.path.join(out_dir, f"stats_{file_ep}_{sequence_id}{suffix_text}.txt"),
                    )

                # Rerun最终体素快照
                if use_rerun:
                    final_rerun_step = episode_step
                    log_voxels_to_rerun(evaluator.actor.mapper.one_map, final_rerun_step)
                    log_similarity_to_rerun(evaluator.actor.mapper.one_map, final_rerun_step)
                    log_trajectory_to_rerun(
                        poses, final_rerun_step, voxel_map=evaluator.actor.mapper.one_map
                    )

                sequence_id += 1
                if (
                    result == Result.SUCCESS and
                    sequence_id < len(episode.obj_sequence) and
                    episode_step < evaluator.max_steps
                ):
                    current_obj = episode.obj_sequence[sequence_id]
                    evaluator.start_next_target(current_obj)
                    if use_rerun and evaluator.reset_map_between_targets:
                        rr.set_time_sequence("step", episode_step)
                        for path in ("world/voxels", "world/frontiers", "world/path", "world/similarity"):
                            rr.log(path, rr.Clear(recursive=True))

    # --- 保存轨迹和结果 ---
    suffix_text = f"_{suffix}" if suffix else ""
    for seq_id, seq in enumerate(results.sequence_poses):
        np.savetxt(
            os.path.join(out_dir, f"trajectory_{file_ep}_{seq_id}{suffix_text}.csv"),
            seq, delimiter=",", header="map_x,map_y,map_z,yaw,hab_x,hab_y,hab_z", comments="",
        )

    with open(os.path.join(out_dir, f"state_{file_ep}{suffix_text}.txt"), "w") as f:
        f.write(",".join(
            str(results.sequence_results[i].value) for i in range(len(results.sequence_results))
        ))

    log_line(f"\nEpisode {episode_id} complete. Results: "
             f"{[r.name for r in results.sequence_results]}", log_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-episode 3D VoxelMap evaluation.")
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--suffix", default="rerun_3d")
    parser.add_argument("--log_path", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--config", default="config/mon/eval_multi_conf_floor.yaml")
    parser.add_argument("--rerun", action="store_true", help="启用rerun 3D实时可视化")
    parser.add_argument("--rerun_interval", type=int, default=1,
                        help="每N帧刷新一次体素点云到rerun (默认20)")
    args, unknown = parser.parse_known_args()

    # Spock config injection
    sys.argv = [sys.argv[0]] + unknown
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", args.config])

    eval_config = load_eval_config()

    # Use MONActor3D
    actor = MONActor3D(eval_config.EvalConf)
    evaluator = HabitatMultiEvaluator(eval_config.EvalConf, actor)

    out_dir = args.out_dir or f"result_3d_ep{args.episode_id}"
    os.makedirs(out_dir, exist_ok=True)

    log_path = args.log_path or os.path.join(out_dir, f"ep{args.episode_id}_{args.suffix}.log")
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    # 启动 TeeLogger: 将关键 print 输出也写入日志文件
    tee = TeeLogger(log_path, sys.stdout)
    sys.stdout = tee

    log_line("=== Single-episode 3D VoxelMap Evaluation ===", log_path)
    log_line(f"VoxelMap: voxel_xy={actor.mapper.one_map.voxel_size:.3f}m, "
             f"voxel_z={actor.mapper.one_map.voxel_size_z:.3f}m, "
             f"n_cells={actor.mapper.one_map.n_cells}", log_path)
    log_line(f"Rerun: {'enabled' if args.rerun else 'disabled'}", log_path)

    if args.rerun:
        rr.init("voxelmap_eval_3d", spawn=True)
        # 设置世界坐标系
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    run_single_episode(
        evaluator, args.episode_id, args.suffix, log_path, out_dir,
        use_rerun=args.rerun, rerun_voxel_interval=args.rerun_interval,
    )

    # 关闭 TeeLogger，恢复 stdout
    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
