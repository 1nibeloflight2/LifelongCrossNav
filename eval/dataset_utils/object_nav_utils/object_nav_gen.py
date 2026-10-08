# from https://github.com/naokiyokoyama/ovon/blob/main/ovon/dataset/objectnav_generator.py


import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs

from dataclasses import dataclass

import numpy as np

import itertools

from typing import Dict, List, Union, Sequence

@dataclass
class VPConf:
    goal_vp_max_dist: float
    goal_vp_cell_size: float
    frame_cov_thresh: float

def geodesic_distance(
    sim: habitat_sim.Simulator,
    position_a: Union[Sequence[float], np.ndarray],
    position_b: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
) :
    path = habitat_sim.MultiGoalShortestPath()
    if isinstance(position_b[0], (Sequence, np.ndarray)):
        path.requested_ends = np.array(position_b, dtype=np.float32)
    else:
        path.requested_ends = np.array([np.array(position_b, dtype=np.float32)])
    path.requested_start = np.array(position_a, dtype=np.float32)
    sim.pathfinder.find_path(path)
    end_pt = path.points[-1] if len(path.points) else np.array([])
    return path.geodesic_distance, end_pt

def build_sim(path_to_hm3d_v0_2, sc, start_poses_tilt_angle, ensure_nav=True):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = path_to_hm3d_v0_2 + sc
    backend_cfg.scene_dataset_config_file = path_to_hm3d_v0_2 + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    backend_cfg.allow_sliding = False
    sensor_specs = []
    for name, sensor_type in zip(
            ["color", "depth", "semantic"],
            [
                habitat_sim.SensorType.COLOR,
                habitat_sim.SensorType.DEPTH,
                habitat_sim.SensorType.SEMANTIC,
            ],
    ):
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = f"{name}_sensor"
        sensor_spec.sensor_type = sensor_type
        sensor_spec.resolution = [640, 640]
        sensor_spec.position = [0.0, 0.88, 0.0]
        sensor_spec.hfov = 90
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(sensor_spec)
    # create agent specifications
    agent_cfg = habitat_sim.AgentConfiguration(
        height=0.88,
        radius=0.15,
        sensor_specifications=sensor_specs,
        action_space={
            "look_up": habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=start_poses_tilt_angle),
            ),
            "look_down": habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=start_poses_tilt_angle),
            ),
        },
    )

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if ensure_nav:
        sim = ensure_navmesh(sim)
    return sim

def ensure_navmesh(sim: habitat_sim.Simulator):
    assert sim.pathfinder.is_loaded, "pathfinder is not loaded!"
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_height = 1.1
    navmesh_settings.agent_radius = 0.17
    navmesh_settings.agent_max_climb = 0.10
    navmesh_settings.cell_height = 0.05
    navmesh_success = sim.recompute_navmesh(
        sim.pathfinder, navmesh_settings, include_static_objects=False
    )
    assert navmesh_success, "Failed to build the navmesh!"
    return sim


def get_geodesic(start_pos: np.ndarray, sim: habitat_sim.Simulator, goal_obj, correct_start=False):
    start_pos = start_pos.astype(np.float32)
    if not correct_start and ((not sim.pathfinder.is_navigable(start_pos)) or sim.pathfinder.island_radius(start_pos) < 1.2):
        print("Invalid start position")
        return None, None
    if correct_start:
        if not sim.pathfinder.is_navigable(start_pos):
            start_pos = sim.pathfinder.snap_point(start_pos)
        # check nan
        if np.isnan(start_pos).any():
            print("Invalid start position after snapping")
            return None, None
    closest_goals = []
    for vps in goal_obj.view_pts:
        geo_dist, closest_point = geodesic_distance(
            sim, start_pos, vps["agent_state"]["position"]
        )
        if closest_point.size != 0:
            closest_goals.append((geo_dist, closest_point))
    if len(closest_goals) > 0:
        geo_dists = sorted(closest_goals, key=lambda x: x[0])
        return geo_dists[0]
    return None, None


def _compute_frame_coverage(obs: List[Dict[str, np.ndarray]], oid: int):
    def _single(obs, oid):
        mask = obs["semantic_sensor"] == oid
        return mask.sum() / mask.size

    if isinstance(obs, list):
        return [_single(o, oid) for o in obs]
    if isinstance(obs, dict):
        return _single(obs, oid)
    else:
        raise TypeError("argument `obs` must be either a list or a dict.")

def make_object_viewpoints(sim: habitat_sim.Simulator, obj: habitat_sim.scene.SemanticObject, vp_conf: VPConf):
    object_position = obj.bbox.center
    eps = 1e-5
    x_len, _, z_len = obj.bbox.sizes / 2.0 + vp_conf.goal_vp_max_dist
    x_bxp = (
            np.arange(-x_len, x_len + eps, step=vp_conf.goal_vp_cell_size)
            + object_position[0]
    )
    z_bxp = (
            np.arange(-z_len, z_len + eps, step=vp_conf.goal_vp_cell_size)
            + object_position[2]
    )
    candiatate_poses = [
        np.array([x, object_position[1], z])
        for x, z in itertools.product(x_bxp, z_bxp)
    ]

    def _down_is_navigable(pt):
        pf = sim.pathfinder

        delta_y = 0.05
        max_steps = int(2 / delta_y)
        step = 0
        is_navigable = pf.is_navigable(pt, 2)
        while not is_navigable:
            pt[1] -= delta_y
            is_navigable = pf.is_navigable(pt)
            step += 1
            if step == max_steps:
                return False
        return True

    def _face_object(object_position: np.array, point: np.ndarray):
        EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
        cam_normal = (object_position - point) + EPS_ARRAY
        cam_normal[1] = 0
        cam_normal = cam_normal / np.linalg.norm(cam_normal)
        return quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)

    def _get_iou(pt):
        obb = habitat_sim.geo.OBB(obj.bbox)
        if obb.distance(pt) > vp_conf.goal_vp_max_dist:
            return -0.5, pt, None

        if not _down_is_navigable(pt):
            return -1.0, pt, None

        pt = np.array(sim.pathfinder.snap_point(pt))
        q = _face_object(object_position, pt)

        cov = 0
        sim.agents[0].set_state(habitat_sim.AgentState(position=pt, rotation=q))
        for act in ["look_down", "look_up", "look_up"]:
            obs = sim.step(act)
            cov += _compute_frame_coverage(obs, obj.semantic_id)

        return cov, pt, q

    candiatate_poses_ious = [_get_iou(pos) for pos in candiatate_poses]
    best_iou = (
        max(v[0] for v in candiatate_poses_ious)
        if len(candiatate_poses_ious) != 0
        else 0
    )
    if best_iou <= 0.0:
        return []

    view_locations = [
        {
            "agent_state": {
                "position": pt.tolist(),
                "rotation": quat_to_coeffs(q).tolist(),
            },
            "iou": iou,
        }
        for iou, pt, q in candiatate_poses_ious
        if iou > vp_conf.frame_cov_thresh
    ]
    view_locations = sorted(view_locations, reverse=True, key=lambda v: v["iou"])

    return view_locations