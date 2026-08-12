"""
Script to generate a multi-object episode dataset.
This is achieved by scanning the objectnav_v2 dataset to
    - find out which objects are accessible in which floor
    - get viable starting positions per floor

We then generate a sequence of 4 distinct object categories and compute the shortest possible path per episode to
any instance of the object category.
"""
from dataclasses import dataclass, field
from distutils.command.build import build
from math import floor

from eval.dataset_utils import HM3DDataset, SemanticObject
from eval.dataset_utils.object_nav_utils.object_nav_gen import make_object_viewpoints, VPConf, get_geodesic, build_sim

import itertools

import numpy as np

from typing import List, Dict, Set, Tuple

import cv2

import habitat_sim
from habitat.utils.visualizations import maps
from habitat_sim import ActionSpec, ActuationSpec

from tqdm import tqdm

import json, gzip

import os

os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'



# configure here
num_objects = 3  # episodes of 3 goals, this will filter our floors with less than 3 object goals
num_eps_per_floor = 4  # out of the remaining floors, how many episodes do we want per floor?
path_to_hm3d_objectnav_v2 = "datasets/objectnav_hm3d_v2/val/content/"
path_to_hm3d_v0_2 = "datasets/scene_datasets/"
viewpoint_dir = "datasets/multi_object_data"

# TODO Check this?
start_poses_tilt_angle = 30

@dataclass
class Floor:
    height: float
    scene_id: str
    object_categories: Set[str] = field(default_factory=set)
    objects: Dict[str, List[SemanticObject]] = field(default_factory=dict) # the actual objects found in the floor
    start_positions: List[np.ndarray] = field(default_factory=list)
    start_rotations: List[np.ndarray] = field(default_factory=list)

@dataclass
class SceneAccumulated:
    floors: Dict[int, Floor] = field(default_factory=dict)

@dataclass
class MultiEpisode:
    start_position: np.ndarray
    start_rotation: np.ndarray
    object_goals: List[str]
    scene_id: str
    floor: int
    best_seq_dists: List[Tuple[float, np.ndarray]] = field(default_factory=list)

    def to_json(self):
        return {
            "start_position": self.start_position.tolist(),
            "start_rotation": self.start_rotation.tolist(),
            "object_goals": self.object_goals,
            "scene_id": self.scene_id,
            "floor": self.floor,
            "best_seq_dists": [(d, p.tolist()) for d, p in self.best_seq_dists]
        }

@dataclass
class ValidCombination:
    start_position: np.ndarray
    start_rotation: np.ndarray
    object_goal: str
    floor: int

def draw_bounding_box(
    sim, top_down_map, sem_obj, line_thickness=4, color=10
):
    center = sem_obj.bbox.center
    x_len, _, z_len = sem_obj.bbox.sizes / 2.0
    # Nodes to draw rectangle
    corners = [
        center + np.array([x, 0, z])
        for x, z in [
            (-x_len, -z_len),
            (-x_len, z_len),
            (x_len, z_len),
            (x_len, -z_len),
            (-x_len, -z_len),
        ]
    ]

    map_corners = [
        maps.to_grid(
            p[2],
            p[0],
            (
                top_down_map.shape[0],
                top_down_map.shape[1],
            ),
            sim=sim,
        )
        for p in corners
    ]

    maps.draw_path(
        top_down_map,
        map_corners,
        color,
        line_thickness,
    )
    return top_down_map

def draw_point(sim, top_down_map, position, point_type, point_padding=2):
    t_x, t_y = maps.to_grid(
        position[2],
        position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=sim,
    )
    top_down_map[
    t_x - point_padding: t_x + point_padding + 1,
    t_y - point_padding: t_y + point_padding + 1,
    ] = point_type
    return top_down_map


def visualize_episodes(datapath: str, scene_path: str, sim_name: str, output_dir: str):
    # Load the generated episodes
    sim_dir = os.path.join(output_dir, sim_name)
    os.makedirs(sim_dir, exist_ok=True)
    episode_file = os.path.join(datapath, scene_path, f"{sim_name}_episodes.json.gz")
    with gzip.open(episode_file, "rt") as f:
        episodes = json.load(f)["episodes"]

    # Build the simulator
    sim = build_sim(path_to_hm3d_v0_2, episodes[0]['scene_id'], start_poses_tilt_angle, True)

    # Load scene data
    scene_data = {}
    episodes_ = []
    episodes_, scene_data = HM3DDataset.load_hm3d_episodes(episodes_, scene_data, path_to_hm3d_objectnav_v2)
    scene_data = HM3DDataset.load_hm3d_objects(scene_data, sim.semantic_scene.objects, episodes[0]['scene_id'])

    for idx, episode in enumerate(episodes):
        # Get the top-down map
        top_down_map = maps.get_topdown_map(
            sim.pathfinder,
            height=episode["start_position"][1],
            map_resolution=512,
            draw_border=True,
        )

        # Draw the start position
        top_down_map = draw_point(
            sim,
            top_down_map,
            np.array(episode["start_position"]),
            maps.MAP_SOURCE_POINT_INDICATOR,
        )

        # Draw the object goals
        for i, obj_category in enumerate(episode["object_goals"]):
            for obj in scene_data[ episodes[0]['scene_id']].object_locations[obj_category]:
                top_down_map = draw_bounding_box(sim, top_down_map, obj, 4,
                                                 int(i * 246.0 / len(episode["object_goals"])) + 10)

        # Draw the path
        for dist, pos in episode["best_seq_dists"]:
            top_down_map = draw_point(
                sim,
                top_down_map,
                np.array(pos),
                maps.MAP_VIEW_POINT_INDICATOR,
            )

        # Colorize and save the map
        top_down_map = maps.colorize_topdown_map(top_down_map)
        output_file = os.path.join(sim_dir, f"episode_{idx}.png")
        cv2.imwrite(output_file, top_down_map)

    sim.close()

def build_episode(sim: habitat_sim.Simulator, start_pos: np.ndarray, object_sequence: List[str], floor_data: Floor):
    # build the possible combinations first
    combinations = list(itertools.product(*[floor_data.objects[obj] for obj in object_sequence]))
    pathes = []
    for comb in combinations:
        start = start_pos
        subpathes = []
        for waypoint in comb:
            dist, next_start = get_geodesic(start, sim, waypoint)
            if dist is None:
                break
            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = next_start
            found_path = sim.pathfinder.find_path(path)
            if not found_path:
                break
            heights = [p.tolist()[1] for p in path.points]
            h_delta = max(heights) - min(heights)

            if h_delta > 0.3:
                break
            subpathes.append((dist, next_start, found_path))
            start = next_start
        if len(subpathes) == len(comb):
            pathes.append(subpathes)
    if len(pathes) == 0:
        return None
    pathes_sorted = sorted(pathes, key=lambda x: sum(b[0] for b in x))
    # for path in pathes_sorted:
    #     sum_of_dists = 0
    #     for segment in path:
    #         sum_of_dists += segment[0]
        # print(sum_of_dists)
    # top_down_map = maps.get_topdown_map(
    #     sim.pathfinder,
    #     height=start_pos[1],
    #     map_resolution=512,
    #     draw_border=True,
    # )
    # for i, obj_c in enumerate(object_sequence):
    #     for obj in floor_data.objects[obj_c]:
    #         draw_bounding_box(sim, top_down_map, obj, 4, int(i * 246.0/len(object_sequence)) + 10)
    # top_down_map = draw_point(
    #     sim,
    #     top_down_map,
    #     start_pos,
    #     maps.MAP_SOURCE_POINT_INDICATOR,
    # )
    # for wp in pathes_sorted[0]:
    #     top_down_map = draw_point(
    #         sim,
    #         top_down_map,
    #         wp[1],
    #         maps.MAP_VIEW_POINT_INDICATOR,
    #     )
    #
    # top_down_map = maps.colorize_topdown_map(top_down_map)
    # cv2.imwrite("pmap.png", top_down_map)
    return pathes_sorted[0]

def load_scenes(episodes, scene_data, valid_start_positions, scene_floors, scenes):
    episodes, scene_data = HM3DDataset.load_hm3d_episodes(episodes, scene_data, path_to_hm3d_objectnav_v2)

    # we fetch all the valid episode characteristics per scene
    for ep in episodes:
        if ep.scene_id not in valid_start_positions.keys():
            valid_start_positions[ep.scene_id] = []
            scene_floors[ep.scene_id] = [ep.start_position[1]]
        floor_id = -1
        for i in range(len(scene_floors[ep.scene_id])):
            if np.abs(scene_floors[ep.scene_id][i] - ep.start_position[1]) < 0.5:
                floor_id = i
                scene_floors[ep.scene_id][i] = scene_floors[ep.scene_id][i]*0.95 + 0.05 * ep.start_position[1]
                break
        if floor_id == -1:
            scene_floors[ep.scene_id].append(ep.start_position[1])
            floor_id = len(scene_floors[ep.scene_id]) - 1
        valid_start_positions[ep.scene_id].append(ValidCombination(ep.start_position, ep.start_rotation,
                                                                   ep.obj_sequence[0], floor_id))
    print(f"Loaded {len(episodes)} episodes")
    # from this, we can compute the start locations and viable object goals per floor per scene,
    # we should be able to freely combine those
    for sc in valid_start_positions.keys():
        scenes[sc] = SceneAccumulated()
        for comb in valid_start_positions[sc]:
            if comb.floor not in scenes[sc].floors.keys():
                scenes[sc].floors[comb.floor] = Floor(scene_floors[sc][comb.floor], sc)
            scenes[sc].floors[comb.floor].object_categories.add(comb.object_goal)
            scenes[sc].floors[comb.floor].start_positions.append(comb.start_position)
            scenes[sc].floors[comb.floor].start_rotations.append(comb.start_rotation)
    # we filter our floors with less than 3 object goal categories
    number_of_floors = 0
    for sc in scenes.keys():
        scenes[sc].floors = {floor: data for floor, data in scenes[sc].floors.items()
                             if len(data.object_categories) >= num_objects}
        number_of_floors += len(scenes[sc].floors)
    return number_of_floors


def store_viewpoints(scenes, scene_id, output_dir):
    viewpoints_data = {}

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    scene = scenes[scene_id]
    viewpoints_data[scene_id] = {}

    for floor_id, floor in scene.floors.items():

        for category, objects in floor.objects.items():
            if category not in viewpoints_data[scene_id]:
                viewpoints_data[scene_id][category] = {}

            for obj in objects:
                viewpoints_data[scene_id][category][
                    obj.object_id] = obj.view_pts


        # Save the data to a JSON file
        sc_short = scene_id.split("/")[-1].split(".")[0]
        output_file = os.path.join(output_dir, f'{sc_short}_viewpoints.json')
        with open(output_file, 'w') as f:
            json.dump(viewpoints_data[scene_id], f, indent=2)

def load_all_scene_data(sc, scenes, scene_data, viewpoint_conf, sim = None):
    # to track the object assignments
    obj_id_to_floors = {}
    needs_save = False
    # Load the scene
    needs_close = False
    if sim is None:
        needs_close = True
        sim = build_sim(path_to_hm3d_v0_2, sc, start_poses_tilt_angle, True)
    # we load all the objects of interest in the semantic scene
    scene_data = HM3DDataset.load_hm3d_objects(scene_data, sim.semantic_scene.objects, sc)
    print(sum(len(scene_data[sc].object_locations[cat]) for cat in scene_data[sc].object_locations.keys()))
    # we go through the floors of the scene and add the object locations to it
    sc_short = sc.split("/")[-1].split(".")[0]
    if os.path.exists(os.path.join(viewpoint_dir, f"{sc_short}_viewpoints.json")):
        with open(os.path.join(viewpoint_dir, f"{sc_short}_viewpoints.json"), "r") as f:
            viewpoints_data = json.load(f)
    else:
        viewpoints_data = {}

    for fl in scenes[sc].floors.keys():
        min_dist = np.inf
        floor_level = scenes[sc].floors[fl].height
        for cat in scenes[sc].floors[fl].object_categories:
            scenes[sc].floors[fl].objects[cat] = []
            for obj in scene_data[sc].object_locations[cat]:
                # get the object bounding box
                obj_bbox = obj.bbox
                obj_center = obj_bbox.center
                # we assume an object to be on a given floor if it's center point is at max 0.2 under the floor
                # or under 1.7 above the floor
                obj_z = obj_center[1]
                # print(obj_z, obj_center, obj.object_id)
                if abs(obj_z - floor_level) < abs(min_dist):
                    min_dist = obj_z - floor_level
                if floor_level - 0.2 <= obj_z <= floor_level + 1.7:
                    if obj.view_pts is None:
                        if cat in viewpoints_data and obj.object_id in viewpoints_data[cat]:
                            obj.view_pts = viewpoints_data[cat][obj.object_id]
                        else:
                            needs_save = True
                            obj.view_pts = make_object_viewpoints(sim, obj, viewpoint_conf)
                    if obj.object_id not in obj_id_to_floors.keys():
                        obj_id_to_floors[obj.object_id] = [fl]
                        scenes[sc].floors[fl].objects[cat].append(obj)

                    else:
                        if obj_id_to_floors[obj.object_id] == [fl]:
                            scenes[sc].floors[fl].objects[cat].append(obj)
                            pass
                        else:
                            print(
                                f"Warning in floor {fl} with height {floor_level}, object {obj.object_id} with center height {obj_z} was already assigned to floors "
                                f"{obj_id_to_floors[obj.object_id]} with level"
                                f" {[scenes[sc].floors[f].height for f in obj_id_to_floors[obj.object_id]]}")
    if needs_close:
        sim.close()
    return needs_save
def generate_dataset(datapath: str):
    viewpoint_conf = VPConf(1.0, 0.1, 0.05)
    # First, we load all the episodes of hm3d_objectnav_v2
    episodes = []
    scene_data = {}
    valid_start_positions = {}
    scene_floors = {}
    scenes = {}
    number_of_floors = load_scenes(episodes, scene_data, valid_start_positions, scene_floors, scenes)


    num_eps = num_eps_per_floor * number_of_floors
    print(f"Found {number_of_floors} valid floors. Will generate"
          f" {num_eps } episodes.")
    print("Loading and sorting all object instances for all floors..")
    # TODO This is the most time consuming part, we should parallelize this, or at least save the results
    with tqdm(total=number_of_floors) as pbar:
        for sc in list(scenes.keys()):
            load_all_scene_data(sc, scenes, scene_data, viewpoint_conf)

    print(f"Successfully loaded all objects, generating episodes")
    max_retries = 30
    with tqdm(total=num_eps) as pbar:
        for sc in scenes.keys():
            # We could save viable object ids here?
            sim = build_sim(path_to_hm3d_v0_2, sc, start_poses_tilt_angle)
            scene_episodes = {"episodes": []}
            for fl in scenes[sc].floors.keys():
                for i in range(num_eps_per_floor):
                    path = None
                    start_pos = None
                    start_rot = None
                    objects = None
                    retries = 0
                    while path is None and retries < max_retries:
                        choice_pos = np.random.choice(len(scenes[sc].floors[fl].start_positions), 1, replace=True)[0]
                        choice_rot = np.random.choice(len(scenes[sc].floors[fl].start_rotations), 1, replace=True)[0]
                        start_pos = scenes[sc].floors[fl].start_positions[choice_pos]
                        start_rot = scenes[sc].floors[fl].start_rotations[choice_rot]
                        objects = np.random.choice(sorted(scenes[sc].floors[fl].object_categories), size=(num_objects, ), replace=False).tolist()
                        path = build_episode(sim, np.array(start_pos), objects, scenes[sc].floors[fl])
                        retries += 1
                        # if path is None:
                        #     print(f"Failed to generate path for {objects}")
                    if path is None:
                        print(f"Failed to generate path for scene {sc}, floor {fl} after {max_retries} retries.")
                        continue
                    seq_dists = [(p[0], p[1]) for p in path]
                    ep = MultiEpisode(np.array(start_pos), np.array(start_rot), objects, sc, fl, seq_dists)
                    ep_json = ep.to_json()
                    scene_episodes["episodes"].append(ep_json)
                    pbar.update(1)
            sim.close()
            # get sim name e.g. from /hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb_episodes.json'
            sim_name = sc.split("/")[-1].split(".")[0]
            scene_path = sc.split("/")[-2]
            # create the directory if it doesn't exist, and create directory for the scene. We copy the name of the scene in hm3d
            if not os.path.exists(datapath):
                os.makedirs(datapath)
            if not os.path.exists(os.path.join(datapath, scene_path)):
                os.makedirs(os.path.join(datapath, scene_path))
            json_str = json.dumps(scene_episodes)
            # gzip and save
            with gzip.open(os.path.join(datapath, scene_path, f"{sim_name}_episodes.json.gz"), "wt") as f:
                f.write(json_str)


# def fix_geodesic():



if __name__ == "__main__":
    generate_dataset("datasets/multiobject_episodes")
    # fix_geodesic("/home/finn/mon_data/multiobject_episodes")
    # datapath = "/home/finn/mon_data/multiobject_episodes"
    # scene_path = "00831-yr17PDCnDDW"
    # sim_name = "yr17PDCnDDW"
    # output_dir = "/home/finn/mon_data/visualized_episodes"
    #
    # os.makedirs(output_dir, exist_ok=True)
    # visualize_episodes(datapath, scene_path, sim_name, output_dir)