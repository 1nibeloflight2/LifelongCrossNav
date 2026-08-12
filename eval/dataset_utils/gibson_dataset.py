# os / file system utils
import os
from os import listdir
import gzip
import json

# eval
from eval.dataset_utils import SceneData, GibsonEpisode
from eval.habitat_utils import FMMPlanner

# numpy
import numpy as np
import quaternion

# typing
from typing import List, Dict

# skimage
import skimage


def compute_gt_path_gibson(episode: GibsonEpisode, dataset_info):
    goal_idx = episode.object_id
    floor_idx = episode.floor_id
    scene_id = episode.scene_id
    file_name = scene_id.split('/')[-1]
    # Remove the file extension
    scene_name = file_name.split('.')[0]
    object_boundary = 1.0
    map_resolution = 5
    scene_info = dataset_info[scene_name]
    sem_map = scene_info[floor_idx]['sem_map']
    map_obj_origin = scene_info[floor_idx]['origin']
    selem = skimage.morphology.disk(2)
    traversible = skimage.morphology.binary_dilation(
        sem_map[0], selem) != True
    traversible = 1 - traversible
    planner = FMMPlanner(traversible)
    selem = skimage.morphology.disk(
        int(object_boundary * 100. / map_resolution))
    goal_map = skimage.morphology.binary_dilation(
        sem_map[goal_idx + 1], selem) != True
    goal_map = 1 - goal_map
    planner.set_multi_goal(goal_map)

    # Get starting loc in GT map coordinates
    pos = episode.start_position
    x = -pos[2]
    y = -pos[0]
    min_x, min_y = map_obj_origin / 100.0
    map_loc = int((-y - min_y) * 20.), int((-x - min_x) * 20.)
    episode.best_dist = planner.fmm_dist[map_loc] \
                        / 20.0 + object_boundary
    return episode

def load_gibson_objects(scene_data: Dict[str, SceneData], dataset_info, scene_id: str):
    for obj in scene_data[scene_id].object_ids.keys():
        obj_id = scene_data[scene_id].object_ids[obj][0]
        file_name = scene_id.split('/')[-1]
        # Remove the file extension
        name = file_name.split('.')[0]

        for fl in dataset_info[name].keys():
            map_origin = dataset_info[name][fl]['origin']
            obj_locs = dataset_info[name][fl]['sem_map'][obj_id + 1]
            obj_locs = np.argwhere(obj_locs == 1).astype(np.float32)
            min_x, min_y = map_origin / 100.0
            obj_locs = np.flip(obj_locs, axis=-1)
            obj_locs[:, 0] = obj_locs[:, 0] / 20. + min_x
            obj_locs[:, 1] = obj_locs[:, 1] / 20. + min_y
            obj_locs = - obj_locs
            scene_data[scene_id].object_locations[obj].append(obj_locs)
    return scene_data


def load_gibson_episodes(episodes: List[GibsonEpisode], scene_data: Dict[str, SceneData], dataset_info,
                         object_nav_path: str, export=True):
    i = 0
    files = listdir(object_nav_path)
    files = sorted(files, key=str.casefold)
    for file in files:
        if file.endswith('.json.gz') and 'glb' not in file.lower():
            file_path = os.path.join(object_nav_path, file)
            # json_data = None
            modified = False
            with gzip.open(file_path, 'r') as f:
                json_data = json.load(f)
                scene_id = json_data['episodes'][0]['scene_id']
                if scene_id not in scene_data:
                    scene_data_ = SceneData(scene_id, {}, {})
                    scene_data[scene_id] = scene_data_
                # modified = False
                for ep in json_data['episodes']:
                    obj_name = ep['object_category']
                    if obj_name not in scene_data[ep['scene_id']].object_locations.keys():
                        scene_data[ep['scene_id']].object_locations[obj_name] = []
                        scene_data[ep['scene_id']].object_ids[obj_name] = []
                    scene_data[ep['scene_id']].object_ids[obj_name].append(ep['object_id'])
                    episode = GibsonEpisode(ep['scene_id'],
                                            i,
                                            ep['start_position'],
                                            quaternion.from_float_array(ep['start_rotation']),
                                            [ep['object_category']],
                                            ep['info']['geodesic_distance'] if 'info' in ep.keys() else -1,
                                            ep['object_id'],
                                            ep['floor_id'])
                    if episode.best_dist == -1:
                        episode = compute_gt_path_gibson(episode, dataset_info)  # TODO Export this so we do not need to recompute all the time
                        ep['info'] = ep.get('info', {})
                        ep['info']['geodesic_distance'] = episode.best_dist
                        modified = True
                    episodes.append(episode)
                    i += 1
            if modified and json_data is not None and export:
                # First, convert the JSON to a string
                json_str = json.dumps(json_data)
                # Then compress and write it
                with gzip.open(file_path, 'wt') as f:
                    f.write(json_str)
    return episodes, scene_data