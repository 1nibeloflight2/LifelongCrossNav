from eval.dataset_utils import Episode, SceneData, SemanticObject

# typing
from typing import Dict, List

from eval.dataset_utils.target_categories import (
    canonical_target_category,
    matches_target_category,
)

# filesystem utils
import os
from os import listdir
import gzip
import json

def load_hm3d_episodes(episodes: List[Episode], scene_data: Dict[str, SceneData], object_nav_path: str):
    i = 0
    files = listdir(object_nav_path)
    files = sorted(files, key=str.casefold)
    for file in files:
        if file.endswith('.json.gz'):
            with gzip.open(os.path.join(object_nav_path, file), 'r') as f:
                json_data = json.load(f)
                scene_id = json_data['episodes'][0]['scene_id']
                if scene_id not in scene_data:
                    scene_data_ = SceneData(scene_id, {}, {})
                    for obj_ in json_data['goals_by_category']:
                        obj = json_data['goals_by_category'][obj_]
                        obj_name = obj[0]['object_category']
                        scene_data_.object_locations[
                            obj_name] = []  # the actual locations and bounding boxes will be loaded later
                        scene_data_.object_ids[obj_name] = []
                        for obj_loc in obj:
                            scene_data_.object_ids[obj_name].append(obj_loc['object_id'])
                    scene_data[scene_id] = scene_data_
                for ep in json_data['episodes']:
                    episode = Episode(ep['scene_id'],
                                      i,
                                      ep['start_position'],
                                      ep['start_rotation'],
                                      [ep['object_category']],
                                      ep['info']['geodesic_distance'])
                    episodes.append(episode)
                    i += 1
    return episodes, scene_data

def load_hm3d_objects(scene_data: Dict[str, SceneData], semantic_objects, scene_id: str):
    for scene_obj in semantic_objects:
        obj_name = scene_obj.category.name()
        for cat in scene_data[scene_id].object_locations.keys():
            if scene_obj.id in scene_data[scene_id].object_locations[cat]:
                continue
            canonical = canonical_target_category(cat)
            if canonical is not None:
                # For benchmark targets, rebuild membership from the strict
                # semantic-name taxonomy. Existing goal IDs may themselves be
                # polluted by the old substring matcher.
                belongs = matches_target_category(canonical, obj_name)
            else:
                belongs = scene_obj.semantic_id in scene_data[scene_id].object_ids[cat]
            if belongs:
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
    return scene_data


if __name__ == '__main__':
    eps, scene_data = load_hm3d_episodes([], {}, "datasets/objectnav_hm3d_v1/val/content")
    print(f"Found {len(eps)} episodes")
    scene_dist = {}
    for ep in eps:
        if ep.scene_id not in scene_dist:
            scene_dist[ep.scene_id] = 1
        else:
            scene_dist[ep.scene_id] += 1

    for sc in scene_dist:
        print(f"Scene {sc}, number of eps {scene_dist[sc]}")

    obj_counts = {}
    for ep in eps:
        for obj in ep.obj_sequence:
            if obj not in obj_counts:
                obj_counts[obj] = 1
            else:
                obj_counts[obj] += 1
    total = sum([obj_counts[obj] for obj in obj_counts])
    for obj in obj_counts:
        print(f"Object {obj}, count {obj_counts[obj]}, percentage {obj_counts[obj] / total}")
