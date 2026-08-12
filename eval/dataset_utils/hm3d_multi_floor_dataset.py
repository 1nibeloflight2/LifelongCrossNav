"""
Dataset loader for the new floor_multi format.
Compatible with HabitatMultiEvaluator.
"""
from eval.dataset_utils import Episode, SceneData, SemanticObject

from typing import Dict, List

import os
from os import listdir
import gzip
import json


def load_hm3d_multi_floor_episodes(episodes: List[Episode], scene_data: Dict[str, SceneData], object_nav_path: str):
    """
    Loads the generated floor_multi episodes from val_floor_multi/content/.
    Each file is {scene_name}.json.gz containing multiple episodes.

    The release format has no precomputed ``best_seq_dists``. Posterior
    evaluation recomputes each segment's geodesic from its recorded start.
    ``Episode.best_dist`` receives explicit null placeholders only for legacy
    interfaces that still expect the dataclass field.
    """
    i = 0
    files = sorted(listdir(object_nav_path))
    for file in files:
        if not file.endswith(".json.gz"):
            continue
        filepath = os.path.join(object_nav_path, file)
        with gzip.open(filepath, 'rt') as f:
            json_data = json.load(f)

        if len(json_data.get('episodes', [])) == 0:
            continue

        scene_id = json_data['episodes'][0]['scene_id']
        if scene_id not in scene_data:
            scene_data_ = SceneData(scene_id, {}, {})
        else:
            scene_data_ = scene_data[scene_id]

        for ep in json_data['episodes']:
            best_dist_placeholder = [
                [None, None] for _ in ep['object_goals']
            ]

            episode = Episode(
                ep['scene_id'],
                i,
                ep['start_position'],
                ep['start_rotation'],
                ep['object_goals'],
                best_dist_placeholder,
                None  # floor_id is None for cross-floor
            )

            episodes.append(episode)
            for obj in ep['object_goals']:
                if obj not in scene_data_.object_locations.keys():
                    scene_data_.object_locations[obj] = []
                    scene_data_.object_ids[obj] = []
            i += 1

        scene_data[scene_id] = scene_data_

    return episodes, scene_data
if __name__ == '__main__':
    eps, scene_data = load_hm3d_multi_floor_episodes(
        [], {}, "datasets/objectnav_hm3d_v2/val_floor_multi/content/")
    print("Found {} episodes".format(len(eps)))

    scene_dist = {}
    for ep in eps:
        if ep.scene_id not in scene_dist:
            scene_dist[ep.scene_id] = 1
        else:
            scene_dist[ep.scene_id] += 1

    for sc in scene_dist:
        print("Scene {}, number of eps {}".format(sc, scene_dist[sc]))

    obj_counts = {}
    for ep in eps:
        for obj in ep.obj_sequence:
            if obj not in obj_counts:
                obj_counts[obj] = 1
            else:
                obj_counts[obj] += 1
    total = sum([obj_counts[obj] for obj in obj_counts])
    for obj in obj_counts:
        print("Object {}, count {}, percentage {:.3f}".format(
            obj, obj_counts[obj], obj_counts[obj] / total))
