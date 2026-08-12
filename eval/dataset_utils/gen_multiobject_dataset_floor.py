"""
Generate a cross-floor multi-goal dataset from HM3D objectnav v2 (val_mini).
- goals_by_category stays the same, each instance gets a floor id.
- episodes reuse start_position/start_rotation and use ordered NUM_GOALS sequences.
- Generation-time geodesics validate that an ordered query is reachable, but
  are not serialized; evaluation recomputes posterior distances from recorded
  trajectory starts.
- 生成多目标数据集，基于原始数据集的起点和目标物体，但重新组合成跨楼层的多目标序列。基于网页的楼层统计
- 最终的数据存在nill，因为有些场景的目标物体之间没有可行路径，或者起点到第一个目标没有可行路径。
"""

import csv
import gzip
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from eval.dataset_utils import HM3DDataset, SceneData
from eval.dataset_utils.target_categories import canonical_target_category
from eval.dataset_utils.object_nav_utils.object_nav_gen import (
    VPConf,
    build_sim,
    get_geodesic,
    make_object_viewpoints,
)


INPUT_DIR = "datasets/objectnav_hm3d_v2/val/content/"
OUTPUT_DIR = "datasets/objectnav_hm3d_v2/val_floor_multi/content/"
PATH_TO_HM3D_V0_2 = "datasets/scene_datasets/hm3d_v0.2/"
FLOOR_META_CSV = "datasets/hm3d_val_floors.csv"
START_POSES_TILT_ANGLE = 30
NUM_GOALS = 3
RANDOM_SEED = 0


def load_floor_counts(csv_path: str) -> Dict[str, int]:
    floor_counts: Dict[str, int] = {}
    if not os.path.exists(csv_path):
        return floor_counts
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row.get("scene_id")
            floors = row.get("floors")
            if not scene_id or floors is None or floors == "":
                continue
            try:
                floor_counts[scene_id] = int(floors)
            except ValueError:
                continue
    return floor_counts


def scene_short_id(scene_id: str) -> str:
    # Example: hm3d_v0.2/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
    parts = scene_id.split("/")
    for part in parts:
        if "-" in part and part[0:5].isdigit():
            return part
    if parts:
        return parts[-1].split(".")[0]
    return scene_id


def cluster_floors_from_goals(goals_by_category: Dict, k: Optional[int]) -> List[float]:
    """
    Determine floor thresholds from goal instance heights using top-(k-1)
    largest gaps in sorted Y values.

    Returns a sorted list of threshold values (length = k-1).
    A point with y < thresholds[0] belongs to floor 0,
    thresholds[0] <= y < thresholds[1] belongs to floor 1, etc.

    If k is None or <= 1, returns an empty list (single floor, no thresholds).
    """
    heights = []
    for _, instances in goals_by_category.items():
        for inst in instances:
            pos = inst.get("position")
            if not pos or len(pos) < 2:
                continue
            heights.append(pos[1])

    if not heights:
        return []

    height_arr = np.array(heights, dtype=float)
    sorted_h = np.sort(np.unique(height_arr))

    if k is None or k <= 1:
        return []

    if len(sorted_h) < 2:
        return []

    num_thresholds = k - 1
    diffs = np.diff(sorted_h)

    # Find top-(k-1) largest gaps as floor boundaries
    num_gaps = min(num_thresholds, len(diffs))
    gap_indices = np.argsort(diffs)[::-1][:num_gaps]
    gap_indices = np.sort(gap_indices)

    thresholds = []
    for idx in gap_indices:
        t = (sorted_h[idx] + sorted_h[idx + 1]) / 2.0
        thresholds.append(float(t))

    return sorted(thresholds)


def cluster_floors_from_objects(
    category_objects: Dict[str, List],
    k: Optional[int],
) -> List[float]:
    """Compute floor thresholds from the strictly classified target objects."""
    goals = {
        category: [
            {"position": [float(v) for v in obj.bbox.center]}
            for obj in objects
        ]
        for category, objects in category_objects.items()
    }
    return cluster_floors_from_goals(goals, k)


def assign_floor_id(obj_center_y: float, floor_thresholds: List[float]) -> int:
    """
    Assign floor id based on thresholds.
    Floor 0: y < thresholds[0]
    Floor i: thresholds[i-1] <= y < thresholds[i]
    Floor N-1 (last): y >= thresholds[-1]
    """
    for i, t in enumerate(floor_thresholds):
        if obj_center_y < t:
            return i
    return len(floor_thresholds)


def build_scene_data(scene_id: str, goals_by_category: Dict) -> Dict[str, SceneData]:
    scene_data = {scene_id: SceneData(scene_id, {}, {})}
    for cat, instances in goals_by_category.items():
        scene_data[scene_id].object_locations[cat] = []
        scene_data[scene_id].object_ids[cat] = [inst["object_id"] for inst in instances]
    return scene_data


def normalize_category_key(category_key: str, instances: List[Dict]) -> str:
    if instances:
        obj_cat = instances[0].get("object_category")
        if obj_cat:
            return canonical_target_category(obj_cat) or obj_cat
    canonical = canonical_target_category(category_key)
    if canonical is not None:
        return canonical
    if "_" in category_key:
        return category_key.rsplit("_", 1)[-1]
    return category_key


def build_semantic_index(scene_data: Dict[str, SceneData], scene_id: str) -> Dict:
    semantic_by_id: Dict = {}
    for cat in scene_data[scene_id].object_locations:
        for obj in scene_data[scene_id].object_locations[cat]:
            if obj.semantic_id is None:
                continue
            semantic_by_id[obj.semantic_id] = obj
            semantic_by_id[str(obj.semantic_id)] = obj
    return semantic_by_id


@dataclass
class GoalInstance:
    object_id: int
    view_pts: List[Dict]


def ensure_viewpoints(sim, obj, vp_conf: VPConf) -> bool:
    if obj.view_pts is None:
        obj.view_pts = make_object_viewpoints(sim, obj, vp_conf)
    return bool(obj.view_pts)


def representative_viewpoint(obj) -> Optional[np.ndarray]:
    if not obj.view_pts:
        return None
    pos = obj.view_pts[0]["agent_state"]["position"]
    return np.array(pos, dtype=np.float32)


def best_target_from_start(
    sim,
    vp_conf: VPConf,
    start_pos: np.ndarray,
    candidates: List[Tuple],
    correct_start: bool,
) -> Optional[Dict]:
    best = None
    for instance_id, obj in candidates:
        if not ensure_viewpoints(sim, obj, vp_conf):
            continue
        dist, view_pt = get_geodesic(start_pos, sim, obj, correct_start=correct_start)
        if dist is None:
            continue
        candidate = {
            "best_instance_id": instance_id,
            "distance": float(dist),
            "view_point": view_pt.tolist(),
        }
        if best is None or candidate["distance"] < best["distance"]:
            best = candidate
    return best


def compute_best_seq_dists(
    sim,
    vp_conf: VPConf,
    start_pos: List[float],
    category_objects: Dict[str, List[Tuple]],
    categories: Tuple[str, ...],
) -> Optional[Dict]:
    candidate_lists = [category_objects.get(cat, []) for cat in categories]
    if any(not candidates for candidates in candidate_lists):
        return None

    start_np = np.array(start_pos, dtype=np.float32)
    best_first = best_target_from_start(
        sim, vp_conf, start_np, candidate_lists[0], correct_start=False
    )
    if best_first is None:
        return None

    best_seq = {"0": best_first}
    for step in range(1, len(categories)):
        from_candidates = candidate_lists[step - 1]
        to_candidates = candidate_lists[step]
        step_entries = []
        for instance_id, obj in from_candidates:
            if not ensure_viewpoints(sim, obj, vp_conf):
                entry = {
                    "from_instance_id": instance_id,
                    "best_instance_id": None,
                    "distance": None,
                    "view_point": None,
                }
                step_entries.append(entry)
                continue
            # Try each viewpoint until we find a reachable path (max 10 attempts)
            best_next = None
            for vp in obj.view_pts[:10]:
                start_from_obj = np.array(vp["agent_state"]["position"], dtype=np.float32)
                best_next = best_target_from_start(
                    sim, vp_conf, start_from_obj, to_candidates, correct_start=True
                )
                if best_next is not None:
                    break
            if best_next is None:
                entry = {
                    "from_instance_id": instance_id,
                    "best_instance_id": None,
                    "distance": None,
                    "view_point": None,
                }
            else:
                entry = {
                    "from_instance_id": instance_id,
                    "best_instance_id": best_next["best_instance_id"],
                    "distance": best_next["distance"],
                    "view_point": best_next["view_point"],
                }
            step_entries.append(entry)
        best_seq[str(step)] = step_entries

    return best_seq


def build_ordered_combos(categories: List[str], target_count: int, rng) -> List[Tuple[str, ...]]:
    combos = list(itertools.permutations(categories, NUM_GOALS))
    if not combos:
        return []
    if len(combos) <= target_count:
        return combos
    indices = rng.choice(len(combos), size=target_count, replace=False)
    return [combos[i] for i in indices]


def add_floor_to_goals(
    goals_by_category: Dict,
    semantic_by_id: Dict,
    floor_heights: List[float],
) -> int:
    missing = 0
    for cat, instances in goals_by_category.items():
        for inst in instances:
            obj = semantic_by_id.get(inst["object_id"])
            if obj is None:
                pos = inst.get("position")
                if pos is not None:
                    inst["floor"] = assign_floor_id(pos[1], floor_heights)
                else:
                    inst["floor"] = -1
                missing += 1
                continue
            inst["floor"] = assign_floor_id(obj.bbox.center[1], floor_heights)
    return missing


def build_category_objects(
    goals_by_category: Dict,
    semantic_by_id: Dict,
    scene_data: Optional[SceneData] = None,
) -> Dict[str, List[Tuple]]:
    category_objects: Dict[str, List[Tuple]] = {}
    for cat, instances in goals_by_category.items():
        normalized = normalize_category_key(cat, instances)
        # The strict loader has already rebuilt all six target lists from raw
        # semantic names. Use those lists directly so generation both removes
        # polluted IDs and adds valid aliases absent from the source JSON.
        if scene_data is not None and normalized in scene_data.object_locations:
            strict_objects = scene_data.object_locations[normalized]
            category_objects[cat] = [
                (obj.semantic_id, obj) for obj in strict_objects
                if obj.semantic_id is not None
            ]
            continue
        objs = []
        for inst in instances:
            obj = semantic_by_id.get(inst["object_id"])
            if obj is None:
                view_pts = inst.get("view_points")
                if view_pts:
                    objs.append((inst["object_id"], GoalInstance(inst["object_id"], view_pts)))
                continue
            objs.append((inst["object_id"], obj))
        category_objects[cat] = objs
    return category_objects


def serialize_strict_goals(
    source_keys: Dict[str, str],
    canonical_locations: Dict[str, List],
    floor_heights: List[float],
    sim,
    vp_conf: VPConf,
) -> Dict:
    """Write goals that use the same six-class taxonomy as path generation."""
    result = {}
    for source_key, category in source_keys.items():
        records = []
        for obj in canonical_locations.get(category, []):
            ensure_viewpoints(sim, obj, vp_conf)
            records.append({
                "position": [float(v) for v in obj.bbox.center],
                "radius": None,
                "object_id": obj.semantic_id,
                "object_name": f"{obj.object_category}_{obj.semantic_id}",
                "object_name_id": None,
                "object_category": category,
                "room_id": None,
                "room_name": None,
                "view_points": obj.view_pts or [],
                "floor": assign_floor_id(float(obj.bbox.center[1]), floor_heights),
            })
        result[source_key] = records
    return result


def generate_scene_dataset(
    json_data: Dict,
    sim,
    vp_conf: VPConf,
    rng,
    floor_counts: Dict[str, int],
) -> Optional[Dict]:
    episodes = json_data.get("episodes", [])
    if not episodes:
        return None

    scene_id = episodes[0]["scene_id"]
    goals_by_category = json_data.get("goals_by_category", {})
    categories = []
    category_name_map = {}
    for key, instances in goals_by_category.items():
        normalized = normalize_category_key(key, instances)
        category_name_map[key] = normalized
        categories.append(normalized)

    scene_key = scene_short_id(scene_id)
    target_floors = floor_counts.get(scene_key)
    scene_data = build_scene_data(scene_id, goals_by_category)
    scene_data = HM3DDataset.load_hm3d_objects(scene_data, sim.semantic_scene.objects, scene_id)
    semantic_by_id = build_semantic_index(scene_data, scene_id)

    # object_locations may use source keys such as
    # ``scene.basis.glb_tv_monitor``. Build a canonical view for generation.
    canonical_locations = {}
    for key, objects in scene_data[scene_id].object_locations.items():
        normalized = category_name_map.get(key, normalize_category_key(key, []))
        canonical_locations.setdefault(normalized, []).extend(objects)
    floor_heights = cluster_floors_from_objects(canonical_locations, target_floors)
    strict_goals_by_category = serialize_strict_goals(
        category_name_map,
        canonical_locations,
        floor_heights,
        sim,
        vp_conf,
    )
    strict_scene_data = SceneData(scene_id, canonical_locations, {})
    category_objects = build_category_objects(
        goals_by_category, semantic_by_id, strict_scene_data
    )
    normalized_objects: Dict[str, List[Tuple]] = {}
    for key, objs in category_objects.items():
        normalized = category_name_map.get(key, key)
        normalized_objects.setdefault(normalized, []).extend(objs)

    combos = build_ordered_combos(categories, len(episodes), rng)
    new_episodes: List[Dict] = []
    for idx, combo in enumerate(combos):
        if idx >= len(episodes):
            break
        start_ep = episodes[idx]
        # Keep the established reachability filter used to construct the
        # benchmark, but do not serialize its precomputed path chain. The
        # released posterior metric recomputes each segment independently from
        # the actual recorded start pose.
        reachable_chain = compute_best_seq_dists(
            sim, vp_conf, start_ep["start_position"], normalized_objects, combo
        )
        if reachable_chain is None:
            continue
        new_episodes.append(
            {
                "scene_id": start_ep["scene_id"],
                "start_position": start_ep["start_position"],
                "start_rotation": start_ep["start_rotation"],
                "object_goals": list(combo),
            }
        )

    return {
        "episodes": new_episodes,
        "goals_by_category": strict_goals_by_category,
        "floor_info": {
            "num_floors": target_floors if target_floors else len(floor_heights) + 1,
            "thresholds": floor_heights,
        },
    }


def generate_dataset(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(input_dir))
    rng = np.random.default_rng(RANDOM_SEED)
    vp_conf = VPConf(1.0, 0.1, 0.05)
    floor_counts = load_floor_counts(FLOOR_META_CSV)

    for file in tqdm(files, desc="Scenes"):
        if not file.endswith(".json.gz"):
            continue
        input_path = os.path.join(input_dir, file)
        with gzip.open(input_path, "rt") as f:
            json_data = json.load(f)
        episodes = json_data.get("episodes", [])
        if not episodes:
            continue

        scene_id = episodes[0]["scene_id"]
        sim_scene_id = scene_id
        if sim_scene_id.startswith("hm3d_v0.2/minival/"):
            sim_scene_id = "val/" + sim_scene_id[len("hm3d_v0.2/minival/"):]
        elif sim_scene_id.startswith("hm3d_v0.2/"):
            sim_scene_id = sim_scene_id[len("hm3d_v0.2/"):]
        sim = build_sim(PATH_TO_HM3D_V0_2, sim_scene_id, START_POSES_TILT_ANGLE, False)
        output_data = generate_scene_dataset(json_data, sim, vp_conf, rng, floor_counts)
        sim.close()

        if output_data is None:
            continue
        output_path = os.path.join(output_dir, file)
        with gzip.open(output_path, "wt") as f:
            json.dump(output_data, f)


if __name__ == "__main__":
    generate_dataset(INPUT_DIR, OUTPUT_DIR)
