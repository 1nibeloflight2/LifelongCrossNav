# agent_max_climb = 0.3（允许跨越 30cm 台阶）
# agent_max_slope = 60°（允许 60° 坡度，楼梯一般 30-40°）
# cell_size = 0.03（水平分辨率 3cm）
# cell_height = 0.05（垂直分辨率 5cm，关键参数——默认 0.2m 无法分辨楼梯台阶）

import os
import sys
import gzip
import json
import argparse
import glob
import numpy as np
import pandas as pd
import habitat_sim

from eval.habitat_multi_evaluator import HabitatMultiEvaluator
from eval.habitat_utils import get_closest_dist, get_dist_to_bbox_3d
from eval.navmesh_utils import recompute_cross_floor_navmesh
from eval.dataset_utils import HM3DDataset
from eval.dataset_utils.target_categories import (
    canonical_target_category,
    matches_target_category,
)
from config import load_eval_config

RESULT_NAMES = {
    1: 'SUCCESS',
    2: 'FAILURE_MISDETECT',
    3: 'FAILURE_STUCK',
    4: 'FAILURE_OOT',
    5: 'FAILURE_NOT_REACHED',
    6: 'FAILURE_ALL_EXPLORED',
}

def is_valid_instance_for_category(category, obj_name):
    """Use the same strict taxonomy as runtime success evaluation."""
    canonical = canonical_target_category(category)
    if canonical is None:
        return True
    return matches_target_category(canonical, obj_name)


def _parse_reader_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read OneMap or LifelongCrossNav single-/multi-target results "
            "and compute metrics using either the OneMap-v1 or posterior protocol"
        )
    )
    parser.add_argument('--type', type=str.lower, choices=('auto', 'lcn', 'onemap'), default='auto')
    parser.add_argument('--out', default=None, help='Result root directory')
    parser.add_argument('--config', default='config/mon/eval_multi_conf_floor.yaml')
    parser.add_argument(
        '--metric_protocol',
        type=str.lower,
        choices=('auto', 'onemap_v1', 'posterior'),
        default='auto',
        help=(
            "Metric protocol. 'auto' uses onemap_v1 for single-target configs "
            "and posterior for multi-target configs."
        ),
    )
    return parser.parse_args(argv)


def _detect_result_type(root, requested):
    if requested != 'auto':
        return requested
    if os.path.isfile(os.path.join(root, 'summary.json')) or glob.glob(os.path.join(root, 'episode_*')):
        return 'lcn'
    if os.path.isdir(os.path.join(root, 'state')):
        return 'onemap'
    raise FileNotFoundError(f"Cannot detect result format under: {root}")


def _build_result_records(root, result_type):
    """Return normalized episode records for both storage layouts."""
    records = []
    if result_type == 'onemap':
        state_dir = os.path.join(root, 'state')
        pose_dir = os.path.join(root, 'trajectories')
        for filename in sorted(
            (f for f in os.listdir(state_dir) if f.startswith('state_') and f.endswith('.txt')),
            key=lambda value: int(value[6:-4]),
        ):
            episode_id = int(filename[6:-4])
            records.append({
                'episode_id': episode_id,
                'label': str(episode_id),
                'state_path': os.path.join(state_dir, filename),
                'pose_pattern': os.path.join(pose_dir, f'poses_{episode_id}_{{seq}}.csv'),
                'result_values': None,
            })
        return records

    summary_path = os.path.join(root, 'summary.json')
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding='utf-8') as handle:
            summary = json.load(handle)
    for entry in summary.get('episodes', []):
        episode_id = entry.get('episode_id')
        label = str(entry.get('episode_label', episode_id))
        episode_dir = os.path.join(root, f'episode_{label}')
        state_matches = sorted(glob.glob(os.path.join(episode_dir, f'state_{label}_*.txt')))
        if not state_matches:
            state_matches = sorted(glob.glob(os.path.join(episode_dir, 'state_*.txt')))
        records.append({
            'episode_id': episode_id,
            'label': label,
            'state_path': state_matches[0] if state_matches else None,
            'pose_pattern': os.path.join(episode_dir, f'trajectory_{label}_{{seq}}_*.csv'),
            'result_values': entry.get('result_values'),
        })
    # Support an interrupted run whose episode directory exists before summary write.
    known = {str(record['label']) for record in records}
    for episode_dir in sorted(glob.glob(os.path.join(root, 'episode_*'))):
        label = os.path.basename(episode_dir)[len('episode_'):]
        if label in known:
            continue
        state_matches = sorted(glob.glob(os.path.join(episode_dir, 'state_*.txt')))
        if not state_matches:
            continue
        try:
            episode_id = int(label)
        except ValueError:
            episode_id = label
        records.append({
            'episode_id': episode_id,
            'label': label,
            'state_path': state_matches[0],
            'pose_pattern': os.path.join(episode_dir, f'trajectory_{label}_{{seq}}_*.csv'),
            'result_values': None,
        })
    return records


def main(argv=None):
    args = _parse_reader_args(argv)
    # Load the evaluation configuration
    # Spock parses process argv itself, so expose only its supported flag.
    sys.argv = [sys.argv[0], '--config', args.config]
    eval_config = load_eval_config()
    evaluator = HabitatMultiEvaluator(eval_config.EvalConf, None)
    seq_len = int(getattr(evaluator, 'num_seq', 3))
    metric_protocol = args.metric_protocol
    if metric_protocol == 'auto':
        metric_protocol = 'posterior' if evaluator.multi_object else 'onemap_v1'
    if metric_protocol == 'onemap_v1' and evaluator.multi_object:
        raise SystemExit(
            "metric_protocol=onemap_v1 is only defined for the single-target "
            "HM3D-v1 benchmark; use --metric_protocol posterior for multi-target data"
        )

    path = os.path.abspath(args.out or 'results_multi_floor')
    result_type = _detect_result_type(path, args.type)
    output_csv = os.path.join(path, 'results.csv')
    records = _build_result_records(path, result_type)
    print(
        f"Result format: {result_type}; metric protocol: {metric_protocol}; "
        f"root: {path}"
    )

    # Load goals_by_category for each scene from the dataset files
    scene_goals = {}  # scene_id -> goals_by_category
    dataset_path = evaluator.object_nav_path
    for file in sorted(os.listdir(dataset_path)):
        if not file.endswith('.json.gz'):
            continue
        with gzip.open(os.path.join(dataset_path, file), 'rt') as f:
            json_data = json.load(f)
        if not json_data.get('episodes'):
            continue
        scene_id = json_data['episodes'][0]['scene_id']
        scene_goals[scene_id] = json_data.get('goals_by_category', {})

    # Build sim helper
    from eval.dataset_utils.object_nav_utils import object_nav_gen
    sim = None
    current_scene = None
    # Cache: scene_id -> {category: [SemanticObject, ...]} (all instances including name-matched)
    scene_all_objects = {}

    def get_or_build_sim(scene_id):
        nonlocal sim, current_scene
        if sim is not None and current_scene == scene_id:
            return sim
        if sim is not None:
            sim.close()
        sim = object_nav_gen.build_sim(evaluator.scene_path, scene_id, 15, False)
        recompute_cross_floor_navmesh(sim)
        current_scene = scene_id
        return sim

    def get_all_objects_for_scene(scene_id):
        """Get all object instances for a scene (JSON-listed + name-matched from semantic mesh)."""
        if scene_id in scene_all_objects:
            return scene_all_objects[scene_id]

        # Build scene_data with categories from JSON
        goals_by_cat = scene_goals.get(scene_id, {})
        from eval.dataset_utils.common import SceneData
        scene_data_local = {}
        sd = SceneData(scene_id, {}, {})
        for cat_key, instances in goals_by_cat.items():
            cat_name = cat_key.split('_', 1)[-1] if '.glb_' in cat_key else cat_key
            sd.object_locations[cat_name] = []
            sd.object_ids[cat_name] = []
            for inst in instances:
                sd.object_ids[cat_name].append(inst['object_id'])
        scene_data_local[scene_id] = sd

        # Load all objects from semantic scene (includes name-matched ones)
        sim_inst = get_or_build_sim(scene_id)
        scene_data_local = HM3DDataset.load_hm3d_objects(
            scene_data_local, sim_inst.semantic_scene.objects, scene_id
        )

        # Filter out invalid substring matches (e.g., "toilet paper" matched to "toilet")
        filtered = {}
        for cat, objs in scene_data_local[scene_id].object_locations.items():
            filtered[cat] = [obj for obj in objs if is_valid_instance_for_category(cat, obj.object_category)]
        scene_all_objects[scene_id] = filtered
        return scene_all_objects[scene_id]

    def get_viewpoints_for_instance(goals_by_cat, semantic_id):
        """Get viewpoints from JSON for a given semantic_id, or None if not listed."""
        for cat_key, instances in goals_by_cat.items():
            for inst in instances:
                if inst['object_id'] == semantic_id:
                    return inst.get('view_points', [])
        return []

    def compute_best_geodesic_to_category(sim_inst, start_pos, category, scene_id, goals_by_cat):
        """Compute the minimum geodesic from this segment's actual CSV start.

        Target endpoints range over all valid instances of the category. The
        dataset's precomputed ``best_seq_dists`` is intentionally not used.
        """
        all_objects = get_all_objects_for_scene(scene_id)
        instances = all_objects.get(category, [])
        if not instances:
            return np.inf, None

        start = np.asarray(start_pos, dtype=np.float32)
        if not sim_inst.pathfinder.is_navigable(start):
            start = sim_inst.pathfinder.snap_point(start)
            if np.isnan(start).any():
                return np.inf, None

        min_dist = np.inf
        best_label = None

        for sem_obj in instances:
            viewpoints = get_viewpoints_for_instance(goals_by_cat, sem_obj.semantic_id)
            if viewpoints:
                target_points = [
                    viewpoint['agent_state']['position']
                    for viewpoint in viewpoints
                ]
                mg_path = habitat_sim.MultiGoalShortestPath()
                mg_path.requested_start = start
                mg_path.requested_ends = np.asarray(target_points, dtype=np.float32)
                sim_inst.pathfinder.find_path(mg_path)
                dist = mg_path.geodesic_distance
            else:
                center = np.asarray(sem_obj.bbox.center, dtype=np.float32)
                snapped = sim_inst.pathfinder.snap_point(center)
                if np.isnan(snapped).any():
                    continue
                shortest_path = habitat_sim.ShortestPath()
                shortest_path.requested_start = start
                shortest_path.requested_end = snapped
                sim_inst.pathfinder.find_path(shortest_path)
                dist = shortest_path.geodesic_distance

            if dist < min_dist:
                min_dist = dist
                best_label = f"{sem_obj.object_category}_{sem_obj.semantic_id}"

        return min_dist, best_label

    def find_closest_instance_at_end(sim_inst, end_pos, category, scene_id, max_vertical_diff=1.0):
        """Associate the trajectory end with the nearest valid target instance.

        Prefer the established same-floor 2D AABB distance.  If no instance
        satisfies the vertical gate, fall back to the nearest 3D AABB across
        all floors.  ``None`` therefore means that strict semantic filtering
        found no valid instance of this category in the scene.
        """
        all_objects = get_all_objects_for_scene(scene_id)
        instances = all_objects.get(category, [])
        if not instances:
            return None

        # Find the closest instance with height+distance constraint
        min_dist = np.inf
        best_label = None
        for sem_obj in instances:
            h = abs(end_pos[1] - sem_obj.bbox.center[1])
            if h > max_vertical_diff:
                continue
            d = get_closest_dist(end_pos, [sem_obj], is_gibson=False)
            if d < min_dist:
                min_dist = d
                best_label = f"{sem_obj.object_category}_{sem_obj.semantic_id}"

        if best_label is not None:
            return best_label

        # No same-floor candidate: retain a useful diagnostic association by
        # selecting the closest valid instance using full 3D point-to-AABB
        # distance, rather than returning None merely because it is upstairs
        # or downstairs from the trajectory endpoint.
        for sem_obj in instances:
            d = get_dist_to_bbox_3d(
                np.asarray(sem_obj.bbox.center, dtype=np.float32),
                np.asarray(sem_obj.bbox.sizes, dtype=np.float32),
                np.asarray(end_pos, dtype=np.float32),
            )
            if d < min_dist:
                min_dist = d
                best_label = f"{sem_obj.object_category}_{sem_obj.semantic_id}"

        return best_label

    rows = []
    total_files = len(records)
    print(f"Processing {total_files} episodes...")

    episodes_by_id = {str(episode.episode_id): episode for episode in evaluator.episodes}
    for idx, record in enumerate(records):
        experiment_num = record['episode_id']

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Progress: {idx + 1}/{total_files} episodes processed")

        episode = episodes_by_id.get(str(experiment_num))
        if episode is None and isinstance(experiment_num, int) and experiment_num < len(evaluator.episodes):
            episode = evaluator.episodes[experiment_num]
        if episode is None:
            print(f"Warning: episode {experiment_num} is not present in the configured dataset")
            continue

        state_values = record.get('result_values')
        if state_values is None and record.get('state_path'):
            with open(record['state_path'], 'r') as file:
                content = file.read().strip()
            state_values = [int(val) for val in content.split(',')] if content else []
        state_values = state_values or []
        scene_id = episode.scene_id
        goals_by_cat = scene_goals.get(scene_id, {})

        # Initialize per-sequence arrays
        SR_per = [None] * seq_len
        instance_per = [None] * seq_len
        result_per = [None] * seq_len
        # Object names describe the episode definition, not only the segments
        # that happened to execute.  Retain the configured sequence length;
        # result_per/SR_per already express early termination explicitly.
        object_per = list(episode.obj_sequence[:seq_len])
        d_short_per = [None] * seq_len
        d_real_per = [None] * seq_len
        SPL_per = [None] * seq_len

        # Fill SR_per based on state file directly (1=success, else=0)
        for seq_num in range(seq_len):
            if seq_num < len(state_values):
                result_value = state_values[seq_num]
                SR_per[seq_num] = 1 if result_value == 1 else 0
                result_per[seq_num] = RESULT_NAMES.get(
                    result_value, f'UNKNOWN_RESULT_{result_value}'
                )
            else:
                break

        # Compute each attempted segment from its own recorded trajectory.
        for seq_num in range(seq_len):
            if SR_per[seq_num] is None:
                continue

            pose_pattern = record['pose_pattern'].format(seq=seq_num)
            pose_matches = sorted(glob.glob(pose_pattern))
            pose_file = pose_matches[0] if pose_matches else pose_pattern
            if not os.path.isfile(pose_file):
                print(f"Warning: Pose file not found for experiment {experiment_num} seq {seq_num}")
                continue

            poses = np.genfromtxt(pose_file, delimiter=",", skip_header=1)
            # OneMap trajectories contain four columns (map xyz + yaw), while
            # LCN trajectories additionally append Habitat xyz and therefore
            # contain seven columns.  A one-row CSV is returned as a 1-D array
            # by NumPy, so preserve its actual column count instead of assuming
            # the legacy four-column layout.
            if poses.size == 0:
                print(
                    f"Warning: Empty pose file for experiment {experiment_num} "
                    f"seq {seq_num}: {pose_file}"
                )
                continue
            poses = np.atleast_2d(poses)
            if poses.shape[1] < 3:
                print(
                    f"Warning: Invalid pose file for experiment {experiment_num} "
                    f"seq {seq_num} (expected at least 3 columns, got "
                    f"{poses.shape[1]}): {pose_file}"
                )
                continue
            # LCN stores authoritative Habitat agent-body xyz in columns 4:7;
            # the first three map columns use camera height.  Validate the
            # coordinates that will actually be used for geodesic evaluation.
            habitat_columns = poses.shape[1] >= 7
            coordinate_slice = slice(4, 7) if habitat_columns else slice(0, 3)
            poses = poses[np.isfinite(poses[:, coordinate_slice]).all(axis=1)]
            if len(poses) == 0:
                print(
                    f"Warning: No valid xyz poses for experiment {experiment_num} "
                    f"seq {seq_num}: {pose_file}"
                )
                continue

            # OneMap-v1 measures travelled distance in the map XY plane.  The
            # posterior multi-floor protocol retains full 3D trajectory length.
            if len(poses) > 1:
                distance_dims = 2 if metric_protocol == 'onemap_v1' else 3
                d_real = np.linalg.norm(
                    poses[1:, :distance_dims] - poses[:-1, :distance_dims], axis=1
                ).sum()
            else:
                d_real = 0.0
            d_real_per[seq_num] = d_real

            # Preserve the established metric protocol. The first segment
            # skips the initialization pose when a second sample exists;
            # subsequent segments start from their first recorded pose. These
            # are real navigable agent positions, not object centres.
            if seq_num == 0 and len(poses) >= 2:
                start_idx = 1
            else:
                start_idx = 0

            if habitat_columns:
                # LCN: use the saved Habitat body coordinates directly.  The
                # map z column is camera height (body y + 0.88 m) and can fail
                # navmesh snapping or snap onto the wrong floor.
                start_pos_sim = np.asarray(poses[start_idx, 4:7], dtype=np.float32)
                end_pos_sim = np.asarray(poses[-1, 4:7], dtype=np.float32)
            else:
                # OneMap legacy format: (-z_hab, -x_hab, y_hab, yaw).
                start_pose = poses[start_idx, :3]
                end_pose = poses[-1, :3]
                start_pos_sim = np.array(
                    [-start_pose[1], start_pose[2], -start_pose[0]], dtype=np.float32
                )
                end_pos_sim = np.array(
                    [-end_pose[1], end_pose[2], -end_pose[0]], dtype=np.float32
                )

            target_cat = episode.obj_sequence[seq_num] if seq_num < len(episode.obj_sequence) else 'unknown'
            if metric_protocol == 'onemap_v1':
                # Reproduce the original OneMap single-target protocol: the
                # HM3D-v1 dataset already stores the oracle geodesic distance
                # in episode.info.geodesic_distance. Do not recompute it from
                # the recorded trajectory or a rebuilt navmesh.
                dataset_best_dist = episode.best_dist
                if isinstance(dataset_best_dist, (list, tuple, np.ndarray)):
                    dataset_best_dist = (
                        dataset_best_dist[seq_num]
                        if seq_num < len(dataset_best_dist)
                        else np.nan
                    )
                try:
                    dataset_best_dist = float(dataset_best_dist)
                except (TypeError, ValueError):
                    dataset_best_dist = np.nan
                d_short_per[seq_num] = (
                    dataset_best_dist if np.isfinite(dataset_best_dist) else np.nan
                )
                # Instance association is not part of the original OneMap-v1
                # metric. Avoid rebuilding a navmesh solely for this optional
                # diagnostic column; object_per still records the target class.
                instance_per[seq_num] = None
            else:
                sim_inst = get_or_build_sim(scene_id)
                best_dist, _ = compute_best_geodesic_to_category(
                    sim_inst, start_pos_sim, target_cat, scene_id, goals_by_cat
                )
                if best_dist < np.inf:
                    d_short_per[seq_num] = best_dist
                else:
                    print(
                        f"Warning: No geodesic path for exp {experiment_num} "
                        f"seq {seq_num} ({target_cat})"
                    )
                    d_short_per[seq_num] = np.nan
                # Instance association is diagnostic output only. It does not
                # replace the next segment's actual CSV start pose.
                instance_per[seq_num] = find_closest_instance_at_end(
                    sim_inst, end_pos_sim, target_cat, scene_id
                )

        # Compute SPL_per
        for seq_num in range(seq_len):
            if SR_per[seq_num] is None:
                SPL_per[seq_num] = None
            elif SR_per[seq_num] == 0:
                SPL_per[seq_num] = 0.0
            else:
                if d_short_per[seq_num] is not None and d_real_per[seq_num] is not None:
                    d_s = d_short_per[seq_num]
                    d_r = d_real_per[seq_num]
                    if max(d_s, d_r) > 0:
                        SPL_per[seq_num] = d_s / max(d_s, d_r)
                    else:
                        SPL_per[seq_num] = 0.0
                else:
                    SPL_per[seq_num] = 0.0

        # Compute episode-level metrics
        non_null_sr = [s for s in SR_per if s is not None]
        SR = 1 if (len(non_null_sr) == seq_len and all(s == 1 for s in non_null_sr)) else 0

        success_count = sum(1 for s in SR_per if s == 1)
        PR = success_count / seq_len

        valid_d_short = [d for d in d_short_per if d is not None and not np.isnan(d)]
        valid_d_real = [d for d in d_real_per if d is not None and not np.isnan(d)]
        if SR == 1 and len(valid_d_short) > 0 and len(valid_d_real) > 0:
            sum_d_short = sum(valid_d_short)
            sum_d_real = sum(valid_d_real)
            SPL = SR * sum_d_short / max(sum_d_short, sum_d_real)
        else:
            SPL = 0.0

        d_short_success = []
        d_real_success = []
        for seq_num in range(seq_len):
            if SR_per[seq_num] == 1 and d_short_per[seq_num] is not None and d_real_per[seq_num] is not None:
                d_short_success.append(d_short_per[seq_num])
                d_real_success.append(d_real_per[seq_num])

        if len(d_short_success) > 0 and len(d_real_success) > 0:
            sum_ds = sum(d_short_success)
            sum_dr = sum(d_real_success)
            PPL = PR * sum_ds / max(sum_ds, sum_dr)
        else:
            PPL = 0.0

        rows.append({
            'experiment': experiment_num,
            'scene': scene_id,
            'SR_per': SR_per,
            'result_per': result_per,
            'object_per': object_per,
            'instance': instance_per,
            'd_short_per': d_short_per,
            'd_real_per': d_real_per,
            'SPL_per': SPL_per,
            'SR': SR,
            'SPL': SPL,
            'PR': PR,
            'PPL': PPL,
        })

    if sim is not None:
        sim.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved per-episode results to {output_csv}")
    print(f"Total episodes evaluated: {len(df)}")

    # === Compute overall metrics ===
    if len(df) == 0:
        print("No data to compute overall metrics.")
        return

    SR_all = df['SR'].mean()
    SPL_all = df['SPL'].mean()
    PR_all = df['PR'].mean()
    PPL_all = df['PPL'].mean()

    # Per-sequence metrics
    N_per_valid = []
    SR_per_all_condition = []
    SPL_per_all_condition = []
    SPL_per_all_global = []
    SR_per_all_global = []

    for seq_num in range(seq_len):
        sr_vals = [row[seq_num] for row in df['SR_per']]
        spl_vals = [row[seq_num] for row in df['SPL_per']]

        valid_sr = [v for v in sr_vals if v is not None]
        valid_spl = [v for v in spl_vals if v is not None]

        N_valid = len(valid_sr)
        N_per_valid.append(N_valid)

        if N_valid > 0:
            SR_cond = sum(1 for v in valid_sr if v == 1) / N_valid
            SR_per_all_condition.append(SR_cond)

            SPL_cond = sum(valid_spl) / len(valid_spl) if len(valid_spl) > 0 else 0.0
            SPL_per_all_condition.append(SPL_cond)

            SPL_global = sum(v if v is not None else 0.0 for v in spl_vals) / len(df)
            SPL_per_all_global.append(SPL_global)

            SR_global = sum(v if v is not None else 0.0 for v in sr_vals) / len(df)
            SR_per_all_global.append(SR_global)
        else:
            SR_per_all_condition.append(0.0)
            SPL_per_all_condition.append(0.0)
            SPL_per_all_global.append(0.0)
            SR_per_all_global.append(0.0)

    print("\n========== Overall Metrics ==========")
    print(f"SR_all:   {SR_all:.4f}")
    print(f"SPL_all:  {SPL_all:.4f}")
    print(f"PR_all:   {PR_all:.4f}")
    print(f"PPL_all:  {PPL_all:.4f}")
    print(f"N_per_valid:            {N_per_valid}")
    print(f"SR_per_all_condition:   {[f'{v:.4f}' for v in SR_per_all_condition]}")
    print(f"SPL_per_all_condition:  {[f'{v:.4f}' for v in SPL_per_all_condition]}")
    print(f"SPL_per_all_global:     {[f'{v:.4f}' for v in SPL_per_all_global]}")
    print(f"SR_per_all_global:      {[f'{v:.4f}' for v in SR_per_all_global]}")

    # Progress distribution histogram
    pr_counts = df['PR'].value_counts().sort_index()
    print("\n========== Progress Distribution ==========")
    for pr_val, count in pr_counts.items():
        print(f"PR = {pr_val:.3f}: {count} episodes")

    # Expand episode rows into one row per attempted target so object-specific
    # failure causes remain available. Scene metrics below stay episode-level.
    sequence_rows = []
    for row in rows:
        for seq_num in range(seq_len):
            result_name = row['result_per'][seq_num]
            if result_name is None:
                continue
            sequence_rows.append({
                'experiment': row['experiment'],
                'sequence': seq_num,
                'scene': row['scene'],
                'object': row['object_per'][seq_num] or 'unknown',
                'result': result_name,
            })

    sequence_df = pd.DataFrame(sequence_rows)
    if len(sequence_df) == 0:
        return

    result_columns = list(RESULT_NAMES.values())

    def build_distribution(group_column):
        counts = pd.crosstab(sequence_df[group_column], sequence_df['result'])
        counts = counts.reindex(columns=result_columns, fill_value=0)
        totals = counts.sum(axis=1)
        display = pd.DataFrame(index=counts.index)
        display['TOTAL'] = totals
        for result_name in result_columns:
            display[result_name] = [
                f"{count} ({count / total:.1%})" if total else "0 (0.0%)"
                for count, total in zip(counts[result_name], totals)
            ]
        display['SUCCESS_RATE'] = counts['SUCCESS'] / totals
        return display.sort_values(
            by=['SUCCESS_RATE', 'TOTAL'], ascending=[False, False]
        )

    object_distribution = build_distribution('object')
    print("\n========== Result Distribution by Object ==========")
    print(object_distribution.to_string(
        formatters={'SUCCESS_RATE': lambda value: f'{value:.1%}'}
    ))

    def summarize_scene(scene_name, scene_df):
        """Compute the same metrics as Overall Metrics for one scene."""
        n_episodes = len(scene_df)
        n_per_valid = []
        sr_per_condition = []
        spl_per_condition = []
        spl_per_global = []
        sr_per_global = []

        for seq_num in range(seq_len):
            sr_vals = [values[seq_num] for values in scene_df['SR_per']]
            spl_vals = [values[seq_num] for values in scene_df['SPL_per']]
            valid_sr = [value for value in sr_vals if value is not None]
            valid_spl = [value for value in spl_vals if value is not None]

            n_valid = len(valid_sr)
            n_per_valid.append(n_valid)
            if n_valid > 0:
                sr_per_condition.append(
                    sum(1 for value in valid_sr if value == 1) / n_valid
                )
                spl_per_condition.append(
                    sum(valid_spl) / len(valid_spl) if valid_spl else 0.0
                )
            else:
                sr_per_condition.append(0.0)
                spl_per_condition.append(0.0)

            sr_per_global.append(
                sum(value if value is not None else 0.0 for value in sr_vals)
                / n_episodes
            )
            spl_per_global.append(
                sum(value if value is not None else 0.0 for value in spl_vals)
                / n_episodes
            )

        format_metric_list = lambda values: '[' + ', '.join(
            f'{value:.4f}' for value in values
        ) + ']'
        return {
            'scene': scene_name,
            'TOTAL_EPISODES': n_episodes,
            'SR_scene_all': scene_df['SR'].mean(),
            'SPL_scene_all': scene_df['SPL'].mean(),
            'PR_scene_all': scene_df['PR'].mean(),
            'PPL_scene_all': scene_df['PPL'].mean(),
            'N_per_valid': str(n_per_valid),
            'SR_per_condition': format_metric_list(sr_per_condition),
            'SPL_per_condition': format_metric_list(spl_per_condition),
            'SPL_per_global': format_metric_list(spl_per_global),
            'SR_per_global': format_metric_list(sr_per_global),
        }

    scene_rows = [
        summarize_scene(scene_name, scene_df)
        for scene_name, scene_df in df.groupby('scene', sort=False)
    ]
    scene_metrics = pd.DataFrame(scene_rows).sort_values(
        by=['SR_scene_all', 'PR_scene_all', 'TOTAL_EPISODES'],
        ascending=[False, False, False],
    )
    print("\n========== Episode-level Metrics by Scene ==========")
    print(scene_metrics.to_string(
        index=False,
        formatters={
            'SR_scene_all': lambda value: f'{value:.4f}',
            'SPL_scene_all': lambda value: f'{value:.4f}',
            'PR_scene_all': lambda value: f'{value:.4f}',
            'PPL_scene_all': lambda value: f'{value:.4f}',
        },
    ))


if __name__ == "__main__":
    main()
