# eval utils
from eval import get_closest_dist, FMMPlanner
from eval.actor import Actor
from eval.dataset_utils.gibson_dataset import load_gibson_episodes
from mapping import rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *
from eval.dataset_utils import HM3DMultiFloorDataset
from eval.navmesh_utils import recompute_cross_floor_navmesh
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt

# os / filsystem
import bz2
import os
from os import listdir
import gzip
import json
import pathlib

# cv2
import cv2

# numpy
import numpy as np

# skimage
import skimage

# dataclasses
from dataclasses import dataclass

# quaternion
import quaternion

# typing
from typing import Tuple, List, Dict
import enum

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# tabulate
from tabulate import tabulate

# rerun
import rerun as rr

# pandas
import pandas as pd

# pickle
import pickle

# scipy
from scipy.spatial.transform import Rotation as R

SEQ_LEN = 3
class Result(enum.Enum):
    SUCCESS = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6


def _mapper_current_detection_habitat_pos(mapper):
    if hasattr(mapper, "current_confirmed_detection_voxel"):
        voxel = getattr(mapper, "current_confirmed_detection_voxel", None)
        if voxel is None or not hasattr(mapper, "one_map"):
            return None
        x, y, z = mapper.one_map.voxel_to_metric(*voxel)
        return np.array([-y, z, -x], dtype=np.float32)
    pos = getattr(mapper, "chosen_detection", None)
    if pos is not None and hasattr(mapper, "one_map"):
        pos_metric = mapper.one_map.px_to_metric(pos[0], pos[1])
        return np.array([-pos_metric[1], -pos_metric[0]], dtype=np.float32)
    return None


class Metrics:
    def __init__(self, ep_id) -> None:
        self.sequence_lengths = []
        self.sequence_results = []
        self.sequence_poses = []
        self.ep_id = ep_id
        self.sequence_object = []

    def add_sequence(self, sequence: np.ndarray, result: Result, target_object: str) -> None:
        start_id = 0
        if len(self.sequence_poses) > 0:
            start_id = sum([len(seq) for seq in self.sequence_poses])
        seq_poses = sequence[start_id:, :]
        self.sequence_poses.append(seq_poses)
        length = np.linalg.norm(seq_poses[1:, :2] - seq_poses[:-1, :2])
        self.sequence_results.append(result)
        self.sequence_lengths.append(length)
        self.sequence_object.append(target_object)

    def get_progress(self):
        return self.sequence_results.count(Result.SUCCESS) /SEQ_LEN


class HabitatMultiEvaluator:
    def __init__(self,
                 config: EvalConf,
                 actor: Actor,
                 ) -> None:
        self.config = config
        self.multi_object = config.multi_object
        self.max_steps = config.max_steps
        self.reset_map_between_targets = bool(getattr(config, "reset_map_between_targets", False))
        self.single_target_height_aware = bool(getattr(config, "single_target_height_aware", False))
        self.max_dist = config.max_dist
        self.controller = config.controller
        self.mapping = config.mapping
        self.planner = config.planner
        self.log_rerun = config.log_rerun
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
        self.is_gibson = config.is_gibson

        self.sim = None
        self.actor = actor
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.control_frequency = config.controller.control_freq
        self.max_vel = config.controller.max_vel
        self.max_ang_vel = config.controller.max_ang_vel
        self.time_step = 1.0 / self.control_frequency
        self.num_seq = SEQ_LEN if self.multi_object else 1
        self.square = config.square_im

        if self.multi_object:
            if 'objectnav_hm3d_MFMON' in self.object_nav_path:
                self.episodes, self.scene_data = HM3DMultiFloorDataset.load_hm3d_multi_floor_episodes(
                    self.episodes, self.scene_data, self.object_nav_path)
            else:
                self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                           self.scene_data,
                                                                                           self.object_nav_path)
        else:
            # The 3D batch entry uses the same simulator/runtime as the
            # multi-object evaluator, but standard HM3D ObjectNav episodes
            # contain a one-element obj_sequence.  Loading them here keeps
            # scene setup, strict category matching and cross-floor navmesh
            # reconstruction identical across both evaluation families.
            self.episodes, self.scene_data = HM3DDataset.load_hm3d_episodes(
                self.episodes,
                self.scene_data,
                self.object_nav_path,
            )
        if self.actor is not None:
            self.logger = rerun_logger.RerunLogger(self.actor.mapper, False, "") if self.log_rerun else None
        if self.is_gibson:
            self.results_path = "/home/finn/active/MON/results_gibson_multi"
        elif not self.multi_object:
            self.results_path = "results_single/"
        elif 'objectnav_hm3d_MFMON' in self.object_nav_path:
            self.results_path = "results_multi_floor/"
        else:
            self.results_path = "results_multi/"
        os.makedirs(os.path.join(self.results_path, 'state'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'trajectories'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'similarities'), exist_ok=True)

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path + scene_id

        backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        hfov = 90 if self.square else 79
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.hfov = hfov
        rgb.position = np.array([0, 0.88, 0])
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        res_x = 640
        res_y = 640 if self.square else 480
        rgb.resolution = [res_y, res_x]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.hfov = hfov
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.position = np.array([0, 0.88, 0])
        depth.resolution = [res_y, res_x]
        agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            turn_left=ActionSpec("turn_left", ActuationSpec(amount=30.0)),
            turn_right=ActionSpec("turn_right", ActuationSpec(amount=30.0)),
            look_up=ActionSpec("look_up", ActuationSpec(amount=30.0)),
            look_down=ActionSpec("look_down", ActuationSpec(amount=30.0)),
        ))
        agent_cfg.sensor_specifications = [rgb, depth]
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        # Runtime motion must use the same cross-floor navigation geometry as
        # post-hoc geodesic evaluation. The bundled HM3D navmesh is too coarse
        # for several stair and landing connections.
        recompute_cross_floor_navmesh(self.sim)
        if self.scene_data[scene_id].objects_loaded:
            return
        self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)

    def start_next_target(self, query: str) -> None:
        """Switch goals using the selected cross-target memory policy."""
        if self.reset_map_between_targets:
            self.actor.reset_for_next_target()
        self.actor.set_query(query, allow_poi=not self.reset_map_between_targets)

    def execute_action(self, action: Dict
                       ):
        if 'discrete' in action.keys():
            # We have a discrete actor
            self.sim.step(action['discrete'])

        elif 'continuous' in action.keys():
            # We have a continuous actor
            self.vel_control.angular_velocity = action['continuous']['angular']
            self.vel_control.linear_velocity = action['continuous']['linear']
            agent_state = self.sim.get_agent(0).state
            previous_rigid_state = habitat_sim.RigidState(
                utils.quat_to_magnum(agent_state.rotation), agent_state.position
            )

            # manually integrate the rigid state
            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            # snap rigid state to navmesh and set state to object/sim
            # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
            end_pos = self.sim.step_filter(
                previous_rigid_state.translation, target_rigid_state.translation
            )

            # set the computed state
            agent_state.position = end_pos
            agent_state.rotation = utils.quat_from_magnum(
                target_rigid_state.rotation
            )
            self.sim.get_agent(0).set_state(agent_state)
            self.sim.step_physics(self.time_step)

    def read_results(self, path, sort_by, data_pkl=None):
        from eval.dataset_utils import gen_multiobject_dataset
        from eval.dataset_utils.object_nav_utils import object_nav_gen
        state_dir = os.path.join(path, 'state')
        state_results = {}

        # Check if the state directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        # Iterate through all files in the state directory
        data = []
        sum_successes = 0
        if data_pkl is None:
            episodes = []
            scene_data = {}
            valid_start_positions = {}
            scene_floors = {}
            scenes = {}
            episodes, scene_data = HM3DDataset.load_hm3d_episodes(episodes, scene_data, gen_multiobject_dataset.path_to_hm3d_objectnav_v2)
            number_of_floors = gen_multiobject_dataset.load_scenes(episodes, scene_data, valid_start_positions, scene_floors, scenes)
            scene_loaded = {}
            sim = None
            for filename in sorted(os.listdir(state_dir)):
                if filename.startswith('state_') and filename.endswith('.txt'):
                    try:
                        # Extract the experiment number from the filename
                        experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'
                        # Read the content of the file
                        # if experiment_num > 10:
                        #     continue
                        with open(os.path.join(state_dir, filename), 'r') as file:
                            content = file.read().strip()

                        # Convert the content to a number (assuming it's a float)
                        state_values = content.split(',')
                        state_values = [int(val) for val in state_values]
                        # Store the result in the dictionary
                        # Create a row for each sequence in the experiment

                        for seq_num, value in enumerate(state_values):
                            spl = 0
                            map_size = 0
                            if value == 1:
                                if seq_num == 2:
                                    sum_successes += 1
                                poses = np.genfromtxt(os.path.join(pose_dir, "poses_" + str(experiment_num) + "_" +
                                                                   str(seq_num) + ".csv"), delimiter=",")
                                if len(poses.shape) == 1:
                                    poses = poses.reshape((1, 4))
                                path_length = np.linalg.norm(poses[1:, :3] - poses[:-1, :3], axis=1).sum()
                                # compute the optimal path length
                                if sim is None or not sim.curr_scene_name in self.episodes[experiment_num].scene_id:
                                    if sim is not None:
                                        sim.close()
                                        sim = None
                                    sim = gen_multiobject_dataset.build_sim(gen_multiobject_dataset.path_to_hm3d_v0_2, self.episodes[experiment_num].scene_id, gen_multiobject_dataset.start_poses_tilt_angle, True)
                                if self.episodes[experiment_num].scene_id not in scene_loaded:
                                    needs_save = gen_multiobject_dataset.load_all_scene_data(self.episodes[experiment_num].scene_id,
                                                                                scenes, scene_data,
                                                                                viewpoint_conf=object_nav_gen.VPConf(1.0, 0.1, 0.05), sim=sim)
                                    if needs_save:
                                        print(f"Storing viewpoints for scene {self.episodes[experiment_num].scene_id}")
                                        gen_multiobject_dataset.store_viewpoints(scenes, self.episodes[experiment_num].scene_id,"datasets/multi_object_data")
                                    scene_loaded[self.episodes[experiment_num].scene_id] = True

                                start_pos = poses[0, :3]
                                pos = np.array([-start_pos[1], start_pos[2], -start_pos[0]])
                                floor_data = scenes[self.episodes[experiment_num].scene_id].floors[self.episodes[experiment_num].floor_id]
                                possible_objs = floor_data.objects[self.episodes[experiment_num].obj_sequence[seq_num]]
                                min_dist = np.inf
                                top_down_map = maps.get_topdown_map(
                                                sim.pathfinder,
                                                height=start_pos[1],
                                                map_resolution=512,
                                                draw_border=True,
                                            )
                                map_size = top_down_map.shape[0] * top_down_map.shape[1]
                                obj_found = False
                                for obj in possible_objs:
                                    dist, next_start = object_nav_gen.get_geodesic(pos, sim, obj, correct_start=True)
                                    if dist is None:
                                        continue
                                    obj_found = True
                                    if dist < min_dist:
                                        min_dist = dist
                                best_dist = min_dist
                                if not obj_found:
                                    print(f"Warning: No object found for sequence {seq_num} in experiment {experiment_num}")
                                spl = min(1.0, 1 * (best_dist/ max(path_length, best_dist)))
                            optimal_total_path_length = sum([d[0] for d in self.episodes[experiment_num].best_dist])
                            data.append({
                                'experiment': experiment_num,
                                'sequence': seq_num,
                                'state': value,
                                'spl': spl / self.num_seq,
                                'map_size': map_size,
                                'opt_path': optimal_total_path_length,
                                'object': self.episodes[experiment_num].obj_sequence[seq_num],
                                'scene': self.episodes[experiment_num].scene_id
                            })

                        # deltas = poses[1:, :3] - poses[:-1, :3]
                        # distance_traveled = np.linalg.norm(deltas, axis=1).sum()
                        # if state_value == 1:
                        #     spl[experiment_num] = self.episodes[experiment_num].best_dist / max(
                        #         self.episodes[experiment_num].best_dist, distance_traveled)
                        # else:
                        #     spl[experiment_num] = 0
                        if self.episodes[experiment_num].episode_id != experiment_num:
                            print(
                                f"Warning, experiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                    except ValueError:
                        print(f"Warning: Skipping {filename} due to invalid format")
                    # except Exception as e:
                    #     print(f"Error reading {filename}: {str(e)}")
            data = pd.DataFrame(data)
        else:
            with open(data_pkl, 'rb') as f:
                data = pickle.load(f)
        # data = data[data['experiment'] < 88]
        states = data["state"].unique()
        # print(sum_successes/236)
        def has_success(group, seq_id):
            return group[(group['sequence'] == seq_id) & (group['state'] == 1)].shape[0] > 0
        def calc_prog_per_episode(group):
            successes = group.groupby('experiment').apply(lambda x: (x['state'] == 1).sum())
            progress = successes / self.num_seq
            return progress

        def calc_spl_per_episode(group):
            spls_per_exp = group.groupby('experiment')['spl'].sum()
            return spls_per_exp

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})
            progress = calc_prog_per_episode(group)
            spl = calc_spl_per_episode(group)
            s = progress[progress == 1]
            result['Progress'] = progress.mean()
            result['SPL'] = spl.mean()
            result['opt_PL'] = group['opt_path'].mean()
            result['Map Size'] = group['map_size'].mean() / 100
            result['s'] = s.sum() / len(progress)
            result['s_spl'] = spl[progress == 1].sum()/len(progress)

            # Calculate average SPL and multiply by 100
            # avg_spl = group['spl'].mean()
            # result['Average SPL'] = avg_spl

            return result

        # Per-object results
        object_results = data.groupby('object').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'object': 'Object'})

        # Per-scene results
        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        # Overall results
        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Function to format percentages
        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        data_per_scene = data.groupby('scene')
        sr_per_scene = []
        spl_per_scene = []
        for scene, scene_data in data_per_scene:
            print(f"\nScene: {scene}")
            success_rates = []
            spl_values = []
            seq_numbers = []
            for i in range(self.num_seq):
                sequences = scene_data[scene_data['sequence'] == i]
                if len(sequences) > 0:
                    successful_experiments = sequences[sequences['state'] == 1]
                    spl = sequences['spl'].mean() * SEQ_LEN
                    success_rate = len(successful_experiments) / len(sequences)

                    success_rates.append(success_rate)
                    spl_values.append(spl)
                    seq_numbers.append(i)
                    print(f"  Sequence {i}:")
                    print(f"    Num of experiments: {len(sequences)}")
                    print(f"    Overall SPL: {spl:.4f}")
                    print(f"    Fraction of successful experiments: {success_rate:.2%}")
                else:
                    print(f"  Sequence {i}: No data")
                    success_rates.append(0)
                    spl_values.append(0)
            sr_per_scene.append(success_rates)
            spl_per_scene.append(spl_values)
        sr_per_scene = np.array(sr_per_scene)
        spl_per_scene = np.array(spl_per_scene)
        sr_per_scene = np.mean(sr_per_scene, axis=0)
        spl_per_scene = np.mean(spl_per_scene, axis=0)
        print(f"SPL: {spl_per_scene}, Success Rate: {sr_per_scene}")
        # Plot Success Rate
        ax1.plot(np.arange(self.num_seq), sr_per_scene, label=scene, marker='o')

        # Plot SPL
        ax2.plot(np.arange(self.num_seq), spl_per_scene, label=scene, marker='o')
            # Set up Success Rate subplot
        ax1.set_xlabel('Sequence Number')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate per Sequence')
        # ax1.legend()
        ax1.grid(True)

        # Set up SPL subplot
        ax2.set_xlabel('Sequence Number')
        ax2.set_ylabel('SPL')
        ax2.set_title('SPL per Sequence')
        # ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('output_plot.png')

        plt.show()
        # selected_experiment_ids = successful_experiments['experiment'].unique()
        # experiments_with_second_success = successful_experiments.groupby('experiment').filter(
        #     lambda x: has_success(x, 1))
        # successful_second_ids = experiments_with_second_success['experiment'].unique()
        # fraction_successful = len(successful_second_ids) / len(selected_experiment_ids) if len(
        #     selected_experiment_ids) > 0 else 0
        #
        # # Calculate conditional SPL for each experiment
        # second_sequences = data[(data['state'] == 1) & (data['sequence'] == 1)]
        # conditional_spl = second_sequences['spl'].mean()
        # print(f"\nOverall Conditional SPL (second sequence, given first success): {conditional_spl:.4f}")
        #
        # print(f"Fraction of successful first experiments: {len(selected_experiment_ids)/len(all_ids):.2%}")
        # print(f"Fraction of successful second, conditioned on first: {fraction_successful:.2%}")
        return data

    def evaluate(self):
        n_eps = 0
        results = []
        for n_ep, episode in enumerate(self.episodes):
            poses = []
            metric = Metrics(episode.episode_id)
            results.append(metric)
            if n_ep in self.exclude_ids:
                continue
            # Skip already completed episodes (resume from interruption)
            state_file = f"{self.results_path}/state/state_{episode.episode_id}.txt"
            if os.path.exists(state_file):
                print(f"Skipping episode {episode.episode_id} (already completed)")
                n_eps += 1
                continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)

            self.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            self.actor.reset()
            print(
                "[Eval3D] Cross-target memory reset: "
                f"{'enabled' if self.reset_map_between_targets else 'disabled'}"
            )
            sequence_id = 0
            current_obj = episode.obj_sequence[sequence_id]
            self.actor.set_query(current_obj, allow_poi=False)
            if self.log_rerun:
                pts = []
                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                    if not self.is_gibson:
                        pt = obj.bbox.center[[0, 2]]
                        pt = (-pt[1], -pt[0])
                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                    else:
                        for pt_ in obj:
                            pt = (pt_[0], pt_[1])
                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                pts = np.array(pts)
                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))
            not_failed = True
            episode_step = 0
            while not_failed and sequence_id < len(episode.obj_sequence):
                if episode_step >= self.max_steps:
                    results[-1].add_sequence(np.array(poses), Result.FAILURE_OOT, episode.obj_sequence[sequence_id])
                    break
                steps = 0
                running = True
                no_goal_frames = 0
                while episode_step < self.max_steps and running:
                    observations = self.sim.get_sensor_observations()
                    # observations['depth'] = fill_depth_holes(observations['depth'])
                    observations['state'] = self.sim.get_agent(0).get_state()
                    agent_pos = observations['state'].position
                    sensor_state = observations['state'].sensor_states.get("depth")
                    if sensor_state is not None:
                        agent_cam_pos = np.asarray(sensor_state.position, dtype=np.float32)
                    else:
                        agent_cam_pos = np.array(agent_pos, dtype=np.float32).copy()
                        agent_cam_pos[1] += 0.88
                    pose = np.zeros((7,))
                    pose[0] = -observations['state'].position[2]
                    pose[1] = -observations['state'].position[0]
                    pose[2] = agent_cam_pos[1]
                    pose[4] = observations['state'].position[0]
                    pose[5] = observations['state'].position[1]
                    pose[6] = observations['state'].position[2]
                    # yaw
                    orientation = observations['state'].rotation
                    q0 = orientation.x
                    q1 = orientation.y
                    q2 = orientation.z
                    q3 = orientation.w
                    r = R.from_quat([q0, q1, q2, q3])
                    # r to euler
                    yaw, _, _1 = r.as_euler("yxz")
                    pose[3] = yaw

                    poses.append(pose)
                    if self.log_rerun:
                        cam_x = -self.sim.get_agent(0).get_state().position[2]
                        cam_y = -self.sim.get_agent(0).get_state().position[0]
                        rr.log("camera/rgb", rr.Image(observations["rgb"]).compress(jpeg_quality=50))
                        # rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                        #         observations["depth"].max() - observations["depth"].min())))
                        self.logger.log_pos(cam_x, cam_y)
                    initializing = getattr(self.actor, "_is_initializing", None)
                    initialization_action = bool(initializing()) if callable(initializing) else False
                    action, called_found = self.actor.act(observations)
                    initialization_action = initialization_action or bool(
                        getattr(self.actor.mapper, "initializing", False)
                    )
                    self.execute_action(action)
                    if self.log_rerun:
                        self.logger.log_map()

                    if steps % 100 == 0:
                        dist = get_closest_dist(
                            agent_cam_pos,
                            self.scene_data[episode.scene_id].object_locations[current_obj],
                            self.is_gibson,
                        )
                        height_info = "height=n/a"
                        if not self.is_gibson:
                            aabbs = self.scene_data[episode.scene_id].object_locations[current_obj]
                            if len(aabbs) > 0:
                                min_height = min(
                                    abs(agent_cam_pos[1] - aabb.bbox.center[1])
                                    for aabb in aabbs
                                )
                                height_info = f"height={min_height:.3f}"
                        print(
                            f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id}, distance to closest object: {dist}, {height_info}")
                    steps += 1
                    episode_step += 1
                    stuck_now = False
                    if not initialization_action and len(poses) >= 30:
                        pos_delta = np.linalg.norm(poses[-1][:2] - poses[-30][:2])
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
                        getattr(self.actor.mapper, "_no_goal_available", False) and
                        len(self.actor.mapper.nav_goals) == 0 and
                        self.actor.mapper.path is None and
                        getattr(self.actor.mapper, "navigation_mode", "explore") != "navigate" and
                        not self.actor.mapper.object_detected and
                        not self.actor.mapper.stair_sem_mode
                    )
                    no_goal_frames = no_goal_frames + 1 if no_goal_now else 0
                    if called_found or stuck_now or no_goal_frames >= 5 or episode_step >= self.max_steps:
                        running = False
                        result = Result.FAILURE_OOT
                        # We will now compute the closest distance to the bounding box of the object
                        if called_found:
                            use_height = self.multi_object or (
                                self.single_target_height_aware and not self.is_gibson
                            )
                            agent_eval_pos = (
                                agent_cam_pos
                                if use_height
                                else np.asarray(agent_pos)[[0, 2]]
                            )
                            dist = get_closest_dist(
                                agent_eval_pos,
                                self.scene_data[episode.scene_id].object_locations[current_obj],
                                self.is_gibson,
                            )
                            det_hab_pos = _mapper_current_detection_habitat_pos(self.actor.mapper)
                            dist_detect = float("inf")
                            if det_hab_pos is not None:
                                detection_eval_pos = (
                                    det_hab_pos
                                    if use_height
                                    else np.asarray(det_hab_pos)[[0, 2]]
                                )
                                dist_detect = get_closest_dist(
                                    detection_eval_pos,
                                    self.scene_data[episode.scene_id].object_locations[current_obj],
                                    self.is_gibson,
                                )
                            height_info = "height=n/a"
                            if not self.is_gibson:
                                aabbs = self.scene_data[episode.scene_id].object_locations[current_obj]
                                if len(aabbs) > 0:
                                    min_height = min(
                                        abs(agent_cam_pos[1] - aabb.bbox.center[1])
                                        for aabb in aabbs
                                    )
                                    height_info = f"height={min_height:.3f}"
                            print(
                                f"Episode {episode.episode_id} seq {sequence_id}: called_found=True "
                                f"dist={dist:.4f} detect_dist={dist_detect:.4f} "
                                f"max_dist={self.max_dist} {height_info}")
                            # Single-target success uses agent-to-target distance,
                            # optionally with the multi-target height gate.
                            # Multi-target also verifies the projected detection.
                            success = dist < self.max_dist
                            if self.multi_object:
                                success = success and dist_detect < self.max_dist
                            if success:
                                result = Result.SUCCESS
                                print("Object found!")
                            else:
                                if dist_detect < self.max_dist:
                                    result = Result.FAILURE_NOT_REACHED
                                else:
                                    result = Result.FAILURE_MISDETECT
                                not_failed = False
                                print(f"Object not found! Dist {dist}, detect dist: {dist_detect}.")
                        else:
                            not_failed = False
                            if stuck_now:
                                result = Result.FAILURE_STUCK
                            if no_goal_frames >= 5:
                                result = Result.FAILURE_ALL_EXPLORED
                        results[-1].add_sequence(np.array(poses), result, current_obj)
                        final_sim = (self.actor.mapper.get_map() + 1.0) / 2.0
                        confs = (self.actor.mapper.one_map.confidence_map > 0).cpu().squeeze().numpy()
                        nav_map = self.actor.mapper.one_map.navigable_map.astype(bool)
                        final_sim = final_sim[0]
                        final_sim = monochannel_to_inferno_rgb(final_sim)

                        final_sim[~confs, :] = [0, 0, 0]
                        # final_sim[(~nav_map) & confs, :] = [0, 0, 0]
                        min_x = np.min(np.where(confs)[0])
                        max_x = np.max(np.where(confs)[0])
                        min_y = np.min(np.where(confs)[1])
                        max_y = np.max(np.where(confs)[1])
                        final_sim = final_sim[min_x:max_x, min_y:max_y]
                        final_sim = final_sim.transpose((1, 0, 2))
                        final_sim = np.flip(final_sim, axis=0)                        # get min and max x and y of confs


                        cv2.imwrite(f"{self.results_path}/similarities/final_sim_{episode.episode_id}_{sequence_id}.png", final_sim)
                        # Create the plot
                        plt.figure(figsize=(10, 10))
                        poses_ = np.array([self.actor.mapper.one_map.metric_to_px(*pos[:2]) for pos in poses])
                        poses_[:, 0] -= min_x
                        poses_[:, 1] -= min_y
                        plt.imshow(final_sim[:, :, ::-1], interpolation='nearest', aspect='equal',
                                  extent=(0, final_sim.shape[1], 0, final_sim.shape[0]))

                        # Color each sequence segment differently
                        seq_colors = ['b', 'g', 'y']  # blue, green, yellow
                        seq_labels = ['Seq 0', 'Seq 1', 'Seq 2']
                        # Compute segment boundaries from sequence_poses
                        seg_starts = [0]
                        for prev_seq in range(sequence_id):
                            seg_starts.append(seg_starts[-1] + len(results[n_ep].sequence_poses[prev_seq]))
                        seg_starts.append(len(poses_))
                        for seg_idx in range(len(seg_starts) - 1):
                            s_start = seg_starts[seg_idx]
                            s_end = seg_starts[seg_idx + 1]
                            color = seq_colors[seg_idx % len(seq_colors)]
                            label = seq_labels[seg_idx % len(seq_labels)]
                            plt.plot(poses_[s_start:s_end, 0], poses_[s_start:s_end, 1],
                                     f'{color}-o', markersize=2, label=label)
                        plt.legend()

                        # Set equal aspect ratio to ensure accurate positions
                        plt.axis('equal')

                        # Add labels and title
                        plt.xlabel('X position')
                        plt.ylabel('Y position')
                        plt.title('Path of Poses')

                        # Add grid for better readability
                        plt.grid(True)

                        # Save the plot as SVG
                        plt.savefig(f"{self.results_path}/similarities/path_{episode.episode_id}_{sequence_id}.svg", format='svg', dpi=300, bbox_inches='tight')

                        # Display the plot (optional, comment out if not needed)
                        plt.show()

                        sequence_id += 1
                        if (
                            result == Result.SUCCESS and
                            sequence_id < len(episode.obj_sequence) and
                            episode_step < self.max_steps
                        ):
                            current_obj = episode.obj_sequence[sequence_id]
                            self.start_next_target(current_obj)
                            if self.log_rerun:
                                pts = []
                                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                                    if not self.is_gibson:
                                        pt = obj.bbox.center[[0, 2]]
                                        pt = (-pt[1], -pt[0])
                                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                                    else:
                                        for pt_ in obj:
                                            pt = (pt_[0], pt_[1])
                                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                                pts = np.array(pts)
                                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))

            for seq_id, seq in enumerate(results[n_ep].sequence_poses):
                np.savetxt(
                    f"{self.results_path}/trajectories/poses_{episode.episode_id}_{seq_id}.csv",
                    seq,
                    delimiter=",",
                    header="map_x,map_y,map_z,yaw,hab_x,hab_y,hab_z",
                    comments="",
                )
            # save final sim to image file

            print(f"Overall progress: {sum([m.get_progress() for m in results]) / (n_eps)}, per object: ")
            # for obj in success_per_obj.keys():
            #     print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            # print(
            #     f"Result distribution: successes: {results.count(Result.SUCCESS)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}")
            # Write result to file
            with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                f.write(','.join(
                    str(results[n_ep].sequence_results[i].value) for i in range(len(results[n_ep].sequence_results))))


    def read_results_floor(self, path, sort_by, data_pkl=None):
        """Read results and compute SPL using real-time geodesic distance calculation.
        Uses the new floor_multi dataset format with goals_by_category."""
        from eval.dataset_utils.object_nav_utils import object_nav_gen
        state_dir = os.path.join(path, 'state')
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return None

        # Load goals_by_category for each scene from the dataset files
        scene_goals = {}  # scene_name -> goals_by_category
        dataset_path = self.object_nav_path
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
        sim = None
        current_scene = None

        def get_or_build_sim(scene_id):
            nonlocal sim, current_scene
            if sim is not None and current_scene == scene_id:
                return sim
            if sim is not None:
                sim.close()
            sim = object_nav_gen.build_sim(self.scene_path, scene_id, 15, False)
            recompute_cross_floor_navmesh(sim)
            current_scene = scene_id
            return sim

        def find_objects_for_category(goals_by_cat, category):
            """Find all instances of a category from goals_by_category."""
            for cat_key, instances in goals_by_cat.items():
                cat_name = cat_key.split('_', 1)[-1] if '.glb_' in cat_key else cat_key
                if cat_name == category:
                    return instances
            return []

        def compute_geodesic_to_category(sim_inst, start_pos, instances):
            """Compute min geodesic distance from start_pos to any instance viewpoint."""
            min_dist = np.inf
            start = np.array(start_pos, dtype=np.float32)
            if not sim_inst.pathfinder.is_navigable(start):
                start = sim_inst.pathfinder.snap_point(start)
                if np.isnan(start).any():
                    return np.inf

            for inst in instances:
                vps = inst.get('view_points', [])
                if not vps:
                    continue
                # Collect all viewpoint positions
                vp_positions = [vp['agent_state']['position'] for vp in vps]
                # Use MultiGoalShortestPath for efficiency
                mg_path = habitat_sim.MultiGoalShortestPath()
                mg_path.requested_start = start
                mg_path.requested_ends = np.array(vp_positions, dtype=np.float32)
                sim_inst.pathfinder.find_path(mg_path)
                if mg_path.geodesic_distance < min_dist:
                    min_dist = mg_path.geodesic_distance

            return min_dist

        data = []
        if data_pkl is not None and os.path.exists(data_pkl):
            with open(data_pkl, 'rb') as f:
                data = pickle.load(f)
        else:
            for filename in sorted(os.listdir(state_dir)):
                if not (filename.startswith('state_') and filename.endswith('.txt')):
                    continue
                try:
                    experiment_num = int(filename[6:-4])
                except ValueError:
                    print(f"Warning: Skipping {filename} due to invalid format")
                    continue

                if experiment_num >= len(self.episodes):
                    print(f"Warning: experiment_num {experiment_num} exceeds episode count {len(self.episodes)}")
                    continue

                with open(os.path.join(state_dir, filename), 'r') as file:
                    content = file.read().strip()

                state_values = [int(val) for val in content.split(',')]
                episode = self.episodes[experiment_num]
                scene_id = episode.scene_id
                goals_by_cat = scene_goals.get(scene_id, {})

                # Compute total optimal path from dataset
                optimal_total_path = 0.0
                for d in episode.best_dist:
                    if d[0] is not None:
                        optimal_total_path += d[0]

                for seq_num, value in enumerate(state_values):
                    spl = 0.0
                    map_size = 0

                    if value == 1:  # SUCCESS
                        pose_file = os.path.join(pose_dir,
                            f"poses_{experiment_num}_{seq_num}.csv")
                        if not os.path.exists(pose_file):
                            continue
                        poses = np.genfromtxt(pose_file, delimiter=",")
                        if len(poses.shape) == 1:
                            poses = poses.reshape((1, 4))
                        path_length = np.linalg.norm(
                            poses[1:, :3] - poses[:-1, :3], axis=1).sum()

                        # Get sim for geodesic computation
                        sim_inst = get_or_build_sim(scene_id)

                        # Get start position for this sub-task
                        # poses are in [-z, -x, y, yaw] format
                        start_pos = poses[0, :3]
                        pos = np.array([-start_pos[1], start_pos[2], -start_pos[0]])

                        # Find target objects
                        target_cat = episode.obj_sequence[seq_num]
                        instances = find_objects_for_category(goals_by_cat, target_cat)

                        if instances:
                            best_dist = compute_geodesic_to_category(
                                sim_inst, pos, instances)
                            if best_dist < np.inf:
                                spl = min(1.0, best_dist / max(path_length, best_dist))
                            else:
                                print(f"Warning: No geodesic path for exp {experiment_num} seq {seq_num}")
                        else:
                            print(f"Warning: No instances for {target_cat} in {scene_id}")

                    data.append({
                        'experiment': experiment_num,
                        'sequence': seq_num,
                        'state': value,
                        'spl': spl / self.num_seq,
                        'map_size': map_size,
                        'opt_path': optimal_total_path,
                        'object': episode.obj_sequence[seq_num] if seq_num < len(episode.obj_sequence) else 'unknown',
                        'scene': scene_id,
                    })

                if self.episodes[experiment_num].episode_id != experiment_num:
                    print(f"Warning: experiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")

            if sim is not None:
                sim.close()

            data = pd.DataFrame(data)
            if data_pkl is not None:
                with open(data_pkl, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Results saved to {data_pkl}")

        # === Compute and display metrics (same as read_results) ===
        states = data["state"].unique()

        def calc_prog_per_episode(group):
            successes = group.groupby('experiment').apply(lambda x: (x['state'] == 1).sum())
            return successes / self.num_seq

        def calc_spl_per_episode(group):
            return group.groupby('experiment')['spl'].sum()

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})
            progress = calc_prog_per_episode(group)
            spl = calc_spl_per_episode(group)
            s = progress[progress == 1]
            result['Progress'] = progress.mean()
            result['SPL'] = spl.mean()
            result['opt_PL'] = group['opt_path'].mean()
            result['Map Size'] = group['map_size'].mean() / 100
            result['s'] = s.sum() / len(progress)
            result['s_spl'] = spl[progress == 1].sum() / len(progress)
            return result

        object_results = data.groupby('object').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'object': 'Object'})

        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        # Per-sequence breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sr_per_scene = []
        spl_per_scene = []
        data_per_scene = data.groupby('scene')
        for scene, scene_data_g in data_per_scene:
            success_rates = []
            spl_values = []
            for i in range(self.num_seq):
                sequences = scene_data_g[scene_data_g['sequence'] == i]
                if len(sequences) > 0:
                    successful = sequences[sequences['state'] == 1]
                    spl_val = sequences['spl'].mean() * SEQ_LEN
                    success_rate = len(successful) / len(sequences)
                    success_rates.append(success_rate)
                    spl_values.append(spl_val)
                else:
                    success_rates.append(0)
                    spl_values.append(0)
            sr_per_scene.append(success_rates)
            spl_per_scene.append(spl_values)

        sr_per_scene = np.array(sr_per_scene)
        spl_per_scene = np.array(spl_per_scene)
        sr_mean = np.mean(sr_per_scene, axis=0)
        spl_mean = np.mean(spl_per_scene, axis=0)
        print(f"SPL: {spl_mean}, Success Rate: {sr_mean}")

        ax1.plot(np.arange(self.num_seq), sr_mean, marker='o')
        ax1.set_xlabel('Sequence Number')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate per Sequence')
        ax1.grid(True)

        ax2.plot(np.arange(self.num_seq), spl_mean, marker='o')
        ax2.set_xlabel('Sequence Number')
        ax2.set_ylabel('SPL')
        ax2.set_title('SPL per Sequence')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('output_plot_floor.png')
        plt.show()

        return data
