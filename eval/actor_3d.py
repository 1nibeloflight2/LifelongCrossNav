"""
MONActor3D: Actor using Navigator3D with pure 3D voxel-based navigation.

Drop-in replacement for MONActor. Uses Navigator3D internally,
outputs the same action format for the evaluator.

Action space (TravExplorer-style, all discrete):
  move_forward: 25cm
  turn_left / turn_right: 30°
  look_up / look_down: 30°
  stop: end episode
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

from mapping.navigator_3d import Navigator3D
from planning import Planning, Controllers
from vision_models.clip_dense import ClipModel
from vision_models.segformer_stair_detector import SegFormerStairDetector
from vision_models.dfine_model import DFineDetector


class Actor(ABC):
    @abstractmethod
    def act(self, observations: Dict[str, any]) -> Tuple[Dict, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_query(self, query: str, allow_poi: bool = True):
        pass


class MONActor3D(Actor):
    """
    Actor using 3D voxel-based Navigator.
    Compatible with HabitatMultiEvaluator interface.

    All actions are discrete (30° turns, 30° pitch, 0.25m forward).
    Pitch state is tracked to ensure horizontal restoration before movement/turning.
    """

    def __init__(self, config):
        model = ClipModel("weights/clip.pth", jetson=False)
        target_detector = DFineDetector(
            confidence_threshold=config.planner.dfine_confidence,
            config_path=config.planner.dfine_config,
            checkpoint_path=config.planner.dfine_checkpoint,
        )
        stair_detector = SegFormerStairDetector(
            model_path=getattr(
                config.planner,
                "segformer_stair_model",
                "weights/segformer_b2_ade20k",
            ),
            pixel_threshold=getattr(
                config.planner, "segformer_stair_pixel_threshold", 0.25
            ),
            min_component_area=getattr(
                config.planner, "segformer_stair_min_area", 80
            ),
            max_components=getattr(
                config.planner, "segformer_stair_max_components", 8
            ),
            use_fp16=getattr(config.planner, "segformer_stair_fp16", True),
        )

        self.square = config.square_im
        self.mapper = Navigator3D(model, target_detector, stair_detector, config)

        # Number of look_down steps during init (controls tilt angle)
        self.init_look_down_steps = 2  # 2 x 30° = 60° down

        # Initialization sequence:
        #   Phase 0: look_down (tilt camera down for ground observation)
        #   Phase 1: rotate 360° while looking down (scan ground/support)
        #   Phase 2: look_up (restore horizontal)
        #   Phase 3: move forward once, then rotate 360° horizontal (scan obstacles)
        # After all phases complete, normal navigation begins.
        self.init_phase = 0          # current phase (0-3)
        self.init_steps_remaining = 0
        self.initialization_kind = "init"
        self._target_init_pitch_restore_remaining = 0
        self._setup_init_phase(0)

        # Pitch state: number of discrete look_down steps applied (positive = looking down).
        # Each step is 30°. Must return to 0 before any move/turn.
        self.pitch_level = 0  # 0=horizontal, 1=down 30°, 2=down 60°, etc.

        # Recovery rotation state
        self.no_path_counter = 0
        self.recovery_rotation_remaining = 0
        self._last_executed_action = None

        hfov = 90 if self.square else 79  # TravExplorer uses 79° FOV
        res_x = 640
        res_y = 640 if self.square else 480
        hfov_rad = np.deg2rad(hfov)
        focal_length = (res_x / 2) / np.tan(hfov_rad / 2)
        principal_point_x = res_x / 2
        principal_point_y = res_y / 2
        K = np.array([
            [focal_length, 0, principal_point_x],
            [0, focal_length, principal_point_y],
            [0, 0, 1]
        ])
        self.mapper.set_camera_matrix(K)

    def _setup_init_phase(self, phase: int):
        """Configure steps remaining for each initialization phase."""
        self.init_phase = phase
        if phase == 0:
            # Phase 0: look down N steps
            self.init_steps_remaining = self.init_look_down_steps
        elif phase == 1:
            # Phase 1: rotate 360° while looking down (12 steps x 30°)
            self.init_steps_remaining = 12
        elif phase == 2:
            # Phase 2: look up to restore horizontal (same N steps)
            self.init_steps_remaining = self.init_look_down_steps
        elif phase == 3:
            # Phase 3: move forward 1 step (离开楼梯底部狭窄区域)
            self.init_steps_remaining = 1
        elif phase == 4:
            # Phase 4: rotate 360° horizontal (12 steps x 30° = 1 circle)
            self.init_steps_remaining = 12
        else:
            # All init phases done
            self.init_steps_remaining = 0

    def _is_initializing(self) -> bool:
        """Return True if still in initialization sequence."""
        return (
            self._target_init_pitch_restore_remaining > 0 or
            (self.init_phase <= 4 and self.init_steps_remaining > 0)
        )

    # Coordinate transform from Habitat WORLD (x=right, y=up, z=backward)
    # to our WORLD frame (x=forward, y=left, z=up):
    #   ours_x = -hab_z,  ours_y = -hab_x,  ours_z = hab_y
    _HAB_TO_OURS = np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ], dtype=np.float64)

    # Coordinate transform from OUR CAMERA local (x=forward, y=left, z=up)
    # to HABITAT SENSOR local (OpenGL: x=right, y=up, z=backward):
    #   hab_sensor_x(right)    = -our_cam_y(left)
    #   hab_sensor_y(up)       =  our_cam_z(up)
    #   hab_sensor_z(backward) = -our_cam_x(forward)
    _OUR_CAM_TO_HAB_SENSOR = np.array([
        [ 0, -1,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
    ], dtype=np.float64)

    def act(self, observations: Dict[str, any]) -> Tuple[Dict, bool]:
        """Run one policy step while preserving the action executed last frame."""
        self.mapper.set_last_executed_action(self._last_executed_action)
        result = self._act_impl(observations)
        action_dict = result[0] if result else None
        self._last_executed_action = (
            action_dict.get("discrete") if isinstance(action_dict, dict) else None
        )
        return result

    def _act_impl(self, observations: Dict[str, any]) -> Tuple[Dict, bool]:
        initialization_action = self._is_initializing()
        self.mapper.set_initializing(initialization_action)
        state = observations["state"]

        # Use SENSOR state (not agent body state) to capture pitch from look_down/look_up.
        # state.rotation only has yaw (agent body), but look_down changes sensor rotation.
        sensor_state = state.sensor_states["depth"]
        sensor_pos = sensor_state.position      # sensor world position in Habitat frame
        sensor_rot = sensor_state.rotation      # sensor world rotation quaternion

        # Build position: Habitat (x=right, y=up, z=backward) -> Ours (x=fwd, y=left, z=up)
        pos = np.array([[-sensor_pos[2]], [-sensor_pos[0]], [sensor_pos[1]]])

        # Build rotation from sensor quaternion (includes pitch)
        q0 = sensor_rot.x
        q1 = sensor_rot.y
        q2 = sensor_rot.z
        q3 = sensor_rot.w

        # Habitat sensor convention: x=right, y=UP, z=BACKWARD (OpenGL-like).
        # state.sensor_states["depth"].rotation gives the quaternion representing
        # the SENSOR-TO-WORLD transform (hab_sensor_local -> hab_world).
        # Habitat look_down uses negative pitch angle (Rx(-30) for 30° down).
        #
        # Transform chain for pts:
        #   pts_our_cam -> pts_hab_sensor (via _OUR_CAM_TO_HAB_SENSOR)
        #   pts_hab_sensor -> pts_hab_world (via r_sensor_to_world from quaternion)
        #   pts_hab_world -> pts_our_world (via _HAB_TO_OURS)
        r_sensor_to_world = R.from_quat([q0, q1, q2, q3]).as_matrix()
        r_mat = self._HAB_TO_OURS @ r_sensor_to_world @ self._OUR_CAM_TO_HAB_SENSOR

        # Extract yaw for 2D navigation (from agent body rotation, not sensor)
        agent_orientation = state.rotation
        r_hab_agent = R.from_quat([
            agent_orientation.x, agent_orientation.y,
            agent_orientation.z, agent_orientation.w
        ]).as_matrix()
        r_agent = self._HAB_TO_OURS @ r_hab_agent @ self._HAB_TO_OURS.T
        yaw = np.arctan2(r_agent[1, 0], r_agent[0, 0])

        transformation_matrix = np.hstack((r_mat, pos))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))

        # Feed data to navigator
        obj_found = self.mapper.add_data(
            observations["rgb"][:, :, :-1].transpose(2, 0, 1),
            observations["depth"].astype(np.float32),
            transformation_matrix,
        )
        if obj_found and not initialization_action:
            return {'discrete': 'turn_left'}, True

        # === INITIALIZATION SEQUENCE ===
        # Phase 0: look down → Phase 1: rotate 360° down → Phase 2: look up → Phase 3: move forward → Phase 4: rotate 360° horizontal
        # === RE-INITIALIZATION REQUEST (from stair sem mode exit) ===
        if self.mapper.request_reinit:
            self.mapper.request_reinit = False
            self.initialization_kind = "reinit"
            # Restart init sequence from current pitch
            # First need to restore pitch to horizontal, then move forward, then rotate scan
            if self.pitch_level > 0:
                self.init_phase = 2  # look_up phase
                self.init_steps_remaining = self.pitch_level
            elif self.pitch_level < 0:
                self.init_phase = 0  # look_down to get to 0 first (unusual)
                self.init_steps_remaining = abs(self.pitch_level)
            else:
                # Already horizontal — move forward then rotate scan
                self.init_phase = 3
                self.init_steps_remaining = 1
            print(f"[Actor3D] Re-initialization triggered (pitch={self.pitch_level}, phase={self.init_phase})")

        # A re-initialization request is consumed after add_data(), so refresh
        # mapper state before choosing the forced scan action.
        if self._is_initializing() and not initialization_action:
            initialization_action = True
            self.mapper.set_initializing(True)

        if self._is_initializing():
            if self._target_init_pitch_restore_remaining > 0:
                action = 'look_up' if self.pitch_level > 0 else 'look_down'
                self.pitch_level += -1 if self.pitch_level > 0 else 1
                self._target_init_pitch_restore_remaining -= 1
                return {'discrete': action}, False

            self.init_steps_remaining -= 1
            if self.init_phase == 0:
                # Look down
                self.pitch_level += 1
                action = 'look_down'
            elif self.init_phase == 1:
                # Rotate while looking down (don't change pitch)
                action = 'turn_left'
            elif self.init_phase == 2:
                # Look up to restore horizontal
                self.pitch_level -= 1
                action = 'look_up'
            elif self.init_phase == 3:
                # Move forward (离开楼梯底部)
                action = 'move_forward'
            elif self.init_phase == 4:
                # Rotate horizontal
                action = 'turn_left'
            else:
                action = 'turn_left'

            # Advance to next phase when current phase is done
            if self.init_steps_remaining <= 0:
                self._setup_init_phase(self.init_phase + 1)

            # The final horizontal turn is still an initialization action.  It
            # is executed after this return; normal planning starts with the
            # following observation, when the full 29-action scan is visible.
            if not self._is_initializing():
                kind = "Target initialization" if self.initialization_kind == "target_init" else "Initialization"
                print(f"[Actor3D] {kind} scan complete (29 actions)")

            return {'discrete': action}, False

        # ObjectNav view/yaw correction actions are explicit one-step commands.
        # They must preempt automatic pitch tracking; otherwise a stale
        # desired_pitch_level can consume the frame and issue an extra look_up.
        if (
            getattr(self.mapper, "_objectnav_align_action", None) is not None or
            getattr(self.mapper, "_objectnav_correction_action", None) is not None or
            (
                getattr(self.mapper, "_objectnav_restore_pending", False) and
                getattr(self.mapper, "_objectnav_restore_action", None) is not None
            )
        ):
            path = self.mapper.get_path()
            if isinstance(path, str):
                if path == "L":
                    return {'discrete': 'turn_left'}, False
                elif path == "R":
                    return {'discrete': 'turn_right'}, False
                elif path == "F":
                    return {'discrete': 'move_forward'}, False
                elif path == "U":
                    self.pitch_level -= 1
                    self.mapper.desired_pitch_level = self.pitch_level
                    return {'discrete': 'look_up'}, False
                elif path == "D":
                    self.pitch_level += 1
                    self.mapper.desired_pitch_level = self.pitch_level
                    if self.pitch_level == self.mapper.desired_pitch_level and self.pitch_level == 2:
                        self.mapper._notify_pitch_reached_for_fake_frontier()
                    return {'discrete': 'look_down'}, False

        # === PITCH ADJUSTMENT (Trajectory-Driven Gaze) ===
        # Pitch is fully controlled by navigator signals:
        #   - desired_pitch_level: target pitch from path geometry / stair frontier
        #   - Positive = looking down, Negative = looking up
        # Actor steps one 30° increment per frame toward the desired pitch.
        desired_pitch = self.mapper.desired_pitch_level
        if self.pitch_level < desired_pitch:
            self.pitch_level += 1
            # Notify navigator when pitch reaches target (for fake frontier confirmation)
            if self.pitch_level == desired_pitch and desired_pitch == 2:
                self.mapper._notify_pitch_reached_for_fake_frontier()
            return {'discrete': 'look_down'}, False
        elif self.pitch_level > desired_pitch:
            self.pitch_level -= 1
            return {'discrete': 'look_up'}, False

        # === STAIR SEM: Force forward steps on arrival at semantic stair frontier ===
        if self.mapper._stair_sem_force_forward > 0:
            self.mapper._stair_sem_force_forward -= 1
            if self.mapper._stair_sem_force_forward <= 0:
                # Complete the transition on the next mapped frame, when the
                # navigator has an up-to-date landing map.  Both directions
                # use the landing probe; this is especially important for a
                # narrow downstairs landing whose opposite edge is not yet
                # visible from the stair exit.
                direction = getattr(self.mapper, "_stair_sem_direction", "down")
                print(
                    f"[Actor3D] StairSem-{'ascent' if direction == 'up' else 'descend'} "
                    f"force-forward complete — pending platform probe"
                )
                self.mapper._post_upstairs_probe_pending = True
            return {'discrete': 'move_forward'}, False

        # === Navigation ===
        path = self.mapper.get_path()

        if isinstance(path, str):
            if path == "L":
                return {'discrete': 'turn_left'}, False
            elif path == "R":
                return {'discrete': 'turn_right'}, False
            elif path == "F":
                return {'discrete': 'move_forward'}, False
            elif path == "U":
                self.pitch_level -= 1
                self.mapper.desired_pitch_level = self.pitch_level
                return {'discrete': 'look_up'}, False
            elif path == "D":
                self.pitch_level += 1
                self.mapper.desired_pitch_level = self.pitch_level
                if self.pitch_level == self.mapper.desired_pitch_level and self.pitch_level == 2:
                    self.mapper._notify_pitch_reached_for_fake_frontier()
                return {'discrete': 'look_down'}, False

        if path is not None and isinstance(path, np.ndarray) and path.shape[0] > 0:
            # Valid path found — reset recovery state
            self.no_path_counter = 0
            self.recovery_rotation_remaining = 0

            # Path is in metric coordinates from Navigator3D
            path_simplified = Planning.simplify_path(path.astype(np.float32))
            path_simplified = np.array(path_simplified, dtype=np.float32)

            if self.mapper.log:
                path_px = np.array([
                    self.mapper.one_map.metric_to_px(p[0], p[1])
                    for p in path_simplified
                ], dtype=np.float32)
                rr.log("map/path_simplified", rr.LineStrips2D(
                    path_px,
                    colors=np.repeat(np.array([[0, 0, 255]]), path_px.shape[0], axis=0)
                ))

            # Discrete path following: decide turn or move_forward
            action = self._discrete_path_follow(pos, yaw, path_simplified)
            return {'discrete': action}, obj_found
        else:
            # No path available — recovery rotation
            self.no_path_counter += 1

            if self.recovery_rotation_remaining > 0:
                self.recovery_rotation_remaining -= 1
                return {'discrete': 'turn_left'}, False

            if self.no_path_counter >= 3:
                # Full 360° spin (12 steps x 30°)
                self.recovery_rotation_remaining = 12
                self.no_path_counter = 0
                if getattr(self.mapper, "navigation_mode", "explore") != "navigate":
                    self.mapper.frontier_clusters.clear()
                    self.mapper.nav_goals.clear()
                    self.mapper.last_nav_goal = None
                    self.mapper.path = None
                    self.mapper.cyclic_checker = self.mapper.cyclic_checker.__class__()
                    self.mapper.blacklisted_goals.clear()
                self.recovery_rotation_remaining -= 1
                return {'discrete': 'turn_left'}, False

            return {'discrete': 'turn_left'}, False

    def _discrete_path_follow(self, pos, yaw, path):
        """Decide discrete action to follow a metric path.

        Strategy:
          - Compute angle to next waypoint
          - If angle error > 15° (half of 30° turn): turn toward it
          - Otherwise: move_forward
        """
        # Find next waypoint (skip points too close)
        current_pos = pos[:2, 0]  # [x, y] in our frame
        target = None
        for i in range(len(path)):
            dist = np.linalg.norm(path[i][:2] - current_pos)
            if dist > 0.15:  # skip waypoints within 15cm
                target = path[i][:2]
                break

        if target is None:
            # All waypoints reached or very close — move forward
            return 'move_forward'

        # Compute desired heading
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        desired_yaw = np.arctan2(dy, dx)

        # Angle difference (wrapped to [-pi, pi])
        angle_diff = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

        # Threshold: 15° (half turn step)
        if angle_diff > np.deg2rad(15):
            return 'turn_left'
        elif angle_diff < -np.deg2rad(15):
            return 'turn_right'
        else:
            return 'move_forward'

    def reset(self):
        self.mapper.reset()
        self.init_phase = 0
        self.init_steps_remaining = 0
        self.initialization_kind = "init"
        self._target_init_pitch_restore_remaining = 0
        self._setup_init_phase(0)
        self.pitch_level = 0
        self.no_path_counter = 0
        self.recovery_rotation_remaining = 0
        self._last_executed_action = None

    def reset_for_next_target(self):
        """Forget the previous target's map while keeping the simulator pose."""
        previous_pitch = self.pitch_level
        previous_voxels = len(self.mapper.one_map.voxels)
        self.reset()
        self.initialization_kind = "target_init"
        self.pitch_level = previous_pitch
        self._target_init_pitch_restore_remaining = abs(previous_pitch)
        print(
            "[Actor3D] Target switch reset: map cleared, agent pose preserved; "
            f"cleared_voxels={previous_voxels}, "
            f"pitch_restore_actions={self._target_init_pitch_restore_remaining}"
        )

    def set_query(self, query: str, allow_poi: bool = True):
        self.mapper.set_query([query], allow_poi=allow_poi)
