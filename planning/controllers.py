# abc
from abc import ABC, abstractmethod

# numpy
import numpy as np

# habitat
import habitat_sim
from habitat_sim.utils import common as utils

# conf
from config import HabitatControllerConf, SpotControllerConf


class Controller(ABC):

    @abstractmethod
    def control(self, pos, yaw, path):
        """
        Executes the control logic for the agent
        :param pos: pos in metric units
        :param path: path in metric units
        :return:
        """
        pass

    # @abstractmethod
    # def turn_left(self):
    #     pass


class HabitatController(Controller):
    def __init__(self, sim, config: HabitatControllerConf):
        self.sim = sim
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True

        self.control_frequency = config.control_freq
        self.max_vel = config.max_vel
        self.max_ang_vel = config.max_ang_vel
        self.time_step = 1.0 / self.control_frequency

    def compute_angle_vel(self, yaw, dx, dy, time_step, max_ang_vel, max_vel):
        desired_angle = np.arctan2(dy, dx)
        angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

        angular_velocity = np.array([0.0, np.clip(angle_diff / time_step, -max_ang_vel, max_ang_vel), 0.0])
        return angular_velocity


    def compute_velocity(self, current_pos, next_pos, yaw, time_step, max_ang_vel, max_vel):
        dx = next_pos[0] - current_pos[0, 0]
        dy = next_pos[1] - current_pos[1, 0]

        desired_angle = np.arctan2(dy, dx)
        angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

        angular_velocity = np.array([0.0, np.clip(angle_diff / time_step, -max_ang_vel, max_ang_vel), 0.0])

        if abs(angle_diff) < 0.005:  # Increased threshold to allow for small corrections
            angular_velocity = np.array([0.0, 0.0, 0.0])
            speed = np.clip(np.linalg.norm([dx, dy]) / time_step, 0, max_vel)
            linear_velocity = np.array([0.0, 0.0, -speed])
        else:
            linear_velocity = np.array([0.0, 0.0, 0.0])

        return angular_velocity, linear_velocity

    def control(self, pos, yaw, path, own_update=True):
        """
        Executes the habitat control logic for the agent
        :param pos: np.ndarray of shape [2, ], pos in metric units
        :param path: path in metric units
        :return:
        """
        pos = pos.astype(np.float32)
        path = path.astype(np.float32)
        if path is not None and path.shape[0] > 0:
            # find closest point on path
            distances = np.linalg.norm(path - pos[:2].T, axis=1)
            next_id = np.argmin(distances)

            next_id += 1
            next_id = min(next_id, path.shape[0] - 1)
            next_pos = path[next_id]
            # determine action
            if own_update:
                self.vel_control.angular_velocity, self.vel_control.linear_velocity = self.compute_velocity(
                    pos, next_pos, yaw, self.time_step, self.max_ang_vel, self.max_vel
                )
            else:
                if next_id == path.shape[0] - 1 and distances[next_id] < 0.001:
                    # then we just need to match the yaw?
                    dx = path[next_id][0] - path[next_id - 1][0]
                    dy = path[next_id][1] - path[next_id - 1][1]
                    # angular = self.compute_angle_vel(yaw, dx, dy,  self.time_step, self.max_ang_vel, self.max_vel)
                    # if np.linalg.norm(angular) < 0.001:
                    angular = np.array(
                        [0.0,  self.max_ang_vel, 0.0])

                    return angular, np.array([0.0, 0.0, 0.0])

                return self.compute_velocity(
                    pos, next_pos, yaw, self.time_step, self.max_ang_vel, self.max_vel)
        else:
            if own_update:
                self.vel_control.angular_velocity = np.array([0.0, 0.0, 0.0])
                self.vel_control.linear_velocity = np.array([0.0, -self.max_vel/5.0, 0.0])
            else:
                return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
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


class SpotController(Controller):
    def __init__(self, cfg):
        self.max_vel = cfg.max_vel
        self.max_ang_vel = cfg.max_ang_vel

    def compute_relative_pose(self, current_pos, next_pos, yaw, desired_yaw, max_ang_vel, max_vel):
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        print("Pos: ", current_pos, " Next pos:", next_pos)
        desired_angle = desired_yaw
        angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

        dx_body = dx * np.cos(yaw) + np.sin(yaw) * dy
        dy_body = -dx * np.sin(yaw) + np.cos(yaw) * dy

        return dx_body, dy_body, angle_diff

    def control(self, pos, yaw, path):
        if path is not None and path.shape[0] > 1:
            next_id = min(5, path.shape[0] - 1)
            next_pos = path[next_id]
            desired_yaw = np.arctan2(next_pos[1] - path[next_id - 1][1], next_pos[0] - path[next_id - 1][0])
            next_pos_center_offset = np.array([-0.4 * np.cos(desired_yaw), -0.4 * np.sin(desired_yaw)])

            next_pos = next_pos + next_pos_center_offset

            curr_pos_center_offset = np.array([-0.4 * np.cos(yaw), -0.4 * np.sin(yaw)])
            # determine action
            pos = pos[:2] + curr_pos_center_offset
            print("YAW: ", yaw)
            x, y, heading = self.compute_relative_pose(
                pos, next_pos, yaw, desired_yaw, self.max_ang_vel, self.max_vel
            )
            norm_pos = np.sqrt(x ** 2 + y ** 2)
            if norm_pos > self.max_vel:
                x *= self.max_vel / norm_pos
                y *= self.max_vel / norm_pos

            if abs(heading) > self.max_ang_vel:
                if heading < 0:
                    heading = -self.max_ang_vel
                else:
                    heading = self.max_ang_vel
        else:
            return np.array([0, 0])
        return np.array([x, y, heading])
