# Adapted the point nav policy of VLFM to be used outside VLFM. See https://github.com/bdaiinstitute/vlfm
from typing import Union, Any, Tuple, Dict

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from torch import Tensor
from depth_camera_filtering import filter_depth

from torch import Tensor
from vlfm.obs_transformers.utils import image_resize

def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """Returns a 2x2 or 3x3 rotation matrix for a given angle; if 3x3, the z-axis is
    rotated."""
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")

def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """Calculates polar coordinates (rho, theta) relative to a given position and
    heading to a given goal position. 'rho' is the distance from the agent to the goal,
    and theta is how many radians the agent must turn (to the left, CCW from above) to
    face the goal. Coordinates are in (x, y), where x is the distance forward/backwards,
    and y is the distance to the left or right (right is negative)

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading.
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
    """
    rotation_matrix = get_rotation_matrix(-curr_heading, ndims=2)
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)

try:
    import habitat
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

    habitat_version = habitat.__version__

    if habitat_version == "0.1.5":
        print("Using habitat 0.1.5; assuming SemExp code is being used")

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                value, action, action_log_probs, rnn_hidden_states = super().act(*args, **kwargs)
                return action, rnn_hidden_states

    else:
        from habitat_baselines.common.tensor_dict import TensorDict
        from habitat_baselines.rl.ppo.policy import PolicyActionData

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                policy_actions: "PolicyActionData" = super().act(*args, **kwargs)
                return policy_actions.actions, policy_actions.rnn_hidden_states

    HABITAT_BASELINES_AVAILABLE = True

except ModuleNotFoundError:
    from onemap_utils.nh_pointnav import (
        PointNavResNetPolicy,
    )

    class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
        """Already outputs a tensor, so no need to convert."""

        pass

    HABITAT_BASELINES_AVAILABLE = False

def load_pointnav_policy(file_path: str, cfg_conf_path: str) -> PointNavResNetTensorOutputPolicy:
    """Loads a PointNavResNetPolicy policy from a .pth file.

    Args:
        file_path (str): The path to the trained weights of the pointnav policy.
    Returns:
        PointNavResNetTensorOutputPolicy: The policy.
    """
    if HABITAT_BASELINES_AVAILABLE:
        obs_space = SpaceDict(
            {
                "depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = Discrete(4)
        if habitat_version == "0.1.5":
            pointnav_policy = PointNavResNetTensorOutputPolicy(
                obs_space,
                action_space,
                hidden_size=512,
                num_recurrent_layers=2,
                rnn_type="LSTM",
                resnet_baseplanes=32,
                backbone="resnet18",
                normalize_visual_inputs=False,
                obs_transform=None,
            )
            # Need to overwrite the visual encoder because it uses an older version of
            # ResNet that calculates the compression size differently
            from onemap_utils.nh_pointnav import (
                PointNavResNetNet,
            )

            # print(pointnav_policy)
            pointnav_policy.net = PointNavResNetNet(discrete_actions=True, no_fwd_dict=True)
            state_dict = torch.load(file_path + ".state_dict", map_location="cpu")
        else:
            state_dict = torch.load(file_path, map_location="cpu")
            cfg_dict = torch.load(cfg_conf_path, map_location="cpu")
            pointnav_policy = PointNavResNetTensorOutputPolicy.from_config(cfg_dict, obs_space, action_space)
            state_dict = state_dict
        pointnav_policy.load_state_dict(state_dict)
        return pointnav_policy

    else:
        ckpt_dict = torch.load(file_path, map_location="cpu")
        pointnav_policy = PointNavResNetTensorOutputPolicy()
        current_state_dict = pointnav_policy.state_dict()
        # Let old checkpoints work with new code
        if "net.prev_action_embedding_cont.bias" not in ckpt_dict.keys():
            ckpt_dict["net.prev_action_embedding_cont.bias"] = ckpt_dict["net.prev_action_embedding.bias"]
        if "net.prev_action_embedding_cont.weights" not in ckpt_dict.keys():
            ckpt_dict["net.prev_action_embedding_cont.weight"] = ckpt_dict["net.prev_action_embedding.weight"]

        pointnav_policy.load_state_dict({k: v for k, v in ckpt_dict.items() if k in current_state_dict})
        unused_keys = [k for k in ckpt_dict.keys() if k not in current_state_dict]
        print(f"The following unused keys were not loaded when loading the pointnav policy: {unused_keys}")
        return pointnav_policy

def move_obs_to_device(
    observations: Dict[str, Any],
    device: torch.device,
    unsqueeze: bool = False,
) -> Dict[str, Tensor]:
    """Moves observations to the given device, converts numpy arrays to torch tensors.

    Args:
        observations (Dict[str, Union[Tensor, np.ndarray]]): The observations.
        device (torch.device): The device to move the observations to.
        unsqueeze (bool): Whether to unsqueeze the tensors or not.
    Returns:
        Dict[str, Tensor]: The observations on the given device as torch tensors.
    """
    # Convert numpy arrays to torch tensors for each dict value
    for k, v in observations.items():
        if isinstance(v, np.ndarray):
            tensor_dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
            observations[k] = torch.from_numpy(v).to(device=device, dtype=tensor_dtype)
            if unsqueeze:
                observations[k] = observations[k].unsqueeze(0)

    return observations

class WrappedPointNavResNetPolicy:
    """
    Wrapper for the PointNavResNetPolicy that allows for easier usage, however it can
    only handle one environment at a time. Automatically updates the hidden state
    and previous action for the policy.
    """

    def __init__(
        self,
        ckpt_path: str,
        point_nav_cfg_path: str,
        device: Union[str, torch.device] = "cuda",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.policy = load_pointnav_policy(ckpt_path,point_nav_cfg_path)
        self._depth_image_shape = np.array([224, 224])
        self.policy.to(device)
        discrete_actions = not hasattr(self.policy.action_distribution, "mu_maybe_std")
        self.pointnav_test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments.
            self.policy.net.num_recurrent_layers,
            512,  # hidden state size
            device=device,
        )
        if discrete_actions:
            num_actions = 1
            action_dtype = torch.long
        else:
            num_actions = 2
            action_dtype = torch.float32
        self.pointnav_prev_actions = torch.zeros(
            1,  # number of environments
            num_actions,
            device=device,
            dtype=action_dtype,
        )
        self.device = device
        self._num_steps = 0


    def act(
        self,
        depth,
        robot_xy,
        robot_heading,
        goal_pt,
        deterministic: bool = False,
    ) -> Tensor:
        """Infers action to take towards the given (rho, theta) based on depth vision.

        Args:
            observations (Union["TensorDict", Dict]): A dictionary containing (at least)
                the following:
                    - "depth" (torch.float32): Depth image tensor (N, H, W, 1).
                    - "pointgoal_with_gps_compass" (torch.float32):
                        PointGoalWithGPSCompassSensor tensor representing a rho and
                        theta w.r.t. to the agent's current pose (N, 2).
            masks (torch.bool): Tensor of masks, with a value of 1 for any step after
                the first in an episode; has 0 for first step.
            deterministic (bool): Whether to select a logit action deterministically.

        Returns:
            Tensor: A tensor denoting the action to take.
        """
        # Convert numpy arrays to torch tensors for each dict value
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        rho, theta = rho_theta(robot_xy, robot_heading, goal_pt)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)

        # observations = move_obs_to_device(observations, self.device)
        depth = torch.Tensor(filter_depth(depth)).unsqueeze(-1).unsqueeze(0).to("cuda")
        obs_pointnav = {
            "depth": image_resize(
                depth,
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        pointnav_action, rnn_hidden_states = self.policy.act(
            obs_pointnav,
            self.pointnav_test_recurrent_hidden_states,
            self.pointnav_prev_actions,
            masks,
            deterministic=deterministic,
        )
        self.pointnav_prev_actions = pointnav_action.clone()
        self.pointnav_test_recurrent_hidden_states = rnn_hidden_states
        return pointnav_action

    def reset(self) -> None:
        """
        Resets the hidden state and previous action for the policy.
        """
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(self.pointnav_test_recurrent_hidden_states)
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)
        self._num_steps = 0


if __name__ == "__main__":
    policy = WrappedPointNavResNetPolicy("/home/finn/External/vlfm/pointnav.pth", "/home/finn/External/vlfm/pointnav_conf.pth", "cuda")
    robot_xy = np.array([2.0, 4.0])
    depth = np.random.random((480, 640))
    curr_heading = 0.7
    curr_goal = np.array([2.0, 7.0])
    policy.act(depth, robot_xy, curr_heading, curr_goal)
    print("a")
