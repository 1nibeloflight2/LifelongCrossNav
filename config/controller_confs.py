from enum import Enum

# numpy
import numpy as np

from spock import spock


@spock
class HabitatControllerConf:
    control_freq: float
    max_vel: float
    max_ang_vel: float


@spock
class SpotControllerConf:
    max_vel: float
    max_ang_vel: float


class ControllerChoice(Enum):
    habitat = HabitatControllerConf
    spot = SpotControllerConf
