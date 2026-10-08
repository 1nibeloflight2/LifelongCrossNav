from enum import Enum

from spock import spock, SpockBuilder
from config import HabitatControllerConf, SpotControllerConf, MappingConf, PlanningConf, ControllerChoice



@spock
class Conf:
    controller: ControllerChoice
    mapping: MappingConf
    planner: PlanningConf
    log_rerun: bool


def load_config():
    return SpockBuilder(Conf, HabitatControllerConf, PlanningConf, MappingConf, SpotControllerConf,
                        desc='Default MON config.').generate()
