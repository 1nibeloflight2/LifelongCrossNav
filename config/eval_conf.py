from spock import spock, SpockBuilder

from config import HabitatControllerConf, MappingConf, PlanningConf


@spock
class EvalConf:
    multi_object: bool
    max_steps: int
    max_dist: float
    log_rerun: bool
    is_gibson: bool
    controller: HabitatControllerConf
    mapping: MappingConf
    planner: PlanningConf
    object_nav_path: str
    scene_path: str
    use_pointnav: bool
    square_im: bool


def load_eval_config():
    return SpockBuilder(EvalConf, HabitatControllerConf, PlanningConf, MappingConf,
                        desc='Default MON config.').generate()
