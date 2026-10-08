from dataclasses import dataclass

# typing
from typing import List, Dict, Union, Optional

import numpy as np

@dataclass
class SemanticObject:
    object_id: str  # unique identifier
    object_category: str
    bbox: List[float]
    semantic_id: Optional = None
    view_pts: Union[List, None] = None

    def __eq__(self, other):
        if isinstance(other, SemanticObject):
            return self.object_id == other.object_id
        elif isinstance(other, str):
            return self.object_id == other
        return False


@dataclass
class SceneData:
    scene_id: str
    object_locations: Dict[str, List[SemanticObject]]
    object_ids: Dict[str, List[str]]
    objects_loaded: bool = False


@dataclass
class Episode:
    scene_id: str
    episode_id: int
    start_position: List[float]
    start_rotation: List[float]
    obj_sequence: List[str]
    best_dist: Union[float, List[float]]
    floor_id: Union[int, None] = None

@dataclass
class GibsonEpisode(Episode):
    object_id: int = 0
    floor_id: int = 0