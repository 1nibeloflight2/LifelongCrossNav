from dataclasses import dataclass

from abc import ABC, abstractmethod


@dataclass
class NavGoal(ABC):

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def get_descr_point(self):
        pass