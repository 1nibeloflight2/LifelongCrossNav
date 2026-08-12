"""Shared state types for 3D navigation."""

from enum import Enum


class StairDirection(str, Enum):
    """Canonical stair travel directions used by logs and new APIs."""

    ASCENT = "ascent"
    DESCENT = "descend"

    @classmethod
    def normalize(cls, value: "StairDirection | str") -> "StairDirection":
        if isinstance(value, cls):
            return value
        value = str(value).lower()
        if value in {"up", "rise", "ascent"}:
            return cls.ASCENT
        if value in {"down", "descend", "descent"}:
            return cls.DESCENT
        raise ValueError(f"Unsupported stair direction: {value}")

    @property
    def legacy(self) -> str:
        return "up" if self is StairDirection.ASCENT else "down"
