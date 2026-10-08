"""Modular components for cross-floor 3D navigation.

Heavy perception dependencies are imported lazily so state/config tooling can
inspect this package without constructing the GPU navigation stack.
"""

__all__ = ["Navigator3D", "StairDirection"]


def __getattr__(name):
    if name == "Navigator3D":
        from .navigator import Navigator3D
        return Navigator3D
    if name == "StairDirection":
        from .state import StairDirection
        return StairDirection
    raise AttributeError(name)
