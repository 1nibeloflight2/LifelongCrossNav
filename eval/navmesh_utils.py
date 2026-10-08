"""Shared Habitat navmesh settings for cross-floor evaluation."""

from __future__ import annotations

import habitat_sim


CROSS_FLOOR_NAVMESH_PARAMS = {
    "agent_height": 1.5,
    "agent_radius": 0.1,
    "agent_max_climb": 0.3,
    "agent_max_slope": 60.0,
    "cell_size": 0.03,
    "cell_height": 0.05,
}


def make_cross_floor_navmesh_settings() -> habitat_sim.NavMeshSettings:
    """Build the single canonical navmesh configuration used by LCN tools."""
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    for name, value in CROSS_FLOOR_NAVMESH_PARAMS.items():
        setattr(settings, name, value)
    return settings


def recompute_cross_floor_navmesh(
    sim: habitat_sim.Simulator,
    *,
    log_prefix: str = "[CrossFloorNavMesh]",
) -> None:
    """Rebuild a simulator's in-memory navmesh with cross-floor parameters."""
    settings = make_cross_floor_navmesh_settings()
    success = sim.recompute_navmesh(
        sim.pathfinder,
        settings,
        include_static_objects=False,
    )
    if not success:
        raise RuntimeError("Habitat failed to recompute the cross-floor navmesh")
    print(
        f"{log_prefix} rebuilt: "
        f"height={settings.agent_height:.2f}, radius={settings.agent_radius:.2f}, "
        f"climb={settings.agent_max_climb:.2f}, slope={settings.agent_max_slope:.1f}, "
        f"cell={settings.cell_size:.2f}x{settings.cell_height:.2f}, "
        f"islands={sim.pathfinder.num_islands}"
    )
