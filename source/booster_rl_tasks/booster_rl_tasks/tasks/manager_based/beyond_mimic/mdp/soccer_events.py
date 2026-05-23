"""Event-manager helpers specific to the soccer kick task.

Most reset/spawn behavior lives inside :class:`SoccerKickCommand` so that ball
+ robot poses are sampled together. This module only adds the domain-
randomization events that act once at scene startup, and a small push event.
"""
from __future__ import annotations

# Re-exported here so env configs can address everything from this module.
from isaaclab.envs.mdp import (  # noqa: F401  (re-export for convenience)
    randomize_rigid_body_material,
    randomize_rigid_body_mass,
    apply_external_force_torque,
    push_by_setting_velocity,
)
