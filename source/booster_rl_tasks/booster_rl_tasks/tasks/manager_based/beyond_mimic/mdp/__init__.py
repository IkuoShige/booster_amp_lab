# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .robustness import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .symmetry import *
from .curriculums import *

# --- Soccer kick task (V1 single-agent) ---
from . import soccer_perception as soccer_perception  # noqa: F401
from . import soccer_commands as soccer_commands  # noqa: F401
from . import soccer_observations as soccer_observations  # noqa: F401
from . import soccer_rewards as soccer_rewards  # noqa: F401
from . import soccer_terminations as soccer_terminations  # noqa: F401
from . import soccer_events as soccer_events  # noqa: F401
from . import soccer_trap_events as soccer_trap_events  # noqa: F401
from . import soccer_trap_rewards as soccer_trap_rewards  # noqa: F401
from . import soccer_curriculums as soccer_curriculums  # noqa: F401
# --- V4 skill library (role one-hot, mode-conditional rewards, history obs) ---
from . import soccer_role_obs as soccer_role_obs  # noqa: F401
# --- V4 Stage 2 — trap skill (single-agent receiver) ---
from . import soccer_trap_commands as soccer_trap_commands  # noqa: F401
from . import soccer_trap_obs as soccer_trap_obs  # noqa: F401
# --- V4 Stage 3 — defend skill (single-agent defender) ---
from . import soccer_defend_commands as soccer_defend_commands  # noqa: F401
from . import soccer_defend_obs as soccer_defend_obs  # noqa: F401
from . import soccer_defend_rewards as soccer_defend_rewards  # noqa: F401
