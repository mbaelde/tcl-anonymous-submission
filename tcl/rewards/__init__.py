"""Reward primitives for the TCL paper.

Only the standalone (A) formulation is exposed here; the shaped (B) reward
lives inline in `agents/sac_tcl.py` for historical reasons (it is bound to
the SAC training loop and re-evaluates beta on every gradient step).
"""

from tcl.rewards.standalone import (
    tcl_standalone_reward,
    tcl_standalone_reward_gaussian,
)

__all__ = ["tcl_standalone_reward", "tcl_standalone_reward_gaussian"]
