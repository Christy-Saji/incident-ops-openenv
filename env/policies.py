"""Baseline (non-learned) policies for the incident triage environment.

Lives in `env` rather than `scripts` or `server` so that the evaluation script,
the FastAPI demo endpoint and any future consumer share one implementation.
Imports nothing heavier than the task configs.
"""

from __future__ import annotations

from tasks.task_config import TASK_CONFIGS


def heuristic_policy(task: str, state: dict) -> str:
    """Deterministic optimal policy — walks the task's optimal_actions list.

    Reads `actions_taken` from the observation. That field must be present on the
    Observation schema (env/models.py); when it was missing this function saw an
    empty history every step and returned optimal_actions[0] forever, turning the
    "optimal baseline" into a policy that repeated one action until max_steps and
    resolved nothing.
    """
    config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
    optimal = config.get("optimal_actions", [])
    done_actions = set(state.get("actions_taken", []))

    for action in optimal:
        if action not in done_actions:
            return action

    # Fallback: resolve if everything else is done, else idle.
    if "resolve_incident" not in done_actions:
        return "resolve_incident"
    return "no_op"
