"""Server-side policy rollout for the /demo endpoint.

Restores the two entry points app.py needs. The previous top-level inference.py
was deleted in 40326bc while app.py still imported from it, which meant the
FastAPI app could not import at all — the server, the CI Docker smoke test and
the Hugging Face Space were all down.

This version is deliberately thin: prompt construction, action parsing and the
heuristic baseline are imported from their existing homes rather than
re-implemented, which is what made the original 374-line module drift out of
sync with training in the first place.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from env.environment import DevOpsEnv
from env.policies import heuristic_policy
from graders.grader import compute_score
from training.prompting import build_prompt
from training.reward_functions import extract_action

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - openai is a base dependency
    OpenAI = None  # type: ignore[assignment]
    OPENAI_AVAILABLE = False


# Consecutive identical actions before the rollout is judged stuck and stopped.
LOOP_STREAK_LIMIT = 5
LOOP_PENALTY_PER_TRIP = 0.03


def create_openai_client_from_env() -> tuple[Optional[Any], Optional[str]]:
    """Build an OpenAI-compatible client from environment variables.

    Returns (None, None) when no credentials are configured, which makes the
    caller fall back to the heuristic baseline instead of failing.
    """
    if not OPENAI_AVAILABLE:
        return None, None

    api_base   = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
    )

    if not api_base or not model_name or not api_key:
        # No key means every request would 401; fall back to the heuristic.
        return None, None

    return OpenAI(base_url=api_base, api_key=api_key), model_name


def _llm_action(state: dict, client: Any, model: str) -> Optional[str]:
    """One LLM call returning a valid action, or None on any failure.

    Uses training/prompting.build_prompt so the served prompt matches the
    trained one exactly.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_prompt(state),
            max_tokens=32,
            temperature=0.3,
        )
        return extract_action(response.choices[0].message.content or "")
    except Exception as exc:
        print(f"[inference] LLM call failed ({type(exc).__name__}: {exc}) — using heuristic")
        return None


def run_episode(
    task_name: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    *,
    partial_obs: bool = False,
    stochastic: bool = False,
) -> dict:
    """Run one full episode and return a JSON-serialisable trajectory.

    Falls back to the heuristic baseline per-step whenever the LLM is
    unavailable or returns an unparseable action, so /demo always returns a
    complete episode.
    """
    env = DevOpsEnv(task=task_name, partial_obs=partial_obs, stochastic=stochastic)
    state = env.reset()
    use_llm = client is not None and model is not None
    policy_label = model if use_llm else "heuristic_baseline"

    rewards: list[str] = []
    steps: list[dict] = []
    repeat_streak = 0
    last_action: Optional[str] = None
    loop_penalty = 0.0

    for step_num in range(1, env.max_steps + 1):
        action = _llm_action(state, client, model) if use_llm else None
        if action is None:
            action = heuristic_policy(task_name, state)

        state, reward, done, info = env.step(action)
        rewards.append(f"{reward:.2f}")
        steps.append({
            "step":     step_num,
            "action":   action,
            "reward":   round(float(reward), 4),
            "done":     done,
            "error":    info.get("error"),
            "phase":    state["incident_phase"],
            "metrics":  state["metrics"],
        })

        if action == last_action:
            repeat_streak += 1
        else:
            repeat_streak = 1
            last_action = action

        if repeat_streak >= LOOP_STREAK_LIMIT and not env._state.get("resolved", False):
            loop_penalty += LOOP_PENALTY_PER_TRIP
            break

        if done:
            break

    score, breakdown = compute_score(task_name, env._state)
    score = max(0.0, score - loop_penalty)

    return {
        "task":            task_name,
        "policy":          "llm" if use_llm else "heuristic_baseline",
        "policy_label":    policy_label,
        "partial_obs":     partial_obs,
        "stochastic":      stochastic,
        "env":             env,
        "score":           round(score, 4),
        "resolved":        env._state["resolved"],
        "steps_taken":     len(steps),
        "rewards":         rewards,
        "score_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        "steps":           steps,
        "episode":         env.episode(),
    }
