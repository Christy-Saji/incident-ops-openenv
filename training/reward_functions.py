"""All 9 GRPO reward signal functions for the Incident Ops environment.

Each function follows the TRL GRPOTrainer signature:
    f(prompts, completions, **kwargs) -> List[float]

Reward signal index:
  1. format_reward_func           — valid action string check
  2. step_reward_func             — environment step reward (task-aware)
  3. anti_cheat_reward_func       — penalise no_op / premature resolve / loops
  4. task_alignment_reward_func   — task-correct diagnostics/mitigations
  5. sequence_progress_reward_func— enforce diagnose→mitigate→communicate→resolve order
  6. progress_delta_reward_func   — dense reward for measurable task progress
  7. communication_gate_reward_func—gate comms until technical progress exists
  8. terminal_outcome_reward_func — primary outcome signal (score delta + resolution)
  9. diversity_reward_func        — penalise GRPO group-level mode collapse
"""

from __future__ import annotations

import json
import re
from typing import List

from env.environment import DevOpsEnv
from env.models import VALID_ACTIONS
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_action(text: str) -> str | None:
    """Extract the first valid action from raw model output.

    Handles outputs like:
      - "inspect_deploy_history"
      - "Action: inspect_deploy_history"
      - JSON snippets containing an action token
    """
    if not isinstance(text, str):
        return None

    cleaned = text.strip().lower().replace("`", " ").replace("\n", " ")
    if cleaned in VALID_ACTIONS:
        return cleaned

    best_action = None
    best_pos = None
    for action in VALID_ACTIONS:
        match = re.search(rf"\b{re.escape(action)}\b", cleaned)
        if match:
            if best_pos is None or match.start() < best_pos:
                best_action = action
                best_pos = match.start()
    return best_action


def _extract_state_from_prompt(prompt: list[dict]) -> dict:
    """Best-effort parse of the user-state JSON from a chat prompt."""
    try:
        if not prompt:
            return {}
        return json.loads(prompt[-1]["content"])
    except Exception:
        return {}


def _replay_env_from_prompt(prompt: list[dict]) -> tuple[DevOpsEnv, dict, str]:
    """Reconstruct approximate environment state by replaying prompt history.

    Preference order for replayed history:
      1) actions_taken  (full episode history)
      2) recent_actions (short sliding window fallback)
    """
    state_dict = _extract_state_from_prompt(prompt)
    task = state_dict.get("task", "easy")
    if task not in TASK_CONFIGS:
        task = "easy"

    env = DevOpsEnv(task=task)
    env.reset()

    history = state_dict.get("actions_taken", state_dict.get("recent_actions", []))
    if not isinstance(history, list):
        history = []

    for prev_action in history:
        if prev_action in VALID_ACTIONS:
            env.step(prev_action)

    return env, state_dict, task


# Actions that are never task-useful
_FILLER_ACTIONS = {"acknowledge_incident"}
_NO_VALUE_ACTIONS = {"no_op"}

# All inspect and mitigation action names (used in multiple reward funcs)
_INSPECT_ACTIONS = {
    "inspect_auth_logs", "inspect_db_metrics", "inspect_deploy_history",
    "inspect_network_topology", "inspect_memory_profile", "inspect_disk_usage",
}
_MIT_ACTIONS = {
    "rollback_auth_deploy", "rollback_service_deploy", "restart_auth_service",
    "scale_db_cluster", "flush_cache", "shift_traffic_canary", "withdraw_bgp_route",
    "archive_old_logs", "reduce_log_verbosity",
}


# ---------------------------------------------------------------------------
# 1. Format reward
# ---------------------------------------------------------------------------

def format_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 1: Did the model output a single valid action string?

    Reduced from 1.0 → 0.1 so this baseline signal no longer dominates
    the task-specific rewards. The model has a real incentive to explore
    diagnostic / mitigation actions rather than just any valid action.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        action = extract_action(text)
        rewards.append(0.1 if action in VALID_ACTIONS else -0.4)
    return rewards


# ---------------------------------------------------------------------------
# 2. Step reward
# ---------------------------------------------------------------------------

def step_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 2: Does this action improve the incident state?

    Restores approximate mid-episode state by replaying recent_actions from the
    serialised observation in the user message before scoring the candidate action.
    NaN safety: the entire env instantiation + step is wrapped in try/except.
    Return value is explicitly clamped to [-1.0, 1.0].
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        try:
            env, _, _ = _replay_env_from_prompt(prompt)
            _, step_reward, _, _ = env.step(action)
            rewards.append(float(max(-1.0, min(1.0, step_reward))))
        except Exception:
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------------------------
# 3. Anti-cheat reward
# ---------------------------------------------------------------------------

def anti_cheat_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 3: Penalise reward hacking (loop detection + no-op spam).

    Priority order (first match wins):
      1. Loop (last 3 all identical)            → -0.8
      2. Consecutive repeat (last action == now) → -0.7
      3. no_op                                  → -0.6
      4. resolve_incident (usually premature)   → -0.3
      5. acknowledge spam (2nd+ time)           → -0.6
      6. Action already done somewhere          → -0.4
      7. Novel action                           → +0.2
    """
    rewards = []
    for i, completion in enumerate(completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        prompt = prompts[i] if i < len(prompts) else []
        state_dict = _extract_state_from_prompt(prompt)
        all_actions_taken: List[str] = state_dict.get(
            "actions_taken", state_dict.get("recent_actions", [])
        )
        if not isinstance(all_actions_taken, list):
            all_actions_taken = []

        last_action = all_actions_taken[-1] if all_actions_taken else None

        if len(all_actions_taken) >= 3 and len(set(all_actions_taken[-3:])) == 1:
            rewards.append(-0.8)
        elif action == last_action:
            rewards.append(-0.7)
        elif action in _NO_VALUE_ACTIONS:
            rewards.append(-0.6)
        elif action == "resolve_incident":
            rewards.append(-0.3)
        elif action in _FILLER_ACTIONS and all_actions_taken.count(action) >= 1:
            rewards.append(-0.6)
        elif action in all_actions_taken:
            rewards.append(-0.4)
        else:
            rewards.append(0.2)
    return rewards


# ---------------------------------------------------------------------------
# 4. Task alignment reward
# ---------------------------------------------------------------------------

def task_alignment_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 4: Bonus for task-correct diagnostics/mitigations; penalty for wrong ones.

    Action categories and rewards:
      required_diag  (not yet done)  → +0.40
      required_mit   (not yet done)  → +0.40
      required_diag/mit (already done)→ -0.25  (discourage duplicate)
      good_followups (comms/resolve)  → +0.10
      inspect_* NOT in required_diag  → -0.50  (wrong diagnostic)
      other task-specific actions wrong→ -0.35
      no_op                           → -0.50
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(-0.3)
            continue

        task = "easy"
        all_actions_taken: List[str] = []
        try:
            state_dict = json.loads(prompt[-1]["content"])
            task = state_dict.get("task", "easy")
            all_actions_taken = state_dict.get(
                "actions_taken", state_dict.get("recent_actions", [])
            )
            if not isinstance(all_actions_taken, list):
                all_actions_taken = []
        except Exception:
            pass

        config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
        required_diag  = config.get("required_diagnostics", [])
        required_mit   = config.get("required_mitigations", [])
        good_followups = config.get("good_followups", [])

        if action in required_diag and action not in all_actions_taken:
            rewards.append(0.40)
        elif action in required_mit and action not in all_actions_taken:
            rewards.append(0.40)
        elif action in required_diag or action in required_mit:
            rewards.append(-0.25)
        elif action in good_followups:
            rewards.append(0.10)
        elif action in _INSPECT_ACTIONS:
            rewards.append(-0.50)
        elif action in _MIT_ACTIONS:
            rewards.append(-0.35)
        elif action in _NO_VALUE_ACTIONS:
            rewards.append(-0.50)
        else:
            rewards.append(-0.15)

    return rewards


# ---------------------------------------------------------------------------
# 5. Sequence progress reward
# ---------------------------------------------------------------------------

def sequence_progress_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 5: Enforce the correct SRE workflow order.

    investigate → mitigate → communicate → resolve
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(0.0)
            continue

        state_dict = _extract_state_from_prompt(prompt)
        task = state_dict.get("task", "easy")
        if task not in TASK_CONFIGS:
            task = "easy"
        all_actions_taken: List[str] = state_dict.get(
            "actions_taken", state_dict.get("recent_actions", [])
        )
        if not isinstance(all_actions_taken, list):
            all_actions_taken = []

        config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
        required_diag = set(config.get("required_diagnostics", []))
        required_mit  = set(config.get("required_mitigations", []))
        done_so_far   = set(all_actions_taken)

        diag_complete = required_diag.issubset(done_so_far)
        mit_complete  = required_mit.issubset(done_so_far)
        is_resolve    = action == "resolve_incident"

        if action in required_diag and not diag_complete:
            rewards.append(0.25)
        elif action in required_mit and diag_complete and not mit_complete:
            rewards.append(0.35)
        elif action in required_mit and not diag_complete:
            rewards.append(-0.35)
        elif is_resolve and diag_complete and mit_complete:
            rewards.append(0.60)
        elif is_resolve and not (diag_complete and mit_complete):
            rewards.append(-0.80)
        else:
            rewards.append(-0.05)

    return rewards


# ---------------------------------------------------------------------------
# 6. Progress delta reward
# ---------------------------------------------------------------------------

def progress_delta_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 6: Dense reward for measurable task progress.

    Compares completion of required diagnostics/mitigations before vs after
    applying the candidate action in the replayed environment.
    """
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")
        if action not in VALID_ACTIONS:
            rewards.append(-0.4)
            continue
        try:
            env, _, task = _replay_env_from_prompt(prompt)
            cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
            req_diag = set(cfg.get("required_diagnostics", []))
            req_mit  = set(cfg.get("required_mitigations", []))

            before = set(env._state.get("actions_taken", []))
            before_progress = (
                len(req_diag.intersection(before)) +
                len(req_mit.intersection(before))
            )

            env.step(action)
            after = set(env._state.get("actions_taken", []))
            after_progress = (
                len(req_diag.intersection(after)) +
                len(req_mit.intersection(after))
            )

            delta = after_progress - before_progress
            if delta > 0:
                rewards.append(0.5)
            elif action in {"acknowledge_incident", "post_status_update"}:
                rewards.append(-0.6)
            elif action in before:
                rewards.append(-0.5)
            else:
                rewards.append(-0.35)
        except Exception:
            rewards.append(-0.4)
    return rewards


# ---------------------------------------------------------------------------
# 7. Communication gate reward
# ---------------------------------------------------------------------------

def communication_gate_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 7: Gate communication actions until technical progress exists."""
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")
        if action not in VALID_ACTIONS:
            rewards.append(-0.3)
            continue
        try:
            env, _, task = _replay_env_from_prompt(prompt)
            cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
            req_diag = set(cfg.get("required_diagnostics", []))
            req_mit  = set(cfg.get("required_mitigations", []))
            taken    = set(env._state.get("actions_taken", []))
            diag_done = len(req_diag.intersection(taken)) > 0
            mit_done  = len(req_mit.intersection(taken)) > 0
            tech_progress = diag_done or mit_done

            if action == "acknowledge_incident":
                rewards.append(-0.8 if "acknowledge_incident" in taken else 0.05)
            elif action == "post_status_update":
                rewards.append(0.15 if tech_progress else -0.8)
            elif action == "resolve_incident":
                rewards.append(0.2 if (diag_done and mit_done) else -1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(-0.3)
    return rewards


# ---------------------------------------------------------------------------
# 8. Terminal outcome reward
# ---------------------------------------------------------------------------

def terminal_outcome_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 8: Primary outcome-first terminal objective.

    Reconstructs episode state from prompt history, simulates the proposed
    action, and rewards score progress + resolution. Strongly favors true
    resolution and penalises farming / looping.
    """
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")
        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        try:
            env, _, task = _replay_env_from_prompt(prompt)
            pre_score, _ = compute_score(task, env._state)
            _, _, _, info = env.step(action)
            post_score, _ = compute_score(task, env._state)

            score_delta = float(post_score - pre_score)
            state = env._state
            cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
            required_mit  = set(cfg.get("required_mitigations", []))
            required_diag = set(cfg.get("required_diagnostics", []))
            done_actions  = set(state.get("actions_taken", []))
            mit_complete  = required_mit.issubset(done_actions)
            stable        = state.get("incident_phase") in {"monitoring", "resolved"}
            resolved      = bool(state.get("resolved", False))

            reward = 1.5 * score_delta

            history = state.get("actions_taken", [])
            comms_done = "post_status_update" in history

            if resolved and stable and mit_complete:
                reward += 3.0 if comms_done else 1.0
            elif action == "resolve_incident" and info.get("error") == "incident_not_stable":
                reward -= 1.5
            elif action == "resolve_incident" and not (stable and mit_complete):
                reward -= 1.3

            # history already set above for comms gate
            if len(history) >= 4 and len(set(history[-4:])) == 1:
                reward -= 1.0
            if action in _FILLER_ACTIONS and len(history) > 2:
                reward -= 0.8
            elif action in _FILLER_ACTIONS:
                reward -= 0.35
            if history.count(action) > 1:
                reward -= 0.6

            if action in required_diag and action not in set(history[:-1]):
                reward += 0.4
            if action in required_mit and action not in set(history[:-1]):
                reward += 0.5

            rewards.append(float(max(-2.5, min(3.5, reward))))
        except Exception:
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------------------------
# 9. Diversity reward
# ---------------------------------------------------------------------------

def diversity_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 9: Penalise GRPO group-level mode collapse.

    GRPO samples num_generations completions per prompt. When all completions
    are identical the reward_std is 0 — there is no gradient signal and the
    model stops learning. This reward detects that collapse at the batch level
    and applies a uniform negative signal so the KL term forces exploration.
    Returns 0.0 when the group shows healthy diversity.
    """
    actions = [extract_action(c[0]["content"] or "") or "" for c in completions]
    valid_actions = [a for a in actions if a in VALID_ACTIONS]
    unique_count = len(set(valid_actions)) if valid_actions else 0

    if unique_count <= 1:
        return [-0.5] * len(completions)
    elif unique_count == 2:
        return [-0.1] * len(completions)
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Convenience list — import this in pipeline.py
# ---------------------------------------------------------------------------

ALL_REWARD_FUNCTIONS = [
    format_reward_func,
    step_reward_func,
    anti_cheat_reward_func,
    task_alignment_reward_func,
    sequence_progress_reward_func,
    progress_delta_reward_func,
    communication_gate_reward_func,
    terminal_outcome_reward_func,
    diversity_reward_func,
]
