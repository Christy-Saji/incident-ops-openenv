"""Dataset builders for SFT warm-start and GRPO curriculum training.

Two public functions:
  generate_sft_dataset()  — optimal trajectory (state, action) pairs for SFT
  generate_grpo_dataset() — mixed initial + mid-episode prompts for GRPO
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

from datasets import Dataset

from env.environment import DevOpsEnv
from env.models import VALID_ACTIONS
from tasks.task_config import TASK_CONFIGS


# System prompt injected into every training sample
SYSTEM_PROMPT = (
    "You are an On-call SRE resolving a live infrastructure incident. "
    "Select the single NEXT best action to take.\n"
    "Valid actions: {actions}\n\n"
    "Rules:\n"
    "1. NEVER repeat an action that already appears in 'actions_taken' or 'recent_actions'.\n"
    "2. Follow the SRE workflow in order: DIAGNOSE first, then MITIGATE, then COMMUNICATE "
    "(post_status_update), then RESOLVE.\n"
    "3. Only call resolve_incident when all services are running and mitigations are done.\n"
    "Output ONLY the action name. No explanation."
).format(actions=", ".join(VALID_ACTIONS))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_prompt(state: dict) -> dict:
    """Wrap a state observation as a chat-format prompt."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(state)},
        ]
    }


def _make_prompt_with_history(state: dict, actions_taken: List[str]) -> dict:
    """Like _make_prompt but injects the full actions_taken list.

    This ensures anti_cheat_reward_func and sequence_progress_reward_func
    can correctly penalise repeated actions in mid-episode prompts.
    """
    state_with_history = {**state, "actions_taken": actions_taken}
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(state_with_history)},
        ]
    }


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

def generate_sft_dataset(seed: int = 42) -> Dataset:
    """Generate supervised fine-tuning dataset from optimal action sequences.

    For each of the 6 tasks we replay the full optimal trajectory and record
    every (observation, optimal_next_action) pair as a prompt/completion.
    This gives the model a strong starting policy before GRPO kicks in.

    Returns a HuggingFace Dataset with columns: prompt, completion.
    """
    random.seed(seed)
    data = []

    for task_name, config in TASK_CONFIGS.items():
        optimal_actions = config.get("optimal_actions", [])
        if not optimal_actions:
            continue

        env = DevOpsEnv(task=task_name)
        state = env.reset()

        for action in optimal_actions:
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(state)},
            ]
            data.append({
                "prompt":     prompt,
                "completion": [{"role": "assistant", "content": action}],
            })
            state, _, done, _ = env.step(action)
            if done:
                break

    random.shuffle(data)
    print(f"  [dataset] SFT: {len(data)} (state, optimal_action) pairs across {len(TASK_CONFIGS)} tasks.")
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# GRPO curriculum dataset
# ---------------------------------------------------------------------------

def generate_grpo_dataset(
    per_task_n: int = 8,
    mid_episode_n: int = 60,
    seed: int = 42,
    tasks: Optional[List[str]] = None,
) -> Dataset:
    """Generate a curriculum-mixed prompt dataset for GRPO.

    Distribution:
      - per_task_n  prompts per task initial state (all tasks unless filtered)
      - mid_episode_n states captured after 1–3 random valid steps

    Mid-episode prompts include the full actions_taken list so reward
    functions can correctly penalise repeated actions.

    Args:
        per_task_n:     Number of initial-state prompts per task.
        mid_episode_n:  Number of mid-episode state prompts (sampled randomly).
        seed:           Random seed for reproducibility.
        tasks:          Optional list of task names to include. Defaults to all 6.

    Returns:
        A shuffled HuggingFace Dataset with column: prompt.
    """
    random.seed(seed)
    all_tasks = tasks or list(TASK_CONFIGS.keys())
    data: List[dict] = []

    # Initial states — actions_taken is empty at episode start
    for task in all_tasks:
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(per_task_n):
            data.append(_make_prompt(state))

    # Mid-episode states — inject full action history for reward functions
    for _ in range(mid_episode_n):
        task = random.choice(all_tasks)
        n_warm = random.randint(1, 3)
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(n_warm):
            action = random.choice(VALID_ACTIONS)
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
                break
        data.append(_make_prompt_with_history(state, env._state["actions_taken"]))

    random.shuffle(data)
    print(
        f"  [dataset] GRPO: {len(data)} prompts "
        f"({per_task_n}/task × {len(all_tasks)} tasks + {mid_episode_n} mid-episode)."
    )
    return Dataset.from_list(data)
