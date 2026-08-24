"""Dataset builders for SFT warm-start and GRPO curriculum training.

Two public functions:
  generate_sft_dataset()  — optimal trajectory (state, action) pairs for SFT
  generate_grpo_dataset() — mixed initial + mid-episode prompts for GRPO

Mid-episode states are built from optimal-prefix warm-ups only. Random actions
are never used, so every mid-episode prompt starts from a clean state with
harmful_action_count == 0 and a recoverable trajectory.

Prompt construction lives in training/prompting.py so that evaluation and
inference build byte-identical prompts.

KNOWN LIMITATION (Tier 1, not fixed here): every prompt this module produces is
a state on an optimal trajectory, which means the GRPO prompt set is a subset of
the SFT training states — 33 unique states, all of them already memorised by the
SFT phase with the optimal action as the label. After SFT the policy is
near-deterministic on them, so all GRPO group samples come out identical and the
advantage (r - group_mean) / group_std is zero. Fixing that needs off-policy
state generation (eps-greedy prefixes, stochastic metrics) and a train/eval task
split. See tests/test_reward_ranking.py for the companion tripwire.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Iterator, List, NamedTuple, Optional

from env.environment import DevOpsEnv
from tasks.task_config import TASK_CONFIGS
from training.prompting import SYSTEM_PROMPT, build_prompt

if TYPE_CHECKING:
    # `datasets` is only in the [train] extra, and is imported lazily inside the
    # two builder functions. Keeping it out of the module body lets tests and
    # scripts import iter_training_states() without installing it.
    from datasets import Dataset

__all__ = [
    "SYSTEM_PROMPT",
    "TrainingState",
    "iter_training_states",
    "generate_sft_dataset",
    "generate_grpo_dataset",
]


# ---------------------------------------------------------------------------
# Training state enumeration
# ---------------------------------------------------------------------------

class TrainingState(NamedTuple):
    """One (state, expected optimal next action) pair from an optimal trajectory."""
    task: str
    prefix: List[str]      # actions already taken to reach this state
    state: dict            # the observation the model is shown
    expected_action: str   # the optimal next action (the SFT label for this state)


def iter_training_states() -> Iterator[TrainingState]:
    """Enumerate every state on every task's optimal trajectory, deterministically.

    This is the exact state space both training datasets draw from:
    generate_sft_dataset walks it in full, and generate_grpo_dataset samples
    from it. Exposing it separately lets tests reason about the reward
    landscape over the same states the model actually trains on, without
    depending on the `datasets` package or on random sampling.
    """
    for task, config in TASK_CONFIGS.items():
        optimal = config.get("optimal_actions", [])
        if not optimal:
            continue

        env = DevOpsEnv(task=task)
        state = env.reset()

        for i, action in enumerate(optimal):
            yield TrainingState(
                task=task,
                prefix=list(optimal[:i]),
                state=state,
                expected_action=action,
            )
            state, _, done, _ = env.step(action)
            if done:
                break


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
    from datasets import Dataset

    random.seed(seed)
    data = []

    for task_name, config in TASK_CONFIGS.items():
        optimal_actions = config.get("optimal_actions", [])
        if not optimal_actions:
            continue

        env = DevOpsEnv(task=task_name)
        state = env.reset()

        for action in optimal_actions:
            data.append({
                "prompt":     build_prompt(state),
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
      - mid_episode_n states captured after 1-3 optimal-prefix steps

    Every prompt carries the full actions_taken history, because the
    observation itself now does (env/models.py), so reward functions can
    reconstruct episode state exactly rather than from the 5-item
    recent_actions window.

    Args:
        per_task_n:     Number of initial-state prompts per task.
        mid_episode_n:  Number of mid-episode state prompts (sampled randomly).
        seed:           Random seed for reproducibility.
        tasks:          Optional list of task names to include. Defaults to all 6.

    Returns:
        A shuffled HuggingFace Dataset with column: prompt.
    """
    from datasets import Dataset

    random.seed(seed)
    all_tasks = tasks or list(TASK_CONFIGS.keys())
    data: List[dict] = []

    # Initial states — actions_taken is empty at episode start.
    # NOTE: these per_task_n copies are all the *same* state, so they add
    # duplicate rows rather than new information (Tier 1: see module docstring).
    for task in all_tasks:
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(per_task_n):
            data.append({"prompt": build_prompt(state)})

    # Mid-episode states — warm-up from optimal-prefix actions only.
    # Using random.choice(VALID_ACTIONS) was discarded because it frequently
    # triggers harmful actions (e.g. rollback_auth_deploy on "medium"), which
    # sets harmful_action_count > 0 before the model acts and permanently
    # suppresses the efficiency component of compute_score for that state.
    # Optimal-prefix warm-up guarantees every mid-episode state is reachable,
    # solvable, and starts with harmful_action_count == 0.
    for _ in range(mid_episode_n):
        task = random.choice(all_tasks)
        cfg = TASK_CONFIGS[task]
        optimal = cfg.get("optimal_actions", [])
        # Take 1 to len(optimal)-1 steps so there is always at least one
        # meaningful action left for the model to learn.
        n_warm = random.randint(1, max(1, len(optimal) - 1))
        env = DevOpsEnv(task=task)
        state = env.reset()
        for i in range(n_warm):
            if i < len(optimal):
                state, _, done, _ = env.step(optimal[i])
                if done:
                    state = env.reset()
                    break
        data.append({"prompt": build_prompt(state)})

    random.shuffle(data)
    unique = len({row["prompt"][-1]["content"] for row in data})
    print(
        f"  [dataset] GRPO: {len(data)} prompts, {unique} unique "
        f"({per_task_n}/task x {len(all_tasks)} tasks + {mid_episode_n} mid-episode)."
    )
    return Dataset.from_list(data)
