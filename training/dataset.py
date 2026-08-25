"""Dataset builders for SFT warm-start and GRPO curriculum training.

Public functions:
  generate_sft_dataset()   — optimal trajectory (state, action) pairs for SFT
  generate_grpo_dataset()  — eps-greedy-walk prompts for GRPO (Tier 1 Phase C)

TIER 1 PHASE C FIX: prior to this, every prompt this module produced was a
state on an optimal trajectory, which meant the GRPO prompt set was a subset
of the 33 SFT training states, all of them already memorised by the SFT phase
with the optimal action as the label. After SFT the policy was
near-deterministic on them, so all GRPO group samples came out identical and
the advantage (r - group_mean) / group_std was zero.

generate_grpo_dataset() now draws from iter_eps_greedy_states(), which walks
each task with probability (1 - epsilon) taking the Q*-optimal action and
epsilon taking a random action from training.qstar.relevant_actions(task) --
off-path states are exactly where a policy that has only seen the optimal
trajectory needs teaching, and Q* (training/qstar.py) is defined over that
same action set so every state visited has an exact label. Held-out tasks
(memory_leak, disk_full by default) never appear in either dataset -- see
DEFAULT_TRAIN_TASKS / DEFAULT_EVAL_TASKS below.

Prompt construction lives in training/prompting.py so that evaluation and
inference build byte-identical prompts.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Iterator, List, NamedTuple, Optional

from env.environment import DevOpsEnv
from tasks.task_config import TASK_CONFIGS
from training.prompting import SYSTEM_PROMPT, build_prompt
from training.qstar import canonical_key, relevant_actions, solve_cached

if TYPE_CHECKING:
    # `datasets` is only in the [train] extra, and is imported lazily inside the
    # two builder functions. Keeping it out of the module body lets tests and
    # scripts import iter_training_states() without installing it.
    from datasets import Dataset

__all__ = [
    "SYSTEM_PROMPT",
    "DEFAULT_TRAIN_TASKS",
    "DEFAULT_EVAL_TASKS",
    "TrainingState",
    "iter_training_states",
    "iter_eps_greedy_states",
    "generate_sft_dataset",
    "generate_grpo_dataset",
]

# Tier 1 decision: train on easy/medium/hard/network, hold out memory_leak
# and disk_full entirely so scripts/evaluate.py's held-out eval is meaningful.
DEFAULT_TRAIN_TASKS = ["easy", "medium", "hard", "network"]
DEFAULT_EVAL_TASKS = ["memory_leak", "disk_full"]


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


def _optimal_trajectory_keys(task: str) -> set:
    """Canonical keys of every state on `task`'s optimal_actions trajectory.

    Mirrors iter_training_states()'s walk but returns keys instead of
    TrainingState objects. Used to exclude SFT states from the GRPO walk.
    """
    env = DevOpsEnv(task=task)
    env.reset()
    keys = {canonical_key(env._state, env.current_step)}
    for action in TASK_CONFIGS[task].get("optimal_actions", []):
        _, _, done, _ = env.step(action)
        keys.add(canonical_key(env._state, env.current_step))
        if done:
            break
    return keys


def iter_eps_greedy_states(
    tasks: Optional[List[str]] = None,
    n_states: int = 1000,
    epsilon: float = 0.3,
    seed: int = 42,
) -> Iterator[TrainingState]:
    """Eps-greedy walk over each task's Q*-solved state space (Tier 1 Phase C).

    From each task's initial state, repeatedly take the Q*-optimal action with
    probability (1 - epsilon), otherwise a random action from
    training.qstar.relevant_actions(task) -- the same restricted action set
    training.qstar.solve() branches on, so every visited state has an exact
    Q* label to draw from. Each unique canonical state (training.qstar.canonical_key)
    is yielded once, labelled with its Q*-optimal action -- not the possibly
    -random action actually taken to reach it, since the label must be correct
    regardless of how the state was reached.

    Unlike iter_training_states() (which walks exactly one fixed trajectory
    per task and backs tests/test_reward_ranking.py's small, deterministic
    state set), this explores broadly and is what generate_grpo_dataset() and
    generate_sft_dataset() draw from for Phase C's expanded training
    distribution.
    """
    rng = random.Random(seed)
    task_list = list(tasks or TASK_CONFIGS)
    per_task_target = max(1, n_states // len(task_list))

    for task in task_list:
        table = solve_cached(task)
        branch_actions = sorted(relevant_actions(task))
        # Every state on the hand-picked optimal trajectory is already an SFT
        # training example (generate_sft_dataset walks exactly this). Seeding
        # `seen` with them means the walk below passes through but never
        # yields them, guaranteeing zero overlap between the SFT and GRPO
        # state sets (see tests/test_dataset.py) instead of leaving it to
        # chance -- the eps-greedy walk starts from the same initial state
        # every task shares with SFT, so without this the overlap is certain,
        # not merely possible.
        seen: set = _optimal_trajectory_keys(task)
        produced = 0
        attempts = 0
        max_attempts = per_task_target * 25

        while produced < per_task_target and attempts < max_attempts:
            attempts += 1
            env = DevOpsEnv(task=task)
            state = env.reset()
            done = False
            while not done:
                key = canonical_key(env._state, env.current_step)
                qvals = table.q_values.get(key)
                if not qvals:
                    break  # terminal, or outside the solved table

                optimal_action = max(qvals, key=qvals.get)

                if key not in seen and produced < per_task_target:
                    seen.add(key)
                    produced += 1
                    yield TrainingState(
                        task=task,
                        prefix=list(env._state["actions_taken"]),
                        state=state,
                        expected_action=optimal_action,
                    )
                    if produced >= per_task_target:
                        break

                action = rng.choice(branch_actions) if rng.random() < epsilon else optimal_action
                state, _, done, _ = env.step(action)


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

def generate_sft_dataset(seed: int = 42, tasks: Optional[List[str]] = None) -> Dataset:
    """Generate supervised fine-tuning dataset from optimal action sequences.

    For each task in `tasks` (default: DEFAULT_TRAIN_TASKS -- Tier 1 holds
    memory_leak/disk_full out of both SFT and GRPO, see module docstring) we
    replay the full optimal trajectory and record every
    (observation, optimal_next_action) pair as a prompt/completion. This gives
    the model a strong starting policy before GRPO kicks in.

    Returns a HuggingFace Dataset with columns: prompt, completion.
    """
    from datasets import Dataset

    random.seed(seed)
    task_list = tasks if tasks is not None else DEFAULT_TRAIN_TASKS
    data = []

    for task_name in task_list:
        config = TASK_CONFIGS[task_name]
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
    print(f"  [dataset] SFT: {len(data)} (state, optimal_action) pairs across {len(task_list)} tasks {task_list}.")
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# GRPO curriculum dataset
# ---------------------------------------------------------------------------

def generate_grpo_dataset(
    n_states: int = 1000,
    epsilon: float = 0.3,
    seed: int = 42,
    tasks: Optional[List[str]] = None,
) -> Dataset:
    """Generate a GRPO prompt dataset via the eps-greedy walk (Tier 1 Phase C).

    Every prompt carries the full actions_taken history, because the
    observation itself now does (env/models.py), so reward functions can
    reconstruct episode state exactly rather than from the 5-item
    recent_actions window.

    Args:
        n_states: Target number of unique prompts (split evenly across tasks).
        epsilon:  Probability of taking a random (off-Q*-path) action at each
                  step of the walk -- see iter_eps_greedy_states().
        seed:     Random seed for reproducibility.
        tasks:    Task names to include. Defaults to DEFAULT_TRAIN_TASKS
                  (memory_leak/disk_full held out).

    Returns:
        A shuffled HuggingFace Dataset with column: prompt.
    """
    from datasets import Dataset

    task_list = tasks if tasks is not None else DEFAULT_TRAIN_TASKS
    data: List[dict] = [
        {"prompt": build_prompt(ts.state)}
        for ts in iter_eps_greedy_states(
            tasks=task_list, n_states=n_states, epsilon=epsilon, seed=seed,
        )
    ]

    random.Random(seed).shuffle(data)
    unique = len({row["prompt"][-1]["content"] for row in data})
    print(
        f"  [dataset] GRPO: {len(data)} prompts, {unique} unique "
        f"(eps={epsilon} walk over {task_list})."
    )
    return Dataset.from_list(data)
