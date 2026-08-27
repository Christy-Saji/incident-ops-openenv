"""Tier-1 tripwire: does the reward stack rank the optimal action first?

GRPO can only teach the policy what the reward function ranks highest. If, for a
given state, summing all reward functions puts action B above the optimal action
A, then training drives the policy toward B — no amount of compute, steps or
learning-rate tuning fixes that. And because the SFT phase labels that same state
with A, the two phases actively fight, with the KL term holding the policy at A
while the reward pulls toward B.

This module measures that agreement over every state on every task's optimal
trajectory (training/dataset.iter_training_states), where "optimal" now means
Q*-optimal: an action is correct if Q*(s,a) == max_a' Q*(s,a') (training/qstar.py),
not membership in a single hand-blessed sequence. That is what makes a genuine
tie (e.g. rollback_auth_deploy vs shift_traffic_canary when both are required
mitigations and order between them is arbitrary) score as correct for either
ordering, rather than the over-specified single-answer check this test used
before Tier 1's Q* redesign (docs/prompts/tier1.md Phase B).

STATUS: qstar_reward_func is the primary signal (see training/reward_functions.py)
and reward(s,a) = Q*(s,a) - max_a' Q*(s,a') by construction, so agreement here is
a structural property of the reward, not something tuned toward -- this is a hard
assert, not an xfail. If it ever goes red, the regression is in training/qstar.py
or in how a reward stack composes qstar_reward_func with the others, not in this
test.

Run standalone for the full per-state table:
    python tests/test_reward_ranking.py
"""

from __future__ import annotations

import os
import sys

import pytest

# conftest.py already does this under pytest; repeated here so the module can also
# be run directly (python tests/test_reward_ranking.py) for the full report.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import DevOpsEnv  # noqa: E402
from env.models import VALID_ACTIONS  # noqa: E402
from training.dataset import iter_training_states  # noqa: E402
from training.prompting import build_prompt  # noqa: E402
from training.qstar import canonical_key, solve_cached  # noqa: E402
from training.reward_functions import (  # noqa: E402
    ALL_REWARD_FUNCTIONS,
    CORE_REWARD_FUNCTIONS,
)

# ALL_REWARD_FUNCTIONS is what training/pipeline.py uses, CORE_REWARD_FUNCTIONS
# is what colab_training.ipynb uses. Both are Q*-based post Tier-1 Phase B and
# both must be checked, since they are different training paths.
REWARD_STACKS = {
    "all": ALL_REWARD_FUNCTIONS,
    "core": CORE_REWARD_FUNCTIONS,
}


def _is_qstar_optimal(task: str, prefix: list[str], action: str) -> bool:
    """Is `action` on a Q*-optimal path from the state reached by `prefix`?

    Replays `prefix` (the exact actions_taken history for this training
    state, from iter_training_states) through a fresh env to get the
    canonical key -- the observation dict alone doesn't carry step_count /
    harmful_action_count / effective_mitigations needed to key the Q* table.
    """
    env = DevOpsEnv(task=task)
    env.reset()
    for step_action in prefix:
        env.step(step_action)

    table = solve_cached(task)
    key = canonical_key(env._state, env.current_step)
    qvals = table.q_values.get(key)
    if not qvals:
        return True  # terminal/unexplored state — nothing to disagree about
    best = max(qvals.values())
    return action in qvals and qvals[action] == best

# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def score_all_actions(state: dict, reward_funcs=None) -> dict[str, float]:
    """Total reward for every valid action from one state.

    Scores all 19 actions as a single batch per reward function, exactly as
    GRPOTrainer would call them, and sums across the stack.

    NOTE — the batch here is 19 DISTINCT actions, which is what makes every
    ranking test in this module blind to a group-relative reward: such a reward
    sees no duplicates, returns the same constant for all 19, and shifts every
    total equally without ever changing the argmax. That is not hypothetical.
    diversity_reward_func was exactly that shape and shipped in
    ALL_REWARD_FUNCTIONS through a full 500-step GRPO run without a single test
    in this file going red, while in real training it was ranking a unanimously
    Q*-optimal group below a diverse wrong one. test_rewards_are_independent_of
    _sibling_completions below is the guard for that class of bug; keep it.
    """
    reward_funcs = ALL_REWARD_FUNCTIONS if reward_funcs is None else reward_funcs
    prompts = [build_prompt(state)] * len(VALID_ACTIONS)
    completions = [[{"role": "assistant", "content": a}] for a in VALID_ACTIONS]

    totals = [0.0] * len(VALID_ACTIONS)
    for reward_func in reward_funcs:
        for i, value in enumerate(reward_func(prompts, completions)):
            totals[i] += value

    return dict(zip(VALID_ACTIONS, totals))


def rank_training_states(reward_funcs=None) -> list[dict]:
    """Score every training state and record whether the argmax is Q*-optimal."""
    results = []
    for ts in iter_training_states():
        scores = score_all_actions(ts.state, reward_funcs)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_action, best_score = ranked[0]
        results.append({
            "task":            ts.task,
            "prefix":          ts.prefix,
            "expected":        ts.expected_action,
            "argmax":          best_action,
            "correct":         _is_qstar_optimal(ts.task, ts.prefix, best_action),
            "expected_score":  scores[ts.expected_action],
            "argmax_score":    best_score,
            "margin":          round(best_score - scores[ts.expected_action], 4),
            "top3":            [(a, round(s, 2)) for a, s in ranked[:3]],
        })
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stack_name", sorted(REWARD_STACKS))
def test_reward_argmax_matches_optimal_action(stack_name):
    """The reward stack must rank a Q*-optimal action first in every state.

    Hard assert (Tier-1 Phase B is done): qstar_reward_func's reward is
    Q*(s,a) - max_a' Q*(s,a') by construction, so this is expected to hold
    structurally rather than by tuning. See module docstring.
    """
    results = rank_training_states(REWARD_STACKS[stack_name])
    total    = len(results)
    matches  = sum(r["correct"] for r in results)
    mismatches = [r for r in results if not r["correct"]]

    detail = "; ".join(
        f"{r['task']}[{len(r['prefix'])}] want={r['expected']} got={r['argmax']} "
        f"(+{r['margin']})"
        for r in mismatches[:6]
    )
    more = f" (+{len(mismatches) - 6} more)" if len(mismatches) > 6 else ""

    assert matches == total, (
        f"[{stack_name}] reward argmax matches Q*-optimal action: {matches}/{total} states. "
        f"Mismatches: {detail}{more}"
    )


@pytest.mark.parametrize("stack_name", sorted(REWARD_STACKS))
def test_rewards_are_independent_of_sibling_completions(stack_name):
    """A completion's reward must not depend on the other completions beside it.

    Every ranking test in this module scores one batch of 19 distinct actions,
    so a reward that scores completion i by how many of its siblings match it
    is a constant across that batch and cannot move the argmax — it is
    invisible here no matter how badly it behaves in training.

    diversity_reward_func was that reward. It returns 0.0 for a completion that
    is unique in its group and -0.5 for one duplicated throughout it, so a GRPO
    group that unanimously played the Q*-optimal action scored
    0.1 + 0.0 - 0.5 = -0.4 while a group spread over eight mostly-wrong actions
    scored about -0.1. Perfect play, ranked last. It survived a full 500-step
    run in ALL_REWARD_FUNCTIONS with this file green throughout.

    The property asserted here is the one the ranking tests silently assume:
    scoring an action alone must equal scoring it among identical siblings.
    Any future group-relative reward fails this immediately.
    """
    for ts in iter_training_states():
        prompt = build_prompt(ts.state)
        completion = [{"role": "assistant", "content": ts.expected_action}]

        for reward_func in REWARD_STACKS[stack_name]:
            solo = reward_func([prompt], [completion])[0]
            collapsed = reward_func([prompt] * 8, [completion] * 8)

            assert all(v == pytest.approx(solo) for v in collapsed), (
                f"[{stack_name}] {reward_func.__name__} is group-relative: "
                f"scoring {ts.expected_action!r} alone gives {solo}, but scoring "
                f"it among 8 identical siblings gives {collapsed}. A reward that "
                f"reads its siblings cannot be ranked by this module's other "
                f"tests (they use a batch of 19 distinct actions) and, if it "
                f"penalises agreement, it ranks unanimous optimal play last."
            )


@pytest.mark.parametrize("stack_name", sorted(REWARD_STACKS))
def test_reward_never_ranks_no_op_first(stack_name):
    """no_op must never be the highest-scoring action in any training state.

    Enforceable today (0 violations). It is the weakest possible guarantee that
    the reward stack points somewhere useful, and it would break loudly if a
    future reward change inverted a sign.
    """
    offenders = [
        (r["task"], r["prefix"])
        for r in rank_training_states(REWARD_STACKS[stack_name])
        if r["argmax"] == "no_op"
    ]
    assert not offenders, (
        f"Reward stack ranks no_op highest in {len(offenders)} state(s): {offenders}. "
        f"A policy trained on this reward would learn to idle."
    )


@pytest.mark.parametrize("stack_name", sorted(REWARD_STACKS))
def test_reward_never_ranks_an_already_taken_action_first(stack_name):
    """The top-ranked action must not be one already in the episode history.

    Enforceable today (0 violations). Repeating a completed action is the
    canonical reward-hacking failure mode for this environment, and the anti-cheat
    and task-alignment rewards exist to prevent exactly it.
    """
    offenders = [
        (r["task"], r["prefix"], r["argmax"])
        for r in rank_training_states(REWARD_STACKS[stack_name])
        if r["argmax"] in r["prefix"]
    ]
    assert not offenders, (
        f"Reward stack ranks an already-taken action highest in "
        f"{len(offenders)} state(s): {offenders}"
    )


# ---------------------------------------------------------------------------
# Standalone report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = rank_training_states()
    total   = len(results)
    matches = sum(r["correct"] for r in results)

    print(f"\nReward ranking vs optimal policy — {matches}/{total} states agree\n")
    print(f"{'':2s} {'task':12s} {'n':>2s}  {'expected':<24s} {'argmax':<24s} margin")
    print("-" * 88)
    for r in results:
        mark = "OK" if r["correct"] else "XX"
        print(
            f"{mark:2s} {r['task']:12s} {len(r['prefix']):2d}  "
            f"{r['expected']:<24s} {r['argmax']:<24s} {r['margin']:+.2f}"
        )

    print("\nMismatched states — top 3 ranked actions:\n")
    for r in results:
        if not r["correct"]:
            print(f"  {r['task']}[{len(r['prefix'])}] want={r['expected']}")
            print(f"      {r['top3']}")
