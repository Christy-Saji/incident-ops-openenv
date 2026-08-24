"""Tier-1 tripwire: does the reward stack rank the optimal action first?

GRPO can only teach the policy what the reward function ranks highest. If, for a
given state, summing all reward functions puts action B above the optimal action
A, then training drives the policy toward B — no amount of compute, steps or
learning-rate tuning fixes that. And because the SFT phase labels that same state
with A, the two phases actively fight, with the KL term holding the policy at A
while the reward pulls toward B.

This module measures that agreement over every state on every task's optimal
trajectory (training/dataset.iter_training_states).

CURRENT STATUS: the reward stack ranks the optimal action first on roughly half
of the states, so test_reward_argmax_matches_optimal_action is expected to XFAIL.
That is recorded debt, not a flaky test. CI runs with -rxX so the live count
prints in the summary on every run.

WHEN THE TIER-1 REWARD REDESIGN IS DONE: replace the pytest.xfail() call with
    assert matches == total, report
so the suite goes red if the reward stack ever regresses. Reaching 100% here is
the definition of done for that redesign.

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

from env.models import VALID_ACTIONS  # noqa: E402
from training.dataset import iter_training_states  # noqa: E402
from training.prompting import build_prompt  # noqa: E402
from training.reward_functions import (  # noqa: E402
    ALL_REWARD_FUNCTIONS,
    CORE_REWARD_FUNCTIONS,
)

# Both stacks are measured: ALL_REWARD_FUNCTIONS is what training/pipeline.py
# uses, CORE_REWARD_FUNCTIONS is what colab_training.ipynb uses. They are
# different training paths and both must be checked.
REWARD_STACKS = {
    "all9":  ALL_REWARD_FUNCTIONS,
    "core3": CORE_REWARD_FUNCTIONS,
}

# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def score_all_actions(state: dict, reward_funcs=None) -> dict[str, float]:
    """Total reward for every valid action from one state.

    Scores all 19 actions as a single batch per reward function, exactly as
    GRPOTrainer would call them, and sums across the stack.
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
    """Score every training state and record whether the argmax is optimal."""
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
            "correct":         best_action == ts.expected_action,
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
    """The reward stack should rank the optimal action first in every state.

    Expected to XFAIL until the Tier-1 reward redesign lands. See module docstring.
    """
    results = rank_training_states(REWARD_STACKS[stack_name])
    total    = len(results)
    matches  = sum(r["correct"] for r in results)
    mismatches = [r for r in results if not r["correct"]]

    if matches == total:
        return  # reward stack and optimal policy agree everywhere

    detail = "; ".join(
        f"{r['task']}[{len(r['prefix'])}] want={r['expected']} got={r['argmax']} "
        f"(+{r['margin']})"
        for r in mismatches[:6]
    )
    more = f" (+{len(mismatches) - 6} more)" if len(mismatches) > 6 else ""

    pytest.xfail(
        reason=(
            f"[{stack_name}] reward argmax matches optimal action: {matches}/{total} states. "
            f"GRPO cannot learn the optimal policy where these disagree. "
            f"Mismatches: {detail}{more}"
        )
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
