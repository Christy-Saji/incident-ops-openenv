"""Unit tests for all 9 reward functions.

Run with:
    pytest tests/test_reward_functions.py -v

These tests run entirely on CPU — no GPU or model loading required.
"""

import json

import pytest

from env.environment import DevOpsEnv

# All reward functions
from training.reward_functions import (
    anti_cheat_reward_func,
    diversity_reward_func,
    extract_action,
    format_reward_func,
    sequence_progress_reward_func,
    task_alignment_reward_func,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYSTEM_MSG = "You are an SRE."


def _make_prompt(task: str = "easy", actions_taken: list = None) -> list[dict]:
    """Build a minimal chat prompt for testing."""
    env = DevOpsEnv(task=task)
    state = env.reset()
    if actions_taken:
        for a in actions_taken:
            state, _, _, _ = env.step(a)
        state["actions_taken"] = env._state["actions_taken"]
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": json.dumps(state)},
    ]


def _make_completion(action: str) -> list[dict]:
    return [{"role": "assistant", "content": action}]


def _batch(task: str, action: str, actions_taken: list = None):
    """Return (prompts, completions) for a single-item batch."""
    prompt = _make_prompt(task, actions_taken)
    completion = _make_completion(action)
    return [prompt], [completion]


# ---------------------------------------------------------------------------
# extract_action
# ---------------------------------------------------------------------------

class TestExtractAction:
    def test_exact_match(self):
        assert extract_action("inspect_deploy_history") == "inspect_deploy_history"

    def test_with_prefix(self):
        assert extract_action("Action: rollback_auth_deploy") == "rollback_auth_deploy"

    def test_with_backticks(self):
        assert extract_action("`scale_db_cluster`") == "scale_db_cluster"

    def test_with_newline(self):
        assert extract_action("  inspect_auth_logs\n") == "inspect_auth_logs"

    def test_invalid_returns_none(self):
        assert extract_action("do something crazy") is None

    def test_empty_string(self):
        assert extract_action("") is None

    def test_picks_earliest_action(self):
        # "inspect_auth_logs" appears before "rollback_auth_deploy"
        assert extract_action("inspect_auth_logs then rollback_auth_deploy") == "inspect_auth_logs"

    def test_case_insensitive(self):
        assert extract_action("INSPECT_AUTH_LOGS") == "inspect_auth_logs"


# ---------------------------------------------------------------------------
# format_reward_func
# ---------------------------------------------------------------------------

class TestFormatReward:
    def test_valid_action_positive(self):
        prompts, completions = _batch("easy", "inspect_deploy_history")
        rewards = format_reward_func(prompts, completions)
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.1)

    def test_invalid_action_negative(self):
        prompts, completions = _batch("easy", "do_something_invalid")
        rewards = format_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(-0.4)

    def test_empty_completion_negative(self):
        prompts = [_make_prompt("easy")]
        completions = [[{"role": "assistant", "content": ""}]]
        rewards = format_reward_func(prompts, completions)
        assert rewards[0] < 0

    @pytest.mark.parametrize("text", [
        "inspect_deploy_history",
        "  inspect_deploy_history\n",
        "`inspect_deploy_history`",
        "**inspect_deploy_history**",
        "inspect_deploy_history.",
    ])
    def test_bare_action_scores_ceiling(self, text):
        """Surrounding decoration still counts as a bare answer."""
        prompts = [_make_prompt("easy")]
        completions = [[{"role": "assistant", "content": text}]]
        assert format_reward_func(prompts, completions)[0] == pytest.approx(0.1)

    @pytest.mark.parametrize("text", [
        "Action: inspect_deploy_history",
        "I would inspect_deploy_history because the error rate spiked",
        "Let me think. First inspect_deploy_history, then roll back.",
    ])
    def test_padded_action_scores_between(self, text):
        """A valid action buried in prose must rank below a bare one.

        The earlier two-tier version paid these the same +0.1 as a clean
        answer, so nothing in the reward stack ever penalised length — and
        from ~step 320 of the first full GRPO run the policy rambled until
        every rollout hit max_completion_length without emitting EOS.
        """
        prompts = [_make_prompt("easy")]
        completions = [[{"role": "assistant", "content": text}]]
        reward = format_reward_func(prompts, completions)[0]
        assert reward == pytest.approx(0.0)
        assert reward < 0.1

    def test_varies_within_a_group(self):
        """Must not be constant across a group, or GRPO's baseline cancels it.

        The advantage is (r_i - group_mean) / group_std, so a reward that is
        uniform across a group contributes exactly zero to every gradient in
        it. The two-tier version logged mean 0.100000 / std 0.000000 on ~90%
        of training steps — it was decorative.
        """
        prompts = [_make_prompt("easy")] * 3
        completions = [
            [{"role": "assistant", "content": "inspect_deploy_history"}],
            [{"role": "assistant", "content": "Action: inspect_deploy_history"}],
            [{"role": "assistant", "content": "no idea what to do here"}],
        ]
        rewards = format_reward_func(prompts, completions)
        assert rewards == [pytest.approx(0.1), pytest.approx(0.0), pytest.approx(-0.4)]
        assert len(set(rewards)) == 3


# ---------------------------------------------------------------------------
# anti_cheat_reward_func
# ---------------------------------------------------------------------------

class TestAntiCheatReward:
    def test_novel_action_positive(self):
        prompts, completions = _batch("easy", "inspect_deploy_history")
        rewards = anti_cheat_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(0.2)

    def test_no_op_negative(self):
        prompts, completions = _batch("easy", "no_op")
        rewards = anti_cheat_reward_func(prompts, completions)
        assert rewards[0] < 0

    def test_loop_detection(self):
        # Last 3 actions all identical → strong penalty
        prompts, completions = _batch(
            "easy",
            "inspect_auth_logs",
            actions_taken=["inspect_auth_logs", "inspect_auth_logs", "inspect_auth_logs"],
        )
        rewards = anti_cheat_reward_func(prompts, completions)
        assert rewards[0] <= -0.7

    def test_consecutive_repeat(self):
        # Last action == proposed action → penalty
        prompts, completions = _batch(
            "easy",
            "inspect_deploy_history",
            actions_taken=["inspect_deploy_history"],
        )
        rewards = anti_cheat_reward_func(prompts, completions)
        assert rewards[0] < 0


# ---------------------------------------------------------------------------
# task_alignment_reward_func
# ---------------------------------------------------------------------------

class TestTaskAlignmentReward:
    def test_required_diag_positive(self):
        # easy task requires inspect_deploy_history
        prompts, completions = _batch("easy", "inspect_deploy_history")
        rewards = task_alignment_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(0.40)

    def test_required_mitigation_positive(self):
        # easy task requires rollback_auth_deploy — but only after diagnosis
        prompts, completions = _batch(
            "easy",
            "rollback_auth_deploy",
            actions_taken=["inspect_deploy_history"],
        )
        rewards = task_alignment_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(0.40)

    def test_wrong_inspect_negative(self):
        # network task — inspecting auth logs is wrong
        prompts, completions = _batch("network", "inspect_auth_logs")
        rewards = task_alignment_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(-0.50)

    def test_duplicate_required_action_negative(self):
        # Already did inspect_deploy_history — doing it again should penalise
        prompts, completions = _batch(
            "easy",
            "inspect_deploy_history",
            actions_taken=["inspect_deploy_history"],
        )
        rewards = task_alignment_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# sequence_progress_reward_func
# ---------------------------------------------------------------------------

class TestSequenceProgressReward:
    def test_correct_diag_before_mitigation(self):
        prompts, completions = _batch("easy", "inspect_deploy_history")
        rewards = sequence_progress_reward_func(prompts, completions)
        assert rewards[0] > 0

    def test_mitigation_before_diagnosis_negative(self):
        # Trying to rollback before inspecting
        prompts, completions = _batch("easy", "rollback_auth_deploy")
        rewards = sequence_progress_reward_func(prompts, completions)
        assert rewards[0] < 0

    def test_premature_resolve_heavy_penalty(self):
        prompts, completions = _batch("easy", "resolve_incident")
        rewards = sequence_progress_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(-0.80)

    def test_valid_resolve_after_work(self):
        # Complete all required steps, then resolve
        prompts, completions = _batch(
            "easy",
            "resolve_incident",
            actions_taken=["inspect_deploy_history", "rollback_auth_deploy"],
        )
        rewards = sequence_progress_reward_func(prompts, completions)
        assert rewards[0] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# diversity_reward_func
# ---------------------------------------------------------------------------

class TestDiversityReward:
    def test_diverse_completions_zero_reward(self):
        prompts = [_make_prompt("easy")] * 3
        completions = [
            [{"role": "assistant", "content": "inspect_deploy_history"}],
            [{"role": "assistant", "content": "rollback_auth_deploy"}],
            [{"role": "assistant", "content": "inspect_auth_logs"}],
        ]
        rewards = diversity_reward_func(prompts, completions)
        assert all(r == 0.0 for r in rewards)

    def test_collapsed_completions_negative(self):
        prompts = [_make_prompt("easy")] * 3
        completions = [
            [{"role": "assistant", "content": "inspect_auth_logs"}],
            [{"role": "assistant", "content": "inspect_auth_logs"}],
            [{"role": "assistant", "content": "inspect_auth_logs"}],
        ]
        rewards = diversity_reward_func(prompts, completions)
        assert all(r < 0 for r in rewards)


# ---------------------------------------------------------------------------
# Batch consistency tests
# ---------------------------------------------------------------------------

class TestBatchConsistency:
    """Reward functions must return exactly len(completions) rewards."""

    REWARD_FUNCS = [
        format_reward_func,
        anti_cheat_reward_func,
        task_alignment_reward_func,
        sequence_progress_reward_func,
        diversity_reward_func,
    ]

    @pytest.mark.parametrize("reward_func", REWARD_FUNCS)
    def test_output_length_matches_batch(self, reward_func):
        tasks = ["easy", "medium", "hard"]
        actions = ["inspect_deploy_history", "scale_db_cluster", "no_op"]
        prompts   = [_make_prompt(t) for t in tasks]
        completions = [_make_completion(a) for a in actions]
        rewards = reward_func(prompts, completions)
        assert len(rewards) == len(completions)

    @pytest.mark.parametrize("reward_func", REWARD_FUNCS)
    def test_rewards_are_finite_floats(self, reward_func):
        prompts, completions = _batch("easy", "inspect_deploy_history")
        rewards = reward_func(prompts, completions)
        for r in rewards:
            assert isinstance(r, (int, float))
            assert r == r  # not NaN


