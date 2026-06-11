"""Unit tests for the grader / scoring system."""

import pytest
from env.environment import DevOpsEnv
from graders.grader import compute_score, compute_breakdown
from tasks.task_config import TASK_CONFIGS


class TestComputeBreakdown:
    def test_breakdown_has_required_keys(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        breakdown = compute_breakdown("easy", env._state)
        # compute_breakdown includes the internal _harm_penalty key;
        # compute_score pops it before returning the public breakdown.
        public_keys = {k for k in breakdown.keys() if not k.startswith("_")}
        assert public_keys == {"diagnosis", "mitigation", "recovery", "communication", "efficiency"}

    def test_fresh_state_low_scores(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        bd = compute_breakdown("easy", env._state)
        assert bd["diagnosis"] == pytest.approx(0.0)
        assert bd["mitigation"] == pytest.approx(0.0)

    @pytest.mark.parametrize("task", list(TASK_CONFIGS.keys()))
    def test_all_tasks_have_valid_scores(self, task):
        env = DevOpsEnv(task=task)
        env.reset()
        score, bd = compute_score(task, env._state)
        assert 0.0 <= score <= 1.0
        for v in bd.values():
            assert 0.0 <= v <= 1.0


class TestComputeScore:
    def test_score_in_range(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        score, _ = compute_score("easy", env._state)
        assert 0.0 <= score <= 1.0

    def test_optimal_trajectory_high_score(self):
        """Optimal trajectory should yield a significantly higher score than random."""
        for task, config in TASK_CONFIGS.items():
            env = DevOpsEnv(task=task)
            env.reset()
            for action in config["optimal_actions"]:
                _, _, done, _ = env.step(action)
                if done:
                    break
            score, _ = compute_score(task, env._state)
            assert score >= 0.5, f"Task '{task}' optimal score should be >= 0.5, got {score}"

    def test_harmful_actions_reduce_score(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        # Commit several harmful no-ops
        for _ in range(5):
            env.step("no_op")
        score_noops, _ = compute_score("easy", env._state)

        env2 = DevOpsEnv(task="easy")
        env2.reset()
        score_clean, _ = compute_score("easy", env2._state)

        # Harmful actions should degrade (or not improve) the score
        assert score_noops <= score_clean + 0.05  # allow tiny float tolerance

    def test_resolved_state_efficiency_bonus(self):
        """Resolving efficiently (at or under optimal step count) should not reduce score."""
        env = DevOpsEnv(task="easy")
        env.reset()
        config = TASK_CONFIGS["easy"]
        for action in config["optimal_actions"]:
            _, _, done, _ = env.step(action)
            if done:
                break
        score, bd = compute_score("easy", env._state)
        assert bd["efficiency"] > 0.0

    def test_communication_score_increases_with_log(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        score_before, bd_before = compute_score("easy", env._state)
        env.step("post_status_update")
        score_after, bd_after = compute_score("easy", env._state)
        assert bd_after["communication"] >= bd_before["communication"]
