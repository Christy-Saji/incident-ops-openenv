"""Unit tests for the DevOpsEnv environment."""

import pytest

from env.environment import DevOpsEnv
from tasks.task_config import TASK_CONFIGS


class TestEnvReset:
    def test_reset_returns_dict(self):
        env = DevOpsEnv(task="easy")
        obs = env.reset()
        assert isinstance(obs, dict)

    def test_reset_clears_state(self):
        env = DevOpsEnv(task="easy")
        env.step("inspect_deploy_history")
        env.reset()
        assert env.current_step == 0
        assert env._state["actions_taken"] == []
        assert env._state["resolved"] is False

    @pytest.mark.parametrize("task", list(TASK_CONFIGS.keys()))
    def test_all_tasks_reset(self, task):
        env = DevOpsEnv(task=task)
        obs = env.reset()
        assert obs["task"] == task
        assert "incident_title" in obs
        assert "available_actions" in obs


class TestEnvStep:
    def test_step_returns_tuple(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        obs, reward, done, info = env.step("inspect_deploy_history")
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_valid_action_increments_step(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        env.step("inspect_deploy_history")
        assert env.current_step == 1

    def test_invalid_action_penalised(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        _, reward, _, info = env.step("not_a_real_action")
        assert reward < 0
        assert info.get("error") == "invalid_action"

    def test_reward_clamped(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        for _ in range(5):
            _, reward, done, _ = env.step("no_op")
            assert -0.25 <= reward <= 1.0
            if done:
                break

    def test_no_op_increments_harmful_count(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        env.step("no_op")
        assert env._state["harmful_action_count"] >= 1

    def test_resolve_before_stable_fails(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        _, _, _, info = env.step("resolve_incident")
        assert info.get("error") == "incident_not_stable"
        assert not env._state["resolved"]

    def test_optimal_sequence_resolves(self):
        """Executing the optimal action sequence should resolve the incident."""
        for task, config in TASK_CONFIGS.items():
            env = DevOpsEnv(task=task)
            env.reset()
            for action in config["optimal_actions"]:
                _, _, done, _ = env.step(action)
                if done:
                    break
            assert env._state["resolved"], f"Task '{task}' should resolve with optimal actions"

    def test_done_when_max_steps_reached(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        done = False
        for _ in range(env.max_steps + 5):
            _, _, done, _ = env.step("no_op")
            if done:
                break
        assert done


class TestMitigationGating:
    """env v2: a mitigation applied before its prerequisite diagnostic is a no-op."""

    def test_blind_mitigation_is_noop(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        _, reward, _, info = env.step("rollback_auth_deploy")
        assert info.get("error") == "mitigation_without_diagnosis"
        assert env._state["service_status"]["auth"] == "degraded"
        assert env._state["harmful_action_count"] == 1
        assert reward < 0

    def test_blind_mitigation_not_in_effective_mitigations(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        env.step("rollback_auth_deploy")
        assert "rollback_auth_deploy" not in env._state["effective_mitigations"]
        assert "rollback_auth_deploy" in env._state["actions_taken"]

    def test_diagnose_then_mitigate_is_effective(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        env.step("inspect_deploy_history")
        env.step("rollback_auth_deploy")
        assert "rollback_auth_deploy" in env._state["effective_mitigations"]
        assert env._state["service_status"]["auth"] == "running"

    def test_blind_mitigation_blocks_resolution(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        env.step("acknowledge_incident")
        env.step("rollback_auth_deploy")  # blind -- blocked
        _, _, _, info = env.step("resolve_incident")
        assert info.get("error") == "incident_not_stable"
        assert not env._state["resolved"]

    def test_shift_traffic_canary_requires_any_diagnostic(self):
        env = DevOpsEnv(task="medium")
        env.reset()
        _, _, _, info = env.step("shift_traffic_canary")
        assert info.get("error") == "mitigation_without_diagnosis"

        env2 = DevOpsEnv(task="medium")
        env2.reset()
        env2.step("inspect_db_metrics")
        env2.step("shift_traffic_canary")
        assert "shift_traffic_canary" in env2._state["effective_mitigations"]


class TestPartialObservability:
    def test_partial_obs_hides_findings(self):
        env = DevOpsEnv(task="easy", partial_obs=True)
        env.reset()
        env.step("inspect_deploy_history")
        obs = env.state()
        assert obs["known_findings"] == ["[hidden — partial observability active]"]

    def test_full_obs_shows_findings(self):
        env = DevOpsEnv(task="easy", partial_obs=False)
        env.reset()
        env.step("inspect_deploy_history")
        obs = env.state()
        assert len(obs["known_findings"]) > 0
        assert "[hidden" not in obs["known_findings"][0]


class TestEpisodeTrace:
    def test_episode_returns_trajectory(self):
        env = DevOpsEnv(task="easy")
        env.reset()
        env.step("inspect_deploy_history")
        episode = env.episode()
        assert "trajectory" in episode
        assert len(episode["trajectory"]) == 1
        assert episode["trajectory"][0]["action"] == "inspect_deploy_history"
