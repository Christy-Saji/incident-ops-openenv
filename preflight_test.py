"""
Pre-flight test — run this LOCALLY (no GPU needed) before the hackathon.
Tests everything that doesn't require Unsloth or a GPU:
  1. Environment reset / step / grader
  2. All reward functions (no model, dummy completions)
  3. Curriculum dataset generation
  4. FastAPI server endpoints (spins up in-process)
  5. Baseline inference scores

Run:
    python preflight_test.py
All tests should print OK. Fix any FAIL before going to the hackathon.
"""

import json
import sys
import traceback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True))
    except Exception as exc:
        print(f"{FAIL} {name}")
        print(f"       {exc}")
        traceback.print_exc()
        results.append((name, False))


# ---------------------------------------------------------------------------
# 1. Environment core
# ---------------------------------------------------------------------------

def test_env_reset():
    from env.environment import DevOpsEnv
    for task in ["easy", "medium", "hard"]:
        env = DevOpsEnv(task=task)
        obs = env.reset()
        assert "task" in obs, f"'task' missing from observation for {task}"
        assert obs["task"] == task


def test_env_step_valid():
    from env.environment import DevOpsEnv
    from env.models import VALID_ACTIONS
    env = DevOpsEnv(task="easy")
    env.reset()
    obs, reward, done, info = env.step("acknowledge_incident")
    assert isinstance(reward, float), "reward must be float"
    assert isinstance(done, bool), "done must be bool"
    assert "acknowledge_incident" in env._state["actions_taken"]


def test_env_step_invalid():
    from env.environment import DevOpsEnv
    env = DevOpsEnv(task="easy")
    env.reset()
    _, reward, _, info = env.step("definitely_not_an_action")
    assert reward < 0, "invalid action should give negative reward"
    assert info.get("error") == "invalid_action"


def test_env_full_episode_easy():
    from env.environment import DevOpsEnv
    env = DevOpsEnv(task="easy")
    env.reset()
    optimal = ["acknowledge_incident", "inspect_deploy_history",
               "rollback_auth_deploy", "resolve_incident"]
    for action in optimal:
        _, _, done, _ = env.step(action)
    assert env._state["resolved"], "easy task should resolve with optimal actions"


def test_grader():
    from env.environment import DevOpsEnv
    from graders.grader import compute_score
    for task in ["easy", "medium", "hard"]:
        env = DevOpsEnv(task=task)
        env.reset()
        score, breakdown = compute_score(task, env._state)
        assert 0.0 <= score <= 1.0, f"score out of range for {task}: {score}"
        assert set(breakdown.keys()) == {"diagnosis", "mitigation", "recovery",
                                          "communication", "efficiency"}


def test_grader_safe_on_partial_state():
    """Regression: grader must not crash if actions_taken / step_count are missing."""
    from graders.grader import compute_score
    from tasks.task_config import TASK_CONFIGS
    # Simulate a state that omits internal tracking keys (mimics GRPO edge case)
    minimal_state = {
        "service_status": {"auth": "degraded", "api": "degraded", "db": "running", "cache": "running"},
        "metrics": {"cpu_usage": 58, "memory_usage": 62, "latency_ms": 210,
                    "error_rate": 18, "request_rate": 520},
        "resolved": False,
        "harmful_action_count": 0,
        # intentionally missing: actions_taken, step_count
    }
    score, breakdown = compute_score("easy", minimal_state)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 2. Reward functions (no GPU — uses dummy completions)
# ---------------------------------------------------------------------------

def test_format_reward_func():
    from train import format_reward_func
    from env.models import VALID_ACTIONS
    completions_valid = [[{"content": VALID_ACTIONS[0]}]]
    completions_invalid = [[{"content": "banana"}]]
    assert format_reward_func([], completions_valid) == [1.0]
    assert format_reward_func([], completions_invalid) == [0.0]


def test_anti_cheat_reward_func():
    from train import anti_cheat_reward_func
    assert anti_cheat_reward_func([], [[{"content": "no_op"}]]) == [-0.2]
    assert anti_cheat_reward_func([], [[{"content": "resolve_incident"}]]) == [-0.1]
    assert anti_cheat_reward_func([], [[{"content": "rollback_auth_deploy"}]]) == [0.1]


def test_task_alignment_reward_func():
    from train import task_alignment_reward_func
    from env.environment import DevOpsEnv
    env = DevOpsEnv(task="easy")
    state = env.reset()

    prompts = [[
        {"role": "system", "content": "system"},
        {"role": "user", "content": json.dumps(state)},
    ]]
    # inspect_deploy_history is required_diagnostics for easy
    r = task_alignment_reward_func(prompts, [[{"content": "inspect_deploy_history"}]])
    assert r == [0.3], f"expected [0.3], got {r}"
    # no_op should penalise
    r = task_alignment_reward_func(prompts, [[{"content": "no_op"}]])
    assert r == [-0.2], f"expected [-0.2], got {r}"


def test_step_reward_func_task_aware():
    """step_reward_func must not hardcode easy task."""
    from train import step_reward_func
    from env.environment import DevOpsEnv

    for task in ["easy", "medium", "hard"]:
        env = DevOpsEnv(task=task)
        state = env.reset()
        prompts = [[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": json.dumps(state)},
        ]]
        completions = [[{"content": "acknowledge_incident"}]]
        rewards = step_reward_func(prompts, completions)
        assert len(rewards) == 1 and isinstance(rewards[0], float), \
            f"step_reward_func failed for task={task}"


# ---------------------------------------------------------------------------
# 3. Curriculum dataset
# ---------------------------------------------------------------------------

def test_generate_prompts_curriculum():
    from train import generate_prompts
    ds = generate_prompts(easy_n=5, medium_n=5, hard_n=3, mid_episode_n=2)
    assert len(ds) == 15, f"Expected 15 samples, got {len(ds)}"
    tasks_seen = set()
    for item in ds:
        user_msg = item["prompt"][-1]["content"]
        state = json.loads(user_msg)
        tasks_seen.add(state.get("task"))
    assert tasks_seen == {"easy", "medium", "hard"}, \
        f"Not all tasks represented: {tasks_seen}"


# ---------------------------------------------------------------------------
# 4. FastAPI server (in-process, no uvicorn needed)
# ---------------------------------------------------------------------------

def test_api_server():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200

    # Tasks list
    r = client.get("/tasks")
    assert r.status_code == 200
    task_names = [t["name"] for t in r.json()["tasks"]]
    assert set(task_names) == {"easy", "medium", "hard"}

    # Reset
    r = client.post("/reset", json={"task": "easy", "session_id": "test"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "test"

    # Step
    r = client.post("/step", json={"name": "acknowledge_incident", "session_id": "test"})
    assert r.status_code == 200
    assert "reward" in r.json()

    # Score
    r = client.get("/score?session_id=test")
    assert r.status_code == 200
    assert "breakdown" in r.json()

    # State
    r = client.get("/state?session_id=test")
    assert r.status_code == 200

    # Step on unknown session → 404
    r = client.post("/step", json={"name": "no_op", "session_id": "ghost"})
    assert r.status_code == 404, f"Expected 404 for unknown session, got {r.status_code}"


# ---------------------------------------------------------------------------
# 5. Baseline inference scores
# ---------------------------------------------------------------------------

def test_baseline_scores():
    """Run the deterministic baseline policy and check known scores."""
    from env.environment import DevOpsEnv
    from graders.grader import compute_score
    from inference import simple_policy

    EXPECTED = {"easy": 0.94, "medium": 0.93, "hard": 0.61}

    for task, expected in EXPECTED.items():
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(env.max_steps):
            state["all_actions_taken"] = list(env._state["actions_taken"])
            action = simple_policy(state)
            state, _, done, _ = env.step(action)
            if done:
                break
        score, _ = compute_score(task, env._state)
        assert abs(score - expected) <= 0.02, \
            f"Baseline score for {task}: expected ~{expected}, got {score:.2f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  DevOps OpenEnv — Pre-flight Test Suite")
    print("=" * 55)

    check("1a. env reset (all tasks)", test_env_reset)
    check("1b. env step — valid action", test_env_step_valid)
    check("1c. env step — invalid action", test_env_step_invalid)
    check("1d. env full episode — easy optimal", test_env_full_episode_easy)
    check("1e. grader — score range all tasks", test_grader)
    check("1f. grader — safe on partial state (GRPO fix)", test_grader_safe_on_partial_state)

    check("2a. format_reward_func", test_format_reward_func)
    check("2b. anti_cheat_reward_func", test_anti_cheat_reward_func)
    check("2c. task_alignment_reward_func", test_task_alignment_reward_func)
    check("2d. step_reward_func — task-aware", test_step_reward_func_task_aware)

    check("3a. generate_prompts — curriculum mix", test_generate_prompts_curriculum)

    check("4a. FastAPI — all endpoints", test_api_server)

    check("5a. baseline scores", test_baseline_scores)

    print()
    print("=" * 55)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"  ALL {total}/{total} TESTS PASSED")
        print("  You are ready for the hackathon.")
    else:
        failed = [(n, ok) for n, ok in results if not ok]
        print(f"  {passed}/{total} passed — {len(failed)} FAILED")
        for name, _ in failed:
            print(f"    • {name}")
        print("\n  Fix failures before the hackathon.")
        sys.exit(1)
    print("=" * 55)
