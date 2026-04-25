"""Quick preflight smoke test — runs without GPU."""
import sys
sys.path.insert(0, ".")

from env.environment import DevOpsEnv
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS

print("=== Smoke-testing all 6 tasks ===")
for task in TASK_CONFIGS:
    env = DevOpsEnv(task=task)
    obs = env.reset()
    _, r, done, info = env.step("acknowledge_incident")
    score, breakdown = compute_score(task, env._state)
    print(f"  [{task:12s}] reward={r:+.2f}  score={score:.2f}")

print()
print("=== Differentiated inspect actions ===")
# network task: auth logs should say 'not the root cause'
env = DevOpsEnv(task="network")
env.reset()
env.step("inspect_auth_logs")
auth_findings = list(env._state["known_findings"])

env2 = DevOpsEnv(task="network")
env2.reset()
env2.step("inspect_network_topology")
net_findings = list(env2._state["known_findings"])

print(f"  inspect_auth_logs      : {auth_findings}")
print(f"  inspect_network_topology: {net_findings}")
assert auth_findings != net_findings, "FAIL: same hints returned!"
assert any("not the root cause" in f or "not involved" in f for f in auth_findings), \
    "FAIL: auth_logs on network task should return 'not root cause' hint"
print("  PASS: differentiated hints working correctly")

# disk_full task: auth logs should say not relevant
env3 = DevOpsEnv(task="disk_full")
env3.reset()
env3.step("inspect_disk_usage")
disk_findings = list(env3._state["known_findings"])
print(f"  inspect_disk_usage on disk_full: {disk_findings}")
assert any("logrotate" in f or "98" in f for f in disk_findings), \
    "FAIL: disk_full task did not return disk-specific hints"
print("  PASS: disk_full task returns disk-specific hints")

print()
print("=== Stochastic mode ===")
env_s = DevOpsEnv(task="easy", stochastic=True)
env_s.reset()
env_s.step("rollback_auth_deploy")
print(f"  stochastic metrics: {env_s._state['metrics']}")
print("  PASS: stochastic mode runs without error")

print()
print("=== Partial observability ===")
env_p = DevOpsEnv(task="hard", partial_obs=True)
obs_p = env_p.reset()
assert obs_p["known_findings"] == ["[hidden — partial observability active]"], \
    "FAIL: partial obs not working"
print("  PASS: known_findings hidden in partial_obs mode")

print()
print("=== Reward functions (no GPU) ===")
import json
from train import (
    format_reward_func, anti_cheat_reward_func,
    task_alignment_reward_func, generate_prompts, generate_sft_dataset,
)
from env.models import VALID_ACTIONS

assert format_reward_func([], [[{"content": "no_op"}]]) == [1.0]
assert format_reward_func([], [[{"content": "banana"}]]) == [0.0]
assert anti_cheat_reward_func([], [[{"content": "no_op"}]]) == [-0.2]
assert anti_cheat_reward_func([], [[{"content": "rollback_auth_deploy"}]]) == [0.1]

env_ta = DevOpsEnv(task="easy")
state_ta = env_ta.reset()
prompts_ta = [[{"role": "system", "content": "sys"},
               {"role": "user",   "content": json.dumps(state_ta)}]]
r_ta = task_alignment_reward_func(prompts_ta, [[{"content": "inspect_deploy_history"}]])
assert r_ta == [0.3], f"Expected [0.3], got {r_ta}"

ds = generate_prompts(per_task_n=2, mid_episode_n=2)
tasks_seen = {json.loads(item["prompt"][-1]["content"])["task"] for item in ds}
assert tasks_seen == {"easy", "medium", "hard", "network", "memory_leak", "disk_full"}, \
    f"Missing tasks: {tasks_seen}"
print(f"  generate_prompts: {len(ds)} prompts across {len(tasks_seen)} tasks  PASS")

sft_ds = generate_sft_dataset()
print(f"  generate_sft_dataset: {len(sft_ds)} (state, action) pairs  PASS")

print()
print("=" * 50)
print("ALL PREFLIGHT TESTS PASSED")
print("=" * 50)
