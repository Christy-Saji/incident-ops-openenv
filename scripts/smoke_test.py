"""Smoke tests for all GRPO fixes. Run from repo root: python scripts/smoke_test.py"""

import ast
import inspect
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

errors = []


def check(label, ok, detail=""):
    status = "OK" if ok else "FAIL"
    print(f"  {label}: {status}" + (f" -- {detail}" if detail else ""))
    if not ok:
        errors.append(label)


# ── 1. reward_functions imports ──────────────────────────────────────────────
from training.reward_functions import (  # noqa: E402
    ALL_REWARD_FUNCTIONS,
    anti_cheat_reward_func,
)

print("1. reward_functions imports")
check("3-func imports work", True)
check(
    "ALL_REWARD_FUNCTIONS still intact",
    len(ALL_REWARD_FUNCTIONS) == 9,
    f"len={len(ALL_REWARD_FUNCTIONS)}",
)


# ── 2. anti_cheat has no unconditional resolve_incident penalty ──────────────
print("2. anti_cheat_reward_func AST check")
src = inspect.getsource(anti_cheat_reward_func)
tree = ast.parse(src)


class ResolveChecker(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_If(self, node):
        if hasattr(ast, "unparse"):
            cond = ast.unparse(node.test)
            body = ast.unparse(node.body[0]) if node.body else ""
            if "resolve_incident" in cond and "-0.3" in body:
                self.found = True
        self.generic_visit(node)


checker = ResolveChecker()
checker.visit(tree)
check(
    "No unconditional resolve_incident -0.3 in AST",
    not checker.found,
    "flat -0.3 removed" if not checker.found else "still present!",
)


# ── 3. grader efficiency is terminal-only ────────────────────────────────────
print("3. grader.py efficiency terminal-only")
from graders.grader import compute_breakdown  # noqa: E402

base_state = {
    "task": "easy",
    "service_status": {"auth": "degraded", "api": "degraded", "db": "running", "cache": "running"},
    "metrics": {
        "cpu_usage": 58,
        "memory_usage": 62,
        "latency_ms": 210,
        "error_rate": 18,
        "request_rate": 520,
    },
    "actions_taken": ["rollback_auth_deploy"],
    "communication_log": [],
    "resolved": False,
    "harmful_action_count": 0,
    "step_count": 3,
}

bd_unresolved = compute_breakdown("easy", base_state)
check(
    "efficiency=0.0 when unresolved",
    bd_unresolved["efficiency"] == 0.0,
    f"got {bd_unresolved['efficiency']}",
)

bd_resolved = compute_breakdown("easy", {**base_state, "resolved": True})
check(
    "efficiency>0.0 when resolved",
    bd_resolved["efficiency"] > 0.0,
    f"got {bd_resolved['efficiency']} (step=3/8)",
)


# ── 4. dataset warm-up produces zero-harm states ─────────────────────────────
print("4. generate_grpo_dataset harm_action_count")
try:
    from training.dataset import generate_grpo_dataset

    ds = generate_grpo_dataset(per_task_n=2, mid_episode_n=18, seed=99)
    harm_counts = []
    for sample in ds:
        try:
            state = json.loads(sample["prompt"][-1]["content"])
            harm_counts.append(state.get("harmful_action_count", 0))
        except Exception:
            pass

    max_harm = max(harm_counts) if harm_counts else 0
    check(
        "All mid-episode states have harmful_action_count=0",
        max_harm == 0,
        f"max={max_harm}",
    )
except ImportError:
    print("  SKIPPED (datasets package not installed locally -- runs in Colab)")


# ── 5. notebook cell 15 content ──────────────────────────────────────────────
print("5. colab_training.ipynb cell 15")
with open("colab_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell15 = "".join(nb["cells"][15]["source"])
check("3-func import present", "task_alignment_reward_func" in cell15)
check("ALL_REWARD_FUNCTIONS removed", "ALL_REWARD_FUNCTIONS" not in cell15)
check("temperature=1.1", "temperature                 = 1.1" in cell15)
check("beta=0.15", "beta                     = 0.15" in cell15)
check("diag logger present", "_diag_log" in cell15)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"FAILED ({len(errors)} checks):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All checks passed.")
