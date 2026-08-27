"""Smoke tests for all GRPO fixes. Run from repo root: python scripts/smoke_test.py"""

import ast
import inspect
import io
import json
import sys
from pathlib import Path

# Running `python scripts/smoke_test.py` puts scripts/ on sys.path, not the repo
# root, so the project packages are not importable without this. Same bootstrap
# as scripts/evaluate.py.
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    # Tier 1 Phase B: qstar_reward_func replaced the 9 hand-tuned signals as
    # the primary training stack (format + qstar + diversity). See
    # LEGACY_REWARD_FUNCTIONS for the retired 7.
    "ALL_REWARD_FUNCTIONS is the Tier-1 Q* stack (format+qstar+diversity)",
    len(ALL_REWARD_FUNCTIONS) == 3,
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


# ── 4. GRPO eps-greedy dataset excludes held-out tasks ───────────────────────
print("4. generate_grpo_dataset held-out tasks")
try:
    from training.dataset import DEFAULT_EVAL_TASKS, generate_grpo_dataset

    ds = generate_grpo_dataset(n_states=40, seed=99)
    seen_tasks = set()
    for sample in ds:
        try:
            state = json.loads(sample["prompt"][-1]["content"])
            seen_tasks.add(state.get("task"))
        except Exception:
            pass

    leaked = seen_tasks & set(DEFAULT_EVAL_TASKS)
    check(
        "No held-out task appears in the GRPO dataset",
        not leaked,
        f"leaked={leaked}",
    )
except ImportError:
    print("  SKIPPED (datasets package not installed locally -- runs in Colab)")


# ── 5. colab_training.ipynb is unified onto the shared pipeline ───────────────
print("5. colab_training.ipynb unified onto training.pipeline")
with open("colab_training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
# The notebook now runs the SAME code path as `python train.py` (training.pipeline.run
# over config/train.yaml) instead of a hand-rolled SFT/GRPO cell, so Colab and the AMD
# run cannot drift apart. Sync is guaranteed by delegation, not by matching inline values.
check("trains via shared pipeline",
      "from training.pipeline import run" in all_src and "run(cfg)" in all_src)
check("loads config from YAML", "TrainConfig.from_yaml" in all_src)
check("selects a hardware profile", "HARDWARE_PROFILE" in all_src)
# The bespoke GRPOConfig and its divergent hyperparameters are gone — the pipeline
# owns them now via config/train.yaml.
check("no hand-rolled GRPOConfig", "GRPOConfig(" not in all_src)
check("no divergent temperature=1.1", "temperature                 = 1.1" not in all_src)
check("no divergent beta=0.15", "beta                     = 0.15" not in all_src)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"FAILED ({len(errors)} checks):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All checks passed.")
