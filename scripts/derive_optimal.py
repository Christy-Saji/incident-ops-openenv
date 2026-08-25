"""Regenerate tasks/task_config.py's `optimal_actions` from the Q* solver.

These sequences used to be hand-written and were measurably wrong -- deleting
`acknowledge_incident` raised the final score on 5 of 6 tasks (see
docs/prompts/tier1.md). Do not hand-edit `optimal_actions` again: rerun this
script after any change to env dynamics, MITIGATION_PREREQS, or task config.

Usage:
    python scripts/derive_optimal.py            # print the derived sequences
    python scripts/derive_optimal.py --write     # rewrite tasks/task_config.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import DevOpsEnv  # noqa: E402
from tasks.task_config import TASK_CONFIGS  # noqa: E402
from training.qstar import solve_cached  # noqa: E402

TASK_CONFIG_PATH = Path(__file__).resolve().parent.parent / "tasks" / "task_config.py"

HEADER = (
    "        # Derived by scripts/derive_optimal.py (Q* argmax trajectory) --\n"
    "        # do not hand-edit; rerun the script after any env/task change.\n"
)


def derive_all() -> dict[str, list[str]]:
    sequences = {}
    for task in TASK_CONFIGS:
        table = solve_cached(task)
        trajectory = table.optimal_trajectory()

        # Sanity check: the derived sequence must actually resolve the task.
        env = DevOpsEnv(task=task)
        env.reset()
        for action in trajectory:
            _, _, done, _ = env.step(action)
            if done:
                break
        if not env._state["resolved"]:
            raise RuntimeError(
                f"Derived trajectory for '{task}' does not resolve: {trajectory}"
            )
        sequences[task] = trajectory
    return sequences


def _format_sequence(actions: list[str]) -> str:
    lines = [HEADER + '        "optimal_actions": [']
    for action in actions:
        lines.append(f'            "{action}",')
    lines.append("        ],")
    return "\n".join(lines)


def write_task_config(sequences: dict[str, list[str]]) -> None:
    text = TASK_CONFIG_PATH.read_text(encoding="utf-8")

    for task, actions in sequences.items():
        # Match this task's existing optimal_actions block (with or without
        # a prior derive_optimal.py header comment), scoped to avoid bleeding
        # into a neighbouring task's block.
        pattern = re.compile(
            r'( *)(?:# Derived by scripts/derive_optimal\.py.*\n *#.*\n)?'
            r' *"optimal_actions": \[\n(?:.*\n)*?\1\],',
        )
        block_start = text.index(f'"{task}": {{')
        block_end = text.index("\n    },", block_start)
        segment = text[block_start:block_end]

        new_block = _format_sequence(actions)
        new_segment, n = pattern.subn(new_block, segment, count=1)
        if n != 1:
            raise RuntimeError(f"Could not locate optimal_actions block for '{task}'")

        text = text[:block_start] + new_segment + text[block_end:]

    TASK_CONFIG_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    sequences = derive_all()
    for task, actions in sequences.items():
        print(f"{task:12s} {actions}")

    if args.write:
        write_task_config(sequences)
        print(f"\nWrote optimal_actions for {len(sequences)} tasks to {TASK_CONFIG_PATH}")
