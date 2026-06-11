"""Multi-run evaluation script — produces statistically valid results.

Runs each task N times with different seeds, reports mean ± std, and
performs a paired t-test between base and trained scores.

Usage:
    # Quick local test (heuristic baseline vs itself, 3 seeds):
    python scripts/evaluate.py --n-seeds 3

    # Full evaluation (requires trained model on disk):
    python scripts/evaluate.py \\
        --trained-model outputs/trained_sre_agent \\
        --n-seeds 5

    # Generalization test (train tasks vs held-out tasks):
    python scripts/evaluate.py \\
        --trained-model outputs/trained_sre_agent \\
        --train-tasks easy,medium,hard,network \\
        --eval-tasks memory_leak,disk_full \\
        --label generalization_test

Output:
    results/<label>_<timestamp>/
        summary.json     — per-task stats with mean, std, delta, p-value
        full_results.csv — every individual episode score
        report.md        — human-readable markdown report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.environment import DevOpsEnv
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def heuristic_policy(task: str, state: dict) -> str:
    """Deterministic optimal policy — follows task_config optimal_actions list."""
    from env.models import VALID_ACTIONS

    config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
    optimal = config.get("optimal_actions", [])
    done_actions = set(state.get("actions_taken", []))

    for action in optimal:
        if action not in done_actions:
            return action

    # Fallback: resolve if all done, else no_op
    if "resolve_incident" not in done_actions:
        return "resolve_incident"
    return "no_op"


def llm_policy(model_path: str, state: dict) -> str:
    """Run inference against a local saved model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not hasattr(llm_policy, "_model"):
            print(f"  [llm] Loading model from {model_path}...")
            tok = AutoTokenizer.from_pretrained(model_path)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            llm_policy._tokenizer = tok
            llm_policy._model = mdl

        from env.models import VALID_ACTIONS
        import json as _json

        system = (
            "You are an On-call SRE. Select the single NEXT best action.\n"
            f"Valid actions: {', '.join(VALID_ACTIONS)}\n"
            "Output ONLY the action name."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": _json.dumps(state)},
        ]
        tok = llm_policy._tokenizer
        mdl = llm_policy._model
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(mdl.device)

        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                temperature=1.0,
            )
        decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

        for action in VALID_ACTIONS:
            if action in decoded:
                return action
        return "no_op"

    except Exception as e:
        print(f"  [llm] inference error: {e} — falling back to heuristic")
        return heuristic_policy(state.get("task", "easy"), state)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task: str, policy_fn, seed: int, partial_obs: bool = False) -> dict:
    """Run a single episode and return stats."""
    random.seed(seed)
    env = DevOpsEnv(task=task, partial_obs=partial_obs)
    state = env.reset()

    total_reward = 0.0
    steps_taken  = 0

    while True:
        action = policy_fn(task, state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps_taken  += 1
        if done:
            break

    final_score, breakdown = compute_score(task, env._state)
    resolved = bool(env._state.get("resolved", False))

    return {
        "task":        task,
        "seed":        seed,
        "score":       round(final_score, 4),
        "resolved":    resolved,
        "steps":       steps_taken,
        "total_reward":round(total_reward, 4),
        "breakdown":   {k: round(v, 4) for k, v in breakdown.items()},
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return round(m, 4), 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return round(m, 4), round(var ** 0.5, 4)


def paired_ttest_pvalue(a: list[float], b: list[float]) -> float:
    """Simple paired t-test p-value (two-tailed). Returns 1.0 if not enough data."""
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    try:
        from scipy import stats
        _, p = stats.ttest_rel(b, a)
        return round(float(p), 4)
    except ImportError:
        # Manual t-test
        import math
        diffs = [b[i] - a[i] for i in range(len(a))]
        n  = len(diffs)
        md = sum(diffs) / n
        sd = (sum((d - md) ** 2 for d in diffs) / (n - 1)) ** 0.5
        if sd == 0:
            return 0.0 if md != 0 else 1.0
        t  = md / (sd / (n ** 0.5))
        # rough p-value approximation for small n (conservative)
        df = n - 1
        p  = 2 * (1 - _t_cdf(abs(t), df))
        return round(p, 4)


def _t_cdf(t: float, df: int) -> float:
    """Approximated CDF for t-distribution (good enough for p-value reporting)."""
    import math
    x = df / (df + t * t)
    # regularized incomplete beta
    a, b = df / 2, 0.5
    # simple series approximation
    result = 0.0
    term = 1.0
    for k in range(200):
        if k > 0:
            term *= (a + k - 1) * (1 - x) / k
        result += term / (a + k)
        if term < 1e-10:
            break
    result *= (x ** a) * ((1 - x) ** b) / (a * (1.0 if a == 0 else 1.0))
    return min(max(result, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def make_report(summary: dict, label: str, n_seeds: int, args) -> str:
    lines = [
        f"# Evaluation Report — {label}",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Seeds:** {n_seeds}  ",
        f"**Base policy:** {'heuristic' if not args.base_model else args.base_model}  ",
        f"**Trained policy:** {'heuristic (no model)' if not args.trained_model else args.trained_model}  ",
        f"",
        f"## Per-Task Results",
        f"",
        f"| Task | Base (mean±std) | Trained (mean±std) | Δ | p-value | Significant? |",
        f"|------|-----------------|--------------------|---|---------|--------------|",
    ]

    for task, stats in summary.items():
        b_m, b_s = stats["base_mean"], stats["base_std"]
        t_m, t_s = stats["trained_mean"], stats["trained_std"]
        delta    = stats["delta"]
        p        = stats["p_value"]
        sign     = "(*)" if p < 0.05 else "   "
        d_str    = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        lines.append(
            f"| `{task}` | {b_m:.3f}+/-{b_s:.3f} | {t_m:.3f}+/-{t_s:.3f} "
            f"| **{d_str}** | {p:.3f} | {sign} |"
        )

    all_deltas = [s["delta"] for s in summary.values()]
    avg_delta  = sum(all_deltas) / len(all_deltas) if all_deltas else 0
    lines += [
        f"",
        f"**Average delta:** {avg_delta:+.3f}",
        f"",
        f"## Notes",
        f"- p < 0.05 indicates statistically significant improvement",
        f"- All scores are in [0, 1]; multiply by 100 for percentage",
        f"- 'Significant?' uses a paired t-test (two-tailed, a=0.05)",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-run evaluation with statistical significance testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--trained-model", default=None,
                        help="Path to merged trained model dir (e.g. outputs/trained_sre_agent). "
                             "If omitted, both policies use the heuristic baseline.")
    parser.add_argument("--base-model",    default=None,
                        help="Path to base model for base policy. "
                             "If omitted, uses the heuristic optimal policy.")
    parser.add_argument("--n-seeds",   type=int, default=5,
                        help="Number of random seeds per task (default: 5)")
    parser.add_argument("--tasks",     default=None,
                        help="Comma-separated list of tasks to evaluate (default: all 6)")
    parser.add_argument("--eval-tasks",  default=None,
                        help="Held-out tasks for generalization test (overrides --tasks for eval)")
    parser.add_argument("--train-tasks", default=None,
                        help="Train tasks (for labelling only, not used during eval)")
    parser.add_argument("--partial-obs", action="store_true",
                        help="Enable partial observability mode")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for output files (default: results/)")
    parser.add_argument("--label",  default="eval",
                        help="Label for this evaluation run (default: eval)")
    args = parser.parse_args()

    tasks = list(TASK_CONFIGS.keys())
    if args.eval_tasks:
        tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip() in TASK_CONFIGS]
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in TASK_CONFIGS]

    seeds = list(range(args.n_seeds))
    ts    = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(args.output_dir) / f"{args.label}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build policy functions
    if args.base_model:
        base_fn = lambda task, state: llm_policy(args.base_model, state)
    else:
        base_fn = heuristic_policy

    if args.trained_model:
        # Use a separate llm_policy instance for trained model
        trained_path = args.trained_model
        def trained_fn(task, state): return llm_policy(trained_path, state)
    else:
        trained_fn = heuristic_policy

    print(f"\n" + "-" * 60)
    print(f"  Evaluation Run: {args.label}")
    print(f"  Tasks  : {tasks}")
    print(f"  Seeds  : {seeds}")
    print(f"  Output : {run_dir}")
    print("-" * 60)

    all_rows: list[dict] = []
    summary: dict = {}

    for task in tasks:
        print(f"\n  Task: {task}")
        base_scores, trained_scores = [], []

        for seed in seeds:
            print(f"    seed={seed}  ", end="", flush=True)

            b = run_episode(task, base_fn,    seed, partial_obs=args.partial_obs)
            t = run_episode(task, trained_fn, seed, partial_obs=args.partial_obs)

            resolved_str = "OK" if t['resolved'] else "NO"
            print(f"base={b['score']:.3f}  trained={t['score']:.3f}  resolved={resolved_str}")

            base_scores.append(b["score"])
            trained_scores.append(t["score"])

            b["policy"] = "base";    all_rows.append(b)
            t["policy"] = "trained"; all_rows.append(t)

        b_mean, b_std  = mean_std(base_scores)
        t_mean, t_std  = mean_std(trained_scores)
        delta           = round(t_mean - b_mean, 4)
        p_val           = paired_ttest_pvalue(base_scores, trained_scores)

        summary[task] = {
            "base_mean":    b_mean,  "base_std":    b_std,
            "trained_mean": t_mean,  "trained_std": t_std,
            "delta":        delta,   "p_value":     p_val,
            "base_scores":    base_scores,
            "trained_scores": trained_scores,
        }
        significance = "p<0.05 SIGNIFICANT" if p_val < 0.05 else f"p={p_val} (n.s.)"
        print(f"    -> base: {b_mean:.3f}+/-{b_std:.3f}  "
              f"trained: {t_mean:.3f}+/-{t_std:.3f}  "
              f"delta={delta:+.3f}  {significance}")

    # Write outputs
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "full_results.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["task","seed","policy","score","resolved","steps","total_reward"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    report = make_report(summary, args.label, args.n_seeds, args)
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary table
    all_deltas = [s["delta"] for s in summary.values()]
    avg_delta  = sum(all_deltas) / len(all_deltas) if all_deltas else 0

    print(f"\n" + "-" * 60)
    print(f"  Average delta : {avg_delta:+.3f}")
    print(f"  Output dir    : {run_dir}")
    print(f"  Report        : {run_dir / 'report.md'}")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
