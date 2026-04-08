"""Validation script for the devops-openenv inference pipeline."""

import argparse
import os
import re
import subprocess
import sys

import yaml


SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "openenv.yaml")
EXPECTED_BASELINE = {
    "easy": 0.94,
    "medium": 0.73,
    "hard": 0.58,
}
START_RE = re.compile(r"^\[START\] task=(?P<task>\w+) env=(?P<env>\w+) model=(?P<model>.+)$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>[\w_]+) reward=(?P<reward>-?\d+\.\d{2}) "
    r"done=(?P<done>true|false) error=(?P<error>.+)$"
)
END_RE = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) score=(?P<score>\d+\.\d{2}) "
    r"rewards=(?P<rewards>[\d\.,-]+)$"
)


def load_spec():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_inference(env_vars):
    result = subprocess.run(
        [sys.executable, "inference.py"],
        env={**os.environ, **env_vars},
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"inference.py failed:\n{result.stderr}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_run(lines, spec):
    tasks = {task["name"] for task in spec["tasks"]}
    actions = {action["name"] for action in spec["actions"]}
    parsed = []
    index = 0

    while index < len(lines):
        start_match = START_RE.match(lines[index])
        if not start_match:
            raise ValueError(f"Expected [START] line, got: {lines[index]}")

        task_name = start_match.group("task")
        if task_name not in tasks:
            raise ValueError(f"Unknown task in output: {task_name}")

        rewards = []
        step_count = 0
        index += 1

        while index < len(lines) and lines[index].startswith("[STEP]"):
            step_match = STEP_RE.match(lines[index])
            if not step_match:
                raise ValueError(f"Malformed [STEP] line: {lines[index]}")

            step_num = int(step_match.group("step"))
            if step_num != step_count + 1:
                raise ValueError(f"Task {task_name} has non-sequential step numbering")

            action = step_match.group("action")
            if action not in actions:
                raise ValueError(f"Task {task_name} emitted invalid action: {action}")

            reward = float(step_match.group("reward"))
            if not -0.25 <= reward <= 1.0:
                raise ValueError(f"Task {task_name} emitted out-of-range reward: {reward}")

            rewards.append(step_match.group("reward"))
            step_count += 1
            index += 1

        if index >= len(lines):
            raise ValueError(f"Task {task_name} is missing its [END] line")

        end_match = END_RE.match(lines[index])
        if not end_match:
            raise ValueError(f"Malformed [END] line: {lines[index]}")

        steps_reported = int(end_match.group("steps"))
        if steps_reported != step_count:
            raise ValueError(
                f"Task {task_name} reported {steps_reported} steps but emitted {step_count}"
            )

        end_rewards = end_match.group("rewards").split(",")
        if end_rewards != rewards:
            raise ValueError(f"Task {task_name} rewards list does not match step rewards")

        score = float(end_match.group("score"))
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Task {task_name} emitted out-of-range score: {score}")

        parsed.append(
            {
                "task": task_name,
                "model": start_match.group("model"),
                "success": end_match.group("success") == "true",
                "steps": steps_reported,
                "score": score,
            }
        )
        index += 1

    seen_tasks = {entry["task"] for entry in parsed}
    if seen_tasks != tasks:
        raise ValueError(f"Expected tasks {sorted(tasks)}, got {sorted(seen_tasks)}")

    return parsed


def print_summary(label, parsed):
    print(f"{label}:")
    for entry in parsed:
        outcome = "RESOLVED" if entry["success"] else "PARTIAL"
        print(
            f"  - {entry['task']}: {outcome}, steps={entry['steps']}, "
            f"score={entry['score']:.2f}, model={entry['model']}"
        )


def require_expected_scores(label, parsed, expected_scores):
    observed = {entry["task"]: entry["score"] for entry in parsed}
    mismatches = []
    for task_name, expected_score in expected_scores.items():
        observed_score = observed.get(task_name)
        if observed_score is None or abs(observed_score - expected_score) > 0.01:
            mismatches.append(f"{task_name}: expected {expected_score:.2f}, got {observed_score!r}")
    if mismatches:
        raise RuntimeError(f"{label} score mismatch: {'; '.join(mismatches)}")


def build_model_env(model_name):
    api_base = os.getenv("API_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    if not api_base or not api_key:
        missing = []
        if not api_base:
            missing.append("API_BASE_URL")
        if not api_key:
            missing.append("OPENAI_API_KEY/API_KEY/HF_TOKEN")
        raise ValueError(f"Missing configuration for external model test: {', '.join(missing)}")

    return {
        "API_BASE_URL": api_base,
        "MODEL_NAME": model_name,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "API_KEY": os.getenv("API_KEY", ""),
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to test against the configured API_BASE_URL.",
    )
    args = parser.parse_args()

    spec = load_spec()
    baseline_lines = run_inference(
        {
            "API_BASE_URL": "",
            "MODEL_NAME": "",
            "OPENAI_API_KEY": "",
            "API_KEY": "",
            "HF_TOKEN": "",
        }
    )
    baseline = parse_run(baseline_lines, spec)
    print_summary("Baseline validation", baseline)
    require_expected_scores("Baseline validation", baseline, EXPECTED_BASELINE)

    model_names = [model.strip() for model in args.models.split(",") if model.strip()]
    for model_name in model_names:
        parsed = parse_run(run_inference(build_model_env(model_name)), spec)
        print_summary(f"Model validation [{model_name}]", parsed)


if __name__ == "__main__":
    main()
