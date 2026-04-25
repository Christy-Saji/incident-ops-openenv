"""Baseline and LLM inference runner for the incident triage environment."""

import json
import os
import re
import time
from typing import Any

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = Any  # type: ignore[misc,assignment]
    OPENAI_AVAILABLE = False

from env.environment import DevOpsEnv
from env.models import VALID_ACTIONS
from graders.grader import compute_score


SYSTEM_PROMPT = f"""You are the primary on-call SRE for a SaaS platform.
Your job is to investigate production incidents, apply safe mitigations, communicate clearly,
and resolve the incident only when the system is actually stable.

Valid actions (pick EXACTLY one):
{json.dumps(VALID_ACTIONS)}

Guidance:
- inspect first when the root cause is unclear
- avoid noisy actions like repeated restarts or no_op
- use rollback_auth_deploy for bad auth deploys
- use scale_db_cluster and shift_traffic_canary for DB saturation
- use post_status_update before resolve_incident when customer impact is ongoing

Return EXACTLY one action name from the list above.
Do not explain. Do not include markdown. Do not include multiple actions."""


BASELINE_PLANS = {
    "easy": [
        "acknowledge_incident",
        "inspect_deploy_history",
        "rollback_auth_deploy",
        "resolve_incident",
    ],
    "medium": [
        "acknowledge_incident",
        "inspect_db_metrics",
        "scale_db_cluster",
        "shift_traffic_canary",
        "post_status_update",
        "resolve_incident",
    ],
    "hard": [
        "acknowledge_incident",
        "inspect_auth_logs",
        "inspect_deploy_history",
        "inspect_db_metrics",
        "rollback_auth_deploy",
        "scale_db_cluster",
        "shift_traffic_canary",
        "flush_cache",
        "post_status_update",
        "resolve_incident",
    ],
    "network": [
        "acknowledge_incident",
        "inspect_network_topology",
        "inspect_deploy_history",
        "withdraw_bgp_route",
        "shift_traffic_canary",
        "post_status_update",
        "resolve_incident",
    ],
    "memory_leak": [
        "acknowledge_incident",
        "inspect_memory_profile",
        "inspect_deploy_history",
        "rollback_service_deploy",
        "scale_db_cluster",
        "post_status_update",
        "resolve_incident",
    ],
    "disk_full": [
        "acknowledge_incident",
        "inspect_disk_usage",
        "inspect_deploy_history",
        "archive_old_logs",
        "reduce_log_verbosity",
        "post_status_update",
        "resolve_incident",
    ],
}

FALLBACK_ACTIONS = {
    "easy": [
        "inspect_auth_logs",
        "post_status_update",
        "restart_auth_service",
        "inspect_db_metrics",
        "no_op",
    ],
    "medium": [
        "inspect_auth_logs",
        "inspect_deploy_history",
        "flush_cache",
        "restart_auth_service",
        "no_op",
    ],
    "hard": [
        "flush_cache",
        "post_status_update",
        "no_op",
    ],
    "network": [
        "inspect_auth_logs",
        "post_status_update",
        "no_op",
    ],
    "memory_leak": [
        "inspect_auth_logs",
        "post_status_update",
        "no_op",
    ],
    "disk_full": [
        "inspect_auth_logs",
        "post_status_update",
        "no_op",
    ],
}


def simple_policy(state: dict) -> str:
    """Deterministic, reproducible baseline policy."""
    all_actions_taken = state.get("all_actions_taken", [])
    plan = BASELINE_PLANS[state["task"]]
    for action in plan:
        if action not in state["recent_actions"] and action not in all_actions_taken:
            return action

    if state["incident_phase"] == "monitoring" and "resolve_incident" not in all_actions_taken:
        return "resolve_incident"

    for action in FALLBACK_ACTIONS[state["task"]]:
        if action == "no_op":
            return action
        if action not in all_actions_taken:
            return action

    return "no_op"


def extract_action(text: str) -> str | None:
    normalized = text.strip().lower()
    normalized = normalized.replace("`", " ")
    normalized = normalized.replace("\n", " ")

    if normalized in VALID_ACTIONS:
        return normalized

    best_action = None
    best_pos = None
    for valid in VALID_ACTIONS:
        match = re.search(rf"\b{re.escape(valid)}\b", normalized)
        if match:
            if best_pos is None or match.start() < best_pos:
                best_action = valid
                best_pos = match.start()

    return best_action


def extract_response_text(response) -> str:
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content") or ""
                if text_value:
                    parts.append(str(text_value))
            else:
                text_value = getattr(item, "text", None) or getattr(item, "content", None)
                if text_value:
                    parts.append(str(text_value))
        if parts:
            return " ".join(parts)

    for attr in ("reasoning_content", "refusal", "audio"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def format_api_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("error", {}).get("message") or body.get("message")
        if message:
            compact = " ".join(str(message).split())[:80]
            return f"{type(exc).__name__}:{status}:{compact}" if status else f"{type(exc).__name__}:{compact}"
    return type(exc).__name__


def llm_policy(state: dict, client: OpenAI, model: str) -> tuple[str | None, str | None]:
    last_error = None

    for attempt in range(1, 3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(state)},
                ],
                max_tokens=24,
                temperature=0.0,
                response_format={"type": "text"},
                extra_body={"provider": "auto"},
            )
            raw_action = extract_response_text(response).strip()
            action = extract_action(raw_action)
            if action in VALID_ACTIONS:
                return action, None
            compact = " ".join(raw_action.split())[:80] or "empty_response"
            return None, f"llm_invalid_action:{compact}"
        except Exception as exc:
            last_error = format_api_error(exc)
            transient = "APIConnectionError" in last_error or "APIStatusError" in last_error
            if attempt < 2 and transient:
                time.sleep(1.0)
                continue
            return None, f"llm_fallback:{last_error}"

    return None, f"llm_fallback:{last_error or 'unknown_error'}"


def create_openai_client_from_env() -> tuple[OpenAI | None, str | None]:
    if not OPENAI_AVAILABLE:
        return None, None

    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
    )

    if not api_base or not model_name:
        return None, None

    return OpenAI(base_url=api_base, api_key=api_key or "no-key"), model_name


def run_episode(
    task_name: str,
    client: OpenAI | None = None,
    model: str | None = None,
    *,
    partial_obs: bool = False,
    stochastic: bool = False,
) -> dict:
    env = DevOpsEnv(task=task_name, partial_obs=partial_obs, stochastic=stochastic)
    state = env.reset()
    use_llm = client is not None and model is not None

    rewards: list[str] = []
    steps: list[dict] = []
    repeat_streak = 0
    last_action = None
    loop_penalty = 0.0
    policy_name = model if use_llm else "heuristic_baseline"

    print(f"[START] task={task_name} env=devops model={policy_name}")

    for step_num in range(1, env.max_steps + 1):
        state["all_actions_taken"] = list(env._state["actions_taken"])

        action = None
        fallback_reason = None
        action_source = "baseline"
        if use_llm:
            action, fallback_reason = llm_policy(state, client, model)
            if action is not None:
                action_source = "llm"
        if action is None:
            action = simple_policy(state)

        state, reward, done, info = env.step(action)
        rewards.append(f"{reward:.2f}")
        error_msg = fallback_reason or info.get("error", "null")

        if action == last_action:
            repeat_streak += 1
        else:
            repeat_streak = 1
            last_action = action

        step_record = {
            "step": step_num,
            "action": action,
            "action_source": action_source,
            "reward": reward,
            "done": done,
            "score": info.get("score"),
            "breakdown": info.get("breakdown"),
            "error": error_msg,
            "observation": state,
        }
        steps.append(step_record)

        print(
            f"[STEP] step={step_num} action={action} reward={reward:.2f} "
            f"done={str(done).lower()} error={error_msg}"
        )

        if repeat_streak >= 3 and not state.get("resolved", False):
            loop_penalty += 0.03
            print("[STEP] loop_detected=true early_stop=true")
            break

        if done:
            break

    score, breakdown = compute_score(task_name, env._state)
    score = max(0.0, score - loop_penalty)
    success = env._state["resolved"]

    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={score:.2f} rewards={','.join(rewards)}"
    )

    return {
        "task": task_name,
        "policy": "llm" if use_llm else "heuristic_baseline",
        "policy_label": policy_name,
        "partial_obs": partial_obs,
        "stochastic": stochastic,
        "env": env,
        "score": round(score, 4),
        "resolved": success,
        "steps_taken": len(steps),
        "rewards": rewards,
        "score_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        "steps": steps,
        "episode": env.episode(),
    }


def run_task(task_name: str, client: OpenAI | None, model: str | None) -> tuple[float, bool]:
    result = run_episode(task_name, client, model)
    return result["score"], result["resolved"]


if __name__ == "__main__":
    client, model_name = create_openai_client_from_env()

    for task in ["easy", "medium", "hard", "network", "memory_leak", "disk_full"]:
        run_task(task, client, model_name)
