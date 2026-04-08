"""Baseline and LLM inference runner for the incident triage environment."""

import json
import os
import re
import time

from openai import OpenAI

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
        "resolve_incident",
    ],
    "hard": [
        "acknowledge_incident",
        "inspect_auth_logs",
        "inspect_deploy_history",
        "rollback_auth_deploy",
        "scale_db_cluster",
        "shift_traffic_canary",
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
        "restart_auth_service",
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

    for valid in VALID_ACTIONS:
        if re.search(rf"\b{re.escape(valid)}\b", normalized):
            return valid

    return None


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


def run_task(task_name: str, client: OpenAI | None, model: str | None) -> tuple[float, bool]:
    env = DevOpsEnv(task=task_name)
    state = env.reset()
    use_llm = client is not None and model is not None

    print(f"[START] task={task_name} env=devops model={model if use_llm else 'baseline'}")

    rewards: list[str] = []
    success = False
    score = 0.0

    for step_num in range(1, env.max_steps + 1):
        state["all_actions_taken"] = list(env._state["actions_taken"])

        action = None
        fallback_reason = None
        if use_llm:
            action, fallback_reason = llm_policy(state, client, model)
        if action is None:
            action = simple_policy(state)

        state, reward, done, info = env.step(action)
        rewards.append(f"{reward:.2f}")
        error_msg = fallback_reason or info.get("error", "null")

        print(
            f"[STEP] step={step_num} action={action} reward={reward:.2f} "
            f"done={str(done).lower()} error={error_msg}"
        )

        if done:
            break

    score, _ = compute_score(task_name, env._state)
    success = env._state["resolved"]

    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={score:.2f} rewards={','.join(rewards)}"
    )

    return score, success


if __name__ == "__main__":
    api_base = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
    )

    client = None
    if api_base and model_name:
        client = OpenAI(base_url=api_base, api_key=api_key or "no-key")

    for task in ["easy", "medium", "hard"]:
        run_task(task, client, model_name)
