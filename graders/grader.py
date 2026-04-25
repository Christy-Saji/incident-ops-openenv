"""Task-specific scoring logic for the incident triage benchmark."""

from typing import Dict, Tuple

from tasks.task_config import TASK_CONFIGS


TASK_WEIGHTS = {
    "easy": {
        "diagnosis": 0.20,
        "mitigation": 0.35,
        "recovery": 0.25,
        "communication": 0.10,
        "efficiency": 0.10,
    },
    "medium": {
        "diagnosis": 0.20,
        "mitigation": 0.30,
        "recovery": 0.25,
        "communication": 0.15,
        "efficiency": 0.10,
    },
    "hard": {
        "diagnosis": 0.25,
        "mitigation": 0.25,
        "recovery": 0.25,
        "communication": 0.15,
        "efficiency": 0.10,
    },
    "network": {
        "diagnosis": 0.25,
        "mitigation": 0.30,
        "recovery": 0.25,
        "communication": 0.15,
        "efficiency": 0.05,
    },
    "memory_leak": {
        "diagnosis": 0.25,
        "mitigation": 0.30,
        "recovery": 0.25,
        "communication": 0.10,
        "efficiency": 0.10,
    },
    "disk_full": {
        "diagnosis": 0.20,
        "mitigation": 0.35,
        "recovery": 0.25,
        "communication": 0.10,
        "efficiency": 0.10,
    },
}


def _recovery_score(state: Dict) -> float:
    services = state["service_status"]
    metrics = state["metrics"]

    service_points = sum(1 for status in services.values() if status == "running")
    service_score = service_points / len(services)

    latency_score = max(0.0, min(1.0, 1 - (metrics["latency_ms"] - 120) / 680))
    error_score = max(0.0, min(1.0, 1 - metrics["error_rate"] / 25))
    cpu_score = max(0.0, min(1.0, 1 - max(metrics["cpu_usage"] - 60, 0) / 40))
    memory_score = max(0.0, min(1.0, 1 - max(metrics["memory_usage"] - 60, 0) / 40))

    return round(
        min(
            1.0,
            (0.4 * service_score)
            + (0.25 * latency_score)
            + (0.2 * error_score)
            + (0.1 * cpu_score)
            + (0.05 * memory_score),
        ),
        4,
    )


def _communication_score(state: Dict) -> float:
    """Score communication quality by log richness, not just presence."""
    comm_log = state.get("communication_log", [])
    if not comm_log:
        return 0.0
    # Full credit for 2+ meaningful entries, partial for 1
    volume_score = min(1.0, len(comm_log) / 2)
    # Richness heuristic: longer messages indicate more meaningful content
    avg_length = sum(len(m) for m in comm_log) / max(len(comm_log), 1)
    richness_score = min(1.0, avg_length / 60)
    return round(0.6 * volume_score + 0.4 * richness_score, 4)


def compute_breakdown(task_name: str, state: Dict) -> Dict[str, float]:
    config = TASK_CONFIGS[task_name]
    max_steps = config["max_steps"]

    actions_taken = state.get("actions_taken", [])
    diagnosis_done = sum(
        1 for action in config["required_diagnostics"] if action in actions_taken
    )
    mitigation_done = sum(
        1 for action in config["required_mitigations"] if action in actions_taken
    )
    good_followups_done = sum(
        1 for action in config["good_followups"] if action in actions_taken
    )

    diagnosis = diagnosis_done / max(len(config["required_diagnostics"]), 1)
    mitigation = mitigation_done / max(len(config["required_mitigations"]), 1)
    recovery = _recovery_score(state)

    # Improved communication: richness-aware rather than raw count
    communication = _communication_score(state)

    step_count = state.get("step_count", 0)
    efficiency = max(0.0, min(1.0, 1 - (step_count / max_steps)))

    # Halve efficiency if the incident was never resolved
    if not state["resolved"]:
        efficiency *= 0.5

    # Harmful-action-rate penalty (ratio-based)
    harm_rate = state.get("harmful_action_count", 0) / max(step_count, 1)
    harm_penalty = min(0.20, harm_rate * 0.25)

    return {
        "diagnosis": round(diagnosis, 4),
        "mitigation": round(mitigation, 4),
        "recovery": round(recovery, 4),
        "communication": round(communication, 4),
        "efficiency": round(efficiency, 4),
        "_harm_penalty": round(harm_penalty, 4),
    }


def compute_score(task_name: str, state: Dict) -> Tuple[float, Dict[str, float]]:
    breakdown = compute_breakdown(task_name, state)
    weights = TASK_WEIGHTS.get(task_name, TASK_WEIGHTS["hard"])

    # Extract internal penalty key before computing public breakdown
    harm_penalty = breakdown.pop("_harm_penalty", 0.0)

    score = sum(breakdown[key] * weights.get(key, 0) for key in breakdown)
    score -= harm_penalty
    score -= min(0.15, state.get("harmful_action_count", 0) * 0.03)
    score = max(0.0, min(1.0, score))

    return round(score, 2), breakdown
