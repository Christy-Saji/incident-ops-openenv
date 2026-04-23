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
        min(1.0, (0.4 * service_score) + (0.25 * latency_score) + (0.2 * error_score) + (0.1 * cpu_score) + (0.05 * memory_score)),
        4,
    )


def compute_breakdown(task_name: str, state: Dict) -> Dict[str, float]:
    config = TASK_CONFIGS[task_name]
    max_steps = config["max_steps"]

    actions_taken = state.get("actions_taken", [])
    diagnosis_done = sum(1 for action in config["required_diagnostics"] if action in actions_taken)
    mitigation_done = sum(1 for action in config["required_mitigations"] if action in actions_taken)
    good_followups_done = sum(1 for action in config["good_followups"] if action in actions_taken)

    diagnosis = diagnosis_done / len(config["required_diagnostics"])
    mitigation = mitigation_done / len(config["required_mitigations"])
    recovery = _recovery_score(state)
    communication = good_followups_done / len(config["good_followups"])
    efficiency = max(0.0, min(1.0, 1 - (state.get("step_count", 0) / max_steps)))

    if not state["resolved"]:
        efficiency *= 0.5

    return {
        "diagnosis": round(diagnosis, 4),
        "mitigation": round(mitigation, 4),
        "recovery": round(recovery, 4),
        "communication": round(communication, 4),
        "efficiency": round(efficiency, 4),
    }


def compute_score(task_name: str, state: Dict) -> Tuple[float, Dict[str, float]]:
    breakdown = compute_breakdown(task_name, state)
    weights = TASK_WEIGHTS[task_name]

    score = sum(breakdown[key] * weights[key] for key in breakdown)
    score -= min(0.15, state.get("harmful_action_count", 0) * 0.03)
    score = max(0.0, min(1.0, score))

    return round(score, 2), breakdown
