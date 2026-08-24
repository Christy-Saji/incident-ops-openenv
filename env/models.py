"""Typed models and action definitions for the incident triage environment."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

VALID_ACTIONS = [
    # ── Universal ──────────────────────────────────────────────────────────
    "acknowledge_incident",
    "post_status_update",
    "resolve_incident",
    "no_op",

    # ── Diagnostics ────────────────────────────────────────────────────────
    "inspect_auth_logs",
    "inspect_db_metrics",
    "inspect_deploy_history",
    "inspect_network_topology",   # new: BGP / routing layer
    "inspect_memory_profile",     # new: heap / OOM diagnosis
    "inspect_disk_usage",         # new: filesystem saturation

    # ── Mitigations ────────────────────────────────────────────────────────
    "rollback_auth_deploy",
    "rollback_service_deploy",    # new: generic service rollback (non-auth)
    "restart_auth_service",
    "scale_db_cluster",
    "flush_cache",
    "shift_traffic_canary",
    "withdraw_bgp_route",         # new: withdraw leaked BGP advertisement
    "archive_old_logs",           # new: compress & remove old log files
    "reduce_log_verbosity",       # new: dial logging back to INFO/WARN
]


class MetricsSnapshot(BaseModel):
    cpu_usage: int = Field(ge=0, le=100)
    memory_usage: int = Field(ge=0, le=100)
    latency_ms: int = Field(ge=0, le=5000)
    error_rate: int = Field(ge=0, le=100)
    request_rate: int = Field(ge=0, le=10000)


class Observation(BaseModel):
    task: str
    incident_title: str
    customer_impact: str
    incident_phase: str
    active_alerts: List[str]
    service_status: Dict[str, str]
    metrics: MetricsSnapshot
    known_findings: List[str]
    communication_log: List[str]
    recent_actions: List[str]
    available_actions: List[str]
    # Full episode action history.
    #
    # recent_actions is only the last 5 actions. Several consumers need the
    # complete history and previously read a key that the observation never
    # carried, silently getting []:
    #   - scripts/evaluate.py heuristic_policy  (looped acknowledge_incident
    #     forever, which made the eval "baseline" a degenerate policy)
    #   - compare_inference.py forbidden-actions hint (never populated)
    #   - training reward functions replaying episode state from the prompt
    # Exposing it here is what makes all three correct.
    actions_taken: List[str] = Field(default_factory=list)
    # Partial-observability flag — set to True on hard mode
    partial_observability: bool = False


class Action(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_action(cls, value: str) -> str:
        if value not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{value}'. Must be one of {VALID_ACTIONS}")
        return value


class RewardInfo(BaseModel):
    reward: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Optional[str]]
