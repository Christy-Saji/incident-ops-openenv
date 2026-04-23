"""Typed models and action definitions for the incident triage environment."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


VALID_ACTIONS = [
    "acknowledge_incident",
    "inspect_auth_logs",
    "inspect_db_metrics",
    "inspect_deploy_history",
    "rollback_auth_deploy",
    "restart_auth_service",
    "scale_db_cluster",
    "flush_cache",
    "shift_traffic_canary",
    "post_status_update",
    "resolve_incident",
    "no_op",
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
