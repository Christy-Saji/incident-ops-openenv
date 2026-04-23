"""FastAPI server for HF Space deployment — session-isolated with /score and /tasks."""

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from env.environment import DevOpsEnv
from env.models import Action
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS


app = FastAPI(
    title="DevOps Incident Triage OpenEnv",
    description=(
        "An OpenEnv benchmark for SaaS production incident response. "
        "Agents investigate alerts, apply mitigations, communicate status, "
        "and resolve incidents through a typed step/reset/state interface."
    ),
    version="2.0",
)

# Session-isolated environments — keyed by session_id string.
# This prevents concurrent users on the HF Space from corrupting each other's state.
sessions: dict[str, DevOpsEnv] = {}

DEFAULT_SESSION = "default"


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    session_id: Optional[str] = DEFAULT_SESSION


class StepRequest(BaseModel):
    name: str                                   # action name (mirrors Action model)
    session_id: Optional[str] = DEFAULT_SESSION


def _get_session(session_id: str) -> DevOpsEnv:
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset first.",
        )
    return env


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {"name": "devops_incident_triage_env", "status": "ok", "version": "2.0"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "active_sessions": len(sessions)}


@app.get("/tasks")
def list_tasks() -> dict:
    """List all available task names and their descriptions."""
    return {
        "tasks": [
            {
                "name": name,
                "description": cfg.get("description", ""),
                "title": cfg.get("title", ""),
                "max_steps": cfg.get("max_steps", 0),
                "expected_steps": len(cfg.get("optimal_actions", [])),
            }
            for name, cfg in TASK_CONFIGS.items()
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict:
    """Reset (or create) a session with the chosen task."""
    task_name = (request.task if request else None) or "easy"
    session_id = (request.session_id if request else None) or DEFAULT_SESSION

    if task_name not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Choose from {list(TASK_CONFIGS)}",
        )

    env = DevOpsEnv(task=task_name)
    observation = env.reset()
    sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": observation,
        "done": False,
    }


@app.post("/step")
def step(request: StepRequest) -> dict:
    """Apply an action to the environment for the given session."""
    session_id = request.session_id or DEFAULT_SESSION
    env = _get_session(session_id)

    # Validate the action via the Action model (raises ValueError for invalid)
    try:
        Action(name=request.name)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.name}'.",
        )

    observation, reward, done, info = env.step(request.name)

    return {
        "session_id": session_id,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = Query(default=DEFAULT_SESSION)) -> dict:
    """Return the current observation snapshot for the given session."""
    env = _get_session(session_id)
    return env.state()


@app.get("/score")
def score(session_id: str = Query(default=DEFAULT_SESSION)) -> dict:
    """Return the current score breakdown for the given session.
    Useful for judges and demos to inspect the scoring live.
    """
    env = _get_session(session_id)
    total_score, breakdown = compute_score(env.task, env._state)
    return {
        "session_id": session_id,
        "task": env.task,
        "score": round(total_score, 4),
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        "resolved": env._state.get("resolved", False),
        "step_count": env._state.get("step_count", 0),
        "max_steps": env.max_steps,
    }
