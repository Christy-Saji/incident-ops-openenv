"""FastAPI server — session-isolated with /episode, /demo, /leaderboard, partial obs."""

import uuid
import copy
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env.environment import DevOpsEnv
from env.models import Action, VALID_ACTIONS
from graders.grader import compute_score
from inference import create_openai_client_from_env, run_episode as run_policy_episode
from tasks.task_config import TASK_CONFIGS

import os, pathlib

app = FastAPI(
    title="DevOps Incident Triage OpenEnv",
    description=(
        "An OpenEnv benchmark for SaaS production incident response. "
        "Agents investigate alerts, apply mitigations, communicate status, "
        "and resolve incidents through a typed step/reset/state interface.\n\n"
        "New in v3: network/memory_leak/disk_full task scenarios, "
        "/episode replay, /demo auto-run, /leaderboard, partial observability."
    ),
    version="3.0",
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

sessions: dict[str, DevOpsEnv] = {}
DEFAULT_SESSION = "default"

# Leaderboard: task -> list of {session_id, score, steps, resolved}
leaderboard: dict[str, list[dict]] = {t: [] for t in TASK_CONFIGS}

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    session_id: Optional[str] = DEFAULT_SESSION
    partial_obs: Optional[bool] = False   # partial observability toggle


class StepRequest(BaseModel):
    name: str
    session_id: Optional[str] = DEFAULT_SESSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> DevOpsEnv:
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call POST /reset first.",
        )
    return env


def _record_leaderboard(session_id: str, env: DevOpsEnv) -> None:
    """Push a finished episode onto the leaderboard for its task."""
    task = env.task
    score, _ = compute_score(task, env._state)
    entry = {
        "session_id": session_id,
        "score": round(score, 4),
        "steps": env._state.get("step_count", 0),
        "resolved": env._state.get("resolved", False),
        "partial_obs": env.partial_obs,
    }
    board = leaderboard.setdefault(task, [])
    board.append(entry)
    # Keep only top-20 per task, sorted by score desc
    leaderboard[task] = sorted(board, key=lambda x: x["score"], reverse=True)[:20]

# ---------------------------------------------------------------------------
# Routes — core
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the interactive web UI."""
    ui_path = pathlib.Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>DevOps Incident Triage OpenEnv v3</h1><p>Visit /docs</p>")


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "active_sessions": len(sessions), "version": "3.0"}


@app.get("/tasks")
def list_tasks() -> dict:
    """List every available task scenario."""
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
    """Reset (or create) a session. Accepts task, session_id, and partial_obs flag."""
    task_name = (request.task if request else None) or "easy"
    session_id = (request.session_id if request else None) or DEFAULT_SESSION
    partial_obs = (request.partial_obs if request else None) or False

    if task_name not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Choose from {list(TASK_CONFIGS)}",
        )

    env = DevOpsEnv(task=task_name, partial_obs=partial_obs)
    observation = env.reset()
    sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": observation,
        "done": False,
        "partial_obs": partial_obs,
    }


@app.post("/step")
def step(request: StepRequest) -> dict:
    """Apply an action to the environment for the given session."""
    session_id = request.session_id or DEFAULT_SESSION
    env = _get_session(session_id)

    try:
        Action(name=request.name)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid action '{request.name}'.")

    observation, reward, done, info = env.step(request.name)

    if done:
        _record_leaderboard(session_id, env)

    return {
        "session_id": session_id,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = Query(default=DEFAULT_SESSION)) -> dict:
    """Return the current observation snapshot."""
    env = _get_session(session_id)
    return env.state()


@app.get("/score")
def score(session_id: str = Query(default=DEFAULT_SESSION)) -> dict:
    """Return the live score breakdown for a session."""
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

# ---------------------------------------------------------------------------
# Routes — new
# ---------------------------------------------------------------------------

@app.get("/episode")
def episode(session_id: str = Query(default=DEFAULT_SESSION)) -> dict:
    """Return the full episode trace — all steps, rewards, final score.

    Use this to replay exactly what the agent did, step by step.
    The trajectory list contains one entry per step with:
      step, action, reward, done, observation snapshot.
    """
    env = _get_session(session_id)
    return {
        "session_id": session_id,
        **env.episode(),
    }


@app.get("/demo")
def demo(task: str = Query(default="easy"), partial_obs: bool = Query(default=False)) -> dict:
    """Run a real policy rollout and return the full trajectory as JSON.

    When API credentials are configured, this uses the live LLM policy.
    Otherwise it falls back to the deterministic heuristic baseline and labels
    the output accordingly.
    """
    if task not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Choose from {list(TASK_CONFIGS)}",
        )

    session_id = f"demo_{task}_{uuid.uuid4().hex[:6]}"
    client, model_name = create_openai_client_from_env()
    result = run_policy_episode(
        task,
        client,
        model_name,
        partial_obs=partial_obs,
        stochastic=True,
    )

    env = result.pop("env")
    sessions[session_id] = env
    _record_leaderboard(session_id, env)

    return {
        "session_id": session_id,
        "task": task,
        "partial_obs": partial_obs,
        "stochastic": result["stochastic"],
        "policy": result["policy"],
        "policy_label": result["policy_label"],
        "total_reward": round(sum(float(value) for value in result["rewards"]), 4),
        "final_score": result["score"],
        "score_breakdown": result["score_breakdown"],
        "resolved": result["resolved"],
        "steps": result["steps"],
    }


@app.get("/leaderboard")
def get_leaderboard(task: str = Query(default=None)) -> dict:
    """Return the best scores per task across all sessions.

    Pass ?task=easy to filter to a single task.
    """
    if task is not None:
        if task not in TASK_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task '{task}'. Choose from {list(TASK_CONFIGS)}",
            )
        return {"task": task, "leaderboard": leaderboard.get(task, [])}

    return {
        "leaderboard": {
            t: entries[:10] for t, entries in leaderboard.items() if entries
        }
    }
