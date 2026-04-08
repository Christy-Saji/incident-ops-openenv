"""Minimal API server for HF Space style deployment."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import DevOpsEnv
from env.models import Action
from tasks.task_config import TASK_CONFIGS


app = FastAPI(title="DevOps Incident Triage OpenEnv")
current_env = DevOpsEnv(task="easy")


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


@app.get("/")
def root() -> dict:
    return {"name": "devops_incident_triage_env", "status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict:
    global current_env
    task_name = (request.task if request else None) or "easy"
    if task_name not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'")
    current_env = DevOpsEnv(task=task_name)
    return {"observation": current_env.reset(), "done": False}


@app.post("/step")
def step(request: Action) -> dict:
    observation, reward, done, info = current_env.step(request.name)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return current_env.state()
