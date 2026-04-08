---
title: Incident Ops OpenEnv
emoji: "🚨"
colorFrom: red
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# DevOps Incident Triage OpenEnv

An OpenEnv benchmark for real-world SaaS incident response. The agent acts like an on-call SRE:
it investigates alerts, inspects evidence, applies mitigations, communicates status, and resolves
the incident only after the system is genuinely stable.

This is intentionally modeled after real production work rather than a toy control problem.

## Why This Environment

Production incident response is a realistic agent task with real operational value:

- alerts are incomplete and sometimes misleading
- agents must gather evidence before acting
- some actions mitigate symptoms while others fix the root cause
- communication matters, not just technical recovery
- partial recovery should score higher than random action spam, but lower than a full resolution

## Tasks

Three deterministic incident scenarios are included:

| Task | Scenario | Expected difficulty |
| --- | --- | --- |
| `easy` | Bad auth deploy causing login failures | Requires rollback after checking deploy history |
| `medium` | DB saturation from traffic spike | Requires metrics inspection plus mitigation sequence |
| `hard` | Cascading outage after auth deploy | Requires diagnosis, multiple mitigations, and incident hygiene |

## Action Space

The agent can choose one of the following actions each step:

- `acknowledge_incident`
- `inspect_auth_logs`
- `inspect_db_metrics`
- `inspect_deploy_history`
- `rollback_auth_deploy`
- `restart_auth_service`
- `scale_db_cluster`
- `flush_cache`
- `shift_traffic_canary`
- `post_status_update`
- `resolve_incident`
- `no_op`

## Observation Space

Each observation includes:

- `task`
- `incident_title`
- `customer_impact`
- `incident_phase`
- `active_alerts`
- `service_status`
- `metrics`
- `known_findings`
- `communication_log`
- `recent_actions`
- `available_actions`

The observation is intentionally partial: the agent sees symptoms immediately, but must inspect logs,
metrics, or deploy history to confirm the root cause.

## Reward Design

Per-step reward is the score delta with explicit penalties for stalling and harmful actions. Step rewards
can be negative, while the underlying task score remains normalized to `[0.0, 1.0]`. The score combines:

- diagnosis quality
- mitigation completion
- recovery quality
- communication quality
- efficiency

The scorer also penalizes clearly poor behavior such as noisy repeated actions, invalid actions,
premature resolution attempts, and excessive idling.

## Typed Models and Interface

The environment exposes:

- typed `Observation`, `Action`, `RewardInfo`, and `StepResult` models in [env/models.py](env/models.py)
- `reset()` in [environment.py](env/environment.py)
- `step(action)` in [environment.py](env/environment.py)
- `state()` in [environment.py](env/environment.py)
- metadata in [openenv.yaml](openenv.yaml)

## Baseline Scores

The included deterministic baseline is intentionally reproducible but not perfect:

| Task | Baseline score | Notes |
| --- | --- | --- |
| `easy` | `0.94` | Correct rollback and clean resolution |
| `medium` | `0.73` | Mitigates the incident but then takes low-value follow-up actions and never resolves cleanly |
| `hard` | `0.58` | Handles the main mitigations but leaves recovery quality on the table |

These are validated by [test_inference.py](test_inference.py).

## Local Setup

```bash
pip install -r requirements.txt
python inference.py
python test_inference.py
```

## Run Against OpenAI-Compatible Models

PowerShell:

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:OPENAI_API_KEY="sk-..."
python inference.py
```

You can also run a model matrix:

```powershell
python test_inference.py --models gpt-4o-mini,gpt-4.1-mini,gpt-4.1
```

For Hugging Face Inference Providers with OpenAI-compatible routing:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN="hf_..."
python inference.py
```

## API Server

The repo includes a small FastAPI server in [app.py](app.py) for container deployment.

Routes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

Example local run:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker

Build:

```bash
docker build -t devops-openenv .
```

Run the server:

```bash
docker run --rm -p 7860:7860 devops-openenv
```

Then verify it responds:

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
```

## Project Structure

```text
app.py               # FastAPI server for Space-style deployment
env/
  environment.py     # OpenEnv environment implementation
  models.py          # Typed observation/action/reward models
graders/
  grader.py          # Task-specific scoring logic
tasks/
  task_config.py     # Realistic incident scenarios and task metadata
inference.py         # Baseline + OpenAI-client runner
test_inference.py    # Structured output validator and baseline score checker
openenv.yaml         # Environment metadata
Dockerfile
requirements.txt
```
