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

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v3.0-6366f1?style=flat-square)](https://huggingface.co/spaces/chritsysajii/incident-ops-openenv)
[![Tasks](https://img.shields.io/badge/tasks-6-22c55e?style=flat-square)](#tasks)
[![Actions](https://img.shields.io/badge/actions-19-f59e0b?style=flat-square)](#action-space)

An OpenEnv benchmark for real-world SaaS incident response. The agent acts like an on-call SRE:
it investigates alerts, inspects evidence, applies mitigations, communicates status, and resolves
the incident only after the system is genuinely stable.

**Try it instantly:** visit the Space root URL — the interactive web UI lets you step through any
incident in the browser. No code required.

**One-click demo:** `GET /demo?task=network` runs the optimal policy and returns the full
trajectory as JSON. Perfect for judges who want to see a complete solved episode.

---

## Tasks

Six deterministic incident scenarios spanning fundamentally different failure modes:

| Task | Scenario | Root Cause | Expected Steps |
|---|---|---|---|
| `easy` | Auth deploy regression — login failures | Bad rollout → rollback | 4 |
| `medium` | DB saturation from flash-sale traffic spike | Scale + shift traffic | 6 |
| `hard` | Cascading outage after auth deploy | Multi-service retry storm | 9 |
| `network` | BGP route leak — global latency spike | Upstream AS path prepend | 7 |
| `memory_leak` | OOM kills causing pod restart loops | Payment-service heap leak | 7 |
| `disk_full` | Log disk saturation blocking API writes | Log rotation failure | 7 |

---

## Partial Observability

Pass `partial_obs: true` in `/reset` to hide `known_findings` from the agent's observation.
The agent must reason from raw alerts and metrics alone — no accumulated clue list.
This demonstrates sophisticated environment design and meaningfully increases task difficulty.

---

## Quick Start

```bash
# 1. Start an incident
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "network", "session_id": "demo1"}'

# 2. Apply an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"name": "inspect_network_topology", "session_id": "demo1"}'

# 3. See the full episode replay
curl http://localhost:7860/episode?session_id=demo1

# 4. Auto-solve with optimal policy (no session needed)
curl "http://localhost:7860/demo?task=network"

# 5. Check the leaderboard
curl http://localhost:7860/leaderboard
```

---

## Action Space (19 actions)

### Universal
- `acknowledge_incident` — claim ownership
- `post_status_update` — publish customer-facing status
- `resolve_incident` — close the incident when stable
- `no_op` — do nothing (penalized)

### Diagnostics
- `inspect_auth_logs` — auth-service log evidence
- `inspect_db_metrics` — database saturation metrics
- `inspect_deploy_history` — deploy history + runbook
- `inspect_network_topology` — BGP routing tables (new)
- `inspect_memory_profile` — heap/OOM analysis (new)
- `inspect_disk_usage` — filesystem utilisation (new)

### Mitigations
- `rollback_auth_deploy` — revert bad auth release
- `rollback_service_deploy` — revert any non-auth service (new)
- `restart_auth_service` — risky restart (sometimes harmful)
- `scale_db_cluster` — add DB capacity
- `flush_cache` — clear stale session cache
- `shift_traffic_canary` — divert traffic from hot shard
- `withdraw_bgp_route` — fix BGP leak (new)
- `archive_old_logs` — reclaim log disk space (new)
- `reduce_log_verbosity` — stop log volume explosion (new)

---

## Observation Space

| Field | Type | Notes |
|---|---|---|
| `task` | str | Active task name |
| `incident_title` | str | Human-readable incident title |
| `customer_impact` | str | Plain-English impact description |
| `incident_phase` | str | `investigating` / `monitoring` / `resolved` |
| `active_alerts` | list[str] | Current firing alerts |
| `service_status` | dict[str,str] | `running` or `degraded` per service |
| `metrics` | dict | cpu, memory, latency_ms, error_rate, request_rate |
| `known_findings` | list[str] | Accumulated diagnostic clues (hidden in partial-obs mode) |
| `communication_log` | list[str] | Status messages posted |
| `recent_actions` | list[str] | Last 5 actions taken |
| `available_actions` | list[str] | Full action list |
| `partial_observability` | bool | True when known_findings are hidden |

---

## Reward Design

Per-step reward = score delta with explicit penalties. Rewards can be negative.

| Component | Weight (hard) | Notes |
|---|---|---|
| diagnosis | 0.25 | Fraction of required diagnostics completed |
| mitigation | 0.25 | Fraction of required mitigations applied |
| recovery | 0.25 | Service status + latency/error/cpu/memory metrics |
| communication | 0.15 | **Richness-aware**: volume + avg message length |
| efficiency | 0.10 | Penalised if resolved late; halved if unresolved |
| harm penalty | — | Ratio-based: harmful_actions / total_steps |

**Efficiency bonus**: resolving in ≤ optimal steps earns up to +0.10 extra reward.

---

## API Reference

| Route | Method | Description |
|---|---|---|
| `/` | GET | Interactive web UI |
| `/reset` | POST | Start session. Body: `{task, session_id, partial_obs}` |
| `/step` | POST | Apply action. Body: `{name, session_id}` |
| `/state` | GET | Current observation. `?session_id=` |
| `/score` | GET | Live score breakdown. `?session_id=` |
| `/episode` | GET | **Full episode trace with trajectory**. `?session_id=` |
| `/demo` | GET | **Optimal policy auto-run**. `?task=&partial_obs=` |
| `/leaderboard` | GET | **Best scores per task**. `?task=` optional |
| `/tasks` | GET | All task names + descriptions |
| `/health` | GET | Health check |

---

## Baseline Scores

| Task | Score | Notes |
|---|---|---|
| `easy` | `0.94` | Correct rollback and clean resolution |
| `medium` | `0.73` | Mitigates incident but leaves efficiency on the table |
| `hard` | `0.58` | Handles main mitigations, recovery quality partial |
| `network` | `0.89` | BGP withdrawal + traffic shift resolves cleanly |
| `memory_leak` | `0.85` | Rollback + DB scale resolves OOM loop |
| `disk_full` | `0.87` | Archive + verbosity reduction clears disk pressure |

---

## Local Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
# Then open http://localhost:7860 for the UI
```

### Run Against OpenAI-Compatible Models

```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:OPENAI_API_KEY="sk-..."
python inference.py
```

### Docker

```bash
docker build -t devops-openenv .
docker run --rm -p 7860:7860 devops-openenv
```

---

## Project Structure

```
app.py                  # FastAPI server — /episode, /demo, /leaderboard, partial obs
static/
  index.html            # Interactive web UI — click through incidents in the browser
env/
  environment.py        # OpenEnv core: reset / step / state / episode
  models.py             # Typed Observation / Action / RewardInfo / StepResult
graders/
  grader.py             # 5-dim scorer + richness-aware communication + harm rate penalty
tasks/
  task_config.py        # 6 incident scenarios: easy/medium/hard/network/memory_leak/disk_full
inference.py            # Baseline policy + OpenAI-compatible LLM runner
compare_inference.py    # Before/after: base vs trained model
train.py                # Unsloth + GRPO training — curriculum dataset, reward funcs
test_inference.py       # Baseline score regression tests
openenv.yaml            # OpenEnv v3 metadata
Dockerfile
requirements.txt
```

---

## Loom Walkthrough

[▶ Watch the demo](https://www.loom.com/share/778e553dfefd4f3b93614d4ce7996e2a)
