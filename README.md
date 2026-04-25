---
title: Incident Ops OpenEnv
emoji: "🚨"
colorFrom: red
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Incident Ops OpenEnv

OpenEnv Hackathon 2026  
Theme 3 - World Modeling  
Sub-theme 3.1 - Professional Tasks

[Live Hugging Face Space](https://huggingface.co/spaces/chritsysajii/incident-ops-openenv)  
[Open in Colab](https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/main/colab_training.ipynb)  
[Training Notebook in Repo](./colab_training.ipynb)

Incident Ops is an OpenEnv benchmark for training an LLM to behave like an on-call SRE during live production incidents. The agent has to inspect evidence, choose targeted mitigations, communicate status, and resolve only when the system is actually stable. The environment is designed to punish shallow tool spam and reward correct sequential incident handling.

## Why This Is Interesting

Most agent benchmarks still look like games, puzzles, or static QA. This environment instead targets a professional workflow with:

- Partial observability
- Typed actions with real failure modes
- Multiple incident families, not one narrow task
- Reward hacking traps such as safe no-op loops or premature resolution
- A training loop that can show real before/after behavior change

This matters because production incident response is exactly the kind of high-pressure, tool-using, multi-step work where "sounds smart" is not enough.

## Environment Design

The agent receives a structured observation each step containing the incident title, customer impact, phase, active alerts, service health, metrics, findings, recent actions, and the available action set.

The action space contains 19 typed actions:

- Universal: `acknowledge_incident`, `post_status_update`, `resolve_incident`, `no_op`
- Diagnostics: `inspect_auth_logs`, `inspect_db_metrics`, `inspect_deploy_history`, `inspect_network_topology`, `inspect_memory_profile`, `inspect_disk_usage`
- Mitigations: `rollback_auth_deploy`, `rollback_service_deploy`, `restart_auth_service`, `scale_db_cluster`, `flush_cache`, `shift_traffic_canary`, `withdraw_bgp_route`, `archive_old_logs`, `reduce_log_verbosity`

The reward is dense and compositional. It combines diagnosis quality, mitigation completion, recovery quality, communication quality, efficiency, and harmful-action penalties. The GRPO training pipeline also adds action-format, anti-loop, task-alignment, sequence-order, progress-delta, communication-gate, and terminal-outcome reward signals.

The point is not just "get to done." The point is to teach the model the right order:

`diagnose -> mitigate -> communicate -> resolve`

## Task Set

The environment currently includes six deterministic incident scenarios:

| Task | Scenario | Core skill being tested |
| --- | --- | --- |
| `easy` | Auth deploy regression | Detect bad deploy and roll back cleanly |
| `medium` | DB saturation from traffic spike | Read metrics and apply capacity mitigation |
| `hard` | Cascading outage after auth deploy | Multi-step diagnosis plus coordinated mitigations |
| `network` | BGP route leak | Network reasoning and routing mitigation |
| `memory_leak` | OOM kill restart loop | Infer memory failure and roll back bad service deploy |
| `disk_full` | Log disk saturation | Diagnose infra pressure and preserve audit-safe recovery |

These tasks are intentionally different enough that the model cannot win by memorizing one diagnostic habit.

## Training Pipeline

The repo includes a full end-to-end training pipeline in [train.py](./train.py) and a rerunnable notebook in [colab_training.ipynb](./colab_training.ipynb).

Pipeline:

1. SFT warm-start on optimal trajectories from all six tasks
2. GRPO RL fine-tuning against the live environment
3. Reward logging to `outputs_grpo/reward_log.csv`
4. Reward curve export to `reward_curve.png`
5. Before/after evaluation with [compare_inference.py](./compare_inference.py)

Default training model:

- `unsloth/Llama-3.2-1B-Instruct`

Default full-run settings:

- `GRPO_MAX_STEPS=300`
- `GRPO_PER_TASK_PROMPTS=8`
- `GRPO_MID_EPISODE_PROMPTS=60`

The notebook also includes a short sanity-run path before the full run, which makes it easier for judges to verify the pipeline quickly.

## Results

### Reward Curve

![Reward Curve](./reward_curve.png)

`reward_curve.png` is committed to the repo so the main training trend is visible even outside Colab.

### Before vs After Evaluation

Use this command after training:

```bash
python compare_inference.py ^
  --base-model unsloth/Llama-3.2-1B-Instruct ^
  --trained-model ./trained_sre_agent
```

This evaluates the untrained base model and the GRPO-trained model on all six tasks and prints a side-by-side comparison table. Fill the table below with the output from your final submission run:

| Task | Base score | Trained score | Delta |
| --- | --- | --- | --- |
| `easy` | pending | pending | pending |
| `medium` | pending | pending | pending |
| `hard` | pending | pending | pending |
| `network` | pending | pending | pending |
| `memory_leak` | pending | pending | pending |
| `disk_full` | pending | pending | pending |
| `average` | pending | pending | pending |

### Demo Endpoints

The environment exposes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /score`
- `GET /episode`
- `GET /demo`
- `GET /leaderboard`
- `GET /tasks`
- `GET /health`

`/demo` runs a live policy rollout. When model credentials are configured it uses the LLM policy; otherwise it falls back to the built-in heuristic baseline and labels the result accordingly.

## Quick Start

### Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860`.

### Training

```bash
pip install -r requirements-train.txt
set GRPO_MAX_STEPS=300
python train.py
```

### Evaluation

```bash
python compare_inference.py
```

### Docker

```bash
docker build -t incident-ops-openenv .
docker run --rm -p 7860:7860 incident-ops-openenv
```

## Colab Notebook

The repository includes [colab_training.ipynb](./colab_training.ipynb), which does the following:

1. Installs the training dependencies
2. Clones the repo
3. Smoke-tests the environment and reward functions
4. Runs a short GRPO sanity pass
5. Runs the full SFT + GRPO training pipeline
6. Generates `reward_curve.png`
7. Runs the before/after comparison script
8. Optionally pushes the trained model to Hugging Face Hub

Direct Colab launch:

[https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/main/colab_training.ipynb](https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/main/colab_training.ipynb)

## Submission Checklist

This repo now covers the core technical submission pieces:

- OpenEnv environment with valid manifest: [openenv.yaml](./openenv.yaml)
- Working training script: [train.py](./train.py)
- Rerunnable Colab notebook: [colab_training.ipynb](./colab_training.ipynb)
- Reward plot committed in repo: [reward_curve.png](./reward_curve.png)
- Hugging Face Space link in README
- Results/evaluation command in README

Before final submission, make sure to also add whichever public storytelling asset you want judges to open first:

- Hugging Face blog post link, or
- YouTube demo link, or
- Public slide deck link

## Project Structure

```text
app.py
compare_inference.py
colab_training.ipynb
env/
graders/
inference.py
openenv.yaml
reward_curve.png
static/
tasks/
train.py
```
