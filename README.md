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

Hugging Face Space: [incident-ops-openenv](https://huggingface.co/spaces/chritsysajii/incident-ops-openenv)  
Open in Colab: [colab_training.ipynb](https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/master/colab_training.ipynb)  
GitHub Notebook: [colab_training.ipynb](https://github.com/Christy-Saji/incident-ops-openenv/blob/master/colab_training.ipynb)  
Training Reward Log CSV: [reward_log.csv](https://github.com/Christy-Saji/incident-ops-openenv/blob/master/reward_log.csv)  
Training Reward Curve PNG: [reward_curve.png](https://github.com/Christy-Saji/incident-ops-openenv/blob/master/reward_curve.png)  
Hugging Face Blog Post: [placeholder](#)

Incident Ops OpenEnv is an OpenEnv benchmark for training language models to perform structured incident response in a professional operations setting. The environment evaluates whether a model can inspect evidence, select appropriate mitigations, communicate status, and resolve only after the system is stable.

## Problem Theme

This project belongs to Theme 3, World Modeling, under Sub-theme 3.1, Professional Tasks. The focus is operational decision-making in a realistic SaaS incident-response workflow rather than a game-like or static prompt-based benchmark.

## Problem Statement

Language models frequently perform poorly in incident-response settings despite producing plausible explanations. Common failure modes include:

- repeating safe but low-value actions
- applying mitigations before diagnosis
- resolving incidents prematurely
- selecting interventions that do not match the actual root cause

This project evaluates whether reinforcement learning can improve sequential decision-making in that setting. The objective is not rhetorical fluency, but better action selection under partial information.

## Environment

The environment models live production incidents in a structured tool-use setting. Each step returns an observation containing the incident title, customer impact, incident phase, active alerts, service health, metrics, findings, recent actions, and the available action space.

The action space contains 19 typed actions:

- Universal: `acknowledge_incident`, `post_status_update`, `resolve_incident`, `no_op`
- Diagnostics: `inspect_auth_logs`, `inspect_db_metrics`, `inspect_deploy_history`, `inspect_network_topology`, `inspect_memory_profile`, `inspect_disk_usage`
- Mitigations: `rollback_auth_deploy`, `rollback_service_deploy`, `restart_auth_service`, `scale_db_cluster`, `flush_cache`, `shift_traffic_canary`, `withdraw_bgp_route`, `archive_old_logs`, `reduce_log_verbosity`

The reward is dense and compositional. It combines diagnosis quality, mitigation completion, recovery quality, communication quality, efficiency, and harmful-action penalties. The GRPO training setup adds auxiliary reward signals for action validity, anti-loop behavior, task alignment, sequence ordering, progress delta, communication gating, and terminal outcome.

The intended decision sequence is:

`diagnose -> mitigate -> communicate -> resolve`

### Task Set

The environment includes six deterministic incident scenarios:

| Task | Scenario | Core capability |
| --- | --- | --- |
| `easy` | Auth deploy regression | Detect a bad deploy and roll back correctly |
| `medium` | Database saturation from traffic spike | Interpret metrics and apply capacity mitigation |
| `hard` | Cascading outage after auth deploy | Coordinate multi-step diagnosis and recovery |
| `network` | BGP route leak | Perform network diagnosis and routing mitigation |
| `memory_leak` | OOM kill restart loop | Infer service instability and roll back appropriately |
| `disk_full` | Log disk saturation | Recover from infrastructure pressure while preserving operational constraints |

The scenarios are intentionally heterogeneous so that a single repeated diagnostic pattern does not score well across tasks.

## Training Pipeline

The repository includes a complete training pipeline in [train.py](./train.py) and a rerunnable notebook in [colab_training.ipynb](./colab_training.ipynb).

Pipeline:

1. SFT warm-start on optimal trajectories from all six tasks
2. GRPO fine-tuning against the live environment
3. Reward logging to `outputs_grpo/reward_log.csv`
4. Reward curve export to `reward_curve.png`
5. Before/after evaluation with [compare_inference.py](./compare_inference.py)

Training configuration used for the full run:

- Base model: `unsloth/Llama-3.2-1B-Instruct`
- `GRPO_MAX_STEPS=300`
- `GRPO_PER_TASK_PROMPTS=8`
- `GRPO_MID_EPISODE_PROMPTS=60`

The notebook also includes a short sanity run before the full training pass to validate the pipeline end-to-end.

## Results

### Reward Curve

[Open reward curve image directly](https://github.com/Christy-Saji/incident-ops-openenv/blob/master/reward_curve.png)

![Reward Curve](https://raw.githubusercontent.com/Christy-Saji/incident-ops-openenv/master/reward_curve.png)

`reward_curve.png` is committed to the repository as a persistent training artifact.
The plot shows a clear upward reward trend over the course of training. Early steps are volatile and frequently negative, which is consistent with an initially weak policy exploring poor actions. As training progresses, the smoothed reward rises steadily and remains substantially above the starting region, indicating that the model is learning action sequences that align better with the environment's reward structure. The continued variance in the raw trace suggests that the task remains challenging and the policy is not fully stable, but the overall trajectory is positive and consistent with measurable improvement rather than noise alone.



### Raw Training Logs

[Open reward_log.csv directly](https://github.com/Christy-Saji/incident-ops-openenv/blob/master/reward_log.csv)

The full numeric training trace is available in `reward_log.csv` and is linked above as the raw evidence underlying the plotted reward curve.

### Before vs After Evaluation

Evaluation command:

```bash
python compare_inference.py ^
  --base-model unsloth/Llama-3.2-1B-Instruct ^
  --trained-model ./trained_sre_agent
```

Observed result from the current training artifact:

> **Average score improvement: +0.23**  
> **Training improved agent performance.**

The resulting policy remains imperfect and still exhibits noisy behavior in some episodes, including repetition and low-value actions. However, the before/after evaluation shows measurable improvement across all six tasks rather than a flat or degraded outcome.

| Task | Base score | Trained score | Delta |
| --- | --- | --- | --- |
| `easy` | 0.35 | 0.36 | +0.01 |
| `medium` | 0.06 | 0.34 | +0.28 |
| `hard` | 0.15 | 0.27 | +0.12 |
| `network` | 0.00 | 0.38 | +0.38 |
| `memory_leak` | 0.00 | 0.38 | +0.38 |
| `disk_full` | 0.14 | 0.36 | +0.22 |
| `average` | 0.12 | 0.35 | +0.23 |

## Why It Matters

Production incidents are a meaningful testbed for LLM training because they require tool use, sequential reasoning, diagnosis under incomplete information, and resistance to superficially safe but incorrect actions. This environment is intended to serve as:

- a realistic OpenEnv benchmark for professional tool use
- a training setup with observable reward improvement
- a demonstration that operational workflows can be represented as learnable environments rather than static prompt tasks

## Colab Notebook

The notebook at [colab_training.ipynb](./colab_training.ipynb) performs the following steps:

1. installs dependencies
2. clones the repository
3. smoke-tests the environment and reward functions
4. runs a short GRPO sanity pass
5. runs the full SFT + GRPO training pipeline
6. generates `reward_curve.png`
7. runs the before/after comparison script
8. optionally pushes the trained model to Hugging Face Hub

Direct Colab launch:

[https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/master/colab_training.ipynb](https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/master/colab_training.ipynb)

## Demo Endpoints

The environment exposes the following endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /score`
- `GET /episode`
- `GET /demo`
- `GET /leaderboard`
- `GET /tasks`
- `GET /health`

`/demo` runs a live policy rollout. When model credentials are configured, it uses the LLM policy; otherwise it falls back to the built-in heuristic baseline and labels the result accordingly.

## Quick Start

### Local Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

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

