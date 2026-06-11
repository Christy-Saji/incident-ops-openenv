---
title: Incident Ops OpenEnv
emoji: "🚨"
colorFrom: red
colorTo: orange
sdk: docker
app_file: app.py
pinned: false
---

# Incident Ops OpenEnv

> Reinforcement learning benchmark for SRE incident triage — training a language model to diagnose, mitigate, and resolve production failures.

[![CI](https://github.com/Christy-Saji/incident-ops-openenv/actions/workflows/ci.yml/badge.svg)](https://github.com/Christy-Saji/incident-ops-openenv/actions)
[![HuggingFace](https://img.shields.io/badge/🤗-Model-orange)](https://huggingface.co/chritsysajii/sre-agent-llama3-grpo)
[![Space](https://img.shields.io/badge/🤗-Space-blue)](https://huggingface.co/spaces/chritsysajii/incident-ops-openenv-final)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://www.python.org)

**Live Demo:** [incident-ops-openenv-final](https://huggingface.co/spaces/chritsysajii/incident-ops-openenv-final)  
**Trained Model:** [sre-agent-llama3-grpo](https://huggingface.co/chritsysajii/sre-agent-llama3-grpo)  
**Training Notebook:** [colab_training.ipynb](https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/master/colab_training.ipynb)

---

## What It Is

A custom OpenAI Gym-style environment where a language model acts as an on-call SRE. The model observes incident state (metrics, alerts, service health, findings) and selects actions from a 19-action space to diagnose, mitigate, and resolve the incident.

The model is trained using a two-phase pipeline:

1. **SFT warm-start** — supervised fine-tuning on optimal trajectories across all 6 scenarios
2. **GRPO** — group relative policy optimisation with 9 independent reward signals

Training uses [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) on Google Colab (T4 16 GB VRAM). The trained model is evaluated locally via the FastAPI server and the multi-run evaluation script.

---

## Results

| Task | Base | Trained | Δ Delta | Resolved |
|------|------|---------|---------|----------|
| `easy` | 0.35 | 0.36 | +0.01 | ✓ |
| `medium` | 0.06 | 0.34 | **+0.28** | ✓ |
| `hard` | 0.15 | 0.27 | +0.12 | — |
| `network` | 0.00 | 0.38 | **+0.38** | ✓ |
| `memory_leak` | 0.00 | 0.38 | **+0.38** | ✓ |
| `disk_full` | 0.14 | 0.36 | +0.22 | ✓ |
| **average** | 0.12 | 0.35 | **+0.23** | 5/6 |

Training reward curve and component analysis committed as artifacts: [`outputs/reward_curve.png`](outputs/reward_curve.png), [`outputs/reward_log.csv`](outputs/reward_log.csv).

---

## Project Structure

```
incident-ops-openenv/
├── train.py                  # Entry point: python train.py --config config/train.yaml
├── app.py                    # FastAPI server entry point
├── compare_inference.py      # Before/after comparison
│
├── config/
│   └── train.yaml            # All training hyperparameters (single source of truth)
│
├── training/                 # Training package (extracted from monolithic train.py)
│   ├── config.py             # TrainConfig dataclass + YAML loader
│   ├── reward_functions.py   # All 9 GRPO reward signal functions
│   ├── dataset.py            # SFT + GRPO curriculum dataset builders
│   ├── callbacks.py          # RewardLoggerCallback + WandbRewardCallback
│   ├── plot.py               # Reward curve and component plotting
│   └── pipeline.py           # SFT → GRPO training loop (~150 lines)
│
├── env/
│   ├── environment.py        # DevOpsEnv (gym-style: reset/step/state/score/episode)
│   └── models.py             # Action types, observation schema
│
├── graders/
│   └── grader.py             # compute_score() — 5-component weighted scoring
│
├── tasks/
│   └── task_config.py        # 6 incident scenario definitions
│
├── server/
│   └── app.py                # FastAPI routes (reset, step, demo, score, leaderboard)
│
├── static/
│   └── index.html            # Interactive ops console UI (3 tabs: sim / results / lb)
│
├── scripts/
│   └── evaluate.py           # Multi-run eval: N seeds, mean±std, paired t-test
│
├── tests/
│   ├── conftest.py
│   ├── test_reward_functions.py  # 67 tests, all reward funcs, CPU-only
│   ├── test_environment.py       # All 6 tasks, reset/step/obs/optimal-trajectory
│   └── test_grader.py            # Score ranges, component validation
│
└── .github/workflows/ci.yml  # Lint (ruff) + pytest + Docker smoke test
```

---

## Scenarios

| Task | Scenario | Core skill tested |
|------|----------|-------------------|
| `easy` | Auth deploy regression | Detect bad deploy and rollback |
| `medium` | Database saturation spike | Interpret metrics, apply capacity mitigation |
| `hard` | Cascading outage | Multi-step coordinated diagnosis + recovery |
| `network` | BGP route leak | Network diagnosis + routing mitigation |
| `memory_leak` | OOM kill restart loop | Infer instability, rollback appropriately |
| `disk_full` | Log disk saturation | Infrastructure pressure + operational constraints |

The scenarios are heterogeneous by design — a repeated diagnostic pattern does not score well across tasks.

---

## Reward System

GRPO training uses 9 independent reward signals:

| # | Signal | Purpose |
|---|--------|---------|
| 1 | `format_reward` | Valid action string check |
| 2 | `step_reward` | Environment step reward (task-aware) |
| 3 | `anti_cheat_reward` | Loop detection, no-op penalty |
| 4 | `task_alignment_reward` | Task-correct diagnostics/mitigations |
| 5 | `sequence_progress_reward` | Enforce diagnose→mitigate→resolve order |
| 6 | `progress_delta_reward` | Dense reward for measurable task progress |
| 7 | `communication_gate_reward` | Gate comms until technical progress exists |
| 8 | `terminal_outcome_reward` | Primary outcome signal (score delta + resolution) |
| 9 | `diversity_reward` | Penalise GRPO group-level mode collapse |

The environment scoring (`compute_score`) uses 5 weighted components: diagnosis quality, mitigation completion, recovery, communication, and efficiency.

---

## Quick Start

### Run the server locally

```bash
pip install -e .
uvicorn app:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860
```

### Run tests (CPU only, no GPU needed)

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Train on Colab

Open the notebook and run all cells:

```
https://colab.research.google.com/github/Christy-saji/incident-ops-openenv/blob/master/colab_training.ipynb
```

Or train locally (requires GPU with CUDA):

```bash
pip install -e ".[train]"
python train.py --config config/train.yaml
# Resume after disconnect automatically — checkpoints saved every 50 steps
```

### Multi-run evaluation (statistical significance)

```bash
pip install -e ".[eval]"

# Heuristic baseline (no GPU, instant):
python scripts/evaluate.py --n-seeds 5 --label baseline_test

# Trained vs base (requires saved model):
python scripts/evaluate.py \
    --trained-model outputs/trained_sre_agent \
    --n-seeds 5 \
    --label grpo_v1
```

Output: `results/<label>_<timestamp>/report.md` with mean±std and p-values per task.

### Docker

```bash
docker build -t incident-ops-openenv .
docker run --rm -p 7860:7860 incident-ops-openenv
```

---

## Training Configuration

Edit [`config/train.yaml`](config/train.yaml) to change any hyperparameter. Key settings:

```yaml
model:
  id: "unsloth/Qwen2.5-1.5B-Instruct"   # recommended for Colab T4
  lora_rank: 32

training:
  grpo_max_steps: 300
  num_generations: 8
  save_steps: 50          # checkpoint every 50 steps (resume if Colab disconnects)

wandb:
  enabled: false          # set to true + add WANDB_API_KEY to Colab secrets
```

All values can also be overridden via environment variables (`GRPO_MAX_STEPS`, `HF_TOKEN`, etc.).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Take an action |
| `GET` | `/state` | Current observation |
| `GET` | `/score` | Current score + breakdown |
| `GET` | `/episode` | Full episode trajectory |
| `GET` | `/demo` | Run a policy rollout automatically |
| `GET` | `/leaderboard` | Top scores per task |
| `GET` | `/tasks` | Available task configs |
| `GET` | `/health` | Health check |

---

## Hardware Notes

- **Training:** Requires GPU. Tested on Colab T4 (16 GB VRAM), ~45–60 min for 300 GRPO steps with Llama 3.2-1B.
- **Inference / eval / server:** CPU or any GPU. The 4-bit quantised model runs on 4 GB VRAM.
- **Tests:** CPU only, no model loading required.
