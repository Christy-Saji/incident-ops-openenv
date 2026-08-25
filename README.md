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

1. **SFT warm-start** — supervised fine-tuning on optimal trajectories across the 4 held-in scenarios (`easy`, `medium`, `hard`, `network`); `memory_leak` and `disk_full` are held out entirely for evaluation
2. **GRPO** — group relative policy optimisation, primarily rewarded by exact Q\* (`training/qstar.py`) over an eps-greedy-expanded state distribution

Training uses [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) on Google Colab (T4 16 GB VRAM). The trained model is evaluated locally via the FastAPI server and the multi-run evaluation script.

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
│   ├── qstar.py               # Exact Q* solver (BFS + backward induction / beam fallback)
│   ├── reward_functions.py   # format + qstar + diversity (+ 7 retired LEGACY_REWARD_FUNCTIONS)
│   ├── dataset.py            # SFT dataset + eps-greedy GRPO dataset, train/eval task split
│   ├── prompting.py          # SYSTEM_PROMPT + build_prompt (shared by train/eval/compare)
│   ├── callbacks.py          # RewardLoggerCallback + WandbRewardCallback
│   ├── plot.py               # Reward curve (with reward_std collapse panel)
│   └── pipeline.py           # SFT → GRPO training loop + prompt-budget guard
│
├── env/
│   ├── environment.py        # DevOpsEnv (gym-style: reset/step/state/score/episode)
│   └── models.py             # Action types, observation schema, MITIGATION_PREREQS gate
│
├── graders/
│   └── grader.py             # compute_score() — 5-component weighted scoring
│
├── tasks/
│   └── task_config.py        # 6 incident scenario definitions (optimal_actions is Q*-derived)
│
├── server/
│   └── app.py                # FastAPI routes (reset, step, demo, score, leaderboard)
│
├── static/
│   └── index.html            # Interactive ops console UI (3 tabs: sim / results / lb)
│
├── scripts/
│   ├── evaluate.py           # Multi-run eval: N seeds, mean±std, paired t-test
│   ├── derive_optimal.py     # Regenerates task_config.py's optimal_actions from Q*
│   └── smoke_test.py         # Quick end-to-end sanity check
│
├── tests/                    # 83 tests, CPU-only, no model loading
│   ├── conftest.py
│   ├── test_reward_functions.py  # 35 tests — reward funcs incl. retired LEGACY set
│   ├── test_environment.py       # 24 tests — all 6 tasks, reset/step/obs/optimal, gating
│   ├── test_grader.py            # 13 tests — score ranges, component validation
│   ├── test_reward_ranking.py    #  6 tests — reward-vs-Q*-optimal tripwire (hard assert)
│   └── test_dataset.py           #  5 tests — GRPO dataset size, SFT/GRPO disjointness, held-out tasks
│
├── .dockerignore
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

GRPO training uses 3 reward signals (`training/reward_functions.py::ALL_REWARD_FUNCTIONS`):

| Signal | Purpose |
|--------|---------|
| `format_reward` | Valid action string check |
| `qstar_reward` | **Primary signal.** `Q*(s,a) - max_a' Q*(s,a')` |
| `diversity_reward` | Penalise duplicate actions within a GRPO group |

`qstar_reward` is exact Q\* over the environment's finite, deterministic state
space (`training/qstar.py`): the best final `compute_score` reachable after
taking action `a` from state `s`, computed by depth-indexed BFS + backward
induction (falling back to a width-60 beam search on tasks where exact search
doesn't terminate in reasonable time — verified equal on every task where both
ran). Because the reward *is* the grader, maximised, reward-grader alignment is
structural rather than something to tune toward: `tests/test_reward_ranking.py`
hard-asserts the reward stack ranks a Q\*-optimal action first in every training
state.

This replaced an earlier 9-hand-tuned-function stack (kept as
`LEGACY_REWARD_FUNCTIONS` for reference/ablation) whose reward argmax disagreed
with the optimal action on roughly half of all training states — GRPO cannot
learn a policy where the reward and the grader disagree, no matter how long you
train.

`diversity_reward` is scored **per completion** (each one penalised by how many
siblings duplicate it), not per group. A reward that assigns the same value to
every completion in a group is cancelled exactly by GRPO's group-mean baseline
and cannot influence a single gradient — which is what the earlier group-uniform
version did.

The environment scoring (`compute_score`) uses 5 weighted components: diagnosis
quality, mitigation completion, recovery, communication, and efficiency.
Mitigation credit requires the diagnosis gate to have been passed first
(`env/models.py::MITIGATION_PREREQS`) — a mitigation applied before its
prerequisite diagnostic is a no-op that increments `harmful_action_count`
rather than a free pass to skip triage.

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

Environment stochasticity is **on by default** (`--stochastic`), and LLM policies
sample at `--temperature 0.7`. Both are required for the seeds to differ: with a
deterministic environment and greedy decoding every seed yields an identical
episode, and the report will mark those tasks `DEGENERATE` rather than print a
p-value. Note that the heuristic baseline follows the optimal trajectory, so it
is a strong reference, not an untrained one — for a base-vs-trained comparison
pass `--base-model` pointing at the untrained checkpoint.

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
  id: "unsloth/Qwen2.5-3B-Instruct"
  lora_rank: 32
  lora_alpha: 64          # alpha = 2*rank; at 16 the LoRA update was scaled by 0.5
  max_seq_length: 1280    # must fit max_prompt_length + max_completion_length

training:
  grpo_max_steps: 500
  train_tasks: ["easy", "medium", "hard", "network"]  # held-in for SFT + GRPO
  eval_tasks: ["memory_leak", "disk_full"]             # held out of both entirely
  epsilon: 0.3             # eps-greedy off-Q*-path action probability (training/dataset.py)
  n_states: 1000           # target unique GRPO prompts
  num_generations: 8
  max_prompt_length: 1024 # prompts are ~700 tok median; at 512 TRL left-truncated them
  learning_rate: 0.00005  # 5e-5 — a LoRA LR, not a full-finetune one
  kl_coef: 0.005          # KL vs the SFT reference; 0.04 anchored the policy in place
  save_steps: 50          # checkpoint every 50 steps (resume if Colab disconnects)

wandb:
  enabled: false          # set to true + add WANDB_API_KEY to Colab secrets
```

All values can also be overridden via environment variables (`GRPO_MAX_STEPS`, `HF_TOKEN`, etc.).

`training/pipeline.py` asserts at startup that the longest training prompt fits
inside `max_prompt_length` and raises if it does not — TRL left-truncates
silently, which removes the system prompt and produces no warning in the logs.

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

- **Training:** Requires GPU. The committed config targets Colab T4 (16 GB VRAM)
  with `Qwen2.5-3B-Instruct` in 4-bit at 500 GRPO steps. Wall-clock time for that
  configuration has not been re-measured since the config changed — the previously
  quoted figure was for a different model and step count and has been removed
  rather than guessed at.
- **Inference / eval / server:** CPU or any GPU. The 4-bit quantised model runs on 4 GB VRAM.
- **Tests:** CPU only, no model loading required — `pytest tests/ -rxX`.
