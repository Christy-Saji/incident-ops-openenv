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

## Status

**Training is not finished.** The first GRPO run plateaued, and a diagnostic pass
found the cause was structural rather than a hyperparameter problem. The
pre-training defects are now fixed (see *Known Limitations*); the training run
itself has not been repeated, so this README deliberately reports **no
base-vs-trained results**. The previous results table was withdrawn because both
of its columns were unsound:

- the "base" column came from a heuristic policy that read an `actions_taken`
  field the observation never carried, so it looped one action for the whole
  episode instead of following the optimal trajectory;
- the p-values came from a deterministic environment evaluated with greedy
  decoding, where every seed produced an identical episode and the paired t-test
  compared a list against a copy of itself.

What *is* measured today, on CPU, reproducibly:

| Property | Value |
|---|---|
| Optimal-trajectory score, all 6 tasks | **0.90 – 0.96** (`easy` 0.94, `medium` 0.93, `hard` 0.90, `network` 0.96, `memory_leak` 0.93, `disk_full` 0.92) |
| Tasks resolved by the optimal trajectory | 6 / 6 |
| Test suite | 70 tests (69 pass, 1 expected-fail tripwire) |
| Reward stack ranks the optimal action first | **21/41** (9-func) · **22/41** (3-func) — see *Known Limitations* |

Reproduce the first two rows with:

```bash
python scripts/evaluate.py --n-seeds 5 --label baseline_check
```

---

## Known Limitations

These are open, measured, and tracked in the test suite — not hidden.

1. **The reward stack disagrees with the optimal policy on about half of the 41
   training states.** `tests/test_reward_ranking.py` scores all 19 actions from
   every state on every optimal trajectory and checks whether the summed reward
   ranks the optimal action first. It does so **21/41** times for the 9-function
   stack (`ALL_REWARD_FUNCTIONS`, used by `training/pipeline.py`) and **22/41**
   for the 3-function stack (`CORE_REWARD_FUNCTIONS`, used by the Colab
   notebook) — so trimming 9 signals to 3 did not address the disagreement. Both
   are measured, and both are marked XFAIL with their live counts. Three clusters
   account for nearly all of it:
   `acknowledge_incident` is ranked last at t=0 despite starting every optimal
   trajectory; `resolve_incident` outranks `post_status_update` at every
   communication step; and mitigation orderings are separated by margins of
   0.02–0.10 on a ~2.5 scale, i.e. noise. GRPO cannot learn a policy the reward
   does not rank first, so this is the next thing to fix.

2. **Every GRPO prompt is a state SFT already memorised.** Both datasets are
   built from optimal-trajectory prefixes, so the 33 unique GRPO prompts are a
   subset of the 41 SFT training states, each labelled with the optimal action.
   After SFT the policy is near-deterministic on them, all group samples come out
   identical, and GRPO's advantage `(r - group_mean) / group_std` is zero. This
   needs off-policy state generation and a train/eval task split.

3. **Training is single-step, not multi-turn.** The model picks one action per
   prompt and the reward is computed by replaying the environment. There is no
   rollout and no credit assignment across a trajectory.

4. **`good_followups` does not affect `compute_score`.** `graders/grader.py`
   computes the count and discards it; follow-up quality reaches the score only
   indirectly through the communication component.

5. **Two training entry points.** `train.py` + `config/train.yaml` +
   `training/pipeline.py` is one path; `colab_training.ipynb` is another, and it
   hardcodes its own `GRPOConfig` rather than reading the YAML. They now agree on
   LoRA scaling, sequence budget and the reward subset (`CORE_REWARD_FUNCTIONS`),
   but they still differ on `temperature` (1.1 vs 0.9), `beta` (0.15 vs 0.005)
   and SFT epochs (2 vs 1). **Check which path you are running before tuning
   anything** — editing `config/train.yaml` does not change a Colab run.

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
│   ├── dataset.py            # SFT + GRPO dataset builders, state enumeration
│   ├── prompting.py          # SYSTEM_PROMPT + build_prompt (shared by train/eval/compare)
│   ├── callbacks.py          # RewardLoggerCallback + WandbRewardCallback
│   ├── plot.py               # Reward curve (with reward_std collapse panel)
│   └── pipeline.py           # SFT → GRPO training loop + prompt-budget guard
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
│   ├── evaluate.py           # Multi-run eval: N seeds, mean±std, paired t-test
│   └── smoke_test.py         # Quick end-to-end sanity check
│
├── tests/                    # 70 tests, CPU-only, no model loading
│   ├── conftest.py
│   ├── test_reward_functions.py  # 35 tests — all 9 reward funcs
│   ├── test_environment.py       # 19 tests — all 6 tasks, reset/step/obs/optimal
│   ├── test_grader.py            # 13 tests — score ranges, component validation
│   └── test_reward_ranking.py    #  3 tests — reward-vs-optimal tripwire (1 XFAIL)
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
| 9 | `diversity_reward` | Penalise duplicate actions within a GRPO group |

`diversity_reward` is scored **per completion** (each one penalised by how many
siblings duplicate it), not per group. A reward that assigns the same value to
every completion in a group is cancelled exactly by GRPO's group-mean baseline
and cannot influence a single gradient — which is what the earlier group-uniform
version did.

The environment scoring (`compute_score`) uses 5 weighted components: diagnosis quality, mitigation completion, recovery, communication, and efficiency. See *Known Limitations* for how well this stack currently ranks the optimal action.

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
