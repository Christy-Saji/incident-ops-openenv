"""
Training script for the DevOps Incident Triage Environment.
Uses Unsloth + TRL for a two-phase training pipeline:

Phase 1 — SFT Warm-start (NEW)
  1 epoch of supervised fine-tuning on the optimal action sequences from all
  6 task scenarios.  This gives the model a non-random starting policy so that
  GRPO receives non-zero rewards from step 1.

Phase 2 — GRPO RL Training
  Curriculum design (all 6 tasks):
    - 15 prompts per task initial state  (easy/medium/hard/network/memory_leak/disk_full)
    - 15 mid-episode prompts (1-2 random steps taken first across all tasks)
  Shuffled before training.  max_steps bumped to 200 for a more visible reward curve.

Reward functions:
  1. format_reward_func         — valid action string check
  2. step_reward_func           — environment step reward (task-aware)
  3. anti_cheat_reward_func     — penalise no_op / premature resolve
  4. task_alignment_reward_func — rewards diagnostics/mitigations matching the task

Note: torch / unsloth / trl are imported lazily inside main() so that the
reward functions and dataset helpers can be imported on machines without a GPU.
"""

import csv
import json
import os
import random
from typing import List

from datasets import Dataset

from env.environment import DevOpsEnv
from env.models import VALID_ACTIONS
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS

# ---------------------------------------------------------------------------
# Hackathon parameters
# ---------------------------------------------------------------------------
# ⚠️  IMPORTANT: whatever MODEL_ID you set here MUST be the same model you
# pass as --base-model in compare_inference.py.  Comparing a trained 1B
# against a base 3B is an unfair comparison — the 3B will always win on
# raw capability regardless of training quality.
#
# Recommended:  "unsloth/Llama-3.2-1B-Instruct"  (T4, ~30 min training)
#               "unsloth/Llama-3.2-3B-Instruct"  (T4, ~70 min training)
#               "unsloth/Meta-Llama-3.1-8B-Instruct"  (A100 / HF Endpoint)
# ---------------------------------------------------------------------------
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"   # change both here AND in compare_inference.py
MAX_SEQ_LENGTH = 1024
LORA_RANK = 32    # bumped from 16 → 32 for better expressivity at 1B scale
REWARD_LOG_PATH = "outputs_grpo/reward_log.csv"

system_prompt = (
    "You are an On-call SRE. Pick exactly one action from the valid actions.\n"
    "Valid actions: {}\n"
    "Do not explain your reasoning. Just output the action word."
).format(", ".join(VALID_ACTIONS))


# ---------------------------------------------------------------------------
# 1. Reward Functions (4 independent signals)
# ---------------------------------------------------------------------------

def format_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 1: Did the model output a single valid action string?

    Reduced from 1.0 → 0.3 so this baseline signal no longer dominates
    the task-specific rewards and the model has a real incentive to explore
    diagnostic / mitigation actions rather than any valid action.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"].strip()
        rewards.append(0.3 if text in VALID_ACTIONS else -0.5)
    return rewards


def step_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 2: Does this action improve the incident state?

    Restores approximate mid-episode state by replaying recent_actions from the
    serialised observation in the user message before scoring the candidate action.
    This fixes the bug where mid-episode prompts were always evaluated from step 0.

    NaN safety: the entire env instantiation + step is wrapped in try/except.
    A bad task name, corrupted prompt, or any env error returns -0.5 instead
    of propagating an exception that would produce a NaN reward in the batch.
    The return value is also explicitly clamped to [-1.0, 1.0].
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = completion[0]["content"].strip()

        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        try:
            task = "easy"
            recent_actions: List[str] = []
            try:
                state_dict = json.loads(prompt[-1]["content"])
                task = state_dict.get("task", "easy")
                if task not in TASK_CONFIGS:   # guard against unknown task names
                    task = "easy"
                recent_actions = state_dict.get("recent_actions", [])
            except Exception:
                pass

            env = DevOpsEnv(task=task)
            env.reset()
            # Replay the recent actions to reconstruct approximate mid-episode state
            for prev in recent_actions:
                if prev in VALID_ACTIONS:
                    env.step(prev)

            _, step_reward, _, _ = env.step(action)
            # Clamp to [-1, 1] — env already clamps to [-0.25, 1.0] but be explicit
            rewards.append(float(max(-1.0, min(1.0, step_reward))))

        except Exception:
            # Fallback: any env error → neutral negative reward, not NaN
            rewards.append(-0.5)

    return rewards


# Actions that give trivially safe small rewards — the model's previous local optimum
_FILLER_ACTIONS = {"acknowledge_incident", "post_status_update"}


def anti_cheat_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 3: Penalise reward hacking.

    BUG FIX (v2): The previous version read from `recent_actions` which the
    environment caps at the last 5 actions.  More critically, most GRPO
    training prompts come from *initial* episode states where recent_actions
    is [], so the repetition penalty (-0.8) never fired during training.

    This version reads `actions_taken` (the full episode history injected into
    mid-episode prompts by generate_prompts), falling back to recent_actions
    when actions_taken is not present.  The loop-detection branch fires an
    even stronger penalty when the last 3 actions are all the same token.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = completion[0]["content"].strip()

        all_actions_taken: List[str] = []
        try:
            state_dict = json.loads(prompt[-1]["content"])
            # Prefer full history; fall back to sliding window.
            all_actions_taken = state_dict.get(
                "actions_taken", state_dict.get("recent_actions", [])
            )
        except Exception:
            pass

        if action == "no_op":
            rewards.append(-0.5)          # strong no-op penalty
        elif action == "resolve_incident":
            rewards.append(-0.2)          # premature resolve penalty
        elif action in _FILLER_ACTIONS:
            rewards.append(-0.15)         # penalise spamming safe filler actions
        elif len(all_actions_taken) >= 3 and len(set(all_actions_taken[-3:])) == 1:
            # Last 3 actions are identical — loop detected, hard break
            rewards.append(-1.0)
        elif action in all_actions_taken:
            # Action already taken somewhere in the episode
            rewards.append(-0.8)
        else:
            rewards.append(0.4)           # reward any NEW genuine diagnostic/mitigation
    return rewards


def task_alignment_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 4: Strong bonus for picking the task-correct diagnostic or mitigation.

    Key fix: reward for required actions raised from 0.3 → 0.8, making the total
    reward for a correct diagnostic action ~1.4 vs ~-0.35 for a filler action.
    This gap is large enough for a 1B model to discover via GRPO exploration.
    Also ensures no reward is given for redundant actions already taken.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = completion[0]["content"].strip()

        if action not in VALID_ACTIONS:
            rewards.append(-0.3)
            continue

        task = "easy"
        recent_actions = []
        try:
            state_dict = json.loads(prompt[-1]["content"])
            task = state_dict.get("task", "easy")
            recent_actions = state_dict.get("recent_actions", [])
        except Exception:
            pass

        if action in recent_actions and action not in _FILLER_ACTIONS:
            # Remove any task alignment incentive to farm repeated actions
            rewards.append(-0.5)
            continue

        config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
        required_diag = config.get("required_diagnostics", [])
        required_mit  = config.get("required_mitigations", [])
        good_followups = config.get("good_followups", [])

        if action in required_diag or action in required_mit:
            rewards.append(0.8)    # massive reward for the right tool on the right task
        elif action in good_followups:
            rewards.append(0.3)    # good follow-up action
        elif action in _FILLER_ACTIONS:
            rewards.append(0.0)    # no alignment credit for filler
        elif action == "no_op":
            rewards.append(-0.5)
        else:
            rewards.append(-0.2)   # valid action but wrong task

    return rewards


def sequence_progress_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 5: Bonus when the agent has completed required diagnostics
    before attempting mitigations — reinforces the correct SRE workflow:
    investigate → mitigate → communicate → resolve.

    BUG FIX (v2): Now reads `actions_taken` (full history) in addition to
    `recent_actions` so that mid-episode prompts carry complete context.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = completion[0]["content"].strip()

        if action not in VALID_ACTIONS:
            rewards.append(0.0)
            continue

        task = "easy"
        all_actions_taken: List[str] = []
        try:
            state_dict = json.loads(prompt[-1]["content"])
            task = state_dict.get("task", "easy")
            all_actions_taken = state_dict.get(
                "actions_taken", state_dict.get("recent_actions", [])
            )
        except Exception:
            pass

        config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
        required_diag = set(config.get("required_diagnostics", []))
        required_mit  = set(config.get("required_mitigations", []))
        done_so_far   = set(all_actions_taken)

        diag_complete = required_diag.issubset(done_so_far)

        if action in required_mit and diag_complete:
            # Agent is attempting mitigation AFTER completing diagnosis — perfect sequence
            rewards.append(0.5)
        elif action in required_diag and not diag_complete:
            # Agent is still in the diagnostic phase — good
            rewards.append(0.2)
        elif action in required_mit and not diag_complete:
            # Attempting mitigation before diagnosis — penalise slightly
            rewards.append(-0.1)
        else:
            rewards.append(0.0)

    return rewards


def diversity_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 6 (NEW): Penalise GRPO group-level mode collapse.

    GRPO samples `num_generations` completions per prompt and learns from
    relative advantage.  When all completions are identical the reward_std
    is 0 and there is NO gradient signal — the model stops learning entirely.

    This function detects that collapse at the batch level and applies a
    uniform negative signal so the KL term pushes the policy back toward
    exploration.  It returns 0.0 when the group shows healthy diversity.
    """
    actions = [c[0]["content"].strip() for c in completions]
    valid_actions = [a for a in actions if a in VALID_ACTIONS]
    unique_count = len(set(valid_actions)) if valid_actions else 0

    if unique_count <= 1:
        # Complete collapse — penalise uniformly to force exploration
        return [-0.25] * len(completions)
    else:
        return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# 2. Reward Curve Logger Callback
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:  # allow import on CPU-only machines without transformers
    _TrainerCallbackBase = object  # type: ignore[misc,assignment]


class RewardLoggerCallback(_TrainerCallbackBase):  # type: ignore[valid-type]
    """Saves per-step reward metrics to a CSV for demo reward curves.

    Handles both TRL naming conventions for per-function reward keys:
      - Old TRL (< ~0.15):  reward_format_reward_func   (underscore prefix)
      - New TRL (>= ~0.15): rewards/format_reward_func  (slash prefix)

    All keys are normalised to the underscore form before writing so the
    CSV column names are stable regardless of which TRL version is installed.
    Only rows where 'reward' is present are written — non-training log
    events (LR warmup, checkpoint saves) are skipped.
    """

    # Canonical CSV column names.
    CSV_COLUMNS = [
        "step",
        "reward",
        "reward_format_reward_func",
        "reward_step_reward_func",
        "reward_anti_cheat_reward_func",
        "reward_task_alignment_reward_func",
        "reward_sequence_progress_reward_func",
    ]

    def __init__(self, log_path: str = REWARD_LOG_PATH):
        self.log_path = log_path
        self._header_written = False
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    @staticmethod
    def _normalize(key: str) -> str:
        """Convert TRL's slash-format to underscore-format.

        rewards/format_reward_func  ->  reward_format_reward_func
        reward_format_reward_func   ->  reward_format_reward_func (unchanged)
        """
        if key.startswith("rewards/"):
            return "reward_" + key[len("rewards/"):]
        return key

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "reward" not in logs:
            return

        # Capture every reward-related key TRL emits and normalize it.
        row: dict = {"step": state.global_step}
        for raw_key, value in logs.items():
            if raw_key == "reward" or raw_key.startswith("reward_") or raw_key.startswith("rewards/"):
                col = self._normalize(raw_key)
                row[col] = value

        # Fill any canonical column that TRL didn't emit this step.
        for col in self.CSV_COLUMNS:
            row.setdefault(col, "")

        write_header = not self._header_written and not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 3. Curriculum Dataset Generation
# ---------------------------------------------------------------------------

def _make_prompt(state: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(state)},
        ]
    }


def generate_sft_dataset(seed: int = 42) -> Dataset:
    """Generate a supervised fine-tuning dataset from optimal action sequences.

    For each of the 6 tasks we replay the full optimal trajectory and record
    every (observation, optimal_next_action) pair as a prompt/completion.
    This gives the model a strong starting policy before GRPO kicks in.
    """
    random.seed(seed)
    data = []

    for task_name, config in TASK_CONFIGS.items():
        optimal_actions = config.get("optimal_actions", [])
        if not optimal_actions:
            continue

        env = DevOpsEnv(task=task_name)
        state = env.reset()

        for action in optimal_actions:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(state)},
            ]
            # SFT target: the optimal action at this point in the episode
            data.append({
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": action}],
            })
            state, _, done, _ = env.step(action)
            if done:
                break

    random.shuffle(data)
    print(f"  SFT dataset: {len(data)} (state, optimal_action) pairs across all 6 tasks.")
    return Dataset.from_list(data)


def _make_prompt_with_history(state: dict, actions_taken: List[str]) -> dict:
    """Like _make_prompt but injects the full actions_taken list so that
    anti_cheat_reward_func and sequence_progress_reward_func can correctly
    penalise repeated actions in mid-episode prompts.

    BUG FIX: Previously mid-episode prompts only carried `recent_actions`
    (the last-5 sliding window), so the anti-cheat penalty never fired for
    actions taken more than 5 steps ago, and the repetition check was always
    empty for initial-state prompts.
    """
    state_with_history = {**state, "actions_taken": actions_taken}
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(state_with_history)},
        ]
    }


def generate_prompts(
    per_task_n: int = 15,
    mid_episode_n: int = 15,
    seed: int = 42,
    easy_n: int = None,
    medium_n: int = None,
    hard_n: int = None,
) -> Dataset:
    """Generate a curriculum-mixed prompt dataset for GRPO.

    Distribution:
      - per_task_n prompts per task initial state (all 6 tasks)
        OR easy_n/medium_n/hard_n for per-task control (easy/medium/hard only)
      - mid_episode_n : states captured after 1-2 random valid steps

    BUG FIX: Mid-episode prompts now include `actions_taken` (the full episode
    history) via _make_prompt_with_history so that reward functions can penalise
    repeated actions correctly.  Initial-state prompts have an empty
    actions_taken by definition and are unchanged.
    """
    random.seed(seed)
    data = []

    # Support per-task counts (easy_n / medium_n / hard_n) or fall back to per_task_n
    per_task_counts = {}
    if easy_n is not None or medium_n is not None or hard_n is not None:
        per_task_counts = {
            "easy": easy_n if easy_n is not None else per_task_n,
            "medium": medium_n if medium_n is not None else per_task_n,
            "hard": hard_n if hard_n is not None else per_task_n,
        }
        all_tasks = list(per_task_counts.keys())
    else:
        all_tasks = list(TASK_CONFIGS.keys())
        per_task_counts = {task: per_task_n for task in all_tasks}

    # Initial states — actions_taken is empty at episode start, so no history needed
    for task in all_tasks:
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(per_task_counts[task]):
            data.append(_make_prompt(state))

    # Mid-episode states — inject full action history for correct anti-cheat scoring
    for _ in range(mid_episode_n):
        task = random.choice(all_tasks)
        n_warm = random.randint(1, 3)
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(n_warm):
            action = random.choice(VALID_ACTIONS)
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
                break
        # Pass the full actions_taken so reward functions see real episode history
        data.append(_make_prompt_with_history(state, env._state["actions_taken"]))

    random.shuffle(data)
    print(f"  GRPO dataset: {len(data)} prompts across {len(all_tasks)} tasks + mid-episode states.")
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# 4. Main Training Loop
# ---------------------------------------------------------------------------

def main():
    # GPU-only imports — kept here so the module can be imported without a GPU
    import torch  # noqa: F401
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    from unsloth import FastLanguageModel, PatchDPOTrainer
    from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig

    print("[1] Loading Unsloth Model...")
    PatchDPOTrainer()  # Required optimisations

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )

    # ── Phase 1: SFT Warm-start ────────────────────────────────────────────
    # Run 1 epoch of supervised fine-tuning on optimal action sequences.
    # This gives GRPO a non-random starting policy so rewards are non-zero
    # from the very first update step.
    print("[2] Phase 1 — SFT Warm-start on optimal trajectories...")
    sft_dataset = generate_sft_dataset()

    def format_sft_sample(example):
        """Convert prompt+completion into a single flat text for SFT."""
        messages = example["prompt"] + example["completion"]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    sft_dataset = sft_dataset.map(format_sft_sample)

    sft_args = SFTConfig(
        output_dir="outputs_sft",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=1,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=sft_dataset,
    )

    print("  Running SFT warm-start epoch...")
    sft_trainer.train()
    print("  SFT warm-start complete.")

    # ── Phase 2: GRPO RL Training ──────────────────────────────────────────
    print("[3] Phase 2 — Generating GRPO Curriculum (all 6 tasks)...")
    grpo_dataset = generate_prompts()

    print("[4] Setting up GRPO Config (max_steps=300)...")
    training_args = GRPOConfig(
        output_dir="outputs_grpo",
        learning_rate=5e-6,              # halved from 1e-5 — reduces KL explosion risk
        lr_scheduler_type="cosine",
        warmup_steps=20,                 # NaN fix: let optimizer calibrate before full LR kicks in
        max_steps=300,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        # NaN prevention — the three settings below address the KL spike → NaN chain
        max_grad_norm=0.1,               # aggressive clip (default=1.0); stops gradient overflow when KL spikes
        # GRPO-specific
        num_generations=8,               # more group samples → reward_std stays non-zero even during partial collapse
        max_prompt_length=512,
        max_completion_length=32,
        temperature=0.9,                 # adds generation stochasticity so completions are not all identical
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            step_reward_func,
            anti_cheat_reward_func,
            task_alignment_reward_func,
            sequence_progress_reward_func,
            diversity_reward_func,       # NEW: breaks GRPO variance collapse
        ],
        args=training_args,
        train_dataset=grpo_dataset,
        callbacks=[RewardLoggerCallback()],
    )

    print("[5] Beginning GRPO RL Training Loop...")
    trainer.train()

    print("[6] Saving merged model...")
    model.save_pretrained_merged("trained_sre_agent", tokenizer, save_method="merged_16bit")
    print(f"Training Complete! Reward log saved to {REWARD_LOG_PATH}")


if __name__ == "__main__":
    main()
