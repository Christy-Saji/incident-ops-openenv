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
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
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
    """Reward 1: Did the model output a single valid action string?"""
    rewards = []
    for completion in completions:
        text = completion[0]["content"].strip()
        rewards.append(1.0 if text in VALID_ACTIONS else 0.0)
    return rewards


def step_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 2: Does this action improve the incident state?

    Restores approximate mid-episode state by replaying recent_actions from the
    serialised observation in the user message before scoring the candidate action.
    This fixes the bug where mid-episode prompts were always evaluated from step 0.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = completion[0]["content"].strip()

        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        task = "easy"
        recent_actions: List[str] = []
        try:
            state_dict = json.loads(prompt[-1]["content"])
            task = state_dict.get("task", "easy")
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
        rewards.append(float(step_reward))

    return rewards


def anti_cheat_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 3: Penalise reward hacking (spamming no_op or resolving prematurely)."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"].strip()
        if text == "no_op":
            rewards.append(-0.2)
        elif text == "resolve_incident":
            rewards.append(-0.1)
        else:
            rewards.append(0.1)
    return rewards


def task_alignment_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 4: Did the model pick an action that matches the task's required
    diagnostics, mitigations, or good follow-ups?
    This prevents the model from gaming format/anti-cheat rewards with
    valid-but-wrong-task actions.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = completion[0]["content"].strip()

        if action not in VALID_ACTIONS:
            rewards.append(-0.2)
            continue

        task = "easy"
        try:
            state_dict = json.loads(prompt[-1]["content"])
            task = state_dict.get("task", "easy")
        except Exception:
            pass

        config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
        required_diag = config.get("required_diagnostics", [])
        required_mit = config.get("required_mitigations", [])
        good_followups = config.get("good_followups", [])

        if action in required_diag or action in required_mit:
            rewards.append(0.3)
        elif action in good_followups:
            rewards.append(0.1)
        elif action == "no_op":
            rewards.append(-0.2)
        else:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# 2. Reward Curve Logger Callback
# ---------------------------------------------------------------------------

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:  # allow import on CPU-only machines without transformers
    _TrainerCallbackBase = object  # type: ignore[misc,assignment]


class RewardLoggerCallback(_TrainerCallbackBase):  # type: ignore[valid-type]
    """Saves per-step reward metrics to a CSV for demo reward curves.

    Inherits from transformers.TrainerCallback at class-definition time so the
    module can still be imported on machines without GPU/transformers installed
    (the fallback base class is plain object).
    """

    REWARD_KEYS = [
        "reward",
        "reward_format_reward_func",
        "reward_step_reward_func",
        "reward_anti_cheat_reward_func",
        "reward_task_alignment_reward_func",
    ]

    def __init__(self, log_path: str = REWARD_LOG_PATH):
        self.log_path = log_path
        self._header_written = False
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        row = {"step": state.global_step}
        for key in self.REWARD_KEYS:
            row[key] = logs.get(key, "")

        write_header = not self._header_written and not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
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

    # Initial states for every task
    for task in all_tasks:
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(per_task_counts[task]):
            data.append(_make_prompt(state))

    # Mid-episode states — vary task and number of warm-up steps
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
        data.append(_make_prompt(state))

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

    print("[4] Setting up GRPO Config (max_steps=200)...")
    training_args = GRPOConfig(
        output_dir="outputs_grpo",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        max_steps=200,           # doubled from 100 for a more visible reward curve
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1,
        # GRPO-specific
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=32,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            step_reward_func,
            anti_cheat_reward_func,
            task_alignment_reward_func,
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
