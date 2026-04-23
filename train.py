"""
Training script for the DevOps Incident Triage Environment.
Uses Unsloth + TRL (GRPO) to iteratively train a small language model.

Curriculum design:
  - 20 prompts from easy task initial state
  - 20 prompts from medium task initial state
  - 10 prompts from hard task initial state
  - 10 mid-episode prompts (1-2 random steps taken first)
  Shuffled before training.

Reward functions:
  1. format_reward_func       — valid action string check
  2. step_reward_func         — environment step reward (task-aware)
  3. anti_cheat_reward_func   — penalise no_op / premature resolve
  4. task_alignment_reward_func — rewards diagnostics/mitigations matching the task
"""

import csv
import json
import os
import random
from typing import List

import torch
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import GRPOTrainer, GRPOConfig

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
    Uses task-aware env so medium/hard tasks get their correct starting state.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = completion[0]["content"].strip()

        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        # Parse the task from the serialised observation in the user message
        task = "easy"
        try:
            state_dict = json.loads(prompt[-1]["content"])
            task = state_dict.get("task", "easy")
        except Exception:
            pass

        env = DevOpsEnv(task=task)
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

class RewardLoggerCallback(TrainerCallback):
    """Saves per-step reward metrics to a CSV for demo reward curves."""

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

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
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


def generate_prompts(
    easy_n: int = 20,
    medium_n: int = 20,
    hard_n: int = 10,
    mid_episode_n: int = 10,
    seed: int = 42,
) -> Dataset:
    """Generate a curriculum-mixed prompt dataset.

    Distribution:
      - easy_n   : easy task initial states
      - medium_n : medium task initial states
      - hard_n   : hard task initial states
      - mid_episode_n : states captured after 1-2 random valid steps
    """
    random.seed(seed)
    data = []

    for task, count in [("easy", easy_n), ("medium", medium_n), ("hard", hard_n)]:
        env = DevOpsEnv(task=task)
        state = env.reset()
        for _ in range(count):
            data.append(_make_prompt(state))

    # Mid-episode states — vary task and number of warm-up steps
    tasks = ["easy", "medium", "hard"]
    for _ in range(mid_episode_n):
        task = random.choice(tasks)
        n_warm = random.randint(1, 2)
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
    print(f"  Dataset: {len(data)} prompts across easy/medium/hard + mid-episode states.")
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# 4. Main Training Loop
# ---------------------------------------------------------------------------

def main():
    print("[1] Loading Unsloth Model Space...")
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

    print("[2] Generating Curriculum Prompts for Environment...")
    dataset = generate_prompts()

    print("[3] Setting up GRPO Config...")
    training_args = GRPOConfig(
        output_dir="outputs_grpo",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        max_steps=100,
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
            task_alignment_reward_func,   # 4th independent reward signal
        ],
        args=training_args,
        train_dataset=dataset,
        callbacks=[RewardLoggerCallback()],
    )

    print("[4] Beginning RL Training Loop...")
    trainer.train()

    print("[5] Saving Model properly (Phase 9 Checklist)...")
    model.save_pretrained_merged("trained_sre_agent", tokenizer, save_method="merged_16bit")
    print(f"Training Complete! Reward log saved to {REWARD_LOG_PATH}")


if __name__ == "__main__":
    main()
