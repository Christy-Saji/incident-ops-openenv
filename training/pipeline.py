"""Two-phase SFT → GRPO training pipeline.

Phase 1: SFT warm-start on optimal trajectories (1 epoch).
Phase 2: GRPO RL training with 9 reward signals.

Usage (from train.py entry point):
    from training.pipeline import run
    run(config)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.config import TrainConfig


def run(config: "TrainConfig") -> None:
    """Execute the full SFT → GRPO training pipeline.

    All heavy imports (torch, unsloth, trl) are deferred here so that the
    training package can be imported on CPU-only machines (e.g. for tests).
    """
    import torch
    from unsloth import FastLanguageModel, PatchDPOTrainer
    from trl import GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig

    from training.callbacks import RewardLoggerCallback, WandbRewardCallback
    from training.dataset import generate_sft_dataset, generate_grpo_dataset
    from training.plot import plot_reward_curve, plot_reward_components
    from training.reward_functions import ALL_REWARD_FUNCTIONS

    # ------------------------------------------------------------------
    # W&B init (if enabled)
    # ------------------------------------------------------------------
    if config.wandb.enabled:
        try:
            import wandb
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name or config.experiment_name,
                config={
                    "model_id":       config.model.id,
                    "lora_rank":      config.model.lora_rank,
                    "grpo_max_steps": config.training.grpo_max_steps,
                    "num_generations":config.training.num_generations,
                    "per_task_n":     config.training.per_task_prompts,
                    "mid_episode_n":  config.training.mid_episode_prompts,
                    "seed":           config.seed,
                },
            )
            print("[W&B] Run initialised.")
        except ImportError:
            print("[W&B] wandb not installed — skipping. Run: pip install wandb")
            config.wandb.enabled = False

    # ------------------------------------------------------------------
    # 1. Load base model with Unsloth
    # ------------------------------------------------------------------
    print("\n[1] Loading base model:", config.model.id)
    PatchDPOTrainer()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.id,
        max_seq_length=config.model.max_seq_length,
        dtype=None,
        load_in_4bit=config.model.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.model.lora_alpha,
        use_gradient_checkpointing="unsloth",
    )

    # ------------------------------------------------------------------
    # 2. Phase 1 — SFT warm-start
    # ------------------------------------------------------------------
    print("\n[2] Phase 1 — SFT warm-start on optimal trajectories...")
    sft_dataset = generate_sft_dataset(seed=config.seed)

    def format_sft_sample(example):
        messages = example["prompt"] + example["completion"]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    sft_dataset = sft_dataset.map(format_sft_sample)

    sft_args = SFTConfig(
        output_dir=config.output.sft_dir,
        num_train_epochs=config.training.sft_epochs,
        per_device_train_batch_size=config.training.sft_batch_size,
        gradient_accumulation_steps=config.training.sft_gradient_accumulation,
        learning_rate=config.training.sft_learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=1,
        dataset_text_field="text",
        max_length=config.model.max_seq_length,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=sft_dataset,
    )
    sft_trainer.train()
    print("  SFT warm-start complete.")

    # ------------------------------------------------------------------
    # 3. Phase 2 — GRPO curriculum training
    # ------------------------------------------------------------------
    print(f"\n[3] Phase 2 — GRPO curriculum (max_steps={config.training.grpo_max_steps})...")
    grpo_dataset = generate_grpo_dataset(
        per_task_n=config.training.per_task_prompts,
        mid_episode_n=config.training.mid_episode_prompts,
        seed=config.seed,
    )

    grpo_args = GRPOConfig(
        output_dir=config.output.grpo_dir,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.grpo_max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        logging_steps=1,
        max_grad_norm=config.training.max_grad_norm,
        num_generations=config.training.num_generations,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        temperature=config.training.temperature,
        beta=config.training.kl_coef,           # KL penalty — prevents catastrophic forgetting
        # Checkpoint resumption
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
    )

    reward_log_path = config.output.reward_log
    os.makedirs(os.path.dirname(reward_log_path), exist_ok=True)

    callbacks = [
        RewardLoggerCallback(log_path=reward_log_path),
        WandbRewardCallback(enabled=config.wandb.enabled),
    ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=ALL_REWARD_FUNCTIONS,
        args=grpo_args,
        train_dataset=grpo_dataset,
        callbacks=callbacks,
    )

    # Support checkpoint resumption — resume from latest checkpoint if exists
    resume_ckpt = _find_latest_checkpoint(config.output.grpo_dir)
    if resume_ckpt:
        print(f"  Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    print("  GRPO training complete.")

    # ------------------------------------------------------------------
    # 4. Save merged model
    # ------------------------------------------------------------------
    print(f"\n[4] Saving merged model to {config.output.model_path}...")
    model.save_pretrained_merged(
        config.output.model_path,
        tokenizer,
        save_method="merged_16bit",
    )

    # ------------------------------------------------------------------
    # 5. Push to HuggingFace Hub (optional)
    # ------------------------------------------------------------------
    if config.model.push_to_hub and config.model.hub_model_id:
        token = config.model.hub_token or os.environ.get("HF_TOKEN")
        print(f"\n[5] Pushing to HuggingFace Hub: {config.model.hub_model_id}")
        model.push_to_hub(config.model.hub_model_id, token=token)
        tokenizer.push_to_hub(config.model.hub_model_id, token=token)
        print("  Push complete.")

    # ------------------------------------------------------------------
    # 6. Generate plots
    # ------------------------------------------------------------------
    print("\n[6] Generating reward plots...")
    plot_reward_curve(
        log_path=reward_log_path,
        out_path=config.output.reward_curve,
        smooth_window=10,
    )
    plot_reward_components(
        log_path=reward_log_path,
        out_path=config.output.reward_components,
    )

    if config.wandb.enabled:
        try:
            import wandb
            wandb.log({
                "reward_curve": wandb.Image(config.output.reward_curve),
                "reward_components": wandb.Image(config.output.reward_components),
            })
            wandb.finish()
        except Exception:
            pass

    print(f"\n✅ Training complete. Model at: {config.output.model_path}")
    print(f"   Reward log : {reward_log_path}")
    print(f"   Reward plot: {config.output.reward_curve}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the most recent checkpoint directory, if any."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, checkpoints[-1])
