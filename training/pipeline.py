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
    # backend.preimport MUST run before `from trl import ...`: the unsloth backend
    # patches transformers/peft/trl at import time and expects to go first. Imported
    # second, it rewrites trl.SFTConfig's eos_token/pad_token defaults to the
    # placeholder "<EOS_TOKEN>" and SFTTrainer then rejects it as out-of-vocabulary
    # (unslothai/unsloth#2797). See training/backend.py::preimport.
    from training.backend import load_model, preimport, save_model

    preimport(config)

    import torch
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    from training.callbacks import RewardLoggerCallback, WandbRewardCallback
    from training.dataset import generate_grpo_dataset, generate_sft_dataset
    from training.plot import plot_reward_components, plot_reward_curve
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
                    "num_generations":config.hardware.num_generations,
                    "train_tasks":    config.training.train_tasks,
                    "n_states":       config.training.n_states,
                    "epsilon":        config.training.epsilon,
                    "seed":           config.seed,
                },
            )
            print("[W&B] Run initialised.")
        except ImportError:
            print("[W&B] wandb not installed — skipping. Run: pip install wandb")
            config.wandb.enabled = False

    # ------------------------------------------------------------------
    # 1. Load base model + LoRA adapter via the backend seam
    # ------------------------------------------------------------------
    # training/backend.py is the only module that touches the backend (Unsloth
    # today). Choosing a different backend/precision later is a swap there, not here.
    print("\n[1] Loading base model:", config.model.id)
    model, tokenizer = load_model(config)

    # ------------------------------------------------------------------
    # 2. Phase 1 — SFT warm-start
    # ------------------------------------------------------------------
    print("\n[2] Phase 1 — SFT warm-start on optimal trajectories...")
    sft_dataset = generate_sft_dataset(seed=config.seed, tasks=config.training.train_tasks)

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
        max_length=config.hardware.max_seq_length,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # Pin EOS/pad to the loaded tokenizer's own tokens. TRL defaults both to None,
        # meaning "take them from the tokenizer" — which is what we want, but the
        # unsloth backend has been seen to rewrite those defaults to the placeholder
        # "<EOS_TOKEN>" (unslothai/unsloth#2797), and SFTTrainer then raises
        # "The specified `eos_token` ('<EOS_TOKEN>') is not found in the vocabulary".
        # backend.preimport() fixes the import order that triggers it; passing the real
        # tokens here makes the pipeline immune regardless.
        **_sft_token_kwargs(SFTConfig, tokenizer),
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
        n_states=config.training.n_states,
        epsilon=config.training.epsilon,
        seed=config.seed,
        tasks=config.training.train_tasks,
    )

    _assert_prompts_fit(
        grpo_dataset,
        tokenizer,
        config.training.max_prompt_length,
        config.hardware.max_seq_length,
        config.training.max_completion_length,
    )

    grpo_args = GRPOConfig(
        output_dir=config.output.grpo_dir,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.grpo_max_steps,
        per_device_train_batch_size=config.hardware.per_device_train_batch_size,
        gradient_accumulation_steps=config.hardware.gradient_accumulation_steps,
        logging_steps=1,
        max_grad_norm=config.training.max_grad_norm,
        num_generations=config.hardware.num_generations,
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
    # 4. Save merged model via the backend seam
    # ------------------------------------------------------------------
    print(f"\n[4] Saving merged model to {config.output.model_path}...")
    save_model(config, model, tokenizer, config.output.model_path)

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

def _sft_token_kwargs(sft_config_cls, tokenizer) -> dict:
    """EOS/pad overrides for ``SFTConfig``, skipping fields the installed TRL lacks.

    ``eos_token`` and ``pad_token`` are recent ``SFTConfig`` fields and pyproject.toml
    admits ``trl>=0.18.2``; passing a keyword the dataclass does not declare is a
    TypeError, so gate each one on the installed version actually having it.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(sft_config_cls)}
    kwargs = {}
    if "eos_token" in fields:
        kwargs["eos_token"] = tokenizer.eos_token
    if "pad_token" in fields:
        kwargs["pad_token"] = tokenizer.pad_token or tokenizer.eos_token
    return kwargs


def _assert_prompts_fit(
    dataset,
    tokenizer,
    max_prompt_length: int,
    max_seq_length: int,
    max_completion_length: int,
) -> None:
    """Fail loudly if any training prompt would be silently truncated.

    TRL left-truncates prompts longer than max_prompt_length, which removes the
    system prompt (rules + action list) and the head of the observation JSON.
    That happened silently for the whole of the first training run: prompts were
    a median ~700 tokens against a 512-token limit, so GRPO was sampling from a
    truncated JSON fragment with no instructions in it, while SFT had trained on
    the full prompt. Nothing in the logs said so.

    Raising here is the guard that stops that from ever recurring.
    """
    longest_tokens = 0
    longest_text = ""
    for row in dataset:
        text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        n_tokens = len(tokenizer(text)["input_ids"])
        if n_tokens > longest_tokens:
            longest_tokens, longest_text = n_tokens, text

    print(f"  [prompt-budget] longest prompt: {longest_tokens} tokens "
          f"(max_prompt_length={max_prompt_length})")

    if longest_tokens > max_prompt_length:
        raise ValueError(
            f"Prompt budget exceeded: longest training prompt is {longest_tokens} tokens "
            f"but max_prompt_length is {max_prompt_length}. TRL would LEFT-truncate this, "
            f"silently removing the system prompt and the start of the observation.\n"
            f"Fix by raising training.max_prompt_length in config/train.yaml (and "
            f"hardware.max_seq_length with it), or by trimming the observation in "
            f"training/prompting.py:PROMPT_EXCLUDED_OBS_KEYS.\n"
            f"Longest prompt began: {longest_text[:200]!r}"
        )

    if max_prompt_length + max_completion_length > max_seq_length:
        raise ValueError(
            f"max_prompt_length ({max_prompt_length}) + max_completion_length "
            f"({max_completion_length}) = {max_prompt_length + max_completion_length} "
            f"exceeds hardware.max_seq_length ({max_seq_length})."
        )


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
