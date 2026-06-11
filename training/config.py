"""Training configuration — dataclass + YAML loader.

Load from file:
    cfg = TrainConfig.from_yaml("config/train.yaml")

Override via env vars (all optional):
    GRPO_MAX_STEPS, GRPO_PER_TASK_PROMPTS, GRPO_MID_EPISODE_PROMPTS
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml  # PyYAML
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model and LoRA configuration."""
    id: str = "unsloth/Llama-3.2-1B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 16
    max_seq_length: int = 512
    load_in_4bit: bool = True

    # For Colab runs: optionally push to HF Hub after training
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None   # e.g. "yourname/sre-agent-llama3-grpo"
    hub_token: Optional[str] = None      # or set HF_TOKEN env var


@dataclass
class TrainingConfig:
    """GRPO and SFT hyperparameters."""
    # GRPO
    grpo_max_steps: int = 300
    per_task_prompts: int = 8
    mid_episode_prompts: int = 60
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4            # number of GRPO rollouts per prompt
    max_prompt_length: int = 512
    max_completion_length: int = 32
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 8
    max_grad_norm: float = 0.3
    temperature: float = 0.9

    # SFT warm-start
    sft_epochs: int = 1
    sft_learning_rate: float = 2e-4
    sft_batch_size: int = 2
    sft_gradient_accumulation: int = 4

    # Checkpoint resumption
    save_steps: int = 50               # checkpoint every N steps
    save_total_limit: int = 3          # keep only the latest 3


@dataclass
class OutputConfig:
    """Paths for outputs."""
    dir: str = "outputs"
    grpo_dir: str = "outputs/grpo"
    sft_dir: str = "outputs/sft"
    reward_log: str = "outputs/reward_log.csv"
    reward_curve: str = "outputs/reward_curve.png"
    reward_components: str = "outputs/reward_components_mean.png"
    model_path: str = "outputs/trained_sre_agent"


@dataclass
class WandbConfig:
    """Weights & Biases experiment tracking."""
    enabled: bool = False
    project: str = "incident-ops-openenv"
    entity: Optional[str] = None       # your W&B username
    name: Optional[str] = None         # run name (auto-generated if None)


@dataclass
class TrainConfig:
    """Top-level training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 42
    experiment_name: str = "grpo_run"

    # ---------------------------------------------------------------------------
    # Environment variable overrides (for backwards compatibility + Colab ease)
    # ---------------------------------------------------------------------------

    def apply_env_overrides(self) -> "TrainConfig":
        """Apply GRPO_* environment variable overrides in-place."""
        if val := os.environ.get("GRPO_MAX_STEPS"):
            self.training.grpo_max_steps = int(val)
        if val := os.environ.get("GRPO_PER_TASK_PROMPTS"):
            self.training.per_task_prompts = int(val)
        if val := os.environ.get("GRPO_MID_EPISODE_PROMPTS"):
            self.training.mid_episode_prompts = int(val)
        if val := os.environ.get("HF_TOKEN"):
            self.model.hub_token = val
        if val := os.environ.get("WANDB_PROJECT"):
            self.wandb.project = val
        return self

    # ---------------------------------------------------------------------------
    # Loaders
    # ---------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load config from a YAML file, then apply env var overrides."""
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required: pip install pyyaml")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            output=OutputConfig(**raw.get("output", {})),
            wandb=WandbConfig(**raw.get("wandb", {})),
            seed=raw.get("seed", 42),
            experiment_name=raw.get("experiment_name", "grpo_run"),
        )
        return cfg.apply_env_overrides()

    @classmethod
    def default(cls) -> "TrainConfig":
        """Create a default config with env var overrides applied."""
        return cls().apply_env_overrides()

    def summary(self) -> str:
        """Human-readable summary for logging."""
        lines = [
            f"Experiment : {self.experiment_name}",
            f"Model      : {self.model.id}",
            f"LoRA rank  : {self.model.lora_rank}",
            f"GRPO steps : {self.training.grpo_max_steps}",
            f"Prompts/task: {self.training.per_task_prompts}",
            f"Mid-episode: {self.training.mid_episode_prompts}",
            f"Batch size : {self.training.per_device_train_batch_size} "
            f"(grad_accum={self.training.gradient_accumulation_steps})",
            f"Generations: {self.training.num_generations}",
            f"Checkpoint : every {self.training.save_steps} steps",
            f"Output dir : {self.output.dir}",
            f"W&B        : {'enabled → ' + self.wandb.project if self.wandb.enabled else 'disabled'}",
            f"Push HF Hub: {self.model.push_to_hub}",
        ]
        return "\n".join(lines)
