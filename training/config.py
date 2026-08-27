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
from typing import List, Optional

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
    """Model + LoRA geometry — the *locked* half of the config.

    Model identity and LoRA rank/alpha/targets are settled and do not move when the
    hardware/backend/precision decision is made later; those knobs live in
    HardwareConfig instead.
    """
    id: str = "unsloth/Llama-3.2-1B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 64          # alpha = 2*rank convention
    # LoRA adapters on attention (q,k,v,o) + MLP (gate,up,down) projections. Locked.
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )

    # For Colab runs: optionally push to HF Hub after training
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None   # e.g. "yourname/sre-agent-llama3-grpo"
    hub_token: Optional[str] = None      # or set HF_TOKEN env var


SUPPORTED_BACKENDS = ("unsloth", "transformers")


@dataclass
class HardwareConfig:
    """Hardware/runtime profile — the *deferred* half of the config.

    Where training runs, which backend loads the model, and 4-bit vs bf16 are one
    coupled decision. These are the knobs that decision edits — the training backend,
    precision, sequence budget, and the VRAM-sized batch/rollout counts — kept in one
    block so nothing else has to move when the hardware is chosen. See
    training/backend.py for the load/save seam that dispatches on ``backend``.

    Two provisions are supported and both live in config/train.yaml as named
    profiles (see TrainConfig.from_yaml):
      - ``unsloth``      — Google Colab / NVIDIA CUDA, 4-bit (bitsandbytes).
      - ``transformers`` — AMD Developer Cloud / ROCm, stock transformers+peft, bf16
                           (bitsandbytes 4-bit is unreliable on ROCm, so keep
                           load_in_4bit false here).
    """
    backend: str = "unsloth"     # one of SUPPORTED_BACKENDS
    load_in_4bit: bool = True    # transformers/ROCm path requires this be false (bf16)
    max_seq_length: int = 1280   # >= max_prompt_length + max_completion_length
    num_generations: int = 4     # number of GRPO rollouts per prompt
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"hardware.backend must be one of {list(SUPPORTED_BACKENDS)}, "
                f"got {self.backend!r}."
            )


def _resolve_hardware(raw_hw: dict) -> HardwareConfig:
    """Build the active HardwareConfig from the YAML ``hardware`` block.

    Two shapes are accepted:

    - *Named profiles* (used by config/train.yaml): a ``profiles`` mapping plus an
      active ``profile`` selector, e.g. ``{profile: colab_t4, profiles: {...}}``. The
      active profile is the ``HARDWARE_PROFILE`` env var if set, otherwise the
      ``profile`` key — so switching Colab ↔ AMD is one line (or one env var) with
      both provisions kept side by side.
    - *Flat* (used by defaults / ad-hoc configs): the HardwareConfig fields directly,
      e.g. ``{backend: transformers, load_in_4bit: false, ...}``.
    """
    if "profiles" not in raw_hw:
        return HardwareConfig(**raw_hw)

    profiles = raw_hw.get("profiles") or {}
    active = os.environ.get("HARDWARE_PROFILE") or raw_hw.get("profile")
    if not active:
        raise ValueError(
            "config hardware block defines 'profiles' but no active profile is "
            "selected. Set hardware.profile in config/train.yaml or the "
            f"HARDWARE_PROFILE env var. Available: {sorted(profiles)}"
        )
    if active not in profiles:
        raise ValueError(
            f"hardware.profile {active!r} is not defined. "
            f"Available profiles: {sorted(profiles)}"
        )
    return HardwareConfig(**profiles[active])


@dataclass
class TrainingConfig:
    """GRPO and SFT hyperparameters."""
    # GRPO
    grpo_max_steps: int = 300
    # Tier 1 Phase C: train/eval task split + eps-greedy dataset sizing.
    # memory_leak/disk_full held out of both SFT and GRPO so
    # scripts/evaluate.py's held-out eval is meaningful.
    train_tasks: List[str] = field(
        default_factory=lambda: ["easy", "medium", "hard", "network"]
    )
    eval_tasks: List[str] = field(
        default_factory=lambda: ["memory_leak", "disk_full"]
    )
    epsilon: float = 0.3        # off-Q*-path action probability in the eps-greedy walk
    n_states: int = 1000        # target unique GRPO prompts (training/dataset.py)
    max_prompt_length: int = 1024  # below ~900 TRL left-truncates real prompts
    max_completion_length: int = 32
    learning_rate: float = 5e-5    # LoRA LR (1e-5 is a full-finetune LR)
    kl_coef: float = 0.005         # KL penalty — passed as `beta` to GRPOConfig (TRL renamed kl_coef → beta)
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
    """Top-level training configuration.

    Split into a *locked* algorithm half (model, training) and a *deferred* hardware
    half (hardware), mirroring config/train.yaml, so the later backend/precision
    decision edits only the hardware block and training/backend.py.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
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
        if val := os.environ.get("GRPO_N_STATES"):
            self.training.n_states = int(val)
        if val := os.environ.get("GRPO_EPSILON"):
            self.training.epsilon = float(val)
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

        # encoding is explicit: config/train.yaml contains non-ASCII characters in
        # its comments, and Python defaults to the locale encoding (cp1252 on
        # Windows), which raises UnicodeDecodeError. Only ever worked on Linux.
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            hardware=_resolve_hardware(raw.get("hardware", {})),
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
            f"Backend    : {self.hardware.backend} "
            f"(4bit={self.hardware.load_in_4bit}, seq_len={self.hardware.max_seq_length})",
            f"LoRA rank  : {self.model.lora_rank}",
            f"GRPO steps : {self.training.grpo_max_steps}",
            f"Train tasks: {self.training.train_tasks}",
            f"Eval tasks : {self.training.eval_tasks}",
            f"GRPO states: {self.training.n_states} (epsilon={self.training.epsilon})",
            f"Batch size : {self.hardware.per_device_train_batch_size} "
            f"(grad_accum={self.hardware.gradient_accumulation_steps})",
            f"Generations: {self.hardware.num_generations}",
            f"Checkpoint : every {self.training.save_steps} steps",
            f"Output dir : {self.output.dir}",
            f"W&B        : {'enabled → ' + self.wandb.project if self.wandb.enabled else 'disabled'}",
            f"Push HF Hub: {self.model.push_to_hub}",
        ]
        return "\n".join(lines)
