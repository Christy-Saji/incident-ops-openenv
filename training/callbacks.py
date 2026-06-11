"""Training callbacks — reward logging and W&B integration."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.config import TrainConfig

try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:
    _TrainerCallbackBase = object  # type: ignore[misc,assignment]


class RewardLoggerCallback(_TrainerCallbackBase):  # type: ignore[valid-type]
    """Saves per-step reward metrics to a CSV for reward curve plotting.

    Handles both TRL naming conventions for per-function reward keys:
      - Old TRL (< ~0.15):  reward_format_reward_func   (underscore prefix)
      - New TRL (>= ~0.15): rewards/format_reward_func  (slash prefix)

    All keys are normalised to the underscore form before writing so the
    CSV column names are stable regardless of which TRL version is installed.
    Only rows where 'reward' is present are written.
    """

    CSV_COLUMNS = [
        "step",
        "reward",
        "reward_format_reward_func",
        "reward_step_reward_func",
        "reward_anti_cheat_reward_func",
        "reward_task_alignment_reward_func",
        "reward_sequence_progress_reward_func",
        "reward_progress_delta_reward_func",
        "reward_communication_gate_reward_func",
        "reward_terminal_outcome_reward_func",
        "reward_diversity_reward_func",
    ]

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._header_written = False
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    @staticmethod
    def _normalize(key: str) -> str:
        """Convert TRL's slash-format to underscore-format."""
        if key.startswith("rewards/"):
            key = "reward_" + key[len("rewards/"):]
        key = key.replace("/", "_")
        if key.endswith("_mean"):
            key = key[:-5]
        return key

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is None or "reward" not in logs:
            return

        row: dict = {"step": state.global_step}
        for raw_key, value in logs.items():
            if raw_key == "reward" or raw_key.startswith("reward_") or raw_key.startswith("rewards/"):
                col = self._normalize(raw_key)
                if col.endswith("_std"):
                    continue
                row[col] = value

        for col in self.CSV_COLUMNS:
            row.setdefault(col, "")

        write_header = not self._header_written and not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


class WandbRewardCallback(_TrainerCallbackBase):  # type: ignore[valid-type]
    """Logs per-step reward component breakdown to Weights & Biases.

    Only active when wandb is enabled in TrainConfig. Each reward function
    is logged as a separate time series so you can see which signals matter.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._wandb = None
        if enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print("[WandbRewardCallback] wandb not installed — disabling W&B logging.")
                self.enabled = False

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not self.enabled or logs is None or "reward" not in logs or self._wandb is None:
            return

        metrics = {"train/step": state.global_step}
        for key, value in logs.items():
            if key == "reward":
                metrics["train/reward"] = value
            elif key.startswith("reward_") or key.startswith("rewards/"):
                # Normalise to a clean W&B key: train/reward/format_reward_func
                clean = key.replace("rewards/", "").replace("reward_", "").replace("_mean", "")
                metrics[f"train/reward_component/{clean}"] = value
            elif key in ("learning_rate", "loss", "grad_norm"):
                metrics[f"train/{key}"] = value

        self._wandb.log(metrics, step=state.global_step)
