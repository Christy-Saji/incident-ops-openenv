"""Training entry point for Incident Ops OpenEnv.

Usage:
    # Use default config (config/train.yaml):
    python train.py

    # Use a different config file:
    python train.py --config config/train_1b.yaml

    # Override specific params via env vars:
    GRPO_MAX_STEPS=100 python train.py

    # Experiment name (shows in W&B and output dir):
    python train.py --experiment grpo_qwen_300steps
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the SRE incident response agent (SFT → GRPO).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config/train.yaml",
        help="Path to YAML config file (default: config/train.yaml)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override experiment name (shown in W&B and output paths)",
    )
    args = parser.parse_args()

    from training.config import TrainConfig

    config_path = Path(args.config)
    if config_path.exists():
        print(f"[config] Loading from {config_path}")
        cfg = TrainConfig.from_yaml(config_path)
    else:
        print(f"[config] {config_path} not found — using defaults with env var overrides.")
        cfg = TrainConfig.default()

    if args.experiment:
        cfg.experiment_name = args.experiment

    print("\n" + "=" * 60)
    print("Incident Ops OpenEnv — Training Pipeline")
    print("=" * 60)
    print(cfg.summary())
    print("=" * 60 + "\n")

    from training.pipeline import run
    run(cfg)


if __name__ == "__main__":
    main()
