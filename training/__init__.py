"""Training package for Incident Ops OpenEnv.

Submodules:
  config           — TrainConfig dataclass and YAML loader
  reward_functions — All 9 GRPO reward signal functions
  dataset          — SFT warm-start and GRPO curriculum dataset builders
  callbacks        — RewardLoggerCallback for per-step CSV logging
  plot             — Reward curve and component plotting utilities
  pipeline         — main() two-phase SFT → GRPO training loop
"""
