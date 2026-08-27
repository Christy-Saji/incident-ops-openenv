"""Tests for TrainConfig loading and the hardware-profile seam.

CPU-only, no torch/unsloth/transformers — these exercise config resolution, which is
what selects the training backend (Colab/Unsloth vs AMD/transformers) and therefore
gates whether the right provision runs. See training/config.py and training/backend.py.
"""

from __future__ import annotations

import pytest

from training.config import (
    SUPPORTED_BACKENDS,
    HardwareConfig,
    TrainConfig,
    _resolve_hardware,
)

REAL_CONFIG = "config/train.yaml"


# ---------------------------------------------------------------------------
# HardwareConfig field / validation
# ---------------------------------------------------------------------------

def test_default_backend_is_unsloth():
    """The flat default profile is the Colab/Unsloth provision."""
    hw = HardwareConfig()
    assert hw.backend == "unsloth"
    assert hw.load_in_4bit is True


def test_invalid_backend_rejected():
    with pytest.raises(ValueError, match="hardware.backend must be one of"):
        HardwareConfig(backend="tensorflow")


def test_supported_backends_cover_both_provisions():
    assert set(SUPPORTED_BACKENDS) == {"unsloth", "transformers"}


# ---------------------------------------------------------------------------
# _resolve_hardware — named-profile shape
# ---------------------------------------------------------------------------

def _profiles_block():
    return {
        "profile": "colab_t4",
        "profiles": {
            "colab_t4": {"backend": "unsloth", "load_in_4bit": True, "num_generations": 8},
            "amd_mi300x": {"backend": "transformers", "load_in_4bit": False, "num_generations": 16},
        },
    }


def test_resolve_selects_named_profile():
    hw = _resolve_hardware(_profiles_block())
    assert hw.backend == "unsloth"
    assert hw.num_generations == 8


def test_env_var_overrides_active_profile(monkeypatch):
    monkeypatch.setenv("HARDWARE_PROFILE", "amd_mi300x")
    hw = _resolve_hardware(_profiles_block())
    assert hw.backend == "transformers"
    assert hw.load_in_4bit is False
    assert hw.num_generations == 16


def test_unknown_profile_raises():
    block = _profiles_block()
    block["profile"] = "nonexistent"
    with pytest.raises(ValueError, match="is not defined"):
        _resolve_hardware(block)


def test_profiles_without_selector_raises():
    block = _profiles_block()
    del block["profile"]
    with pytest.raises(ValueError, match="no active profile"):
        _resolve_hardware(block)


def test_resolve_flat_shape():
    """A flat hardware block (no `profiles`) is constructed directly."""
    hw = _resolve_hardware({"backend": "transformers", "load_in_4bit": False})
    assert hw.backend == "transformers"


# ---------------------------------------------------------------------------
# Real config/train.yaml — both provisions load, locked block is shared
# ---------------------------------------------------------------------------

def test_real_config_default_profile_is_colab():
    cfg = TrainConfig.from_yaml(REAL_CONFIG)
    assert cfg.hardware.backend == "unsloth"
    assert cfg.hardware.load_in_4bit is True


def test_real_config_amd_profile_is_bf16_transformers(monkeypatch):
    monkeypatch.setenv("HARDWARE_PROFILE", "amd_mi300x")
    cfg = TrainConfig.from_yaml(REAL_CONFIG)
    assert cfg.hardware.backend == "transformers"
    assert cfg.hardware.load_in_4bit is False
    # Global batch must be divisible by num_generations (TRL GRPO constraint).
    hw = cfg.hardware
    global_batch = hw.per_device_train_batch_size * hw.gradient_accumulation_steps
    assert global_batch % hw.num_generations == 0


def test_real_config_kaggle_profile_matches_colab_settings(monkeypatch):
    """Kaggle's T4/P100 is the same 16 GB VRAM class as Colab's T4."""
    colab = TrainConfig.from_yaml(REAL_CONFIG)  # default profile, no env var set
    monkeypatch.setenv("HARDWARE_PROFILE", "kaggle_t4")
    kaggle = TrainConfig.from_yaml(REAL_CONFIG)
    assert kaggle.hardware.backend == colab.hardware.backend == "unsloth"
    assert kaggle.hardware.load_in_4bit is True
    hw = kaggle.hardware
    global_batch = hw.per_device_train_batch_size * hw.gradient_accumulation_steps
    assert global_batch % hw.num_generations == 0


def test_real_config_local_profile_is_4bit_unsloth_with_minimal_footprint(monkeypatch):
    monkeypatch.setenv("HARDWARE_PROFILE", "local_rtx3050")
    cfg = TrainConfig.from_yaml(REAL_CONFIG)
    assert cfg.hardware.backend == "unsloth"
    assert cfg.hardware.load_in_4bit is True
    assert cfg.hardware.num_generations == 2
    hw = cfg.hardware
    global_batch = hw.per_device_train_batch_size * hw.gradient_accumulation_steps
    assert global_batch % hw.num_generations == 0


def test_locked_block_is_independent_of_hardware_profile(monkeypatch):
    """Switching hardware profiles must not change the locked algorithm knobs."""
    colab = TrainConfig.from_yaml(REAL_CONFIG)
    for profile in ("kaggle_t4", "amd_mi300x", "local_rtx3050"):
        monkeypatch.setenv("HARDWARE_PROFILE", profile)
        other = TrainConfig.from_yaml(REAL_CONFIG)

        assert colab.model.id == other.model.id
        assert colab.model.lora_rank == other.model.lora_rank
        assert colab.model.lora_alpha == other.model.lora_alpha
        assert colab.model.lora_target_modules == other.model.lora_target_modules
        assert colab.training.train_tasks == other.training.train_tasks
        assert colab.training.kl_coef == other.training.kl_coef
        assert colab.training.max_prompt_length == other.training.max_prompt_length


def test_max_seq_length_fits_prompt_budget_in_all_profiles(monkeypatch):
    """Every profile must satisfy the pipeline's prompt-budget assertion."""
    for profile in ("colab_t4", "kaggle_t4", "amd_mi300x", "local_rtx3050"):
        monkeypatch.setenv("HARDWARE_PROFILE", profile)
        cfg = TrainConfig.from_yaml(REAL_CONFIG)
        budget = cfg.training.max_prompt_length + cfg.training.max_completion_length
        assert cfg.hardware.max_seq_length >= budget, profile
