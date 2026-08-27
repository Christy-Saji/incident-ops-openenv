"""Tests for the backend seam's import-ordering guard and SFT token overrides.

CPU-only: nothing here imports torch, unsloth or trl. What it protects is the
ordering constraint that killed a Kaggle run — Unsloth patches transformers/peft/trl
at import time and must be imported *before* them. Imported second, it rewrote
``trl.SFTConfig``'s ``eos_token`` default to the placeholder ``"<EOS_TOKEN>"`` and
``SFTTrainer`` rejected it as out-of-vocabulary (unslothai/unsloth#2797). The failure
only shows up on a GPU box several minutes into a run, so it is worth a source-level
tripwire here. See training/backend.py::preimport and training/pipeline.py.
"""

from __future__ import annotations

import dataclasses
import inspect

from training.backend import preimport
from training.config import TrainConfig
from training.pipeline import _sft_token_kwargs


def test_preimport_is_a_noop_for_the_transformers_backend():
    """The AMD/ROCm provision does no import-time patching, so there is nothing to do."""
    cfg = TrainConfig.default()
    cfg.hardware.backend = "transformers"
    assert preimport(cfg) is None


def test_preimport_precedes_the_trl_import_in_the_pipeline():
    """Ordering tripwire: `preimport(config)` must run before `from trl import ...`."""
    from training import pipeline

    # Match the real import statement, not the explanatory comment above it, which
    # also spells "from trl import ...".
    src = inspect.getsource(pipeline.run)
    assert "preimport(config)" in src, "pipeline.run() no longer calls backend.preimport()"
    assert src.index("preimport(config)") < src.index("from trl import GRPOConfig"), (
        "backend.preimport(config) must be called BEFORE `from trl import ...` — "
        "importing unsloth after trl corrupts SFTConfig's eos_token default "
        "(unslothai/unsloth#2797)."
    )


def test_sft_token_kwargs_pins_the_tokenizers_own_tokens():
    @dataclasses.dataclass
    class FakeSFTConfig:
        eos_token: str = None
        pad_token: str = None

    class FakeTokenizer:
        eos_token = "<|im_end|>"
        pad_token = "<|endoftext|>"

    assert _sft_token_kwargs(FakeSFTConfig, FakeTokenizer()) == {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    }


def test_sft_token_kwargs_falls_back_to_eos_when_the_tokenizer_has_no_pad():
    @dataclasses.dataclass
    class FakeSFTConfig:
        eos_token: str = None
        pad_token: str = None

    class FakeTokenizer:
        eos_token = "<|im_end|>"
        pad_token = None

    assert _sft_token_kwargs(FakeSFTConfig, FakeTokenizer())["pad_token"] == "<|im_end|>"


def test_sft_token_kwargs_skips_fields_an_older_trl_does_not_declare():
    """pyproject admits trl>=0.18.2; an undeclared keyword would be a TypeError."""
    @dataclasses.dataclass
    class OldSFTConfig:
        max_length: int = 1024

    class FakeTokenizer:
        eos_token = "<|im_end|>"
        pad_token = None

    assert _sft_token_kwargs(OldSFTConfig, FakeTokenizer()) == {}
