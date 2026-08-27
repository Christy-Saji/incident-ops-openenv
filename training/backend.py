"""Training backend seam — the one place a training backend is touched.

The SFT → GRPO pipeline (``training/pipeline.py``) routes through this module for
the three operations that are backend-specific: importing the backend ahead of TRL,
loading a LoRA-wrapped model + tokenizer, and saving the merged model. Everything else
in the pipeline — dataset construction, reward functions, the TRL
``SFTTrainer``/``GRPOTrainer`` — is backend-agnostic and stays in ``pipeline.py``.

Why the seam exists
-------------------
Where training runs, which backend loads the model, and whether weights are 4-bit or
bf16 are one coupled decision. Isolating it behind :func:`preimport` /
:func:`load_model` / :func:`save_model` means each provision is a self-contained branch
here and the pipeline never changes. Two provisions are supported, selected by
``config.hardware.backend`` (set via a named profile in config/train.yaml):

- ``unsloth``      — Google Colab / NVIDIA CUDA. Unsloth 4-bit (bitsandbytes),
                     ``FastLanguageModel`` + ``save_pretrained_merged``.
- ``transformers`` — AMD Developer Cloud / ROCm (MI300X). Stock ``transformers`` +
                     ``peft`` in bf16; no Unsloth, no bitsandbytes — 4-bit
                     bitsandbytes is unreliable on ROCm, so this path runs bf16 and
                     rejects ``load_in_4bit``.

The seam splits its inputs the same way config/train.yaml does: model identity and
LoRA geometry (rank / alpha / target modules) come from the *locked* ``config.model``
block, while backend / precision / sequence length come from the *deferred*
``config.hardware`` block.

All heavy imports (``torch``, ``unsloth``, ``transformers``, ``peft``) are deferred
into the function bodies so this module — and therefore the whole ``training``
package — stays importable on CPU-only machines, which is what lets the CPU-only
tests import it without a GPU or either backend installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from training.config import TrainConfig


def preimport(config: "TrainConfig") -> None:
    """Import the backend's own libraries before TRL/transformers are imported.

    Unsloth patches ``transformers``, ``peft`` and ``trl`` at import time and expects
    to be imported *first*. Importing it after ``trl`` leaves TRL half-patched: in
    particular ``trl.SFTConfig``'s ``eos_token``/``pad_token`` dataclass defaults come
    back rewritten to the literal placeholder string ``"<EOS_TOKEN>"``, which is in no
    real vocabulary, so ``SFTTrainer.__init__`` dies with::

        ValueError: The specified `eos_token` ('<EOS_TOKEN>') is not found in the
        vocabulary of the given `processing_class` (Qwen2TokenizerFast).

    (unslothai/unsloth#2797.) ``training/pipeline.py`` calls this before its own
    ``from trl import ...`` so the ordering holds no matter who invoked the pipeline.
    ``pipeline.py`` also pins the two tokens explicitly on ``SFTConfig``, so the run
    survives even if a future Unsloth finds another way to clobber those defaults.

    No-op for the ``transformers`` backend, which does no import-time patching.
    """
    if config.hardware.backend == "unsloth":
        import unsloth  # noqa: F401  (imported for its import-time patches only)


def load_model(config: "TrainConfig") -> Tuple[Any, Any]:
    """Load the base model and wrap it with a LoRA adapter, per ``hardware.backend``.

    Returns ``(model, tokenizer)``. Reads model identity + LoRA rank/alpha/targets
    from the locked ``config.model`` block and the backend / precision /
    sequence-length knobs from the deferred ``config.hardware`` block.
    """
    backend = config.hardware.backend
    if backend == "unsloth":
        return _load_unsloth(config)
    if backend == "transformers":
        return _load_transformers(config)
    raise ValueError(f"Unsupported hardware.backend: {backend!r}")


def save_model(config: "TrainConfig", model: Any, tokenizer: Any, path: str) -> None:
    """Persist the merged model + tokenizer to ``path``, per ``hardware.backend``.

    Both backends write a standalone (LoRA-merged) model directory that
    ``scripts/evaluate.py`` and ``server/inference.py`` can load — the trained-model
    → eval handoff is identical regardless of which backend produced it.
    """
    backend = config.hardware.backend
    if backend == "unsloth":
        _save_unsloth(model, tokenizer, path)
    elif backend == "transformers":
        _save_transformers(model, tokenizer, path)
    else:
        raise ValueError(f"Unsupported hardware.backend: {backend!r}")


# ---------------------------------------------------------------------------
# unsloth backend — Google Colab / NVIDIA CUDA, 4-bit
# ---------------------------------------------------------------------------

def _load_unsloth(config: "TrainConfig") -> Tuple[Any, Any]:
    from unsloth import FastLanguageModel, PatchDPOTrainer

    PatchDPOTrainer()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.id,
        max_seq_length=config.hardware.max_seq_length,
        dtype=None,
        load_in_4bit=config.hardware.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        target_modules=list(config.model.lora_target_modules),
        lora_alpha=config.model.lora_alpha,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def _save_unsloth(model: Any, tokenizer: Any, path: str) -> None:
    # Unsloth materialises the LoRA-merged fp16 weights in one call.
    model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")


# ---------------------------------------------------------------------------
# transformers backend — AMD Developer Cloud / ROCm (MI300X), bf16
# ---------------------------------------------------------------------------

def _load_transformers(config: "TrainConfig") -> Tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.hardware.load_in_4bit:
        raise ValueError(
            "hardware.backend='transformers' does not support load_in_4bit=true: "
            "bitsandbytes 4-bit is unreliable on AMD ROCm. Set load_in_4bit: false "
            "(bf16) in the active hardware profile."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.id)
    if tokenizer.pad_token is None:
        # GRPO/SFT batching needs a pad token; Qwen/Llama chat tokenizers ship none.
        tokenizer.pad_token = tokenizer.eos_token

    # No device_map: let the TRL trainer + accelerate place the model. bf16 is native
    # on MI300X/MI325X; there is no bitsandbytes quantisation on this path.
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False          # incompatible with gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()      # needed for grad-checkpointing + LoRA

    lora = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=list(config.model.lora_target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model, tokenizer


def _save_transformers(model: Any, tokenizer: Any, path: str) -> None:
    # Fold the LoRA adapter back into the base weights so the saved directory is a
    # plain HF model, loadable by scripts/evaluate.py without peft.
    merged = model.merge_and_unload()
    merged.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
