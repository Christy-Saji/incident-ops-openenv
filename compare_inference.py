"""
Before/After comparison script for the DevOps Incident Triage RL agent.

Runs the base model and the trained LoRA model on all 3 tasks (easy/medium/hard)
and prints a side-by-side comparison table — the core artefact for your demo.

Usage:
    python compare_inference.py
    python compare_inference.py --base-model unsloth/Llama-3.2-1B-Instruct \
                                --trained-model ./trained_sre_agent
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

from env.environment import DevOpsEnv
from env.models import VALID_ACTIONS
from graders.grader import compute_score

SYSTEM_PROMPT = (
    "You are an On-call SRE. Pick exactly one action from the valid actions.\n"
    "Valid actions: {}\n"
    "Do not explain your reasoning. Just output the action word."
).format(", ".join(VALID_ACTIONS))

TASKS = ["easy", "medium", "hard", "network", "memory_leak", "disk_full"]


# ---------------------------------------------------------------------------
# Model runner helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    """Load a model with Unsloth. Returns (model, tokenizer)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth is not installed. Run: pip install unsloth")
        sys.exit(1)

    print(f"  Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate_action_hf_api(model_id: str, hf_token: str, state: dict) -> str:
    """Call the HuggingFace Serverless Inference API instead of loading locally.

    Uses the text-generation task endpoint.  No GPU required — HF credits
    are deducted from your account automatically.

    Requires:  pip install huggingface_hub
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    client = InferenceClient(model=model_id, token=hf_token)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(state)},
    ]

    # Retry up to 3 times for transient API errors
    for attempt in range(3):
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=32,
                temperature=0.01,   # near-greedy; API does not always support temp=0
            )
            raw = response.choices[0].message.content.strip().lower()
            break
        except Exception as e:
            if attempt == 2:
                print(f"  HF API error after 3 attempts: {e}")
                return "no_op"
            time.sleep(2 ** attempt)   # exponential back-off

    for action in VALID_ACTIONS:
        if action in raw:
            return action
    return "no_op"


def _generate_action(model, tokenizer, state: dict) -> str:
    """Run one greedy forward pass and extract an action."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(state)},
    ]

    # Use the tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,   # match training max_completion_length=32
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()

    # Match against valid actions
    for action in VALID_ACTIONS:
        if action in raw:
            return action

    return "no_op"  # fallback if nothing matched


def run_episode(
    model_or_id,
    tokenizer_or_token,
    task: str,
    label: str,
    use_hf_api: bool = False,
) -> tuple[float, bool, list]:
    """Run a full episode. Works in both local-GPU mode and HF API mode."""
    env = DevOpsEnv(task=task)
    state = env.reset()
    actions_log = []

    print(f"\n  [{label}] task={task}")
    for step_num in range(1, env.max_steps + 1):
        if use_hf_api:
            action = _generate_action_hf_api(model_or_id, tokenizer_or_token, state)
        else:
            action = _generate_action(model_or_id, tokenizer_or_token, state)
        state, reward, done, info = env.step(action)
        actions_log.append(action)
        print(f"    step={step_num:2d}  action={action:<28s}  reward={reward:+.2f}")
        if done:
            break

    score, _ = compute_score(task, env._state)
    resolved = env._state["resolved"]
    print(f"  [{label}] FINAL score={score:.2f}  resolved={resolved}")
    return score, resolved, actions_log


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare base vs trained SRE agent")
    parser.add_argument(
        "--base-model",
        # ⚠️ Keep this in sync with MODEL_ID in train.py for a fair comparison!
        default="unsloth/Llama-3.2-1B-Instruct",
        help="HF model ID or local path for the base (untrained) model",
    )
    parser.add_argument(
        "--trained-model",
        default="./trained_sre_agent",
        help="Local path to the merged trained model, OR a HF repo ID if --use-hf-api",
    )
    parser.add_argument(
        "--use-hf-api",
        action="store_true",
        help="Use HuggingFace Serverless Inference API instead of loading models locally. "
             "Requires --hf-token and --trained-model to be a HF repo ID (e.g. yourname/sre-agent).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace API token (or set HF_TOKEN env var). Required for --use-hf-api.",
    )
    args = parser.parse_args()

    # Results store — initialized before either phase writes to it
    results = {task: {} for task in TASKS}

    if args.use_hf_api and not args.hf_token:
        print("ERROR: --use-hf-api requires --hf-token or the HF_TOKEN environment variable.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Run base model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: Base Model (no RL training)")
    print("=" * 60)

    if args.use_hf_api:
        print(f"  [HF API mode] Base model: {args.base_model}")
        for task in TASKS:
            score, resolved, actions = run_episode(
                args.base_model, args.hf_token, task, "BASE", use_hf_api=True
            )
            results[task]["base_score"] = score
            results[task]["base_resolved"] = resolved
            results[task]["base_actions"] = actions
    else:
        base_model, base_tokenizer = _load_model(args.base_model)
        for task in TASKS:
            score, resolved, actions = run_episode(base_model, base_tokenizer, task, "BASE")
            results[task]["base_score"] = score
            results[task]["base_resolved"] = resolved
            results[task]["base_actions"] = actions
        del base_model, base_tokenizer  # free VRAM before loading next model

    # -----------------------------------------------------------------------
    # Run trained model
    # -----------------------------------------------------------------------
    if args.use_hf_api:
        print("\n" + "=" * 60)
        print("PHASE 2: Trained Model (after GRPO RL) — via HF API")
        print("=" * 60)
        print(f"  [HF API mode] Trained model: {args.trained_model}")
        for task in TASKS:
            score, resolved, actions = run_episode(
                args.trained_model, args.hf_token, task, "TRAINED", use_hf_api=True
            )
            results[task]["trained_score"] = score
            results[task]["trained_resolved"] = resolved
            results[task]["trained_actions"] = actions
    else:
        trained_available = os.path.exists(args.trained_model)
        if not trained_available:
            print(f"\nWARNING: Trained model not found at '{args.trained_model}'. "
                  "Skipping trained model run. Train first with: python train.py")
        else:
            print("\n" + "=" * 60)
            print("PHASE 2: Trained Model (after GRPO RL)")
            print("=" * 60)
            trained_model, trained_tokenizer = _load_model(args.trained_model)
            for task in TASKS:
                score, resolved, actions = run_episode(
                    trained_model, trained_tokenizer, task, "TRAINED"
                )
                results[task]["trained_score"] = score
                results[task]["trained_resolved"] = resolved
                results[task]["trained_actions"] = actions
            del trained_model, trained_tokenizer

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    header = (
        f"{'Task':<8} | {'Base Score':>10} | {'Trained Score':>13} | "
        f"{'Base ✓':>6} | {'Trained ✓':>9} | {'Δ Score':>8}"
    )
    print(header)
    print("-" * 70)

    for task in TASKS:
        r = results[task]
        base_score = r.get("base_score", 0.0)
        trained_score = r.get("trained_score", None)
        base_resolved = "✅" if r.get("base_resolved", False) else "❌"
        trained_resolved = "✅" if r.get("trained_resolved", False) else "❌"

        if trained_score is not None:
            delta = trained_score - base_score
            delta_str = f"{delta:+.2f}"
            trained_str = f"{trained_score:.2f}"
        else:
            delta_str = "N/A"
            trained_str = "N/A"

        print(
            f"{task:<8} | {base_score:>10.2f} | {trained_str:>13} | "
            f"{base_resolved:>6} | {trained_resolved:>9} | {delta_str:>8}"
        )

    print("=" * 70)

    if "trained_score" in list(results.values())[0] or args.use_hf_api:
        improvements = [
            results[t]["trained_score"] - results[t]["base_score"]
            for t in TASKS
            if "trained_score" in results[t]
        ]
        avg_delta = sum(improvements) / len(improvements) if improvements else 0.0
        print(f"\nAverage score improvement: {avg_delta:+.2f}")

        if avg_delta > 0:
            print("✅ Training improved agent performance!")
        elif avg_delta == 0:
            print("⚠️  No change in performance. Consider more training steps.")
        else:
            print("❌ Training degraded performance. Check reward functions.")

    print("\nDone. Use outputs_grpo/reward_log.csv for reward curve plots.")


if __name__ == "__main__":
    main()
