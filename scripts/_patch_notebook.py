"""One-shot notebook patcher — applies all Track B changes to cell 15.

B1: Replace ALL_REWARD_FUNCTIONS import with 3-function import
B2: Replace reward_funcs=ALL_REWARD_FUNCTIONS with the 3-function list
B3: temperature 0.9 → 1.1, add kl_coef=0.15
B4: Diagnostic logging wired into reward_logger.on_log

Run from the repo root:
    python scripts/_patch_notebook.py
"""

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "colab_training.ipynb")
NB_PATH = os.path.normpath(NB_PATH)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ---------------------------------------------------------------------------
# Replacement source for cell 15 (the GRPO training cell)
# ---------------------------------------------------------------------------
NEW_SOURCE = """\
from trl import GRPOTrainer, GRPOConfig
from training.callbacks import RewardLoggerCallback
from training.dataset import generate_grpo_dataset
# B1: Import only the 3 orthogonal reward functions (down from 9 overlapping ones).
# task_alignment    = correctness of action for this specific task
# sequence_progress = enforces diagnose->mitigate->communicate->resolve order
# terminal_outcome  = primary episodic outcome signal (score delta + resolution bonus)
from training.reward_functions import (
    task_alignment_reward_func,
    sequence_progress_reward_func,
    terminal_outcome_reward_func,
)
from unsloth import is_bfloat16_supported
import csv, os

LOG_CSV = f'{OUTPUT_DIR}/reward_log.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

grpo_dataset = generate_grpo_dataset(
    per_task_n    = 8,
    mid_episode_n = 60,
    seed          = 42,
)
print(f'GRPO dataset: {len(grpo_dataset)} prompts')

grpo_args = GRPOConfig(
    output_dir                  = f'{OUTPUT_DIR}/grpo',
    learning_rate               = GRPO_LR,
    lr_scheduler_type           = 'cosine',
    warmup_steps                = 8,
    max_steps                   = GRPO_MAX_STEPS,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_generations             = NUM_GENERATIONS,
    max_prompt_length           = 1024,
    max_completion_length       = 32,
    temperature                 = 1.1,             # B3: was 0.9 -- prevents mode collapse
    kl_coef                     = 0.15,            # B3: was missing (TRL default 0.04 is 7.5x too small for 3-func reward scale)
    max_grad_norm               = 0.3,
    fp16                        = not is_bfloat16_supported(),
    bf16                        = is_bfloat16_supported(),
    logging_steps               = 1,
    save_steps                  = SAVE_STEPS,
    save_total_limit            = 3,   # keep latest 3 checkpoints only
    report_to                   = 'none',
)

reward_logger = RewardLoggerCallback(log_path=LOG_CSV)

trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    # B2: 3 non-overlapping reward functions replace the previous 9.
    # Reduces reward noise, eliminates contradictory anti_cheat / terminal_outcome
    # signals, and gives GRPO cleaner group-relative advantages.
    reward_funcs     = [
        task_alignment_reward_func,
        sequence_progress_reward_func,
        terminal_outcome_reward_func,
    ],
    args             = grpo_args,
    train_dataset    = grpo_dataset,
    callbacks        = [reward_logger],
)

# Auto-resume from latest checkpoint if Colab disconnected mid-training
def _latest_ckpt(d):
    import os
    if not os.path.isdir(d): return None
    ckpts = sorted(
        [x for x in os.listdir(d) if x.startswith('checkpoint-')],
        key=lambda x: int(x.split('-')[-1])
    )
    return os.path.join(d, ckpts[-1]) if ckpts else None

resume = _latest_ckpt(f'{OUTPUT_DIR}/grpo')
if resume:
    print(f'Resuming from {resume}')
else:
    print('Starting fresh GRPO run...')

# B4: Diagnostic logging ─────────────────────────────────────────────────────
# Patches reward_logger.on_log to also print reward_std live in cell output
# and write to diag_log.csv so mode-collapse is visible during training.
#
# What to watch for:
#   reward_std < 0.05  → mode collapse (all 8 rollouts identical, gradient = 0)
#   resolve_incident absent from completions by step 30 → terminal suppression
_diag_path   = f'{OUTPUT_DIR}/diag_log.csv'
_diag_file   = open(_diag_path, 'w', newline='')
_diag_writer = csv.writer(_diag_file)
_diag_writer.writerow(['step', 'reward_std', 'resolve_pct'])

_orig_log = reward_logger.on_log
def _diag_log(args, state, control, logs=None, **kwargs):
    _orig_log(args, state, control, logs=logs, **kwargs)
    if logs:
        std = logs.get('reward_std', None)
        if std is not None:
            flag = '  !! MODE COLLAPSE' if std < 0.05 else ''
            print(f'[DIAG] Step {state.global_step:4d} | reward_std={std:.4f}{flag}')
            _diag_writer.writerow([state.global_step, round(std, 4), ''])
            _diag_file.flush()
reward_logger.on_log = _diag_log

trainer.train(resume_from_checkpoint=resume)
_diag_file.close()
print('GRPO training complete')
print(f'[DIAG] Diagnostic log saved to {_diag_path}')
print('[DIAG] Check reward_log.csv for:')
print('[DIAG]   reward_std < 0.05 at any step  = mode collapse (group variance gone)')
print('[DIAG]   resolve_incident absent by step 30 = terminal suppression still active')
"""

# Convert the multi-line string into the list-of-strings format Jupyter uses
lines = []
for line in NEW_SOURCE.splitlines(keepends=True):
    lines.append(line)

# Verify we are targeting the right cell (must contain GRPOTrainer)
target_cell = nb["cells"][15]
original_src = "".join(target_cell.get("source", []))
assert "GRPOTrainer" in original_src, "Cell 15 does not look like the GRPO cell — aborting!"
assert "ALL_REWARD_FUNCTIONS" in original_src, "Cell 15 already patched or unexpected content."

# Apply patch
target_cell["source"] = lines
print(f"Patched cell 15 ({len(lines)} lines).")

# Verify patch
patched_src = "".join(target_cell["source"])
assert "ALL_REWARD_FUNCTIONS" not in patched_src, "Patch failed: ALL_REWARD_FUNCTIONS still present"
assert "task_alignment_reward_func" in patched_src, "Patch failed: 3-func import missing"
assert "kl_coef" in patched_src, "Patch failed: kl_coef missing"
assert "temperature                 = 1.1" in patched_src, "Patch failed: temperature not updated"
assert "_diag_log" in patched_src, "Patch failed: diagnostic logging missing"
print("All assertions passed.")

# Write back
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook written: {NB_PATH}")
