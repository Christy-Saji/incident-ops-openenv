# DevOps Incident Triage — Hackathon Colab Notebook
# Copy this cell-by-cell into a new Google Colab notebook (T4 GPU runtime)
# ============================================================
# CELL 1: Install dependencies
# ============================================================
"""
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install trl datasets pydantic pyyaml fastapi uvicorn openai httpx -q
"""

# ============================================================
# CELL 2: Clone repo
# ============================================================
"""
import os
REPO = "https://github.com/Christy-saji/incident-ops-openenv"
!git clone {REPO} /content/incident-ops-openenv
%cd /content/incident-ops-openenv
!git log --oneline -5
"""

# ============================================================
# CELL 3: Smoke-test environment (no GPU needed)
# ============================================================
"""
import sys
sys.path.insert(0, "/content/incident-ops-openenv")

from env.environment import DevOpsEnv
from graders.grader import compute_score

for task in ["easy", "medium", "hard"]:
    env = DevOpsEnv(task=task)
    obs = env.reset()
    _, r, done, info = env.step("acknowledge_incident")
    score, breakdown = compute_score(task, env._state)
    print(f"[{task}] step_reward={r:.2f} score={score:.2f} breakdown={breakdown}")

print("\\n✅ Environment smoke test PASSED")
"""

# ============================================================
# CELL 4: Smoke-test reward functions (no GPU)
# ============================================================
"""
import json
from train import (
    format_reward_func,
    step_reward_func,
    anti_cheat_reward_func,
    task_alignment_reward_func,
    generate_prompts,
)
from env.models import VALID_ACTIONS
from env.environment import DevOpsEnv

# format reward
assert format_reward_func([], [[{"content": "no_op"}]]) == [1.0]
assert format_reward_func([], [[{"content": "banana"}]]) == [0.0]

# anti-cheat
assert anti_cheat_reward_func([], [[{"content": "no_op"}]]) == [-0.2]
assert anti_cheat_reward_func([], [[{"content": "rollback_auth_deploy"}]]) == [0.1]

# task alignment
env = DevOpsEnv(task="easy")
state = env.reset()
prompts = [[{"role": "system", "content": "sys"},
            {"role": "user", "content": json.dumps(state)}]]
r = task_alignment_reward_func(prompts, [[{"content": "inspect_deploy_history"}]])
assert r == [0.3], f"Expected [0.3], got {r}"

# curriculum dataset
ds = generate_prompts(easy_n=4, medium_n=4, hard_n=2, mid_episode_n=2)
assert len(ds) == 12
tasks = {json.loads(item["prompt"][-1]["content"])["task"] for item in ds}
assert tasks == {"easy", "medium", "hard"}

print("✅ Reward functions and dataset PASSED")
"""

# ============================================================
# CELL 5: QUICK SANITY TRAIN (5 steps only — costs ~2 min)
# Tests the entire train.py pipeline without wasting credits.
# ============================================================
"""
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import GRPOTrainer, GRPOConfig
from train import (
    format_reward_func, step_reward_func,
    anti_cheat_reward_func, task_alignment_reward_func,
    generate_prompts, RewardLoggerCallback,
)
import os

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

PatchDPOTrainer()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=512, dtype=None, load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(
    model, r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

dataset = generate_prompts(easy_n=4, medium_n=4, hard_n=2, mid_episode_n=2)

training_args = GRPOConfig(
    output_dir="outputs_sanity",
    learning_rate=1e-5,
    max_steps=5,                  # ← tiny: just proves it runs
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=1,
    num_generations=2,
    max_prompt_length=256,
    max_completion_length=16,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward_func, step_reward_func,
                  anti_cheat_reward_func, task_alignment_reward_func],
    args=training_args,
    train_dataset=dataset,
    callbacks=[RewardLoggerCallback(log_path="outputs_sanity/reward_log.csv")],
)

trainer.train()
assert os.path.exists("outputs_sanity/reward_log.csv"), "reward_log.csv not written!"

import pandas as pd
df = pd.read_csv("outputs_sanity/reward_log.csv")
print(df)
print("\\n✅ SANITY TRAIN PASSED — pipeline is working. Now run the full train.")
"""

# ============================================================
# CELL 6: FULL TRAIN (100 steps — uses the HF credits)
# Run only after Cell 5 passes.
# ============================================================
"""
# Clean up sanity model to free VRAM
import gc, torch
del model, tokenizer, trainer
gc.collect()
torch.cuda.empty_cache()

# Full training run
%run train.py

import os
assert os.path.exists("outputs_grpo/reward_log.csv")
import pandas as pd
df = pd.read_csv("outputs_grpo/reward_log.csv")
print(f"Training steps logged: {len(df)}")
print(df.tail())
"""

# ============================================================
# CELL 7: Plot reward curve (for demo / blog)
# ============================================================
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs_grpo/reward_log.csv")
df = df[df["reward"] != ""]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("GRPO Training — DevOps Incident Triage Agent", fontsize=14)

reward_cols = [
    ("reward",                          "Overall Reward",         axes[0, 0]),
    ("reward_format_reward_func",       "Format Reward",          axes[0, 1]),
    ("reward_step_reward_func",         "Step Reward",            axes[1, 0]),
    ("reward_task_alignment_reward_func","Task Alignment Reward", axes[1, 1]),
]

for col, title, ax in reward_cols:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.plot(series.index, series.values, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Reward curve saved to reward_curve.png")
"""

# ============================================================
# CELL 8: Before/After comparison (the demo table)
# ============================================================
"""
!python compare_inference.py \
    --base-model unsloth/Llama-3.2-1B-Instruct \
    --trained-model ./trained_sre_agent
"""

# ============================================================
# CELL 9: Test the FastAPI server (optional, for Space testing)
# ============================================================
"""
import subprocess, time, httpx

proc = subprocess.Popen(
    ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(3)

base = "http://localhost:7860"

# Health
r = httpx.get(f"{base}/health"); print("health:", r.json())

# Tasks
r = httpx.get(f"{base}/tasks"); print("tasks:", [t["name"] for t in r.json()["tasks"]])

# Reset
r = httpx.post(f"{base}/reset", json={"task": "hard", "session_id": "demo"})
print("reset:", r.json()["observation"]["incident_title"])

# Step
r = httpx.post(f"{base}/step", json={"name": "inspect_auth_logs", "session_id": "demo"})
print("step reward:", r.json()["reward"])

# Score
r = httpx.get(f"{base}/score?session_id=demo"); print("score:", r.json())

proc.terminate()
print("\\n✅ FastAPI server test PASSED")
"""
