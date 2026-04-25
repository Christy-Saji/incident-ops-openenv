"""
Direct HF Space deployment — bypasses git entirely.
Uploads every file via the Hub API, which creates a real commit
and forces HF Space to rebuild.

Usage:  python hf_deploy.py
"""
import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
SPACE_ID  = "chritsysajii/incident-ops-openenv"

# Files to SKIP (binary / training-only / local-only)
SKIP = {
    "reward_curve.png",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "devopsenv",
    "outputs_grpo",
    "outputs_sft",
    "outputs_sanity",
    "trained_sre_agent",
    "hf_deploy.py",   # don't upload this script itself
}

REPO_ROOT = Path(__file__).parent

def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP or part.endswith(".pyc"):
            return True
    if path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return True
    return False

def main():
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in the environment before running hf_deploy.py")

    api = HfApi(token=HF_TOKEN)

    print(f"Uploading to Space: {SPACE_ID}")
    files = []
    for f in REPO_ROOT.rglob("*"):
        if f.is_file() and not should_skip(f.relative_to(REPO_ROOT)):
            files.append(f)

    print(f"Found {len(files)} files to upload")

    # Upload all at once as a single commit — triggers ONE rebuild
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=SPACE_ID,
        repo_type="space",
        ignore_patterns=[
            "*.png", "*.jpg", "*.pyc", "__pycache__",
            ".git/*", "*.git", "venv/*", ".venv/*",
            "outputs_grpo/*", "outputs_sft/*", "trained_sre_agent/*",
            "reward_curve.png", "hf_deploy.py",
        ],
        commit_message="deploy: fresh upload via HF API — force rebuild with latest code",
    )
    print("✅ Upload complete — Space is rebuilding now.")
    print(f"   View at: https://huggingface.co/spaces/{SPACE_ID}")

if __name__ == "__main__":
    main()
