"""Shared prompt construction for training, evaluation and comparison.

This module is deliberately dependency-light — it imports only ``json`` and
``env.models``. No ``datasets``, no ``torch``, no ``transformers``. That lets
``scripts/evaluate.py`` and ``compare_inference.py`` build prompts that are
byte-identical to the training ones without pulling in the ``[train]`` extra.

Every prompt in the project must be built here. Three copies of the system
prompt used to exist (training/dataset.py, scripts/evaluate.py,
compare_inference.py) and they had drifted apart, so the model was evaluated on
a different prompt than it was trained on.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from env.models import VALID_ACTIONS

# Observation keys stripped before serialising into a prompt.
#
# available_actions is ~420 chars of the ~1075-char observation JSON and simply
# repeats the action list that is already in the system prompt. It stays on the
# Observation schema (the web UI and the env tests both read it) but paying for
# it twice in every prompt is what pushed prompts past max_prompt_length and got
# them left-truncated during GRPO.
PROMPT_EXCLUDED_OBS_KEYS = ("available_actions",)


SYSTEM_PROMPT = (
    "You are an On-call SRE resolving a live infrastructure incident. "
    "Select the single NEXT best action to take.\n"
    "Valid actions: {actions}\n\n"
    "Rules:\n"
    "1. NEVER repeat an action that already appears in 'actions_taken' or 'recent_actions'.\n"
    "2. Follow the SRE workflow in order: DIAGNOSE first, then MITIGATE, then COMMUNICATE "
    "(post_status_update), then RESOLVE.\n"
    "3. Only call resolve_incident when all services are running and mitigations are done.\n"
    "Output ONLY the action name. No explanation."
).format(actions=", ".join(VALID_ACTIONS))


def serialize_observation(state: Dict[str, Any]) -> str:
    """Serialise an observation into the exact JSON the model is shown.

    Drops PROMPT_EXCLUDED_OBS_KEYS. The observation already carries
    ``actions_taken`` (see env/models.py), so no history needs to be injected
    by the caller.
    """
    payload = {k: v for k, v in state.items() if k not in PROMPT_EXCLUDED_OBS_KEYS}
    return json.dumps(payload)


def build_prompt(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the [system, user] chat prompt for one observation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": serialize_observation(state)},
    ]
