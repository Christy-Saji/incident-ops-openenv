"""GRPO reward signal functions for the Incident Ops environment.

Each function follows the TRL GRPOTrainer signature:
    f(prompts, completions, **kwargs) -> List[float]

Tier 1, Phase B replaced the original 9 hand-tuned signals with exact Q*
(training/qstar.py) as the primary signal, since it is the grader itself,
maximised -- alignment with compute_score is structural rather than
something to tune toward. See docs/prompts/tier1.md.

Active stack (ALL_REWARD_FUNCTIONS / CORE_REWARD_FUNCTIONS):
  format_reward_func   — bare valid action string check (+0.1 / 0.0 / -0.4)
  qstar_reward_func    — primary signal: Q*(s,a) - max_a' Q*(s,a')

Both are <= their ceiling by construction and the stack tops out at +0.1, so a
healthy GRPO run has a NEGATIVE logged reward rising toward +0.1. Read
rewards/qstar_reward_func/mean (target 0.0) as the measure of policy quality;
the total is not on a 0-1 scale and never was.

diversity_reward_func was a third member of this stack for one full 500-step
run and has been retired -- it pays for entropy without reference to
correctness and ranks unanimous optimal play below diverse wrong play. It is
kept importable and tested; see its docstring.

The original 7 (LEGACY_REWARD_FUNCTIONS) are retained, individually tested,
but no longer wired into training -- see that constant's docstring below.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import List

from env.environment import DevOpsEnv
from env.models import VALID_ACTIONS
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS
from training.qstar import QStarTable, _rebuild_env, canonical_key, solve_cached

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_action(text: str) -> str | None:
    """Extract the first valid action from raw model output.

    Handles outputs like:
      - "inspect_deploy_history"
      - "Action: inspect_deploy_history"
      - JSON snippets containing an action token
    """
    if not isinstance(text, str):
        return None

    cleaned = text.strip().lower().replace("`", " ").replace("\n", " ")
    if cleaned in VALID_ACTIONS:
        return cleaned

    best_action = None
    best_pos = None
    for action in VALID_ACTIONS:
        match = re.search(rf"\b{re.escape(action)}\b", cleaned)
        if match:
            if best_pos is None or match.start() < best_pos:
                best_action = action
                best_pos = match.start()
    return best_action


def _extract_state_from_prompt(prompt: list[dict]) -> dict:
    """Best-effort parse of the user-state JSON from a chat prompt."""
    try:
        if not prompt:
            return {}
        return json.loads(prompt[-1]["content"])
    except Exception:
        return {}


def _replay_env_from_prompt(prompt: list[dict]) -> tuple[DevOpsEnv, dict, str]:
    """Reconstruct approximate environment state by replaying prompt history.

    Preference order for replayed history:
      1) actions_taken  (full episode history)
      2) recent_actions (short sliding window fallback)
    """
    state_dict = _extract_state_from_prompt(prompt)
    task = state_dict.get("task", "easy")
    if task not in TASK_CONFIGS:
        task = "easy"

    env = DevOpsEnv(task=task)
    env.reset()

    history = state_dict.get("actions_taken", state_dict.get("recent_actions", []))
    if not isinstance(history, list):
        history = []

    for prev_action in history:
        if prev_action in VALID_ACTIONS:
            env.step(prev_action)

    return env, state_dict, task


# Actions that are never task-useful
_FILLER_ACTIONS = {"acknowledge_incident"}
_NO_VALUE_ACTIONS = {"no_op"}

# All inspect and mitigation action names (used in multiple reward funcs)
_INSPECT_ACTIONS = {
    "inspect_auth_logs", "inspect_db_metrics", "inspect_deploy_history",
    "inspect_network_topology", "inspect_memory_profile", "inspect_disk_usage",
}
_MIT_ACTIONS = {
    "rollback_auth_deploy", "rollback_service_deploy", "restart_auth_service",
    "scale_db_cluster", "flush_cache", "shift_traffic_canary", "withdraw_bgp_route",
    "archive_old_logs", "reduce_log_verbosity",
}


# ---------------------------------------------------------------------------
# 1. Format reward
# ---------------------------------------------------------------------------

# Decoration stripped from the ends of a completion before asking whether it is a
# *bare* action: backticks, quotes, markdown emphasis, trailing punctuation.
_BARE_DECORATION = " \t\n\r`\"'*.:"


def format_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 1: Did the model output a single BARE valid action string?

    Three tiers, not two:

        +0.1  the completion IS the action, modulo surrounding decoration
         0.0  a valid action is in there, but padded with prose
        -0.4  no valid action at all

    Reduced from 1.0 → 0.1 so this baseline signal no longer dominates the
    task-specific rewards.

    Why the middle tier exists. The earlier two-tier version paid +0.1 for any
    completion ``extract_action`` could find an action in *anywhere*, and both
    consequences were visible in the 500-step run:

      1. It never varied within a GRPO group — logged mean 0.100000 with
         std 0.000000 on ~90% of steps. GRPO's advantage is
         ``(r_i - group_mean) / group_std``, so a reward that is uniform across
         a group contributes exactly ZERO to every gradient in it. The signal
         was decorative; the effective stack was one function short of what it
         looked like.
      2. Because ``extract_action`` regex-searches the whole string, a 32-token
         ramble that happened to contain a valid action name scored the same
         +0.1 as a clean one-token answer. Nothing anywhere in the stack
         penalised length, and from ~step 320 the policy drifted into rambling
         until every rollout hit ``max_completion_length`` having never emitted
         EOS at all (clipped_ratio 1.0, mean_terminated_length 0.0).

    Ranking the bare answer above the padded one addresses both: it varies
    within a group whenever some rollouts are clean and some are not, so it
    survives the group-mean baseline, and it puts real pressure back on brevity.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] or ""
        action = extract_action(text)
        if action not in VALID_ACTIONS:
            rewards.append(-0.4)
        elif text.lower().strip(_BARE_DECORATION) == action:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# 2. Step reward
# ---------------------------------------------------------------------------

def step_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 2: Does this action improve the incident state?

    Restores approximate mid-episode state by replaying recent_actions from the
    serialised observation in the user message before scoring the candidate action.
    NaN safety: the entire env instantiation + step is wrapped in try/except.
    Return value is explicitly clamped to [-1.0, 1.0].
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        try:
            env, _, _ = _replay_env_from_prompt(prompt)
            _, step_reward, _, _ = env.step(action)
            rewards.append(float(max(-1.0, min(1.0, step_reward))))
        except Exception:
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------------------------
# 3. Anti-cheat reward
# ---------------------------------------------------------------------------

def anti_cheat_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 3: Penalise reward hacking (loop detection + no-op spam).

    Priority order (first match wins):
      1. Loop (last 3 all identical)            → -0.8
      2. Consecutive repeat (last action == now) → -0.7
      3. no_op                                  → -0.6
      4. acknowledge spam (2nd+ time)           → -0.6
      5. Action already done somewhere          → -0.4
      6. Novel action                           → +0.2

    NOTE: resolve_incident is intentionally NOT penalised here.
    It is handled with full episode context (stable/unstable, mit_complete)
    in terminal_outcome_reward_func, which gives +3.0 for a correct terminal
    call and -1.5 for a premature one. A blanket penalty here would contradict
    that signal and suppress the learning of the terminal action.
    A second call to resolve_incident is still caught by case 5 above
    (action in all_actions_taken → -0.4).
    """
    rewards = []
    for i, completion in enumerate(completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        prompt = prompts[i] if i < len(prompts) else []
        state_dict = _extract_state_from_prompt(prompt)
        all_actions_taken: List[str] = state_dict.get(
            "actions_taken", state_dict.get("recent_actions", [])
        )
        if not isinstance(all_actions_taken, list):
            all_actions_taken = []

        last_action = all_actions_taken[-1] if all_actions_taken else None

        if len(all_actions_taken) >= 3 and len(set(all_actions_taken[-3:])) == 1:
            rewards.append(-0.8)
        elif action == last_action:
            rewards.append(-0.7)
        elif action in _NO_VALUE_ACTIONS:
            rewards.append(-0.6)
        elif action in _FILLER_ACTIONS and all_actions_taken.count(action) >= 1:
            rewards.append(-0.6)
        elif action in all_actions_taken:
            rewards.append(-0.4)
        else:
            rewards.append(0.2)
    return rewards


# ---------------------------------------------------------------------------
# 4. Task alignment reward
# ---------------------------------------------------------------------------

def task_alignment_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 4: Bonus for task-correct diagnostics/mitigations; penalty for wrong ones.

    Action categories and rewards:
      required_diag  (not yet done)  → +0.40
      required_mit   (not yet done)  → +0.40
      required_diag/mit (already done)→ -0.25  (discourage duplicate)
      good_followups (comms/resolve)  → +0.10
      inspect_* NOT in required_diag  → -0.50  (wrong diagnostic)
      other task-specific actions wrong→ -0.35
      no_op                           → -0.50
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(-0.3)
            continue

        task = "easy"
        all_actions_taken: List[str] = []
        try:
            state_dict = json.loads(prompt[-1]["content"])
            task = state_dict.get("task", "easy")
            all_actions_taken = state_dict.get(
                "actions_taken", state_dict.get("recent_actions", [])
            )
            if not isinstance(all_actions_taken, list):
                all_actions_taken = []
        except Exception:
            pass

        config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
        required_diag  = config.get("required_diagnostics", [])
        required_mit   = config.get("required_mitigations", [])
        good_followups = config.get("good_followups", [])

        if action in required_diag and action not in all_actions_taken:
            rewards.append(0.40)
        elif action in required_mit and action not in all_actions_taken:
            rewards.append(0.40)
        elif action in required_diag or action in required_mit:
            rewards.append(-0.25)
        elif action in good_followups:
            rewards.append(0.10)
        elif action in _INSPECT_ACTIONS:
            rewards.append(-0.50)
        elif action in _MIT_ACTIONS:
            rewards.append(-0.35)
        elif action in _NO_VALUE_ACTIONS:
            rewards.append(-0.50)
        else:
            rewards.append(-0.15)

    return rewards


# ---------------------------------------------------------------------------
# 5. Sequence progress reward
# ---------------------------------------------------------------------------

def sequence_progress_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 5: Enforce the correct SRE workflow order.

    investigate → mitigate → communicate → resolve
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")

        if action not in VALID_ACTIONS:
            rewards.append(0.0)
            continue

        state_dict = _extract_state_from_prompt(prompt)
        task = state_dict.get("task", "easy")
        if task not in TASK_CONFIGS:
            task = "easy"
        all_actions_taken: List[str] = state_dict.get(
            "actions_taken", state_dict.get("recent_actions", [])
        )
        if not isinstance(all_actions_taken, list):
            all_actions_taken = []

        config = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
        required_diag = set(config.get("required_diagnostics", []))
        required_mit  = set(config.get("required_mitigations", []))
        done_so_far   = set(all_actions_taken)

        diag_complete = required_diag.issubset(done_so_far)
        mit_complete  = required_mit.issubset(done_so_far)
        is_resolve    = action == "resolve_incident"

        if action in required_diag and not diag_complete:
            rewards.append(0.25)
        elif action in required_mit and diag_complete and not mit_complete:
            rewards.append(0.35)
        elif action in required_mit and not diag_complete:
            rewards.append(-0.35)
        elif is_resolve and diag_complete and mit_complete:
            rewards.append(0.60)
        elif is_resolve and not (diag_complete and mit_complete):
            rewards.append(-0.80)
        else:
            rewards.append(-0.05)

    return rewards


# ---------------------------------------------------------------------------
# 6. Progress delta reward
# ---------------------------------------------------------------------------

def progress_delta_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 6: Dense reward for measurable task progress.

    Compares completion of required diagnostics/mitigations before vs after
    applying the candidate action in the replayed environment.
    """
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")
        if action not in VALID_ACTIONS:
            rewards.append(-0.4)
            continue
        try:
            env, _, task = _replay_env_from_prompt(prompt)
            cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
            req_diag = set(cfg.get("required_diagnostics", []))
            req_mit  = set(cfg.get("required_mitigations", []))

            before = set(env._state.get("actions_taken", []))
            before_progress = (
                len(req_diag.intersection(before)) +
                len(req_mit.intersection(before))
            )

            env.step(action)
            after = set(env._state.get("actions_taken", []))
            after_progress = (
                len(req_diag.intersection(after)) +
                len(req_mit.intersection(after))
            )

            delta = after_progress - before_progress
            if delta > 0:
                rewards.append(0.5)
            elif action in {"acknowledge_incident", "post_status_update"}:
                rewards.append(-0.6)
            elif action in before:
                rewards.append(-0.5)
            else:
                rewards.append(-0.35)
        except Exception:
            rewards.append(-0.4)
    return rewards


# ---------------------------------------------------------------------------
# 7. Communication gate reward
# ---------------------------------------------------------------------------

def communication_gate_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 7: Gate communication actions until technical progress exists."""
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")
        if action not in VALID_ACTIONS:
            rewards.append(-0.3)
            continue
        try:
            env, _, task = _replay_env_from_prompt(prompt)
            cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
            req_diag = set(cfg.get("required_diagnostics", []))
            req_mit  = set(cfg.get("required_mitigations", []))
            taken    = set(env._state.get("actions_taken", []))
            diag_done = len(req_diag.intersection(taken)) > 0
            mit_done  = len(req_mit.intersection(taken)) > 0
            tech_progress = diag_done or mit_done

            if action == "acknowledge_incident":
                rewards.append(-0.8 if "acknowledge_incident" in taken else 0.05)
            elif action == "post_status_update":
                rewards.append(0.15 if tech_progress else -0.8)
            elif action == "resolve_incident":
                rewards.append(0.2 if (diag_done and mit_done) else -1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(-0.3)
    return rewards


# ---------------------------------------------------------------------------
# 8. Terminal outcome reward
# ---------------------------------------------------------------------------

def terminal_outcome_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 8: Primary outcome-first terminal objective.

    Reconstructs episode state from prompt history, simulates the proposed
    action, and rewards score progress + resolution. Strongly favors true
    resolution and penalises farming / looping.
    """
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")
        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue

        try:
            env, _, task = _replay_env_from_prompt(prompt)
            pre_score, _ = compute_score(task, env._state)
            _, _, _, info = env.step(action)
            post_score, _ = compute_score(task, env._state)

            score_delta = float(post_score - pre_score)
            state = env._state
            cfg = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"])
            required_mit  = set(cfg.get("required_mitigations", []))
            required_diag = set(cfg.get("required_diagnostics", []))
            done_actions  = set(state.get("actions_taken", []))
            mit_complete  = required_mit.issubset(done_actions)
            stable        = state.get("incident_phase") in {"monitoring", "resolved"}
            resolved      = bool(state.get("resolved", False))

            reward = 1.5 * score_delta

            history = state.get("actions_taken", [])
            comms_done = "post_status_update" in history

            if resolved and stable and mit_complete:
                reward += 3.0 if comms_done else 1.0
            elif action == "resolve_incident" and info.get("error") == "incident_not_stable":
                reward -= 1.5
            elif action == "resolve_incident" and not (stable and mit_complete):
                reward -= 1.3

            # history already set above for comms gate
            if len(history) >= 4 and len(set(history[-4:])) == 1:
                reward -= 1.0
            if action in _FILLER_ACTIONS and len(history) > 2:
                reward -= 0.8
            elif action in _FILLER_ACTIONS:
                reward -= 0.35
            if history.count(action) > 1:
                reward -= 0.6

            if action in required_diag and action not in set(history[:-1]):
                reward += 0.4
            if action in required_mit and action not in set(history[:-1]):
                reward += 0.5

            rewards.append(float(max(-2.5, min(3.5, reward))))
        except Exception:
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------------------------
# Q* reward (Tier 1, Phase B) -- primary training signal
# ---------------------------------------------------------------------------

_QSTAR_TABLES: dict[str, QStarTable] = {}


def _qstar_table(task: str) -> QStarTable:
    table = _QSTAR_TABLES.get(task)
    if table is None:
        table = solve_cached(task)
        _QSTAR_TABLES[task] = table
    return table


def _qstar_best_value(table: QStarTable, task: str, state: dict) -> float:
    """max_a' Q*(state, a'). table.values[key] already IS this max, by
    construction of the backward induction in training/qstar.py::solve."""
    key = canonical_key(state, state["step_count"])
    if key in table.values:
        return table.values[key]
    score, _ = compute_score(task, state)
    return score


def _qstar_action_value(table: QStarTable, task: str, state: dict, action: str) -> float:
    """Q*(state, action).

    Exact whenever `state` and `action` are both in the precomputed table
    (true for every state training/dataset.py's eps-greedy walk can produce,
    since it samples off-path actions from the same training.qstar.relevant_actions
    set the table was solved over). Falls back to a one-step simulate + lookup
    of the successor's cached value for any other action the policy might
    emit during GRPO sampling (the policy's action vocabulary is the full 19,
    not just the relevant set) -- exact if the successor state is itself
    cached, a safe (compute_score-based) underestimate of the true reachable
    value otherwise.
    """
    key = canonical_key(state, state["step_count"])
    qvals = table.q_values.get(key)
    if qvals and action in qvals:
        return qvals[action]

    env = _rebuild_env(task, state)
    env.step(action)
    nkey = canonical_key(env._state, env.current_step)
    if nkey in table.values:
        return table.values[nkey]
    score, _ = compute_score(task, env._state)
    return score


def qstar_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Primary reward: reward(s,a) = Q*(s,a) - max_a' Q*(s,a').

    <= 0 everywhere, exactly 0 on every action that lies on a score-maximising
    path from the replayed state. This is the grader itself, maximised --
    alignment with compute_score is structural rather than something to tune
    toward (see training/qstar.py and docs/prompts/tier1.md Phase B).
    """
    rewards: List[float] = []
    for prompt, completion in zip(prompts, completions):
        action = extract_action(completion[0]["content"] or "")
        if action not in VALID_ACTIONS:
            rewards.append(-0.5)
            continue
        try:
            env, _, task = _replay_env_from_prompt(prompt)
            table = _qstar_table(task)
            state = env._state
            best = _qstar_best_value(table, task, state)
            q_sa = _qstar_action_value(table, task, state, action)
            rewards.append(float(max(-1.0, min(0.0, q_sa - best))))
        except Exception:
            rewards.append(-0.5)
    return rewards


# ---------------------------------------------------------------------------
# 9. Diversity reward
# ---------------------------------------------------------------------------

def diversity_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward 9: Penalise GRPO group-level mode collapse. **RETIRED — do not
    put this back in a training stack.**

    It was removed from ALL_REWARD_FUNCTIONS after the first full 500-step GRPO
    run, because it pays for entropy without reference to correctness and so
    fights qstar_reward_func directly.

    The arithmetic. A group of 8 rollouts that all emit the SAME Q*-optimal
    action scores 0.1 (format) + 0.0 (qstar) - 0.5 (this) = **-0.4** — that is
    perfect play, scored at the bottom of the range. A group that spreads over 8
    different actions, most of them wrong, scores around 0.1 + -0.2 + 0.0 =
    -0.1. Same state, worse play, +0.3 more reward. Step 1 of the run logged
    exactly this: qstar 0.000000 / std 0.000000, diversity -0.500000, total
    -0.400000. Steps 15, 23, 31, 33, 43, 71, 87 and 93 are identical.

    Its gradient points at whichever action is *rarest in the group*, which is
    independent of whether that action is any good, and its within-group spread
    was comparable to qstar's for most of the run. Windowed over that run,
    total reward improved from about -0.46 (steps 1-50) to -0.16 (steps
    451-500), but qstar — the only term that measures policy quality — moved
    only -0.21 → -0.155. Roughly 70% of the apparent improvement was the model
    learning to sample more randomly, and under 20% was it getting better at
    the task.

    Collapse onto the *correct* action is the goal, not a pathology. Collapse
    onto a wrong action does produce a zero-variance group and hence no
    gradient, which is the real hazard this was meant to address — but the
    remedies for that are the sampling temperature, num_generations, and the
    eps-greedy state spread, not a reward that pays for disagreement. Detect it
    via the reward_std column logged by RewardLoggerCallback.

    Kept importable and unit-tested for the record and for ablation.

    Below: why, given that it *was* wired in, it had to be per-completion.

    IMPORTANT — why this is per-completion and must stay that way:

    GRPO computes the advantage as (r_i - group_mean) / group_std. Any reward
    that assigns the *same* value to every completion in a group shifts the mean
    by exactly that value and therefore contributes exactly ZERO to every
    advantage. It is invisible to the optimiser.

    The previous version returned a single uniform value for the whole group
    (-0.5 on collapse, 0.0 otherwise), so despite being the designated
    anti-collapse mechanism it could never influence a single gradient. Its only
    observable effect was to make the logged mean reward dip on collapse, which
    made the reward curve *look* responsive while nothing was happening.

    The fix: penalise each completion by how many siblings duplicate it. A
    completion echoing the majority action is now scored below a minority one,
    so the value varies within the group and survives the group-mean baseline.

        reward_i = -0.5 * (count(action_i) - 1) / (group_size - 1)

    Range [-0.5, 0.0]: 0.0 when an action is unique in the group, -0.5 when
    every completion in the group emitted it.

    Total collapse (all completions identical) still yields a uniform value and
    therefore zero advantage. That is correct and unavoidable — when every
    sample is the same there is genuinely no signal to learn from. Detecting it
    is the job of the reward_std column logged by RewardLoggerCallback, not of
    this function.
    """
    actions = [
        extract_action(c[0]["content"] or "") or "<invalid>" for c in completions
    ]
    counts = Counter(actions)
    group_size = len(actions)
    if group_size <= 1:
        return [0.0] * group_size

    # + 0.0 normalises the -0.0 that float negation produces for unique actions,
    # so the logged CSV column reads 0.0 rather than -0.0.
    return [
        round(-0.5 * (counts[action] - 1) / (group_size - 1), 4) + 0.0
        for action in actions
    ]


# ---------------------------------------------------------------------------
# Convenience list — import this in pipeline.py
# ---------------------------------------------------------------------------

# Tier 1, Phase B: qstar_reward_func is the primary training signal.
# format_reward_func rides alongside it as a malformed-output / brevity guard;
# it does not overlap with Q* (it polices output syntax, not action choice) and
# it stays at its +0.1 ceiling on a well-formed batch.
#
# diversity_reward_func was the third member here for the first full 500-step
# run and has been REMOVED -- it pays for entropy regardless of correctness, so
# it scores a group that unanimously plays the Q*-optimal action (-0.4) below
# one that spreads over eight mostly-wrong actions (~-0.1). See that function's
# docstring for the numbers. Anything added to this list from here must be
# per-completion in the strict sense: a completion's reward may not depend on
# its siblings. tests/test_reward_ranking.py asserts that.
#
# The stack's ceiling is +0.1 (format 0.1 + qstar 0.0). qstar_reward_func is
# <= 0 by construction, so a healthy run shows the LOGGED REWARD RISING TOWARD
# +0.1, never above it, and rewards/qstar_reward_func/mean approaching 0.0 --
# that column, not the total, is the measure of policy quality.
#
# Used by training/pipeline.py.
ALL_REWARD_FUNCTIONS = [
    format_reward_func,
    qstar_reward_func,
]

# The reduced set used by colab_training.ipynb: Q* alone, no format/diversity
# shaping. Defined here rather than inline in the notebook so the notebook,
# the pipeline and tests/test_reward_ranking.py cannot drift apart. Both
# stacks are measured by that test.
CORE_REWARD_FUNCTIONS = [
    qstar_reward_func,
]

# The original 9 hand-tuned functions this file used before Tier 1's Q*
# redesign (docs/prompts/tier1.md Phase B). Retired from ALL_REWARD_FUNCTIONS
# and CORE_REWARD_FUNCTIONS -- test_reward_ranking.py showed the reward
# argmax disagreed with the optimal action on roughly half of all training
# states, and reward-grader alignment with this stack was something to
# hand-tune toward rather than a structural property. Kept importable, and
# individually unit-tested in tests/test_reward_functions.py, purely for a
# future ablation comparison against qstar_reward_func.
LEGACY_REWARD_FUNCTIONS = [
    step_reward_func,
    anti_cheat_reward_func,
    task_alignment_reward_func,
    sequence_progress_reward_func,
    progress_delta_reward_func,
    communication_gate_reward_func,
    terminal_outcome_reward_func,
]
