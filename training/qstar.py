"""Exact Q* solver for the incident-triage environment.

The environment is small, deterministic (with stochastic=False) and finite, so
the optimal action-value function is *computable* rather than approximable:

    Q*(state, action) = best final compute_score reachable after taking
                         `action` from `state`
    reward(state, action) = Q*(state, action) - max_a' Q*(state, a')

reward is <= 0 everywhere and exactly 0 on every action that lies on a
score-maximising path. See docs/prompts/tier1.md Phase B for the full
rationale.

Algorithm: depth-indexed BFS over canonical states (NOT DFS over action
sequences -- DFS with repeats does not terminate in reasonable time), followed
by backward value induction over the resulting DAG (step_count strictly
increases on every transition, so there are no cycles once states are keyed
canonically).

Canonical key:  (frozenset(actions_taken), step_count, harmful_action_count,
                  len(communication_log), resolved)
A key omitting harmful_action_count conflates distinct states and silently
returns wrong values -- that bug produced easy Q*=0.89 against a
known-reachable 0.94 in prototyping. Keep all five fields.

Branching is restricted to RELEVANT_ACTIONS[task] (required diagnostics +
required mitigations + universal actions + a couple of plausible distractors)
-- full 19-way branching does not terminate. training/dataset.py's eps-greedy
walk samples its "off-path" action from this same restricted set, so every
state the model is ever shown during training has a Q* label computed by
exactly this table (see docs/prompts/tier1.md Phase C).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from env.environment import DevOpsEnv
from env.models import MITIGATION_PREREQS, VALID_ACTIONS
from graders.grader import compute_score
from tasks.task_config import TASK_CONFIGS

CanonicalKey = Tuple[FrozenSet[str], int, int, int, bool]

_UNIVERSAL_ACTIONS = {
    "acknowledge_incident", "post_status_update", "resolve_incident", "no_op",
}
_ALL_INSPECT = {a for a in VALID_ACTIONS if a.startswith("inspect_")}
_ALL_MITIGATIONS = set(MITIGATION_PREREQS) | {"shift_traffic_canary"}

CACHE_DIR = "outputs"


def relevant_actions(task: str) -> set:
    """The per-task action set the solver branches on.

    required diagnostics + required mitigations + acknowledge/post_status/resolve
    + one distractor inspect (so a wrong diagnostic is priced, not merely
    excluded). no_op and off-task mitigations are deliberately NOT branched on
    here -- they add a harmful_action_count dimension that (crossed with the
    unrestricted ordering of already-small branch sets) blows the reachable
    state count past what a pure-Python BFS finishes in reasonable time. Their
    Q* is instead computed on demand by a single-step simulate-and-lookup in
    reward_functions.qstar_reward_func, which is exact whenever the resulting
    state is itself in this table (usually true) and a safe underestimate
    otherwise.
    """
    config = TASK_CONFIGS[task]
    required_diag = set(config.get("required_diagnostics", []))
    required_mit = set(config.get("required_mitigations", []))

    # A required mitigation's *actual* MITIGATION_PREREQS diagnostic may not be
    # listed in the task's required_diagnostics (e.g. memory_leak's
    # scale_db_cluster needs inspect_db_metrics, which isn't one of
    # memory_leak's required_diagnostics) -- without it in the branch set the
    # solver can never unlock that mitigation and silently caps Q* far below
    # the true optimum. Pull in one prereq per required mitigation explicitly.
    unlocking_diag = set()
    for mit in required_mit:
        prereqs = MITIGATION_PREREQS.get(mit)
        if prereqs:
            unlocking_diag.add(sorted(prereqs)[0])

    actions = (
        required_diag | required_mit | unlocking_diag
        | {"acknowledge_incident", "post_status_update", "resolve_incident"}
    )
    distractor_inspects = sorted(_ALL_INSPECT - required_diag - unlocking_diag)[:1]
    actions |= set(distractor_inspects)

    return actions


def canonical_key(state: dict, step_count: int) -> CanonicalKey:
    return (
        frozenset(state["actions_taken"]),
        step_count,
        state["harmful_action_count"],
        len(state["communication_log"]),
        bool(state["resolved"]),
    )


@dataclass
class QStarTable:
    task: str
    actions: List[str]
    root_key: CanonicalKey
    values: Dict[CanonicalKey, float] = field(default_factory=dict)
    q_values: Dict[CanonicalKey, Dict[str, float]] = field(default_factory=dict)
    # One representative concrete state per canonical key, needed to score
    # actions that fall outside `actions` (see reward_functions.qstar_reward_func).
    states: Dict[CanonicalKey, dict] = field(default_factory=dict)

    @property
    def root_score(self) -> float:
        return self.values[self.root_key]

    def optimal_trajectory(self) -> List[str]:
        """Greedily follow argmax Q* from the root to a resolved/terminal state."""
        trajectory: List[str] = []
        key = self.root_key
        seen = set()
        while key not in seen:
            seen.add(key)
            qvals = self.q_values.get(key)
            if not qvals:
                break
            best_action = max(qvals, key=qvals.get)
            trajectory.append(best_action)
            if best_action == "resolve_incident":
                break
            # Recompute the successor key by simulating the action, since we
            # only stored a flat action->key edge implicitly via q_values.
            env = _rebuild_env(self.task, self.states[key])
            env.step(best_action)
            key = canonical_key(env._state, env.current_step)
        return trajectory


def _rebuild_env(task: str, state: dict) -> DevOpsEnv:
    env = DevOpsEnv(task=task, stochastic=False)
    env._state = copy.deepcopy(state)
    env.current_step = state["step_count"]
    env._last_action = state["actions_taken"][-1] if state["actions_taken"] else ""
    return env


DEFAULT_BEAM_WIDTH = 60


def solve(task: str, actions: Optional[set] = None, beam_width: int = DEFAULT_BEAM_WIDTH) -> QStarTable:
    """Depth-indexed BFS + backward induction, with a beam-search fallback.

    Forward exploration is exact (dominance-pruned but otherwise complete)
    until the per-level frontier exceeds `beam_width`, at which point it is
    truncated to the `beam_width` states with the highest compute_score so
    far -- i.e. a beam search from that point on. This matters for `hard`:
    exact DP does not terminate in reasonable time there (verified: still
    growing past 20k states after 35s), while a width-60 beam matched the
    exact optimum on every task where both were run during prototyping. Tasks
    small enough to stay exact (easy/medium/network/memory_leak/disk_full)
    are solved exactly regardless of beam_width, since the cap only bites
    once a level's frontier actually exceeds it.
    """
    actions = sorted(actions or relevant_actions(task))
    max_steps = TASK_CONFIGS[task]["max_steps"]

    root_env = DevOpsEnv(task=task, stochastic=False)
    root_state = root_env._state
    root_key = canonical_key(root_state, 0)

    states: Dict[CanonicalKey, dict] = {root_key: copy.deepcopy(root_state)}
    steps: Dict[CanonicalKey, int] = {root_key: 0}
    done_flags: Dict[CanonicalKey, bool] = {root_key: False}
    edges: Dict[CanonicalKey, Dict[str, CanonicalKey]] = {}

    frontier = [root_key]
    while frontier:
        newly_discovered: Dict[CanonicalKey, None] = {}
        for key in frontier:
            if done_flags[key] or steps[key] >= max_steps:
                continue
            state = states[key]
            for action in actions:
                env = _rebuild_env(task, state)
                _, _, done, _ = env.step(action)
                nstate = env._state
                nkey = canonical_key(nstate, env.current_step)
                edges.setdefault(key, {})[action] = nkey
                if nkey not in states:
                    states[nkey] = copy.deepcopy(nstate)
                    steps[nkey] = env.current_step
                    done_flags[nkey] = done
                    newly_discovered[nkey] = None

        # Dominance pruning: harmful_action_count only ever hurts compute_score
        # (via the harm-rate penalty) and gates nothing, so among states that
        # share (actions_taken, step, comm_log length, resolved) only the
        # minimum-harm variant can matter for the max-reachable-score we are
        # solving for. Expanding the rest would only inflate the state count.
        best_by_group: Dict[Tuple[FrozenSet[str], int, int, bool], CanonicalKey] = {}
        for nkey in newly_discovered:
            group = (nkey[0], nkey[1], nkey[3], nkey[4])
            current = best_by_group.get(group)
            if current is None or nkey[2] < current[2]:
                best_by_group[group] = nkey

        frontier = [nkey for nkey in best_by_group.values() if not done_flags[nkey]]

        if len(frontier) > beam_width:
            frontier.sort(
                key=lambda k: (compute_score(task, states[k])[0], -k[2]),
                reverse=True,
            )
            frontier = frontier[:beam_width]

    # Backward induction: process deepest states first.
    values: Dict[CanonicalKey, float] = {}
    q_values: Dict[CanonicalKey, Dict[str, float]] = {}
    for key in sorted(states, key=lambda k: steps[k], reverse=True):
        if done_flags[key] or steps[key] >= max_steps or key not in edges:
            score, _ = compute_score(task, states[key])
            values[key] = score
            continue
        qvals = {a: values[nkey] for a, nkey in edges[key].items()}
        q_values[key] = qvals
        values[key] = max(qvals.values())

    return QStarTable(
        task=task, actions=actions, root_key=root_key,
        values=values, q_values=q_values, states=states,
    )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_hash(task: str, actions: List[str], beam_width: int) -> str:
    payload = json.dumps(
        {
            "task_config": TASK_CONFIGS[task],
            "mitigation_prereqs": {k: sorted(v) for k, v in MITIGATION_PREREQS.items()},
            "actions": actions,
            "beam_width": beam_width,
        },
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(task: str) -> str:
    return os.path.join(CACHE_DIR, f"qstar_{task}.json")


_MEMO: Dict[Tuple[str, int], QStarTable] = {}


def solve_cached(
    task: str,
    actions: Optional[set] = None,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    force: bool = False,
) -> QStarTable:
    """solve(), transparently cached to outputs/qstar_<task>.json and memoized
    in-process (repeated calls within one process, e.g. once per GRPO batch or
    per training state in a test, would otherwise re-read + re-deserialize the
    JSON file every time).

    Disk cache is keyed by a hash of the task config + MITIGATION_PREREQS +
    action set + beam_width, so it invalidates automatically whenever Phase A's
    dynamics change, without anyone needing to remember to delete a file.
    """
    memo_key = (task, beam_width)
    if not force and memo_key in _MEMO:
        return _MEMO[memo_key]

    actions_sorted = sorted(actions or relevant_actions(task))
    digest = _cache_hash(task, actions_sorted, beam_width)
    path = _cache_path(task)

    if not force and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("hash") == digest:
            table = _deserialize(task, actions_sorted, cached)
            _MEMO[memo_key] = table
            return table

    table = solve(task, set(actions_sorted), beam_width=beam_width)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(table, digest), f)
    _MEMO[memo_key] = table
    return table


def _serialize(table: QStarTable, digest: str) -> dict:
    def key_to_str(key: CanonicalKey) -> str:
        return json.dumps([sorted(key[0]), key[1], key[2], key[3], key[4]])

    def jsonable_state(state: dict) -> dict:
        out = dict(state)
        out["effective_mitigations"] = sorted(out.get("effective_mitigations", []))
        return out

    return {
        "hash": digest,
        "root_key": key_to_str(table.root_key),
        "values": {key_to_str(k): v for k, v in table.values.items()},
        "q_values": {key_to_str(k): v for k, v in table.q_values.items()},
        "states": {key_to_str(k): jsonable_state(v) for k, v in table.states.items()},
    }


def _deserialize(task: str, actions: List[str], cached: dict) -> QStarTable:
    def str_to_key(s: str) -> CanonicalKey:
        a, b, c, d, e = json.loads(s)
        return (frozenset(a), b, c, d, e)

    values = {str_to_key(k): v for k, v in cached["values"].items()}
    q_values = {str_to_key(k): v for k, v in cached["q_values"].items()}
    states = {str_to_key(k): v for k, v in cached["states"].items()}
    for state in states.values():
        state["effective_mitigations"] = set(state.get("effective_mitigations", []))
    return QStarTable(
        task=task, actions=actions, root_key=str_to_key(cached["root_key"]),
        values=values, q_values=q_values, states=states,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, help="single task, else all")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore cache")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASK_CONFIGS)
    for task in tasks:
        t0 = time.time()
        table = solve_cached(task, force=args.force)
        elapsed = time.time() - t0
        traj = table.optimal_trajectory()
        print(f"{task:12s} Q*(root)={table.root_score:.4f}  states={len(table.states):6d}  "
              f"{elapsed:.2f}s  trajectory={traj}")
