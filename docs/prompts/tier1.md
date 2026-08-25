# Tier 1 — Make GRPO Able to Learn

> Tier 0 (merged as [PR #2](https://github.com/Christy-Saji/incident-ops-openenv/pull/2), `e136673`) fixed the
> infrastructure, the eval, and the config damping. CI is green for the first time.
> This plan fixes the reason the reward curve was flat.

## Context

Tier 0 left one tripwire deliberately failing:

```
[all9]  reward argmax matches optimal action: 21/41 states
[core3] reward argmax matches optimal action: 22/41 states
```

Investigating *why* those 20 states disagree turned up something bigger than a
reward-tuning problem. Three layers of this project each define "the right
action" differently, and **all three are wrong in different ways**:

| Layer | Claims the optimal action is… | Verdict |
|---|---|---|
| `compute_score` (grader) | whatever maximises the weighted score | **the exploit is here** |
| `optimal_actions` (SFT labels) | a hand-written sequence per task | **suboptimal — measurably** |
| 9 reward functions | diagnose → mitigate → communicate → resolve | enforces an order the grader does not reward |

Evidence gathered this session, all reproducible against the merged code:

**1. The hand-written `optimal_actions` are not optimal.** Deleting
`acknowledge_incident` *raises* the final score on 5 of 6 tasks:

```
task         full optimal    no ack
medium          0.93/6s      0.95/5s
hard            0.90/10s     0.92/9s
network         0.96/7s      0.97/6s
memory_leak     0.93/7s      0.94/6s
disk_full       0.92/7s      0.93/6s
easy            0.94/4s      0.92/3s   <- only task where it pays
```

The reward functions were *right* to rank it last; the SFT labels were wrong. The
two phases were fighting, and SFT was the one holding the wrong end.

**2. The benchmark can be reward-hacked by skipping diagnosis.** A beam search
over the env beats the hand-written trajectory on 5 of 6 tasks, and every winning
sequence **mitigates before diagnosing**:

```
medium  beam 0.95: [scale_db_cluster, inspect_db_metrics, shift_traffic_canary, post_status_update, resolve]
network beam 0.97: [withdraw_bgp_route, shift_traffic_canary, inspect_deploy_history, ...]
```

`compute_score` credits diagnosis by set-membership, and `_apply_action` never
gates a mitigation on the matching diagnostic — so guessing the fix and
back-filling the inspects resolves in fewer steps and earns more efficiency
credit. **A capable model trained against this grader will learn to skip
triage**, which is the one skill the benchmark exists to measure.

**3. Ties are being scored as errors.** `rollback_auth_deploy` vs
`shift_traffic_canary` on `hard[4]` differ by 0.06 out of ~2.5. Both are required
mitigations; the grader only checks set membership, so their order is genuinely
arbitrary. The ranking test treats one blessed ordering as correct and marks the
other wrong — the test itself is over-specified.

### Intended outcome

Stop hand-tuning reward functions against a grader that rewards the wrong thing.
Instead: make the environment reward real triage, derive the optimal policy by
search rather than by hand, and define the reward as **exact Q\*** so that
reward-grader alignment is a property of the construction rather than something
to be tuned toward.

### Decisions taken (confirmed)

- **Gate mitigations on diagnosis.** Env becomes v2; every published score changes.
- **Move to full-episode rollouts.** Sequenced last, behind a go/no-go gate.
- **Hold out `memory_leak` + `disk_full`.** Train on `easy`/`medium`/`hard`/`network`.

---

## Phase A — Fix the environment (this is the load-bearing change)

Nothing downstream is meaningful until diagnose-before-mitigate is genuinely
optimal, because "the optimal action" is the quantity every later phase depends on.

**`env/models.py`** — add a module-level prerequisite map. The relationship is
physical, not task-specific (you cannot archive logs before looking at disk usage),
so it belongs beside `VALID_ACTIONS`, not in per-task config:

```python
MITIGATION_PREREQS: dict[str, set[str]] = {
    "rollback_auth_deploy":   {"inspect_deploy_history", "inspect_auth_logs"},
    "rollback_service_deploy":{"inspect_deploy_history", "inspect_memory_profile"},
    "restart_auth_service":   {"inspect_auth_logs"},
    "scale_db_cluster":       {"inspect_db_metrics"},
    "flush_cache":            {"inspect_db_metrics"},
    "withdraw_bgp_route":     {"inspect_network_topology"},
    "archive_old_logs":       {"inspect_disk_usage"},
    "reduce_log_verbosity":   {"inspect_disk_usage"},
    # shift_traffic_canary is generic load-shedding — any completed diagnostic unlocks it
}
```
Semantics: **any one** listed diagnostic unlocks the mitigation.

**`env/environment.py`** — in `_apply_action`, before applying any mitigation
effect, check the prereq against `self._state["actions_taken"]`. If unmet: apply
no state change, increment `harmful_action_count`, append a finding explaining the
blind fix, and set `info["error"] = "mitigation_without_diagnosis"`. Add the
matching penalty branch in `step()` alongside the existing `incident_not_stable`
one. `_apply_background_dynamics` must respect this too — it currently keys off
`set(actions_taken)` and would re-apply the effect the gate just blocked, so it
needs the same guard (a shared `_mitigation_effective(action)` helper used by both).

**Regenerate `optimal_actions` by search, do not hand-write them.** Add
`scripts/derive_optimal.py` that runs the Q\* solver (Phase B) and writes the
argmax trajectory per task back into `tasks/task_config.py`. Commit the generated
sequences with a header comment naming the script, so nobody edits them by hand
again. This also produces the true per-task ceiling for the README.

**Retire `good_followups` or wire it in.** Tier 0 documented that
`graders/grader.py` computes `good_followups_done` and discards it. With the env
becoming v2 anyway, resolve it now rather than carrying a dead config key.

---

## Phase B — Replace 9 hand-tuned rewards with exact Q\*

The core idea. The environment is small, deterministic and finite, so the optimal
action-value function is *computable*, not approximable:

```
Q*(state, action) = best final compute_score reachable after taking `action` from `state`
reward(state, action) = Q*(state, action) - max_a' Q*(state, a')     # <= 0, 0 for optimal
```

Why this is strictly better than tuning nine functions:

- **Alignment is structural.** The reward *is* the grader, maximised. It cannot
  disagree with `compute_score`, so `test_reward_ranking` goes to 41/41 by
  construction rather than by tuning.
- **Ties resolve correctly and for free.** Every action on *a* score-maximising
  path gets reward 0. The `hard[4]` ordering complaint disappears without a rule.
- **It prices mistakes by their actual cost.** A wasted step scores slightly
  below 0; skipping a required mitigation scores far below. That is a dense,
  correctly-scaled signal — exactly what nine hand-weighted functions were
  failing to approximate.
- **It deletes code.** Most of `training/reward_functions.py` becomes redundant.

**`training/qstar.py`** (new). Depth-indexed BFS over canonical env states — **not**
DFS over action sequences (verified this session: DFS with repeats does not
terminate in reasonable time; the same job with a coarser key finished all 6 tasks
in ~3s, so the approach is sound and the key is what matters).

- Canonical key: `(frozenset(actions_taken), step_count, harmful_action_count,
  len(communication_log), resolved)`. A key omitting `harmful_action_count`
  conflates distinct states and silently returns wrong values — that bug produced
  `easy Q*=0.89` against a known-reachable 0.94 in prototyping.
- Restrict branching to a per-task relevant action set (required diagnostics +
  required mitigations + comms + resolve + the task's plausible distractors).
  Full 19-way branching does not terminate.
- Cache to `outputs/qstar_<task>.json`, keyed by a hash of the env + task config
  so it invalidates automatically when Phase A changes the dynamics.

**First implementation step is a timing check.** If exact DP proves too slow on
`hard` after gating, fall back to the beam search already prototyped
(width 60, matched the exact optimum on every task where both ran) and record in
the plan that the reward is near-optimal rather than exact.

**`training/reward_functions.py`** — new `qstar_reward_func` becomes the primary
signal. Keep `format_reward_func` (small, ±0.1, catches malformed output) and
`diversity_reward_func` (per-completion, from Tier 0). Delete the other seven, or
demote them behind a config flag for the ablation table. Keep
`ALL_REWARD_FUNCTIONS` / `CORE_REWARD_FUNCTIONS` exported so the notebook and
tests keep working.

**`tests/test_reward_ranking.py`** — two changes:
1. Compare against **Q\*-optimality**, not a blessed sequence: an action is correct
   if `Q*(s,a) == max_a' Q*(s,a')`. This is what fixes the over-specified ties.
2. Flip `pytest.xfail(...)` to `assert matches == total`. **This is the definition
   of done for Phase B.** Add a third stack `qstar` to `REWARD_STACKS`.

---

## Phase C — Fix the training distribution

Tier 0 measured that all 33 unique GRPO prompts are states SFT already memorised,
with the optimal action as the label. After SFT the policy is near-deterministic
on them, so every group sample is identical and the advantage is exactly zero.
Q\* rewards do not help if every prompt is a state the policy has already
overfitted.

**`training/dataset.py`**

- `iter_training_states()` gains an ε-greedy walk generator: from each task,
  sample trajectories that follow the Q\*-optimal action with probability 1−ε and
  a random valid action otherwise (ε ≈ 0.3). Target **500–2000 unique states**, up
  from 33. Off-path states are where the policy actually needs teaching, and Q\*
  is defined everywhere so they are all correctly labelled.
- Drop the duplicate-prompt behaviour: `per_task_n` currently appends the *same*
  initial state N times (48 of 108 rows were 6 states repeated).
- Add `train_tasks` / `eval_tasks` parameters. Default train set
  `easy, medium, hard, network`; `memory_leak` and `disk_full` held out.
- **SFT must use only the train split**, or the held-out tasks are not held out.

**`config/train.yaml` + `training/config.py`** — add `train_tasks`, `eval_tasks`,
`epsilon`, `n_states`. `TrainConfig.from_yaml` uses `**raw.get(...)`, so every new
YAML key needs a matching dataclass field or it raises `TypeError`.

**`colab_training.ipynb`** — must be updated in the same pass. It hardcodes its own
`GRPOConfig` and does not read the YAML (Tier 0 finding); changes here do not
reach a Colab run otherwise.

---

## Phase D — Full-episode rollouts (gated: only after A–C are green)

**TRL 0.24.0 has no `rollout_func` hook** (verified: `GRPOConfig` has no such
field, zero mentions in `grpo_trainer.py`). Multi-turn therefore needs an override.

Approach: subclass `GRPOTrainer` and override **`_generate(self, prompts, images)`**
— the narrowest available seam — to run an interactive episode per prompt, feeding
real observations back between turns. The hard part is the completion mask:
environment-injected observation tokens must be excluded from the loss, or the
model trains on its own inputs.

**Sequence this last and treat it as optional.** With Q\* rewards the marginal
value drops considerably — Q\* already encodes the full downstream consequence of
an action, which is most of what credit assignment buys. Phases A–C should be
landed, trained and measured *first*, so there is a working baseline before
touching TRL internals. Go/no-go: only start D if A–C training shows
`reward_std > 0` and a rising reward curve.

**Do not spend AMD credits on D before A–C have produced a result.**

---

## Files

```
env/models.py                  A  MITIGATION_PREREQS
env/environment.py             A  gate + _mitigation_effective() shared guard
graders/grader.py              A  resolve the dead good_followups key
tasks/task_config.py           A  optimal_actions regenerated by script
scripts/derive_optimal.py      A  new — search-derived trajectories
training/qstar.py              B  new — depth-indexed BFS solver + cache
training/reward_functions.py   B  qstar_reward_func; retire 7 others
tests/test_reward_ranking.py   B  Q*-optimality; xfail -> hard assert
tests/test_environment.py      A  new gating tests
training/dataset.py            C  eps-greedy states, train/eval split
training/config.py             C  new fields (from_yaml needs them)
config/train.yaml              C  train_tasks/eval_tasks/epsilon/n_states
colab_training.ipynb           C  mirror config; it bypasses the YAML
training/pipeline.py           D  rollout trainer behind a flag
README.md                      —  update Status; env v2 changes all scores
```

---

## Verification

**Phase A**
1. `pytest tests/ -q -rxX` green; new tests assert a mitigation without its
   prerequisite is a no-op and increments `harmful_action_count`.
2. Re-run the beam search — the winning trajectory for every task must now
   diagnose before mitigating. **This is the check that the exploit is closed.**
3. Regenerate `optimal_actions`; confirm `heuristic_policy` still resolves 6/6.

**Phase B**
4. `python -m training.qstar --task hard --time` completes in reasonable time;
   record it. Fall back to beam search if not.
5. `pytest tests/test_reward_ranking.py -v` — **41/41 on all three stacks, as a
   hard assert, no xfail.** This is the gate for starting any training.
6. Spot-check `Q*(root)` per task against the beam-search optimum; they must agree.

**Phase C**
7. `generate_grpo_dataset()` reports ≥500 unique prompts and 0 overlap with the
   SFT state set — assert both in a test, since silent overlap is exactly what
   caused the original plateau.
8. Confirm `memory_leak` / `disk_full` appear in **no** SFT or GRPO prompt.

**Training (only after 1–8 pass)**
9. 50-step debug run first. Check `reward_std > 0` and non-flat entropy in W&B
   **before** committing to 500 steps. A flat curve with `reward_std ≈ 0` means
   stop and fix, not train longer.
10. `python scripts/evaluate.py --base-model <untrained> --trained-model <out>
    --eval-tasks memory_leak,disk_full --n-seeds 5` — non-zero std per task, no
    `DEGENERATE` rows. Note `heuristic_policy` is an *oracle*, not a baseline;
    `--base-model` must point at the untrained checkpoint.

**Phase D** — only if 9–10 produced a real result.