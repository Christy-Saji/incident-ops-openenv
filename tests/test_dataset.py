"""Tier-1 Phase C tripwires for the training dataset builders.

Tier 0 measured that all 33 GRPO prompts were states SFT had already
memorised with the optimal action as the label, so every GRPO group sample
came out identical and the advantage was exactly zero (see
training/dataset.py's module docstring). These tests assert the two
properties that fix that: the GRPO prompt set is large and does not overlap
the SFT state set, and the held-out tasks (memory_leak, disk_full) never leak
into either.
"""

import json

from training.dataset import (
    DEFAULT_EVAL_TASKS,
    DEFAULT_TRAIN_TASKS,
    generate_grpo_dataset,
    generate_sft_dataset,
)


def _prompt_states(dataset) -> set[str]:
    return {row["prompt"][-1]["content"] for row in dataset}


def _tasks_in(dataset) -> set[str]:
    return {json.loads(row["prompt"][-1]["content"])["task"] for row in dataset}


class TestGrpoDatasetSize:
    def test_at_least_500_unique_prompts_by_default(self):
        grpo = generate_grpo_dataset(seed=1)
        assert len(_prompt_states(grpo)) >= 500


class TestSftGrpoDisjoint:
    def test_zero_overlap_with_sft_state_set(self):
        sft = generate_sft_dataset(seed=1)
        grpo = generate_grpo_dataset(n_states=600, seed=1)
        overlap = _prompt_states(sft) & _prompt_states(grpo)
        assert not overlap, f"{len(overlap)} state(s) appear in both SFT and GRPO datasets"


class TestHeldOutTasks:
    def test_held_out_tasks_absent_from_sft(self):
        sft = generate_sft_dataset(seed=1)
        assert not (_tasks_in(sft) & set(DEFAULT_EVAL_TASKS))

    def test_held_out_tasks_absent_from_grpo(self):
        grpo = generate_grpo_dataset(n_states=600, seed=1)
        assert not (_tasks_in(grpo) & set(DEFAULT_EVAL_TASKS))

    def test_sft_covers_exactly_the_train_split(self):
        sft = generate_sft_dataset(seed=1)
        assert _tasks_in(sft) == set(DEFAULT_TRAIN_TASKS)
