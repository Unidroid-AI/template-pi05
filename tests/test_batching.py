# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batching helpers of the pi0.5 brain. They need neither JAX nor OpenPI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brain"))

import adapt  # noqa: E402
import policy  # noqa: E402


@pytest.mark.parametrize(
    ("rows", "expected"), [(1, 1), (2, 4), (4, 4), (5, 16), (13, 16), (16, 16)]
)
def test_padded_rows_depends_only_on_the_row_count(rows, expected):
    assert adapt.padded_rows(rows) == expected


def test_row_slices_cover_every_row_once():
    slices = adapt.row_slices(37)
    assert [len(range(37)[part]) for part in slices] == [16, 16, 5]
    assert adapt.row_slices(0) == []


def test_stack_padded_repeats_the_first_row():
    rows = [{"state": np.full(3, index), "images": {"a": np.full((2, 2), index)}} for index in (1, 2, 3)]
    batch = adapt.stack_padded(rows, 4)
    assert batch["state"].shape == (4, 3)
    assert batch["images"]["a"].shape == (4, 2, 2)
    assert batch["state"][:, 0].tolist() == [1, 2, 3, 1]


def test_stack_padded_rejects_too_many_rows():
    with pytest.raises(ValueError):
        adapt.stack_padded([np.zeros(1)] * 3, 2)


def test_unbatchable_policy_answers_each_row_in_order():
    class OneAtATime:
        def infer(self, example, noise=None):
            return {"actions": np.full((15, 8), example["id"] + noise.mean())}

    examples = [{"id": index} for index in range(5)]
    noises = [np.full((15, 32), 0.5, dtype=np.float32)] * 5
    actions = policy._infer(OneAtATime(), examples, noises)
    assert [float(row[0, 0]) for row in actions] == [0.5, 1.5, 2.5, 3.5, 4.5]


def test_gpu_memory_budget_becomes_a_jax_fraction(monkeypatch):
    monkeypatch.setattr(policy, "_gpu_total_gib", lambda: 48.0)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    policy._limit_gpu_memory(16)
    assert float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) == pytest.approx(16 / 48, abs=1e-4)


def test_unknown_gpu_size_keeps_the_preset_fraction(monkeypatch):
    monkeypatch.setattr(policy, "_gpu_total_gib", lambda: None)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    policy._limit_gpu_memory(16)
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"
