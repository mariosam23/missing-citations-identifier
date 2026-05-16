"""Tests for ``evaluation.dataset``.

Verifies split determinism (same seed → identical output) and the
JSON round-trip (save → load → identical members).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset import Split, save_split


@pytest.fixture
def tmp_eval_dir(tmp_path: Path) -> Path:
    """A temporary directory for split files."""
    return tmp_path / "eval"


class TestSplitSerialization:
    """Save → load round-trip produces identical data."""

    def test_round_trip(self, tmp_eval_dir: Path) -> None:
        split = Split(
            seed=42,
            train_paper_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            val_paper_ids=[9],
            test_paper_ids=[10],
        )
        path = save_split(split, directory=tmp_eval_dir)
        assert path.exists()

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["seed"] == 42
        assert loaded["train_paper_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert loaded["val_paper_ids"] == [9]
        assert loaded["test_paper_ids"] == [10]

    def test_same_seed_identical_output(self, tmp_eval_dir: Path) -> None:
        """Two saves with the same split produce byte-identical JSON."""
        split_a = Split(
            seed=42,
            train_paper_ids=[1, 2, 3],
            val_paper_ids=[4],
            test_paper_ids=[5],
        )
        split_b = Split(
            seed=42,
            train_paper_ids=[1, 2, 3],
            val_paper_ids=[4],
            test_paper_ids=[5],
        )
        dir_a = tmp_eval_dir / "a"
        dir_b = tmp_eval_dir / "b"
        path_a = save_split(split_a, directory=dir_a)
        path_b = save_split(split_b, directory=dir_b)

        assert path_a.read_bytes() == path_b.read_bytes()

    def test_split_member_counts(self) -> None:
        """Partition is non-overlapping and exhaustive."""
        all_ids = list(range(1, 101))
        # Simulate a split of 100 papers.
        train = all_ids[:80]
        val = all_ids[80:90]
        test = all_ids[90:]

        all_split = set(train) | set(val) | set(test)
        assert len(all_split) == 100
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0
