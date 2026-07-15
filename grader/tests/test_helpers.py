"""Tests for the deterministic helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from grading import AgentFault
from grading import helpers


def test_file_exists(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    assert helpers.file_exists(p) is False
    p.write_text("")
    assert helpers.file_exists(p) is True
    assert helpers.file_exists(p, non_empty=True) is False
    p.write_text("hi")
    assert helpers.file_exists(p, non_empty=True) is True


def test_file_contains(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("Hello, World!")
    assert helpers.file_contains(p, "Hello") is True
    assert helpers.file_contains(p, "hello") is False
    assert helpers.file_contains(p, "hello", case_sensitive=False) is True


def test_regex_search() -> None:
    assert helpers.regex_search("hello world", r"\bworld\b") is True
    assert helpers.regex_search("hello", r"\d+") is False
    assert helpers.regex_search("x", r"[invalid") is False  # bad regex returns False


def test_exact_match_string() -> None:
    assert helpers.exact_match("yes", "yes") == 1.0
    assert helpers.exact_match("yes", "no") == 0.0


def test_exact_match_number() -> None:
    assert helpers.exact_match("3.14", 3.14, value_type="number") == 1.0
    assert helpers.exact_match("nope", 3.14, value_type="number") == 0.0


def test_exact_match_set() -> None:
    assert helpers.exact_match("a,b,c", "c,b,a", value_type="set") == 1.0
    assert helpers.exact_match("a,b", "a,b,c", value_type="set") == 0.0


def test_abs_error_anchored() -> None:
    assert helpers.abs_error(0.0, 0.0, tolerance=0.1) == 1.0
    assert helpers.abs_error(0.05, 0.0, tolerance=0.1) == 0.5
    assert helpers.abs_error(1.0, 0.0, tolerance=0.1) == 0.0
    assert helpers.abs_error(0.0, 0.0, tolerance=0) == 1.0
    assert helpers.abs_error(0.1, 0.0, tolerance=0) == 0.0


def test_kendall_tau_identity() -> None:
    assert helpers.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_kendall_tau_reversed() -> None:
    # Fully reversed: tau = -1, mapped to 0.0
    assert helpers.kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == 0.0


def test_kendall_tau_empty_returns_zero() -> None:
    assert helpers.kendall_tau([], ["a"]) == 0.0


def test_jaccard() -> None:
    assert helpers.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert helpers.jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert helpers.jaccard(set(), set()) == 1.0


def test_json_path_basic() -> None:
    data = {"items": [{"name": "x"}, {"name": "y"}]}
    assert helpers.json_path(data, "$.items[0].name") == "x"
    assert helpers.json_path(data, "$.items.1.name") == "y"
    assert helpers.json_path(data, "$.items[5].name") is None


def test_json_path_filter() -> None:
    data = {"rows": [{"sym": "AAA", "v": 1}, {"sym": "BBB", "v": 2}]}
    assert helpers.json_path(data, "$.rows[?sym=BBB].v") == 2


def test_transcript_contains() -> None:
    t = "Human: foo\n\nAssistant: bar"
    assert helpers.transcript_contains(t, "Bar", case_sensitive=False) is True
    assert helpers.transcript_contains(t, "Bar", case_sensitive=True) is False
    assert helpers.transcript_contains(None, "anything") is False


def test_load_json(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}')
    assert helpers.load_json(p) == {"a": 1}
    assert helpers.load_json(tmp_path / "missing.json") is None
    p.write_text("not json")
    assert helpers.load_json(p) is None


def test_load_submission_or_fault_accepts_valid_csv(tmp_path: Path) -> None:
    p = tmp_path / "submission.csv"
    p.write_text("id,y\n1,0.5\n2,0.8\n")

    df = helpers.load_submission_or_fault(
        p,
        required_columns=["id", "y"],
        numeric_columns=["y"],
        unique_key_column="id",
        n_rows=2,
    )

    assert list(df["y"]) == [0.5, 0.8]


def test_load_submission_or_fault_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "truth.csv"
    target.write_text("id,y\n1,1.0\n")
    link = tmp_path / "submission.csv"
    os.symlink(target, link)

    with pytest.raises(AgentFault, match="not a regular file"):
        helpers.load_submission_or_fault(link, required_columns=["id", "y"])


def test_load_submission_or_fault_rejects_extra_columns(tmp_path: Path) -> None:
    p = tmp_path / "submission.csv"
    p.write_text("id,y,junk\n1,0.5,nope\n")

    with pytest.raises(AgentFault, match="unexpected column"):
        helpers.load_submission_or_fault(
            p,
            required_columns=["id", "y"],
            numeric_columns=["y"],
        )
