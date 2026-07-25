"""Tests for the sim_policy rollout scaffolding (grading.policy_eval)."""

from __future__ import annotations

import time

import pytest

from grading import policy_eval
from grading.policy_eval import aggregate, run_seeds


class _StubPolicy:
    """Minimal policy proxy with a reset() that records calls."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def _patch_loader(monkeypatch, policy):
    """Make run_seeds' load_submitted_policy return ``policy`` (no sandbox)."""
    captured: dict[str, object] = {}

    def _fake_load(**kwargs):
        captured.update(kwargs)
        return policy

    monkeypatch.setattr(policy_eval, "load_submitted_policy", _fake_load)
    return captured


def test_run_seeds_deterministic_seeds_and_reset(monkeypatch) -> None:
    policy = _StubPolicy()
    _patch_loader(monkeypatch, policy)
    seen: list[int] = []

    def roll_one_seed(pol, seed):
        seen.append(seed)
        return {"score": float(seed)}

    results = run_seeds(roll_one_seed, n_seeds=5, base_seed=10_000)

    assert seen == [10_000, 10_001, 10_002, 10_003, 10_004]
    assert [r["seed"] for r in results["per_seed"]] == seen
    assert results["n_seeds"] == 5
    assert results["n_failed"] == 0
    assert policy.reset_calls == 5
    assert all("seed_elapsed_s" in r for r in results["per_seed"])


def test_run_seeds_forwards_loader_kwargs(monkeypatch) -> None:
    captured = _patch_loader(monkeypatch, _StubPolicy())

    run_seeds(
        lambda pol, seed: {"ok": True},
        n_seeds=1,
        policy_path="/tmp/output/custom.py",
        policy_factory="make_policy",
    )

    assert captured["path"] == "/tmp/output/custom.py"
    assert captured["factory_name"] == "make_policy"
    assert "source" not in captured  # path mode does not pass source


def test_run_seeds_source_mode_passes_source(monkeypatch) -> None:
    captured = _patch_loader(monkeypatch, _StubPolicy())

    run_seeds(
        lambda pol, seed: {"ok": True},
        n_seeds=1,
        policy_source="def load_policy():\n    ...",
    )

    assert captured["source"] == "def load_policy():\n    ..."


def test_run_seeds_on_error_record_captures_and_continues(monkeypatch) -> None:
    _patch_loader(monkeypatch, _StubPolicy())

    def roll_one_seed(pol, seed):
        if seed == 10_001:
            raise RuntimeError("boom")
        return {"score": 1.0}

    results = run_seeds(roll_one_seed, n_seeds=3, base_seed=10_000)

    assert results["n_failed"] == 1
    failed = [r for r in results["per_seed"] if "error" in r]
    assert len(failed) == 1
    assert failed[0]["seed"] == 10_001
    assert "RuntimeError: boom" in failed[0]["error"]
    assert "traceback" in failed[0]
    # The other two seeds still produced records.
    assert sum(1 for r in results["per_seed"] if r.get("score") == 1.0) == 2


def test_run_seeds_on_error_raise_propagates(monkeypatch) -> None:
    _patch_loader(monkeypatch, _StubPolicy())

    def roll_one_seed(pol, seed):
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        run_seeds(roll_one_seed, n_seeds=2, on_error="raise")


def test_run_seeds_rejects_bad_on_error(monkeypatch) -> None:
    _patch_loader(monkeypatch, _StubPolicy())
    with pytest.raises(ValueError, match="on_error"):
        run_seeds(lambda pol, seed: {}, n_seeds=1, on_error="explode")


def test_run_seeds_reset_failure_recorded(monkeypatch) -> None:
    class _BadReset:
        def reset(self):
            raise RuntimeError("reset broke")

    _patch_loader(monkeypatch, _BadReset())
    results = run_seeds(lambda pol, seed: {"score": 1.0}, n_seeds=2)

    assert results["n_failed"] == 2
    assert all("reset broke" in r["error"] for r in results["per_seed"])


def test_run_seeds_soft_timeout_promoted_to_failure(monkeypatch) -> None:
    _patch_loader(monkeypatch, _StubPolicy())

    def slow_roll(pol, seed):
        time.sleep(0.02)
        return {"score": 1.0}

    results = run_seeds(slow_roll, n_seeds=1, timeout_per_seed_s=0.001)

    rec = results["per_seed"][0]
    assert rec["soft_timeout_exceeded"] is True
    assert "soft_timeout" in rec["error"]
    assert results["n_failed"] == 1


def test_aggregate_failed_fill_floors_crash_below_skip() -> None:
    # 9 good seeds at 1.0 + 1 crashed seed.
    per_seed = [{"score": 1.0, "seed": i} for i in range(9)]
    per_seed.append({"error": "boom", "seed": 9})

    skipped = aggregate(per_seed, ["score"])  # skip_failed default
    floored = aggregate(per_seed, ["score"], failed_fill={"score": 0.0})

    assert skipped["score"] == 1.0  # crash skipped -> mean unaffected
    assert floored["score"] == pytest.approx(0.9)  # crash floored at 0.0
    assert floored["score"] < skipped["score"]  # the anti-hack guarantee


def test_aggregate_bool_keys_fraction_of_true() -> None:
    per_seed = [
        {"finished": True},
        {"finished": False},
        {"finished": True},
        {"finished": True},
    ]
    assert aggregate(per_seed, ["finished"])["finished"] == pytest.approx(0.75)


def test_aggregate_missing_key_skipped_and_empty_is_none() -> None:
    per_seed = [{"a": 2.0}, {"b": 5.0}, {"a": 4.0}]
    out = aggregate(per_seed, ["a", "missing"])
    assert out["a"] == pytest.approx(3.0)  # only the two records with "a"
    assert out["missing"] is None


def test_aggregate_failed_fill_only_named_key() -> None:
    # "gates" floored, "finished" not in failed_fill -> governed by skip_failed.
    per_seed = [
        {"finished": True, "gates": 10},
        {"error": "boom"},
    ]
    out = aggregate(
        per_seed,
        ["finished", "gates"],
        failed_fill={"gates": 0},
    )
    assert out["finished"] == pytest.approx(1.0)  # crash skipped for this key
    assert out["gates"] == pytest.approx(5.0)  # (10 + 0) / 2
