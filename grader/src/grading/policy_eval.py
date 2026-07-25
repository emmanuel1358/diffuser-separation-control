"""Standard rollout scaffolding for sim_policy graders.

A sim_policy task's `compute_score` typically does the same four things:

  1. Load the held-out eval simulator from /mcp_server/data/.
  2. Load the agent's submitted policy from /tmp/output/policy.py in a
     sandboxed worker.
  3. Roll N held-out seeds: per seed, reset the sim, repeatedly query the
     policy with (obs, info), step the sim, capture per-step / per-episode
     metrics; aggregate.
  4. Reduce the aggregates to a final score via the scoring helpers.

This module wraps step (2) + (3): a single helper that loads the policy, runs
N seeds with a task-supplied per-step closure, and returns per-seed records
plus aggregate metrics. Task-specific details (the observation shape, the
action shape, the reset signature, the episode-termination criteria, the
per-episode metrics) live in two small closures the caller supplies.

The helper is deliberately thin -- it does not prescribe a fixed obs/action
interface, doesn't assume Gymnasium semantics, doesn't bake in a single
metric set. Robotics-control envs vary enough that a one-size-fits-all
interface would either be too restrictive (Gymnasium-only) or too lax (no
real abstraction at all). The author writes their own per-step closure; the
helper handles loading, the seed loop, isolated-exception fallback, and the
results dict shape.

Typical use:

    from grading.env_loading import load_env_module
    from grading.policy_eval import run_seeds, aggregate
    from grading.calibration import piecewise_linear_score, progress_higher_better

    def compute_score() -> float:
        eval_sim_mod = load_env_module("/mcp_server/data/eval_sim.py",
                                        factory_name="make_eval_sim")

        def roll_one_seed(policy, seed):
            sim = eval_sim_mod.make_eval_sim(seed=seed)
            obs = sim.reset()
            done = False
            steps = 0
            while not done and steps < MAX_STEPS:
                action = policy.act(obs, {"step": steps})
                obs, reward, done, info = sim.step(action)
                steps += 1
            return {"finished": bool(info["success"]),
                    "gates": int(info["gates"]),
                    "time": float(steps * sim.dt)}

        results = run_seeds(roll_one_seed, n_seeds=20, base_seed=10_000)
        # Floor a crashed seed at the worst legitimate value instead of
        # skipping it, so crashing hard seeds can't lift the mean:
        agg = aggregate(results["per_seed"], ["finished", "gates"],
                        failed_fill={"finished": False, "gates": 0})
        ...
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Callable, Iterable, Mapping

from grading.policy_runner import load_submitted_policy

logger = logging.getLogger(__name__)


def run_seeds(
    roll_one_seed: Callable[[object, int], Mapping[str, object]],
    *,
    n_seeds: int,
    base_seed: int = 10_000,
    policy_path: str = "/tmp/output/policy.py",
    policy_factory: str = "load_policy",
    policy_source: str | None = None,
    timeout_per_seed_s: float | None = None,
    on_error: str = "record",
) -> dict:
    """Load the agent's submitted policy, run `roll_one_seed(policy, seed)`
    for each of `n_seeds` deterministic seeds, return per-seed records plus
    standard aggregates.

    Args:
        roll_one_seed: callable taking (policy_proxy, seed) and returning a
            mapping with per-seed metrics (e.g. {"finished": bool, "time":
            float, ...}). The callable is responsible for instantiating the
            eval sim with the given seed, looping the policy through, and
            extracting the metrics. The helper does not constrain the keys;
            whatever you return is collected verbatim into per_seed.
        n_seeds: how many seeds to roll.
        base_seed: seeds used are base_seed + i for i in range(n_seeds).
            Pick a base far from anything the agent might use during their
            own training so eval is genuinely held out.
        policy_path: path to the agent's policy file (default
            /tmp/output/policy.py). Forwarded to load_submitted_policy.
        policy_factory: name of the factory function inside policy_path
            (default "load_policy"). Forwarded to load_submitted_policy.
        policy_source: if given, load the policy from this source string
            (grader-supplied wrapper) instead of policy_path. See
            grading.policy_runner.load_submitted_policy.
        timeout_per_seed_s: optional soft wall-clock budget for one seed.
            Observed AFTER the seed completes (the helper does NOT
            preemptively kill a running seed; that would require a third
            process layer). When a seed's roll exceeds the budget, the
            helper logs a warning, sets {"soft_timeout_exceeded": True}
            on the record AND sets {"error": "soft_timeout after Xs"} so
            the record is treated as a failed seed by aggregate(skip_failed
            =True) and counted in n_failed. Rationale: a roll that ran
            past its budget almost certainly produced suspect output;
            pinning it to FLOOR is safer than averaging it in.
        on_error: "record" (default) catches Exception inside roll_one_seed
            AND inside the optional policy.reset() call, writing
            {"error": "...", "traceback": "..."} into that seed's record;
            "raise" propagates either failure. Use "record" for grading
            (a single bad seed shouldn't zero the whole run); "raise"
            during local dev to see failures fast (including broken
            reset() implementations).

    The agent's policy may optionally define `reset()`; this helper calls
    it (when present) before each seed's roll. Reset failures are governed
    by `on_error` just like roll failures.

    The submitted policy is loaded via the repo's sandboxed
    `grading.policy_runner.load_submitted_policy`, which runs the agent's
    factory in a privilege-dropped worker and raises
    `grading.faults.AgentFault` for agent-caused load failures. That
    AgentFault is deliberately NOT caught here: a failed load is a kept 0.0
    for the whole submission, distinct from a single per-seed crash.

    Returns:
        dict with:
            "per_seed":     list[dict]   one entry per seed in seed-order,
                                          with whatever keys roll_one_seed
                                          returned plus the seed itself.
                                          On any failure (raised exception,
                                          OR soft-timeout exceeded):
                                          {"error": str,
                                           "traceback": str (raised only),
                                           "seed": int, ...}.
            "n_seeds":      int          total seeds attempted.
            "n_failed":     int          seeds with an "error" key in
                                          per_seed (covers both raised
                                          exceptions and soft-timeout
                                          failures). Matches the count a
                                          caller would compute via
                                          `sum(1 for r in per_seed if
                                          "error" in r)`.
            "elapsed_s":    float        wall-clock for the whole run.
    """
    if on_error not in {"record", "raise"}:
        raise ValueError(
            f"on_error must be 'record' or 'raise'; got {on_error!r}"
        )

    if policy_source is not None:
        policy = load_submitted_policy(
            path=policy_path, source=policy_source, factory_name=policy_factory,
        )
    else:
        policy = load_submitted_policy(
            path=policy_path, factory_name=policy_factory,
        )

    per_seed: list[dict] = []
    t0 = time.perf_counter()

    for i in range(n_seeds):
        seed = base_seed + i
        seed_t0 = time.perf_counter()
        # Note: the policy proxy's RPCs have their own per-call timeout
        # (see grading.policy_runner). The optional timeout_per_seed_s
        # below is a soft outer cap, observed AFTER the seed completes --
        # we do NOT preemptively kill a running seed (that would require
        # a third process layer).
        #
        # Both policy.reset() (when defined) and roll_one_seed run inside
        # the same try/except, so on_error="raise" surfaces broken reset()
        # implementations and on_error="record" tags either failure mode
        # uniformly with an "error" key.
        try:
            if hasattr(policy, "reset"):
                policy.reset()
            record = dict(roll_one_seed(policy, seed))
        except Exception as e:
            if on_error == "raise":
                raise
            record = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        record.setdefault("seed", seed)
        elapsed = time.perf_counter() - seed_t0
        record.setdefault("seed_elapsed_s", elapsed)
        if (timeout_per_seed_s is not None
                and elapsed > timeout_per_seed_s
                and "error" not in record):
            logger.warning(
                f"seed {seed}: roll exceeded soft timeout "
                f"{timeout_per_seed_s:.1f}s ({elapsed:.1f}s); promoting to "
                "failure so aggregate(skip_failed=True) skips it and "
                "n_failed counts it."
            )
            record["soft_timeout_exceeded"] = True
            record["error"] = (
                f"soft_timeout after {elapsed:.1f}s "
                f"(budget {timeout_per_seed_s:.1f}s)"
            )
        per_seed.append(record)

    # Count any seed with an "error" key (covers both raised exceptions
    # and promoted soft-timeout failures). Matches the per_seed format
    # documented above so callers can use results["n_failed"] directly
    # instead of recomputing it from per_seed.
    n_failed = sum(1 for r in per_seed if "error" in r)
    return {
        "per_seed": per_seed,
        "n_seeds": n_seeds,
        "n_failed": n_failed,
        "elapsed_s": time.perf_counter() - t0,
    }


def aggregate(
    per_seed: Iterable[Mapping[str, object]],
    keys: Iterable[str],
    *,
    skip_failed: bool = True,
    failed_fill: Mapping[str, object] | None = None,
) -> dict:
    """Mean / fraction-of-true across seeds for the named keys. Convenience
    wrapper so callers don't reinvent the same loop.

    For numeric keys -> arithmetic mean (over seeds where the key is
    present and not in an error record).
    For bool keys     -> fraction-of-true.
    Missing-from-record entries are skipped.

    failed_fill: optional ``{key: value}``. When given, a failed seed (one with
    an ``"error"`` record from run_seeds) contributes ``failed_fill[key]`` for
    each named key instead of being skipped -- so a crashed episode counts at
    that value, typically the per-key FLOOR (the worst legitimate value). This
    FLOORS a crash rather than excluding it, which both preserves granularity
    (9 good seeds + 1 floored beats 1 good seed) and removes the incentive to
    strategically crash out of hard seeds (skipping a crash would otherwise
    RAISE the mean). For a key not in ``failed_fill``, ``skip_failed`` governs as
    before. Default (no ``failed_fill``) is unchanged.

    Returns a dict of {key: aggregate_value}. Empty when no seed has a value
    for the key.
    """
    per_seed = list(per_seed)
    fill = dict(failed_fill) if failed_fill is not None else {}
    out = {}
    for key in keys:
        values = []
        for r in per_seed:
            failed = "error" in r
            if failed and key in fill:
                values.append(fill[key])
                continue
            if failed and skip_failed:
                continue
            if key not in r:
                continue
            values.append(r[key])
        if not values:
            out[key] = None
            continue
        if all(isinstance(v, bool) for v in values):
            out[key] = sum(1 for v in values if v) / len(values)
        else:
            out[key] = sum(values) / len(values)
    return out


__all__ = ["run_seeds", "aggregate"]
