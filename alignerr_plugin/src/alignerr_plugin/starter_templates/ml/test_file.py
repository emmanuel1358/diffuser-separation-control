"""Continuous-scoring grader for this task (fill in the TODOs).

``compute_score()`` takes no arguments, reads the submission from ``/tmp/output``
and the root-only held-out truth from ``/mcp_server/data``, and returns a float in
[0, 1] (a score dict whose ``score`` is authoritative is also accepted). Raise
``AgentFault`` for agent-controlled failures (kept 0.0); let author/infra failures
propagate (the run is discarded).

See ``examples/mle-tabular-classification/test_file.py`` for a complete grader.
"""

from __future__ import annotations

from pathlib import Path

from grading import calibration
from grading.faults import AgentFault
from grading.helpers import load_submission_or_fault

SUBMISSION_DIR = Path("/tmp/output")  # agent's submission
PRIVATE_DATA = Path("/mcp_server/data")  # root-only held-out truth (data/private/)

# Calibration anchors. FLOOR = worst plausible raw metric; REF = reference
# solution's metric (-> 0.5); PERFECT = the optimum.
# TODO: replace with your task's real anchors.
FLOOR = 1.0
REF = 0.5
PERFECT = 0.0


def compute_score() -> float:
    # Author data: a load failure is infra, so let it propagate.
    # TODO: load your held-out truth from PRIVATE_DATA.
    _truth = PRIVATE_DATA  # noqa: F841 -- replace with the real load

    # Read the submission through the sanctioned loader (raises AgentFault).
    try:
        sub = load_submission_or_fault(
            SUBMISSION_DIR / "submission.csv",
            required_columns=["id", "pred"],  # TODO: your columns
        )
    except AgentFault:
        raise  # agent fault -> kept 0.0

    # TODO: compute your raw metric from `sub` and the held-out truth. The
    # scaffold just echoes the row count and returns the reference score.
    print(f"submission rows: {len(sub)}")
    raw_metric = REF

    # Map the raw metric onto [0, 1] progress, then through the sanctioned curve
    # (use progress_higher_better when a larger metric is better).
    x = calibration.progress_lower_better(raw_metric, floor=FLOOR, perfect=PERFECT)
    x_ref = calibration.progress_lower_better(REF, floor=FLOOR, perfect=PERFECT)
    return calibration.PiecewiseLinearCurve.from_reference(x_ref).score(x)
