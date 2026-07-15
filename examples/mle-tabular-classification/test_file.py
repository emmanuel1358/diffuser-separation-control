"""Continuous-scoring grader for the tabular regression + classification example.

``compute_score()`` takes no arguments, reads the submission from ``/tmp/output``
and the root-only held-out truth from ``/mcp_server/data``, and returns a score
dict whose ``score`` is authoritative. Agent-controlled failures raise
``AgentFault`` (kept 0.0); author/infra failures propagate (discarded).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grading import calibration
from grading.faults import AgentFault
from grading.helpers import load_submission_or_fault

SUBMISSION_DIR = Path("/tmp/output")
PRIVATE_DATA = Path("/mcp_server/data")

# Per-target anchors: floor = worst plausible metric, ref = reference, perfect =
# optimum. Re-measure if you change the reference, a baseline, or the generator.
T1_FLOOR = 0.2661
T1_REF = 0.0330
T1_PERFECT = 0.0

T2_FLOOR = 0.7770
T2_REF = 0.2107
T2_PERFECT = 0.0

LABEL_FLOOR = 0.9231
LABEL_REF = 0.9939
LABEL_PERFECT = 1.0

# Per-target weights.
W_T1 = 0.35
W_T2 = 0.35
W_LABEL = 0.30
assert abs(W_T1 + W_T2 + W_LABEL - 1.0) < 1e-9


def _sre(pred: Any, true: Any) -> float:
    """Standardized RMSE: RMSE(pred, true) / std(true)."""
    import numpy as np

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    denom = float(np.std(true, ddof=0))
    return rmse / denom if denom > 0 else rmse


# Progress + anchor curve come from the shared calibration toolkit.
# PiecewiseLinearCurve is the only sanctioned curve.
_progress_higher_better = calibration.progress_higher_better
_progress_lower_better = calibration.progress_lower_better


def _reference_aggregate_x() -> float:
    xt1 = _progress_lower_better(T1_REF, T1_FLOOR, T1_PERFECT)
    xt2 = _progress_lower_better(T2_REF, T2_FLOOR, T2_PERFECT)
    xlb = _progress_higher_better(LABEL_REF, LABEL_FLOOR, LABEL_PERFECT)
    return W_T1 * xt1 + W_T2 * xt2 + W_LABEL * xlb


_CURVE = calibration.PiecewiseLinearCurve.from_reference(_reference_aggregate_x())


def compute_score() -> dict[str, Any]:
    """Score ``/tmp/output/submission.csv`` against the hidden test targets."""
    import pandas as pd

    # Author data: a load failure is infra, so let it propagate.
    truth = pd.read_parquet(PRIVATE_DATA / "test_target.parquet")

    # Read the submission with the sanctioned loader (raises AgentFault).
    try:
        sub = load_submission_or_fault(
            SUBMISSION_DIR / "submission.csv",
            required_columns=["t1", "t2", "label"],
            allow_extra_columns=True,
        )
    except AgentFault:
        raise
    except Exception as exc:  # noqa: BLE001 - submission boundary
        raise AgentFault(
            f"could not load submission: {type(exc).__name__}: {exc}"
        ) from exc

    if len(sub) != len(truth):
        raise AgentFault(f"row count mismatch (sub={len(sub)}, truth={len(truth)})")

    # sklearn is an infra dependency: an import failure should propagate.
    from sklearn.metrics import f1_score

    # Read truth columns OUTSIDE the agent try so a broken target propagates as infra.
    truth_t1 = truth["t1"].to_numpy()
    truth_t2 = truth["t2"].to_numpy()
    truth_label = truth["label"].astype(int).to_numpy()

    try:
        t1_sre = _sre(sub["t1"].to_numpy(), truth_t1)
        t2_sre = _sre(sub["t2"].to_numpy(), truth_t2)
        label_pred = sub["label"].astype(int).clip(0, 1).to_numpy()
        label_f1 = float(f1_score(truth_label, label_pred, average="binary"))
    except AgentFault:
        raise
    except Exception as exc:  # noqa: BLE001 - submission boundary
        raise AgentFault(
            f"could not compute metrics from submission: {type(exc).__name__}: {exc}"
        ) from exc

    xt1 = _progress_lower_better(t1_sre, T1_FLOOR, T1_PERFECT)
    xt2 = _progress_lower_better(t2_sre, T2_FLOOR, T2_PERFECT)
    xlb = _progress_higher_better(label_f1, LABEL_FLOOR, LABEL_PERFECT)
    x_agg = W_T1 * xt1 + W_T2 * xt2 + W_LABEL * xlb
    final = _CURVE.score(x_agg)

    print("=" * 64)
    print("Per-target raw metrics:")
    print(f"  t1     SRE = {t1_sre:7.4f}  (floor={T1_FLOOR}, ref={T1_REF}, perfect={T1_PERFECT})")
    print(f"  t2     SRE = {t2_sre:7.4f}  (floor={T2_FLOOR}, ref={T2_REF}, perfect={T2_PERFECT})")
    print(f"  label  F1  = {label_f1:7.4f}  (floor={LABEL_FLOOR}, ref={LABEL_REF}, perfect={LABEL_PERFECT})")
    print("Per-target progress x_i in [0, 1]:")
    print(f"  xt1 = {xt1:.4f}  w={W_T1}")
    print(f"  xt2 = {xt2:.4f}  w={W_T2}")
    print(f"  xlb = {xlb:.4f}  w={W_LABEL}")
    print(f"Aggregate x     = {x_agg:.4f}  (x_ref = {_CURVE.x_ref:.4f})")
    print(f"Final score     = {final:.4f}")
    print("=" * 64)

    return {
        "score": final,
        "subscores": {"t1_progress": xt1, "t2_progress": xt2, "label_progress": xlb},
        "weights": {"t1_progress": W_T1, "t2_progress": W_T2, "label_progress": W_LABEL},
        "metadata": {
            "return_shape": "continuous_score_dict",
            "raw_metrics": {"t1_sre": t1_sre, "t2_sre": t2_sre, "label_f1": label_f1},
            "anchors": {
                "t1": {"floor": T1_FLOOR, "reference": T1_REF, "perfect": T1_PERFECT},
                "t2": {"floor": T2_FLOOR, "reference": T2_REF, "perfect": T2_PERFECT},
                "label": {"floor": LABEL_FLOOR, "reference": LABEL_REF, "perfect": LABEL_PERFECT},
            },
            "aggregate_progress": x_agg,
            "reference_aggregate_progress": _CURVE.x_ref,
        },
    }
