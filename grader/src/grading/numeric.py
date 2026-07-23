"""Finite numeric primitives shared by declarative graders.

Task scorers should not hand-roll float coercion, denominator handling, weight
normalization, or clamping. These helpers reject booleans and non-finite values,
catch Python's large-integer ``OverflowError``, and make empty/zero behavior an
explicit policy choice.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal


class NumericContractError(ValueError):
    """A value violates a declared numeric contract."""


def finite_number(
    value: Any,
    *,
    label: str = "value",
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a finite float satisfying optional inclusive bounds."""
    if isinstance(value, bool):
        raise NumericContractError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericContractError(
            f"{label} is not a finite numeric value: {type(exc).__name__}: {exc}"
        ) from exc
    if not math.isfinite(number):
        raise NumericContractError(f"{label} must be finite, got {number!r}")
    if minimum is not None and number < minimum:
        raise NumericContractError(f"{label}={number} is below minimum {minimum}")
    if maximum is not None and number > maximum:
        raise NumericContractError(f"{label}={number} exceeds maximum {maximum}")
    return number


def score01(value: Any, *, label: str = "score") -> float:
    """Return a finite score clamped to ``[0, 1]``."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return max(0.0, min(1.0, finite_number(value, label=label)))


def safe_ratio(
    numerator: Any,
    denominator: Any,
    *,
    label: str = "ratio",
    zero: Literal["zero", "one", "error"] = "error",
) -> float:
    """Divide finite values with an explicit zero-denominator policy."""
    top = finite_number(numerator, label=f"{label} numerator")
    bottom = finite_number(denominator, label=f"{label} denominator")
    if bottom == 0.0:
        if zero == "zero":
            return 0.0
        if zero == "one":
            return 1.0
        if zero == "error":
            raise NumericContractError(f"{label} denominator is zero")
        raise ValueError(f"unknown zero-denominator policy {zero!r}")
    result = top / bottom
    if not math.isfinite(result):
        raise NumericContractError(f"{label} produced non-finite result {result!r}")
    return result


def safe_mean(
    values: Iterable[Any],
    *,
    label: str = "mean",
    empty: Literal["zero", "error"] = "error",
) -> float:
    """Average finite values with an explicit empty-input policy."""
    numbers = [
        finite_number(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    if not numbers:
        if empty == "zero":
            return 0.0
        if empty == "error":
            raise NumericContractError(f"{label} has no values")
        raise ValueError(f"unknown empty-input policy {empty!r}")
    return sum(numbers) / len(numbers)


def normalized_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    """Validate positive finite weights and normalize them to sum to one."""
    if not weights:
        raise NumericContractError("rubric must declare at least one weight")
    parsed = {
        str(name): finite_number(value, label=f"weight[{name!r}]", minimum=0.0)
        for name, value in weights.items()
    }
    if any(value <= 0.0 for value in parsed.values()):
        raise NumericContractError("rubric weights must all be strictly positive")
    total = sum(parsed.values())
    if not math.isfinite(total) or total <= 0.0:
        raise NumericContractError("rubric weight total must be finite and positive")
    normalized = {name: value / total for name, value in parsed.items()}
    # Assign floating-point residue deterministically to the last criterion.
    residue = 1.0 - sum(normalized.values())
    if residue:
        last = next(reversed(normalized))
        normalized[last] += residue
    return normalized


__all__ = [
    "NumericContractError",
    "finite_number",
    "normalized_weights",
    "safe_mean",
    "safe_ratio",
    "score01",
]
