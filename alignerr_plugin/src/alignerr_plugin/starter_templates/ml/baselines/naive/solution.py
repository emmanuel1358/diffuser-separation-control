"""Weak baseline: a naive untrained predictor (mean/median, or majority-class /
random). Its output must score clearly BELOW the reference -- that gap is the
learnability gate. Add more baselines as baselines/<name>/ if useful.
"""

from __future__ import annotations

from pathlib import Path

OUTPUT = Path("/tmp/output")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # TODO: write a weak but input-dependent predictor.py. Constants belong in
    # the generated no-information audit family and should receive zero.
    raise NotImplementedError("fill in the naive baseline")


if __name__ == "__main__":
    main()
