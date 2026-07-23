#!/usr/bin/env python3
"""Generate the private challenge from trusted entropy.

The required seed is intentionally not committed or exposed to the agent.
Trusted CI regenerates this artifact and promotes it through the private mount.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

N_CHALLENGE = 1000


def generate(output_root: Path) -> None:
    secret = os.environ.get("LBX_PRIVATE_CHALLENGE_SEED")
    if not secret:
        raise RuntimeError(
            "LBX_PRIVATE_CHALLENGE_SEED is required for trusted challenge generation"
        )
    seed = int.from_bytes(hashlib.sha256(secret.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "x1": rng.uniform(-1.0, 3.0, size=N_CHALLENGE),
            "x2": rng.uniform(-1.0, 1.0, size=N_CHALLENGE),
            "x3": rng.uniform(0.0, 2.0 * np.pi, size=N_CHALLENGE),
        }
    )
    targets = pd.DataFrame(
        {
            "t1": (
                0.5 * features["x1"] ** 2
                + 0.3 * features["x1"] * features["x2"]
                + rng.normal(scale=0.05, size=N_CHALLENGE)
            ),
            "t2": (
                features["x1"] * np.sin(features["x3"])
                + rng.normal(scale=0.10, size=N_CHALLENGE)
            ),
            "label": (
                features["x2"] * np.cos(features["x3"]) > 0
            ).astype(np.int64),
        }
    )
    destination = output_root / "data" / "private" / "challenge.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([features, targets], axis=1).to_parquet(destination, index=False)
    print(f"wrote trusted private challenge {destination} ({N_CHALLENGE} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()
    generate(args.output_root.resolve())


if __name__ == "__main__":
    main()
