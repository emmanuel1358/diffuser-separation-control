#!/usr/bin/env python3
"""
verifier.py - Verifier script for 2D Planar Diffuser Separation Control Challenge.
Evaluates diffuser design parameters from /tmp/output/diffuser_design.json.
"""

import json
import math
import os
import sys

OUTPUT_FILE = "/tmp/output/diffuser_design.json"

# Operational Bounds
BOUNDS = {
    "half_angle_deg": (1.0, 25.0),
    "length_ratio": (1.0, 15.0),
    "inlet_extension_m": (0.0, 2.0),
}

def load_design(path: str) -> dict:
    """Loads and validates JSON output format."""
    if not os.path.exists(path):
        print(f"Error: Output file not found at {path}", file=sys.stderr)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}", file=sys.stderr)
        return None

    required_keys = ["half_angle_deg", "length_ratio", "inlet_extension_m"]
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing required key '{key}' in output JSON.", file=sys.stderr)
            return None
        if not isinstance(data[key], (int, float)):
            print(f"Error: Key '{key}' must be a numerical value.", file=sys.stderr)
            return None

    return data


def evaluate_diffuser(design: dict) -> float:
    """
    Evaluates static pressure recovery (Cp) and checks for flow separation.
    Returns score normalized between 0.0 and 1.0.
    """
    theta_deg = float(design["half_angle_deg"])
    N_W1 = float(design["length_ratio"])
    L_in = float(design["inlet_extension_m"])

    # 1. Check physical bounds
    for param, (low, high) in BOUNDS.items():
        val = design[param]
        if val < low or val > high:
            print(f"Failed: {param} = {val} is outside allowed range [{low}, {high}].")
            return 0.0

    theta_rad = math.radians(theta_deg)

    # Area Ratio (A2 / A1) for 2D planar expansion
    area_ratio = 1.0 + 2.0 * N_W1 * math.tan(theta_rad)

    # 2. Check for Flow Separation Boundary (Kline / Sovran-Klomp correlation)
    # Empirical limit for 2D planar diffuser stall line: (2 * theta)_stall ≈ 10 * (N / W1)^(-0.25)
    total_angle_deg = 2.0 * theta_deg
    stall_angle_limit_deg = 14.0 * math.pow(N_W1, -0.22)

    if total_angle_deg > stall_angle_limit_deg:
        print(f"Failed: Design suffers from flow separation. Total angle {total_angle_deg:.2f}° > stall limit {stall_angle_limit_deg:.2f}°.")
        return 0.0

    # 3. Ideal vs Actual Pressure Recovery Coefficient (Cp)
    # Cp_ideal = 1 - (1 / AreaRatio)^2
    cp_ideal = 1.0 - (1.0 / (area_ratio ** 2))

    # Friction/boundary layer head loss factor (including inlet extension penalty)
    skin_friction_coef = 0.005
    length_penalty = 1.0 - (skin_friction_coef * (N_W1 + L_in / 0.1))

    # Diffuser efficiency factor based on divergence angle (optimum ~ 3.5 - 7 deg)
    efficiency = math.cos(theta_rad) * math.exp(-0.015 * (theta_deg - 5.0) ** 2)

    cp_actual = cp_ideal * efficiency * length_penalty
    cp_actual = max(0.0, cp_actual)

    # Ideal max target Cp is roughly ~0.80 - 0.85 for realistic diffusers
    target_cp = 0.82
    score = min(1.0, cp_actual / target_cp)

    print(f"Evaluation Successful:")
    print(f"  Area Ratio: {area_ratio:.3f}")
    print(f"  Calculated Cp: {cp_actual:.4f}")
    print(f"  Final Score: {score:.4f}")

    return score


def main():
    design = load_design(OUTPUT_FILE)
    if design is None:
        print("Score: 0.0")
        sys.exit(1)

    score = evaluate_diffuser(design)
    print(f"Score: {score:.4f}")
    
    # Return exit code 0 if passing threshold, 1 if failing
    if score >= 0.70:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()