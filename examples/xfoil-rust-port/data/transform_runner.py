#!/usr/bin/env python3
"""Public source/candidate differential runner for the XFOIL port task."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PROTOCOL = "transform/v1"
DEFAULT_REPO = Path("/tmp/output/repo")
_POLAR_ROW = re.compile(r"^\s*[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?\s+")
_DIFF_ATOLERANCES: dict[str, dict[str, float]] = {
    "naca_geometry": {"coordinates": 5e-4},
    "analyze_inviscid": {"cl": 0.02, "cm": 0.01},
    "analyze_viscous": {
        "cl": 0.035,
        "cd": 0.0015,
        "cm": 0.02,
        "xtr_upper": 0.08,
        "xtr_lower": 0.08,
    },
    "polar": {
        "alpha": 1e-6,
        "cl": 0.04,
        "cd": 0.002,
        "cm": 0.025,
    },
}


def _base_response(request: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "case_id": str(request.get("case_id", "")),
        "status": status,
        "observations": {},
        "events": [],
        "output_files": {},
    }


def _graphics_off() -> list[str]:
    return ["PLOP", "G", ""]


def _panel_commands(panels: int) -> list[str]:
    if panels != 140:
        return ["PPAR", "N", str(panels), "", "PANE"]
    return ["PANE"]


def _run_xfoil(commands: list[str], workdir: Path, timeout: float = 180.0) -> str:
    payload = "\n".join([*commands, "QUIT", ""]) + "\n"
    result = subprocess.run(
        ["xfoil"],
        input=payload,
        cwd=workdir,
        text=True,
        capture_output=True,
        timeout=timeout,
        env={
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(workdir),
            "LC_ALL": "C",
        },
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"reference XFOIL failed with {result.returncode}: "
            f"{result.stderr[-500:] or result.stdout[-500:]}"
        )
    return result.stdout


def _parse_coordinates(path: Path) -> list[list[float]]:
    coordinates: list[list[float]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0].replace("D", "E"))
            y = float(parts[1].replace("D", "E"))
        except ValueError:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            raise RuntimeError("reference XFOIL emitted non-finite coordinates")
        coordinates.append([x, y])
    if len(coordinates) < 3:
        raise RuntimeError("reference XFOIL did not emit a valid coordinate file")
    return coordinates


def _naca4_coordinates(designation: int, nside: int) -> list[list[float]]:
    """Generate XFOIL's NACA4 buffer coordinates at the requested resolution."""
    if not 0 <= designation <= 9999:
        raise ValueError("designation must be a four-digit NACA number")
    if not 2 <= nside <= 10_000:
        raise ValueError("nside must be between 2 and 10000")

    n4 = designation // 1000
    n3 = (designation - n4 * 1000) // 100
    n2 = (designation - n4 * 1000 - n3 * 100) // 10
    n1 = designation - n4 * 1000 - n3 * 100 - n2 * 10
    camber = n4 / 100.0
    camber_position = n3 / 10.0
    thickness = (n2 * 10 + n1) / 100.0

    x_values: list[float] = []
    thickness_values: list[float] = []
    camber_values: list[float] = []
    bunching = 1.5
    bunching_plus_one = bunching + 1.0
    for index in range(nside):
        fraction = index / (nside - 1)
        x = (
            1.0
            if index == nside - 1
            else 1.0
            - bunching_plus_one * fraction * (1.0 - fraction) ** bunching
            - (1.0 - fraction) ** bunching_plus_one
        )
        surface_thickness = (
            (
                0.29690 * math.sqrt(x)
                - 0.12600 * x
                - 0.35160 * x**2
                + 0.28430 * x**3
                - 0.10150 * x**4
            )
            * thickness
            / 0.20
        )
        surface_camber = 0.0
        if camber_position > 0.0:
            if x < camber_position:
                surface_camber = (
                    camber / camber_position**2 * (2.0 * camber_position * x - x**2)
                )
            else:
                surface_camber = (
                    camber
                    / (1.0 - camber_position) ** 2
                    * (1.0 - 2.0 * camber_position + 2.0 * camber_position * x - x**2)
                )
        x_values.append(x)
        thickness_values.append(surface_thickness)
        camber_values.append(surface_camber)

    coordinates = [
        [x_values[index], camber_values[index] + thickness_values[index]]
        for index in range(nside - 1, -1, -1)
    ]
    coordinates.extend(
        [x_values[index], camber_values[index] - thickness_values[index]]
        for index in range(1, nside)
    )
    return coordinates


def _parse_polar(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not _POLAR_ROW.match(line):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            values = [float(part.replace("D", "E")) for part in parts[:7]]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in values):
            continue
        rows.append(
            {
                "alpha": values[0],
                "cl": values[1],
                "cd": values[2],
                "cdp": values[3],
                "cm": values[4],
                "xtr_upper": values[5],
                "xtr_lower": values[6],
            }
        )
    return rows


def _flow_commands(request: dict[str, Any], polar_path: Path) -> list[str]:
    commands = [
        *_graphics_off(),
        f"NACA {int(request['designation']):04d}",
        *_panel_commands(int(request.get("panels", 140))),
        "OPER",
        "MACH",
        str(float(request.get("mach", 0.0))),
    ]
    operation = str(request["operation"])
    if operation in {"analyze_viscous", "polar"}:
        commands.extend(
            [
                "VISC",
                str(float(request["reynolds"])),
                "VPAR",
                "N",
                str(float(request.get("ncrit", 9.0))),
                "",
            ]
        )
    commands.extend(
        [
            "ITER",
            str(int(request.get("iterations", 100))),
            "PACC",
            str(polar_path),
            "",
        ]
    )
    if operation in {"analyze_inviscid", "analyze_viscous"}:
        commands.extend(["ALFA", str(float(request["alpha"]))])
    elif operation == "polar":
        commands.extend(
            [
                "ASEQ",
                str(float(request["alpha_start"])),
                str(float(request["alpha_end"])),
                str(float(request["alpha_step"])),
            ]
        )
    else:
        raise ValueError(f"unsupported flow operation {operation!r}")
    commands.extend(["PACC", ""])
    return commands


def legacy_response(request: dict[str, Any]) -> dict[str, Any]:
    """Run pinned XFOIL and normalize one request."""

    if request.get("protocol") != PROTOCOL:
        raise ValueError("request protocol must be transform/v1")
    operation = str(request.get("operation", ""))
    with tempfile.TemporaryDirectory(prefix="xfoil-reference-") as tmp:
        workdir = Path(tmp)
        if operation == "naca_geometry":
            coordinates = _naca4_coordinates(
                int(request["designation"]),
                int(request.get("nside", 123)),
            )
            response = _base_response(request, "ok")
            response["observations"] = {
                "coordinates": coordinates,
            }
            return response

        polar_path = workdir / "polar.out"
        _run_xfoil(_flow_commands(request, polar_path), workdir)
        rows = _parse_polar(polar_path)
        response = _base_response(request, "ok" if rows else "nonconverged")
        if operation == "polar":
            response["observations"] = {
                key: [row[key] for row in rows]
                for key in (
                    "alpha",
                    "cl",
                    "cd",
                    "cdp",
                    "cm",
                    "xtr_upper",
                    "xtr_lower",
                )
            }
        elif rows:
            response["observations"] = rows[-1]
        return response


def _load_requests(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("case file must contain a JSON list of request objects")
    return data


def _candidate_responses(
    requests: list[dict[str, Any]],
    repo: Path,
) -> list[dict[str, Any]]:
    build_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LC_ALL": "C",
        "CARGO_HOME": "/opt/cargo-home",
        "CARGO_NET_OFFLINE": "true",
    }
    build = subprocess.run(
        ["cargo", "build", "--release", "--offline", "--bin", "transform-candidate"],
        cwd=repo,
        env=build_env,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError(f"candidate build failed with {build.returncode}")
    binary = repo / "target" / "release" / "transform-candidate"
    payload = (
        "\n".join(
            json.dumps(request, sort_keys=True, separators=(",", ":"))
            for request in requests
        )
        + "\n"
    )
    run = subprocess.run(
        [str(binary)],
        input=payload,
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=600,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LC_ALL": "C",
        },
        check=False,
    )
    if run.returncode != 0:
        raise RuntimeError(
            f"candidate exited with {run.returncode}: {run.stderr[-500:]}"
        )
    responses: list[dict[str, Any]] = []
    for line in run.stdout.splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("candidate emitted a non-object JSON response")
            responses.append(value)
    return responses


def _numeric_values_equal(actual: Any, expected: Any, atol: float) -> bool:
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _numeric_values_equal(actual_item, expected_item, atol)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    if (
        isinstance(actual, list)
        or isinstance(actual, bool)
        or isinstance(expected, bool)
    ):
        return False
    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(actual_number)
        and math.isfinite(expected_number)
        and abs(actual_number - expected_number) <= atol
    )


def _responses_equal(
    request: dict[str, Any],
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
    """Match the hidden grader's full-credit comparison rules for public cases."""
    if not isinstance(actual, dict):
        return False
    required = {
        "protocol",
        "case_id",
        "status",
        "observations",
        "events",
        "output_files",
    }
    if not required.issubset(actual):
        return False
    if (
        actual["protocol"] != PROTOCOL
        or str(actual["case_id"]) != str(expected["case_id"])
        or actual["status"] != expected["status"]
        or not isinstance(actual["observations"], dict)
        or not isinstance(actual["events"], list)
        or not isinstance(actual["output_files"], dict)
    ):
        return False

    if expected["status"] != "ok":
        return True
    tolerances = _DIFF_ATOLERANCES.get(str(request.get("operation", "")))
    if tolerances is None:
        return actual == expected
    expected_observations = expected.get("observations")
    if not isinstance(expected_observations, dict):
        return False
    for field, atol in tolerances.items():
        if field not in actual["observations"] or field not in expected_observations:
            return False
        if not _numeric_values_equal(
            actual["observations"][field],
            expected_observations[field],
            atol,
        ):
            return False
    return True


def _summarize_response_for_diff(response: dict[str, Any] | None) -> dict[str, Any]:
    """Compact agent-facing diff payload; avoid dumping full geometry arrays."""
    if not isinstance(response, dict):
        return {"present": False}
    observations = response.get("observations")
    summary: dict[str, Any] = {
        "present": True,
        "status": response.get("status"),
        "case_id": response.get("case_id"),
    }
    if not isinstance(observations, dict):
        summary["observations"] = None
        return summary
    compact: dict[str, Any] = {}
    for key, value in observations.items():
        if isinstance(value, list) and key in {"x", "y", "xp", "yp", "nx", "ny"}:
            compact[key] = {
                "len": len(value),
                "head": value[:3],
                "tail": value[-3:] if len(value) > 3 else value,
            }
        else:
            compact[key] = value
    summary["observations"] = compact
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("legacy", "candidate", "diff"))
    parser.add_argument("cases", type=Path)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()

    requests = _load_requests(args.cases)
    if args.mode == "legacy":
        for response in (legacy_response(request) for request in requests):
            print(json.dumps(response, sort_keys=True))
        return

    candidate = _candidate_responses(requests, args.repo)
    if args.mode == "candidate":
        for response in candidate:
            print(json.dumps(response, sort_keys=True))
        return

    legacy = [legacy_response(request) for request in requests]
    candidate_by_id = {
        str(response.get("case_id", "")): response for response in candidate
    }
    for request, expected in zip(requests, legacy, strict=True):
        case_id = str(expected["case_id"])
        actual = candidate_by_id.get(case_id)
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "legacy": _summarize_response_for_diff(expected),
                    "candidate": _summarize_response_for_diff(actual),
                    "equal": _responses_equal(request, actual, expected),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
