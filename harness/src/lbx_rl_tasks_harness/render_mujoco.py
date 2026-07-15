from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_DURATION_SEC = 5.0


class Policy(Protocol):
    def act(self, obs: dict[str, Any]) -> Any:
        ...


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a MuJoCo oracle rollout.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--controller", type=Path, help="Deprecated alias for --policy."
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    args = parser.parse_args(argv)

    if not args.model.exists():
        raise FileNotFoundError(f"model not found: {args.model}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MuJoCo videos")

    hooks = _load_module(args.config) if args.config else None
    policy_path = args.policy or args.controller
    policy = load_policy(policy_path) if policy_path else None
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    if hooks and hasattr(hooks, "initialize"):
        hooks.initialize(model, data)
    else:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
    reset_policy(policy, seed=0, metadata={"model_path": str(args.model)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    steps_per_frame = max(
        1, int(round((1.0 / args.fps) / max(model.opt.timestep, 1e-4)))
    )

    with tempfile.TemporaryDirectory() as td:
        frame_dir = Path(td)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        try:
            sim_step = 0
            for idx in range(int(args.fps * args.duration_sec)):
                for _ in range(steps_per_frame):
                    _step(model, data, hooks, policy, step=sim_step)
                    sim_step += 1
                if hooks and hasattr(hooks, "update_scene"):
                    hooks.update_scene(renderer, model, data)
                else:
                    renderer.update_scene(data)
                _write_ppm(frame_dir / f"frame_{idx:04d}.ppm", renderer.render())
        finally:
            renderer.close()

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(frame_dir / "frame_%04d.ppm"),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output),
            ],
            check=True,
        )
    return 0


def _step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hooks: Any,
    policy: Any,
    *,
    step: int,
) -> None:
    if hooks and hasattr(hooks, "before_step"):
        hooks.before_step(model, data, policy)
    elif policy is not None:
        obs = build_observation(model, data, step=step)
        if hooks and hasattr(hooks, "observation"):
            extra = hooks.observation(model, data, obs)
            if isinstance(extra, dict):
                obs.update(extra)
        action = policy.act(obs)
        if hooks and hasattr(hooks, "apply_action"):
            hooks.apply_action(model, data, action)
        else:
            apply_action(model, data, action)
    mujoco.mj_step(model, data)


def build_observation(
    model: mujoco.MjModel, data: mujoco.MjData, *, step: int | None
) -> dict[str, Any]:
    return {
        "time": float(data.time),
        "step": int(step or 0),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "sensordata": data.sensordata.copy(),
        "ctrl": data.ctrl.copy(),
        "nu": int(model.nu),
        "nq": int(model.nq),
        "nv": int(model.nv),
    }


def apply_action(model: mujoco.MjModel, data: mujoco.MjData, action: Any) -> None:
    values = np.asarray(action, dtype=float).reshape(-1)
    if values.size == 1 and model.nu >= 1:
        values = np.repeat(values, model.nu)
    if values.size != model.nu:
        raise ValueError(
            f"policy action size {values.size} does not match model.nu {model.nu}"
        )
    if not np.isfinite(values).all():
        raise ValueError("policy action contains non-finite values")
    for idx in range(model.nu):
        value = float(values[idx])
        if bool(model.actuator_ctrllimited[idx]):
            lo, hi = model.actuator_ctrlrange[idx]
            value = float(np.clip(value, lo, hi))
        data.ctrl[idx] = value


def load_policy(path: Path | None) -> Policy:
    module = _load_module(path)
    if hasattr(module, "Policy"):
        policy = module.Policy()
        if not callable(getattr(policy, "act", None)):
            raise TypeError(f"{path} Policy class must define act(obs)")
        return policy
    if callable(getattr(module, "act", None)):
        return module
    raise TypeError(f"{path} must define act(obs) or class Policy with act(obs)")


def reset_policy(
    policy: Any, *, seed: int | None, metadata: dict[str, Any] | None
) -> None:
    if policy is not None and callable(getattr(policy, "reset", None)):
        policy.reset(seed=seed, metadata=metadata)


def _load_module(path: Path | None) -> Any:
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(path.parent) in sys.path:
            sys.path.remove(str(path.parent))
    return module


def _write_ppm(path: Path, frame: np.ndarray) -> None:
    height, width, _channels = frame.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.asarray(frame, dtype=np.uint8).tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
