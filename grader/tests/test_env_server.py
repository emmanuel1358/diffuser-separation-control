from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env_server import _is_env_task, hidden_env_mode
from env_server import server as env_server_module
from env_server.config import EnvConfig
from env_server.protocol import _pack, _unpack
from env_server.server import EnvServer, _prepend_env_deps, _reject_private_paths


_ENV_SOURCE = """
class Env:
    _env_public_methods = frozenset({"reset", "step", "n_arms"})

    def __init__(self, seed=0, scale=1):
        self.seed = int(seed)
        self.scale = int(scale)
        self.t = 0

    def n_arms(self):
        return 4

    def reset(self, seed=None):
        self.t = 0
        return {"obs": [0, 0]}

    def step(self, action):
        self.t += 1
        return {"obs": [action * self.scale, self.t]}

    def best(self):
        return 3  # grader-only; NOT in _env_public_methods


def make_env(**kwargs):
    return Env(**kwargs)
"""


def _server(tmp_path: Path, *, allowed=None) -> EnvServer:
    env_py = tmp_path / "env.py"
    env_py.write_text(_ENV_SOURCE)
    cfg = EnvConfig(
        module_path=env_py.resolve(),
        factory_name="make_env",
        allowed_env_kwargs=frozenset(allowed) if allowed is not None else None,
    )
    return EnvServer(cfg, tmp_path / "env.sock")


def test_protocol_round_trips_rich_types() -> None:
    payload = {
        "arr": np.arange(6, dtype=np.float64).reshape(2, 3),
        "s": {1, 2, 3},
        "p": Path("/tmp/x"),
    }
    out = _unpack(_pack(payload))
    np.testing.assert_array_equal(out["arr"], payload["arr"])
    assert out["arr"].dtype == np.float64
    assert sorted(out["s"]) == [1, 2, 3]
    assert out["p"] == "/tmp/x"


def test_create_call_destroy(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    assert srv._handle({"method": "__create__", "instance_id": 0, "args": {}})["ok"]
    reply = srv._handle(
        {"method": "step", "instance_id": 0, "args": {"action": 2}}
    )
    assert reply["ok"] and reply["result"]["obs"] == [2, 1]
    assert srv._handle({"method": "__destroy__", "instance_id": 0, "args": {}})["ok"]


def test_underscore_methods_rejected(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    srv._handle({"method": "__create__", "instance_id": 0, "args": {}})
    with pytest.raises(ValueError, match="private"):
        srv._handle({"method": "_secret", "instance_id": 0, "args": {}})


def test_public_method_allow_list_blocks_grader_only(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    srv._handle({"method": "__create__", "instance_id": 0, "args": {}})
    # 'best' is public but NOT in _env_public_methods -> rejected over the socket.
    with pytest.raises(ValueError, match="public API"):
        srv._handle({"method": "best", "instance_id": 0, "args": {}})


def test_allowed_env_kwargs_enforced(tmp_path: Path) -> None:
    srv = _server(tmp_path, allowed=["seed"])
    with pytest.raises(ValueError, match="allowed_env_kwargs"):
        srv._handle(
            {"method": "__create__", "instance_id": 0, "args": {"env_kwargs": {"scale": 9}}}
        )


def test_private_path_kwargs_rejected() -> None:
    with pytest.raises(ValueError, match="server-private path"):
        _reject_private_paths({"states_path": "/mcp_server/data/truth.npz"})
    # A bare relative token (a split name) is fine.
    _reject_private_paths({"split": "test"})


def test_bool_instance_id_rejected(tmp_path: Path) -> None:
    srv = _server(tmp_path)
    with pytest.raises(ValueError, match="instance_id must be an integer"):
        srv._handle({"method": "__create__", "instance_id": True, "args": {}})


def test_env_config_load_defaults_and_override(tmp_path: Path) -> None:
    (tmp_path / "env.py").write_text(_ENV_SOURCE)
    cfg = EnvConfig.load(data_dir=tmp_path, config_path=tmp_path / "missing.json")
    assert cfg.module_path == (tmp_path / "env.py").resolve()
    assert cfg.factory_name == "make_env"
    assert cfg.allowed_env_kwargs is None

    (tmp_path / "env_config.json").write_text(
        '{"module": "env.py", "factory": "make_env", "allowed_env_kwargs": ["seed"]}'
    )
    cfg2 = EnvConfig.load(data_dir=tmp_path, config_path=tmp_path / "env_config.json")
    assert cfg2.allowed_env_kwargs == frozenset({"seed"})


def test_socket_round_trip_with_client(tmp_path: Path) -> None:
    """End-to-end: real Unix socket + the canonical env_client over the wire."""
    import os
    import tempfile
    import threading
    import time

    from env_server import env_client as ec

    srv = _server(tmp_path)
    # AF_UNIX sun_path is ~104 chars; pytest tmp_path is too long on macOS, so
    # bind a short /tmp socket path instead.
    sock_path = Path(tempfile.mktemp(prefix="es", suffix=".sock", dir="/tmp"))
    srv.socket_path = sock_path
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        # The socket FILE appears at bind() but connections are only accepted
        # after listen(), so polling for existence races a fast client into
        # ECONNREFUSED. Retry the actual connection until the server is up.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                client = ec.Env(
                    instance_id=0, env_kwargs={"seed": 1}, socket_path=str(sock_path)
                )
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        with client:
            assert client.call("n_arms") == 4
            assert client.call("step", action=3)["obs"] == [3, 1]
            # grader-only method is not reachable over the socket
            with pytest.raises(ec.EnvError, match="public API"):
                client.call("best")
    finally:
        srv.stop()
        thread.join(timeout=5)
        if sock_path.exists():
            os.unlink(sock_path)


def test_activation_reads_task_toml(tmp_path: Path) -> None:
    toml = tmp_path / "task.toml"
    toml.write_text('[environment]\nhidden_env = "env"\n')
    assert hidden_env_mode(toml) == "env"
    assert _is_env_task(toml) is True

    toml.write_text('[environment]\nhidden_env = ""\n')
    assert _is_env_task(toml) is False
    assert _is_env_task(tmp_path / "nope.toml") is False


def test_activation_normalizes_casing(tmp_path: Path) -> None:
    # Schema validation lowercases hidden_env, but the BAKED task.toml keeps the
    # author's raw value, so the runtime gate must normalize it the same way.
    toml = tmp_path / "task.toml"
    toml.write_text('[environment]\nhidden_env = "ENV"\n')
    assert hidden_env_mode(toml) == "env"
    assert _is_env_task(toml) is True
    toml.write_text('[environment]\nhidden_env = "  Hybrid  "\n')
    assert hidden_env_mode(toml) == "hybrid"
    assert _is_env_task(toml) is True


# --- env_dependencies: server-only deps on the env server's sys.path -------


def test_prepend_env_deps_adds_existing_dir(tmp_path: Path, monkeypatch) -> None:
    import sys

    deps = tmp_path / "env_deps"
    deps.mkdir()
    monkeypatch.setattr(env_server_module, "ENV_DEPS_DIR", deps)
    saved = list(sys.path)  # save/restore so the test doesn't leak the entry
    try:
        _prepend_env_deps()
        assert sys.path[0] == str(deps)
        _prepend_env_deps()  # idempotent: no duplicate
        assert sys.path.count(str(deps)) == 1
    finally:
        sys.path[:] = saved


def test_prepend_env_deps_noop_when_absent(tmp_path: Path, monkeypatch) -> None:
    import sys

    monkeypatch.setattr(env_server_module, "ENV_DEPS_DIR", tmp_path / "nope")
    before = list(sys.path)
    _prepend_env_deps()
    assert sys.path == before


def test_load_env_module_resolves_env_only_dependency(tmp_path: Path, monkeypatch) -> None:
    """An env.py that imports a server-only dep (living ONLY in /mcp_server/env_deps)
    must load via load_env_module -- the choke point the env-server subprocess, the
    supervisor pre-flight, AND the grade-time in-process load all funnel through.
    Regression guard: env_dependencies was previously importable only inside the
    socket subprocess, so the pre-flight and grade-time env loads ImportError'd."""
    import sys

    from env_server import config as env_config_module
    from grading.env_loading import load_env_module

    # A "server-only simulator" that is NOT on the default sys.path.
    deps_dir = tmp_path / "env_deps"
    deps_dir.mkdir()
    (deps_dir / "server_only_sim_pkg.py").write_text("VALUE = 7\n")

    # A trusted env.py that imports the simulator AT MODULE SCOPE (the canonical
    # `import myosuite` pattern) -- fails to load unless env_deps is on the path.
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    env_py = task_dir / "env.py"
    env_py.write_text(
        "import server_only_sim_pkg\n"
        "def make_env(**kw):\n"
        "    return server_only_sim_pkg.VALUE\n"
    )

    saved_path = list(sys.path)
    try:
        # Negative control: with env_deps pointed elsewhere, the import fails --
        # proving the dep is not otherwise reachable and the test is meaningful.
        monkeypatch.setattr(env_config_module, "ENV_DEPS_DIR", tmp_path / "absent")
        with pytest.raises(ModuleNotFoundError):
            load_env_module(env_py, trusted_roots=(task_dir,), module_name="env_dep_neg")

        # Positive: point ENV_DEPS_DIR at the dir holding the simulator; the load
        # now succeeds and make_env is callable.
        monkeypatch.setattr(env_config_module, "ENV_DEPS_DIR", deps_dir)
        mod = load_env_module(env_py, trusted_roots=(task_dir,), module_name="env_dep_pos")
        assert mod.make_env() == 7
        assert str(deps_dir) in sys.path
    finally:
        sys.path[:] = saved_path
        for name in ("server_only_sim_pkg", "env_dep_neg", "env_dep_pos"):
            sys.modules.pop(name, None)
