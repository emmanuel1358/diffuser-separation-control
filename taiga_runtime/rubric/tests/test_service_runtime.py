from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
from rubric.capsule_runtime import CapsuleBundle, CapsuleImageArchive
from rubric.service_config import (
    CaptureHook,
    RuntimeOperatorRoots,
    ServiceArtifact,
    ServiceSpec,
    TaskServiceConfig,
    ToolEndpoint,
    WorkspaceRuntimeSpec,
    load_task_service_config,
)
from rubric.service_runtime import (
    CaptureInfrastructureError,
    CommandResult,
    ServiceSecurityError,
    ServiceStartupError,
    SubprocessCommandRunner,
    TaskServiceRuntime,
    VerifierInfrastructureError,
)


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 987654
        self.returncode = None

    def poll(self) -> int | None:
        return self.returncode


class FakeRunner:
    def __init__(
        self,
        *,
        compose_services: Mapping[str, Mapping[str, object]],
        image_digests: Mapping[str, str],
        files: Mapping[str, bytes] | None = None,
        failing_capture: str | None = None,
        verifier_exit: int = 0,
        dynamic_reward: bool = False,
    ) -> None:
        self.compose_services = {
            name: dict(value) for name, value in compose_services.items()
        }
        self.image_digests = dict(image_digests)
        self.files = dict(files or {})
        self.failing_capture = failing_capture
        self.verifier_exit = verifier_exit
        self.dynamic_reward = dynamic_reward
        self.captured_input_size = 0
        self.commands: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []
        self.inputs: list[bytes | None] = []
        self.started: list[tuple[str, ...]] = []
        self.stop_count = 0
        self.process = FakeProcess()
        self.state_calls: dict[str, int] = {}
        self.verifier_started = False
        self.fail_init = False
        self.main_uid = 1000

    @staticmethod
    def _compose_action(argv: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
        project_index = argv.index("--project-name")
        action_index = project_index + 2
        return argv[action_index], argv[action_index + 1 :]

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_s: float,
        max_output_bytes: int,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        del timeout_s, max_output_bytes
        args = tuple(str(value) for value in argv)
        self.commands.append(args)
        self.environments.append(dict(env))
        self.inputs.append(input_bytes)

        if args[:2] == ("docker", "info"):
            return CommandResult(args, 0, stdout='"fake"\n')
        if args[:3] == ("docker", "image", "load"):
            return CommandResult(args, 0, stdout="Loaded\n")
        if args[:3] == ("docker", "image", "inspect"):
            image_ref = args[-1]
            if "--format" in args:
                return CommandResult(
                    args,
                    0,
                    stdout=json.dumps(
                        {
                            "Cmd": ["/verify"],
                            "Entrypoint": [],
                            "WorkingDir": "/verifier-work",
                        }
                    ),
                )
            return CommandResult(
                args,
                0,
                stdout=json.dumps(
                    [
                        {
                            "Id": self.image_digests[image_ref],
                            "RepoDigests": [],
                        }
                    ]
                ),
            )
        if args[:2] == ("docker", "compose"):
            action, remainder = self._compose_action(args)
            if action == "config":
                return CommandResult(
                    args,
                    0,
                    stdout=json.dumps({"services": self.compose_services}),
                )
            if action == "ps":
                service = remainder[-1]
                return CommandResult(args, 0, stdout=f"{service}-id\n")
            if action == "start":
                self.verifier_started = True
                override_indexes = [
                    index for index, value in enumerate(args) if value == "-f"
                ]
                override_path = Path(args[override_indexes[-1] + 1])
                override = json.loads(override_path.read_text())
                volumes = override["services"]["verifier"]["volumes"]
                for volume in volumes:
                    target = volume["target"]
                    source = Path(volume["source"])
                    content = self.files.get(f"verifier-id:{target}")
                    if self.dynamic_reward and target == "/logs/verifier/reward.json":
                        content = json.dumps(
                            {"score": self.captured_input_size / 100.0}
                        ).encode()
                    if content is not None and not volume.get("read_only"):
                        source.write_bytes(content)
            if (
                action == "exec"
                and self.failing_capture
                and self.failing_capture in args
            ):
                return CommandResult(args, 7, stderr="capture failed")
            if action == "exec" and args[-2:] == ("id", "-u"):
                return CommandResult(args, 0, stdout=f"{self.main_uid}\n")
            if action == "exec" and any('base64 < "$1"' in value for value in args):
                return CommandResult(
                    args,
                    0,
                    stdout=base64.b64encode(b"\x00\xffbinary").decode(),
                )
            if action == "exec" and any('test -f "$1"' in value for value in args):
                return CommandResult(args, 0, stdout="old value\n")
            return CommandResult(args, 0)
        if args[:3] == ("docker", "inspect", "--format"):
            container_id = args[-1]
            service = container_id.removesuffix("-id")
            if "Config.WorkingDir" in args[-2]:
                return CommandResult(args, 0, stdout='"/verifier-work"\n')
            if "NetworkSettings.Networks" in args[-2]:
                return CommandResult(
                    args,
                    0,
                    stdout=json.dumps(
                        {
                            "lbx": {
                                "IPAddress": (
                                    "172.30.0.10" if service == "api" else "172.30.0.11"
                                )
                            }
                        }
                    ),
                )
            call = self.state_calls.get(service, 0)
            self.state_calls[service] = call + 1
            if service == "migrate":
                state = (
                    {"Status": "running", "Running": True}
                    if call == 0
                    else {
                        "Status": "exited",
                        "Running": False,
                        "ExitCode": 9 if self.fail_init else 0,
                    }
                )
            elif service == "main":
                state = (
                    {
                        "Status": "running",
                        "Running": True,
                        "Health": {"Status": "starting"},
                    }
                    if call == 0
                    else {
                        "Status": "running",
                        "Running": True,
                        "Health": {"Status": "healthy"},
                    }
                )
            elif service == "verifier":
                state = (
                    {
                        "Status": "exited",
                        "Running": False,
                        "ExitCode": self.verifier_exit,
                    }
                    if self.verifier_started
                    else {"Status": "created", "Running": False}
                )
            else:
                state = {
                    "Status": "running",
                    "Running": True,
                    "Health": {"Status": "healthy"},
                }
            return CommandResult(args, 0, stdout=json.dumps(state))
        if args[:2] == ("docker", "cp"):
            source, destination = args[-2:]
            content = self.files.get(source)
            if content is not None:
                destination_path = Path(destination)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                destination_path.write_bytes(content)
                if source.startswith("main-id:"):
                    self.captured_input_size = len(content)
                return CommandResult(args, 0)
            if ":" in source:
                return CommandResult(args, 1, stderr=f"missing {source}")
            return CommandResult(args, 0)
        if args[:2] == ("docker", "exec"):
            return CommandResult(args, 0)
        raise AssertionError(f"unexpected fake command: {args}")

    def stream_to_file(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        output_path: Path,
        timeout_s: float,
        max_bytes: int,
        max_error_bytes: int,
    ) -> CommandResult:
        del timeout_s, max_error_bytes
        args = tuple(str(value) for value in argv)
        self.commands.append(args)
        self.environments.append(dict(env))
        self.inputs.append(None)
        source = args[-2]
        content = self.files.get(source)
        if content is None:
            return CommandResult(args, 1, stderr=f"missing {source}")
        with tarfile.open(output_path, "w") as archive:
            member = tarfile.TarInfo(name=PurePosixPath(source.split(":", 1)[1]).name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
        if source.startswith("main-id:"):
            self.captured_input_size = len(content)
        if output_path.stat().st_size > max_bytes:
            return CommandResult(args, 1, stdout_truncated=True)
        return CommandResult(args, 0)

    def start(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        output_path: Path,
        max_output_bytes: int,
    ) -> FakeProcess:
        del output_path, max_output_bytes
        args = tuple(str(value) for value in argv)
        self.started.append(args)
        self.environments.append(dict(env))
        return self.process

    def stop_process_group(self, process: FakeProcess, *, grace_s: float) -> None:
        del grace_s
        assert process is self.process
        self.stop_count += 1
        process.returncode = -15


def _runtime_parts(
    tmp_path: Path,
    *,
    captures: tuple[CaptureHook, ...] = (),
    artifacts: tuple[ServiceArtifact, ...] = (),
    tools: tuple[ToolEndpoint, ...] = (),
    verifier: bool = True,
    compose_override: Mapping[str, Mapping[str, object]] | None = None,
    files: Mapping[str, bytes] | None = None,
    failing_capture: str | None = None,
    verifier_exit: int = 0,
    dynamic_reward: bool = False,
    workspace: WorkspaceRuntimeSpec | None = None,
) -> tuple[TaskServiceRuntime, FakeRunner]:
    services = (
        ServiceSpec(
            name="main",
            role="main",
            user="1000:1000",
            workdir="/workspace",
            shell="/bin/bash",
        ),
        ServiceSpec(name="api", role="sidecar"),
        ServiceSpec(name="migrate", role="init"),
        *((ServiceSpec(name="verifier", role="verifier"),) if verifier else ()),
    )
    roots = RuntimeOperatorRoots(
        capsule=tmp_path / "capsule",
        state=tmp_path / "state-root",
        sealed=tmp_path / "sealed-root",
    )
    config = TaskServiceConfig(
        task_toml_path=tmp_path / "task.toml",
        operator_roots=roots,
        capsule_dir=tmp_path / "capsule",
        capsule_manifest="manifest.json",
        compose_file=None,
        state_dir=roots.state / "lbx-test",
        sealed_dir=roots.sealed / "lbx-test",
        project_name="lbx-test",
        services=services,
        main_service="main",
        verifier_service="verifier" if verifier else None,
        agent_user="1000:1000",
        agent_workdir="/workspace",
        agent_shell="/bin/bash",
        startup_timeout_s=3.0,
        daemon_timeout_s=3.0,
        command_timeout_s=5.0,
        verifier_timeout_s=3.0,
        max_output_bytes=1024 * 1024,
        editor_max_bytes=1024 * 1024,
        workspace=workspace,
        captures=captures,
        artifacts=artifacts,
        tools=tools,
        verifier_result_paths=("/logs/verifier/reward.json",),
        verifier_reward_path="/logs/verifier/reward.json",
        primary_reward="score",
        subscores_key="subscores",
    )
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    compose_path = capsule / "docker-compose.yaml"
    compose_path.write_text("services: {}\n")
    archive_path = capsule / "images.tar"
    archive_path.write_bytes(b"archive")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    images: list[CapsuleImageArchive] = []
    image_digests: dict[str, str] = {}
    compose_services: dict[str, dict[str, object]] = {}
    for index, service in enumerate(services, start=1):
        image_ref = f"lbx/{service.name}:locked"
        image_digest = f"sha256:{index:064x}"
        image_digests[image_ref] = image_digest
        images.append(
            CapsuleImageArchive(
                service=service.name,
                role=service.role,
                archive_path=archive_path,
                archive_sha256=archive_sha,
                image_ref=image_ref,
                image_digest=image_digest,
            )
        )
        compose_services[service.name] = {
            "cap_drop": ["ALL"],
            "image": image_ref,
            "pull_policy": "never",
            "security_opt": ["no-new-privileges:true"],
        }
        if service.role == "verifier":
            compose_services[service.name]["network_mode"] = "none"
    if compose_override:
        for service, row in compose_override.items():
            compose_services[service] = dict(row)
    bundle = CapsuleBundle(
        root=capsule,
        manifest_path=capsule / "manifest.json",
        compose_path=compose_path,
        images=tuple(images),
    )
    runner = FakeRunner(
        compose_services=compose_services,
        image_digests=image_digests,
        files=files,
        failing_capture=failing_capture,
        verifier_exit=verifier_exit,
        dynamic_reward=dynamic_reward,
    )
    runtime = TaskServiceRuntime(
        config,
        runner=runner,
        bundle=bundle,
        sleep=lambda _seconds: None,
    )
    return runtime, runner


def _index(commands: list[tuple[str, ...]], predicate) -> int:
    return next(index for index, command in enumerate(commands) if predicate(command))


def test_host_command_runner_bounds_output_and_kills_timed_out_group() -> None:
    runner = SubprocessCommandRunner()

    noisy = runner.run(
        (sys.executable, "-c", "print('x' * 4096)"),
        env=os.environ,
        timeout_s=2.0,
        max_output_bytes=64,
    )
    timed_out = runner.run(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        env=os.environ,
        timeout_s=0.05,
        max_output_bytes=64,
    )

    assert noisy.returncode == 0
    assert noisy.stdout_truncated
    assert "output truncated" in noisy.stdout
    assert timed_out.timed_out
    assert timed_out.returncode != 0


def test_lifecycle_loads_once_then_waits_for_health_and_init(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(tmp_path)

    runtime.start()
    runtime.start()

    assert len(runner.started) == 1
    assert runner.started[0][0] == "dockerd"
    assert (
        sum(command[:3] == ("docker", "image", "load") for command in runner.commands)
        == 1
    )
    config_index = _index(runner.commands, lambda command: "config" in command)
    up_index = _index(runner.commands, lambda command: "up" in command)
    inspect_index = _index(
        runner.commands,
        lambda command: command[:3] == ("docker", "inspect", "--format"),
    )
    assert config_index < up_index < inspect_index
    assert runner.state_calls["main"] >= 2
    assert runner.state_calls["migrate"] >= 2
    up_command = runner.commands[up_index]
    assert "--pull" in up_command
    assert up_command[up_command.index("--pull") + 1] == "never"


def test_failed_init_job_is_an_infrastructure_startup_failure(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(tmp_path)
    runner.fail_init = True

    with pytest.raises(ServiceStartupError, match="init container exited"):
        runtime.start()

    assert runner.stop_count == 1


def test_command_proxy_uses_argv_and_unprivileged_main_user(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(tmp_path)
    runtime.start()

    result = runtime.exec_main("printf '%s' hello")

    assert result.returncode == 0
    command = runner.commands[-1]
    assert "exec" in command
    assert command[command.index("--user") + 1] == "1000:1000"
    assert command[command.index("--workdir") + 1] == "/workspace"
    assert command[-3:] == (
        "/bin/bash",
        "-lc",
        "printf '%s' hello",
    )
    assert all("docker.sock" not in value for value in command)


def test_workspace_seed_clean_and_checkpoint_contract_is_executed(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "answer.txt").write_text("candidate")
    (tmp_path / "task.toml").write_text("[task]\nname='workspace'\n")
    workspace = WorkspaceRuntimeSpec(
        seed="seed",
        root="/workspace",
        agent_cwd="/workspace",
        init_policy="copy",
        git_baseline=False,
        clean_paths=("build",),
        checkpoint_restore=True,
    )
    runtime, runner = _runtime_parts(tmp_path, workspace=workspace)

    runtime.start()

    assert any(
        command[-5:] == ("tar", "-xf", "-", "-C", "/workspace")
        for command in runner.commands
    )
    assert any(
        command[-4:] == ("rm", "-rf", "--", "/workspace/build")
        for command in runner.commands
    )
    assert any(data is not None and b"answer.txt" in data for data in runner.inputs)
    assert any(
        data is not None and b'"checkpoint_restore":true' in data
        for data in runner.inputs
    )


def test_editor_reads_and_atomically_writes_as_unprivileged_user(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(tmp_path)
    runtime.start()

    output = runtime.edit_main_file(
        command="str_replace",
        path="/workspace/note.txt",
        old_str="old",
        new_str="new",
    )

    assert output == "updated /workspace/note.txt"
    read_index = _index(
        runner.commands,
        lambda command: any('test -f "$1"' in value for value in command),
    )
    write_index = _index(
        runner.commands,
        lambda command: any(".lbx-edit-$$" in value for value in command),
    )
    assert read_index < write_index
    write_command = runner.commands[write_index]
    assert write_command[write_command.index("--user") + 1] == "1000:1000"
    assert runner.inputs[write_index] == b"new value\n"


def test_binary_file_copy_uses_supervisor_proxy_without_socket(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(tmp_path)
    runtime.start()
    source = tmp_path / "payload.bin"
    source.write_bytes(b"\x01\xfehost")
    destination = tmp_path / "copied.bin"

    runtime.copy_file_to_main(source, "/workspace/payload.bin")
    runtime.copy_file_from_main("/workspace/result.bin", destination)

    write_index = _index(
        runner.commands,
        lambda command: any(".lbx-edit-$$" in value for value in command),
    )
    assert runner.inputs[write_index] == b"\x01\xfehost"
    assert destination.read_bytes() == b"\x00\xffbinary"
    assert all(
        "/var/run/docker.sock" not in value
        for command in runner.commands
        for value in command
    )


def test_compose_socket_mount_is_rejected_before_service_start(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(
        tmp_path,
        compose_override={
            "main": {
                "image": "lbx/main:locked",
                "pull_policy": "never",
                "volumes": [
                    {
                        "type": "bind",
                        "source": "/var/run/docker.sock",
                        "target": "/run/docker.sock",
                    }
                ],
            }
        },
    )

    with pytest.raises(ServiceSecurityError, match="bind mount"):
        runtime.start()

    assert not any("up" in command for command in runner.commands)
    assert runner.stop_count == 1


@pytest.mark.parametrize(
    ("unsafe_field", "expected"),
    [
        ({"volumes": ["/host:/data"]}, "bind mount"),
        ({"volumes_from": ["outer"]}, "volumes_from"),
        ({"devices": ["/dev/kvm:/dev/kvm"]}, "devices"),
        ({"privileged": True}, "privileged"),
        ({"cap_add": ["CAP_SYS_ADMIN"]}, "dangerous capabilities"),
        ({"pid": "host"}, "host pid namespace"),
        ({"security_opt": ["seccomp:unconfined"]}, "disable container"),
    ],
)
def test_compose_privilege_escape_paths_fail_closed(
    tmp_path: Path,
    unsafe_field: dict[str, object],
    expected: str,
) -> None:
    runtime, runner = _runtime_parts(
        tmp_path,
        compose_override={
            "main": {
                "image": "lbx/main:locked",
                "pull_policy": "never",
                **unsafe_field,
            }
        },
    )

    with pytest.raises(ServiceSecurityError, match=expected):
        runtime.start()

    assert not any("up" in command for command in runner.commands)


def test_undeclared_compose_dependency_is_rejected(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(
        tmp_path,
        compose_override={
            "escape": {
                "image": "attacker:latest",
                "privileged": True,
                "volumes": ["/var/run/docker.sock:/var/run/docker.sock"],
            }
        },
    )

    with pytest.raises(ServiceSecurityError, match="undeclared services"):
        runtime.start()

    assert not any("up" in command for command in runner.commands)


@pytest.mark.parametrize(
    ("verifier_extra", "main_extra", "expected"),
    [
        ({"network_mode": "bridge"}, {}, "network_mode='none'"),
        (
            {"depends_on": {"main": {"condition": "service_started"}}},
            {},
            "cannot depend",
        ),
        (
            {
                "volumes": [
                    {
                        "type": "volume",
                        "source": "shared",
                        "target": "/verifier-data",
                    }
                ]
            },
            {
                "volumes": [
                    {
                        "type": "volume",
                        "source": "shared",
                        "target": "/agent-data",
                    }
                ]
            },
            "cannot share volumes",
        ),
    ],
)
def test_verifier_is_networkless_and_isolated_from_agent_graph(
    tmp_path: Path,
    verifier_extra: dict[str, object],
    main_extra: dict[str, object],
    expected: str,
) -> None:
    runtime, _runner = _runtime_parts(
        tmp_path,
        compose_override={
            "main": {
                "cap_drop": ["ALL"],
                "image": "lbx/main:locked",
                "pull_policy": "never",
                "security_opt": ["no-new-privileges:true"],
                **main_extra,
            },
            "verifier": {
                "cap_drop": ["ALL"],
                "image": "lbx/verifier:locked",
                "network_mode": "none",
                "pull_policy": "never",
                "security_opt": ["no-new-privileges:true"],
                **verifier_extra,
            },
        },
    )

    with pytest.raises(ServiceSecurityError, match=expected):
        runtime.start()


@pytest.mark.parametrize(
    ("main_policy", "expected"),
    [
        ({"cap_drop": ["ALL"]}, "no-new-privileges"),
        (
            {"security_opt": ["no-new-privileges:true"]},
            "drop all capabilities",
        ),
    ],
)
def test_main_requires_privilege_hardening(
    tmp_path: Path,
    main_policy: dict[str, object],
    expected: str,
) -> None:
    runtime, _runner = _runtime_parts(
        tmp_path,
        compose_override={
            "main": {
                "image": "lbx/main:locked",
                "pull_policy": "never",
                **main_policy,
            }
        },
    )

    with pytest.raises(ServiceSecurityError, match=expected):
        runtime.start()


def test_effective_root_uid_is_rejected(tmp_path: Path) -> None:
    runtime, runner = _runtime_parts(tmp_path)
    runner.main_uid = 0

    with pytest.raises(ServiceSecurityError, match="nonzero UID"):
        runtime.start()


def test_capture_order_freezes_main_before_sidecars_and_runs_verifier(
    tmp_path: Path,
) -> None:
    captures = (
        CaptureHook(
            service="api",
            command="capture-api",
            timeout_s=10.0,
            user=None,
            accepted_exit_codes=(0,),
            atomic_destination=None,
            failure_policy="infrastructure",
        ),
        CaptureHook(
            service="main",
            command="capture-main",
            timeout_s=10.0,
            user="1000:1000",
            accepted_exit_codes=(0,),
            atomic_destination="/tmp/main.capture",
            failure_policy="infrastructure",
        ),
    )
    artifacts = (
        ServiceArtifact(
            kind="file",
            source="/workspace/result.txt",
            service="main",
            destination="main/result.txt",
            required=True,
            exclude=(),
            max_bytes=1024,
            max_files=10,
            max_depth=3,
            preserve_mode=False,
        ),
        ServiceArtifact(
            kind="service",
            source="/tmp/api.json",
            service="api",
            destination="api/api.json",
            required=True,
            exclude=(),
            max_bytes=1024,
            max_files=10,
            max_depth=3,
            preserve_mode=False,
        ),
    )
    runtime, runner = _runtime_parts(
        tmp_path,
        captures=captures,
        artifacts=artifacts,
        files={
            "main-id:/workspace/result.txt": b"agent output\n",
            "api-id:/tmp/api.json": b'{"requests": 4}\n',
            "verifier-id:/logs/verifier/reward.json": b'{"score": 0.75}\n',
        },
    )
    runtime.start()

    result = runtime.finalize_and_verify()
    again = runtime.finalize_and_verify()

    assert result is not None
    assert result.payload["score"] == pytest.approx(0.75)
    assert again is result
    atomic_clear = _index(
        runner.commands,
        lambda command: "/tmp/main.capture" in command
        and any("rm -f" in value for value in command),
    )
    main_capture = _index(runner.commands, lambda command: "capture-main" in command)
    atomic_check = _index(
        runner.commands,
        lambda command: "/tmp/main.capture" in command
        and any("test -e" in value for value in command),
    )
    main_copy = _index(
        runner.commands,
        lambda command: "main-id:/workspace/result.txt" in command,
    )
    graph_pause = _index(runner.commands, lambda command: "pause" in command)
    api_capture = _index(runner.commands, lambda command: "capture-api" in command)
    api_copy = _index(
        runner.commands, lambda command: "api-id:/tmp/api.json" in command
    )
    stop = _index(runner.commands, lambda command: "stop" in command)
    create = _index(runner.commands, lambda command: "create" in command)
    assert (
        api_capture
        < atomic_clear
        < main_capture
        < atomic_check
        < graph_pause
        < main_copy
        < api_copy
        < stop
        < create
    )
    start_verifier = _index(
        runner.commands,
        lambda command: "start" in command and "verifier" in command,
    )
    assert create < start_verifier
    pause_command = runner.commands[graph_pause]
    assert pause_command[-2:] == ("main", "api")
    create_command = runner.commands[create]
    assert "--no-deps" in create_command
    override = json.loads(
        (runtime.config.state_dir / "verifier-override.json").read_text()
    )
    verifier_override = override["services"]["verifier"]
    assert "working_dir" not in verifier_override
    assert verifier_override["network_mode"] == "none"
    assert verifier_override["entrypoint"] == [
        "/bin/sh",
        "/lbx/runtime/verifier-wrapper.sh",
    ]
    assert verifier_override["command"] == ["/verify"]
    assert verifier_override["volumes"][0]["source"] == str(
        runtime.artifact_snapshot.root
    )
    assert verifier_override["volumes"][0]["read_only"] is True
    wrapper = (runtime.config.state_dir / "verifier-wrapper.sh").read_text()
    assert ": > /logs/verifier/reward.json" in wrapper
    assert 'exec "$@"' in wrapper


def test_nonzero_verifier_exit_rejects_even_existing_reward(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(
        tmp_path,
        verifier_exit=9,
        files={
            "verifier-id:/logs/verifier/reward.json": b'{"score": 1.0}\n',
        },
    )
    runtime.start()

    with pytest.raises(VerifierInfrastructureError, match="status 9"):
        runtime.finalize_and_verify()

    assert not any(
        command[:2] == ("docker", "cp")
        and command[-2] == "verifier-id:/logs/verifier/reward.json"
        for command in runner.commands
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"other": 0.5},
        {"score": float("nan")},
        {"score": float("inf")},
    ],
)
def test_canonical_reward_requires_exact_finite_reward_key(
    tmp_path: Path,
    payload: dict[str, float],
) -> None:
    runtime, _runner = _runtime_parts(tmp_path)
    reward = tmp_path / "reward.json"
    reward.write_text(json.dumps(payload))

    with pytest.raises(
        VerifierInfrastructureError,
        match="reward_key|non-finite|invalid verifier JSON",
    ):
        runtime._canonical_payload({"/logs/verifier/reward.json": reward})

    runtime.config = replace(runtime.config, primary_reward=None)
    reward.write_text('{"score": 0.5}')
    assert runtime.config.primary_reward == "score"
    assert runtime._canonical_payload({"/logs/verifier/reward.json": reward})[
        "score"
    ] == pytest.approx(0.5)


def test_verifier_without_result_uses_schema_default_dynamic_score(
    tmp_path: Path,
) -> None:
    task_toml = tmp_path / "task.toml"
    task_toml.write_text("""
[task]
name = "default-verifier-reward"

[[services]]
name = "main"
role = "main"
user = "1000:1000"

[[services]]
name = "verifier"
role = "verifier"
""")
    roots = RuntimeOperatorRoots(
        capsule=tmp_path / "capsule",
        state=tmp_path / "state",
        sealed=tmp_path / "sealed",
    )
    config = load_task_service_config(task_toml, operator_roots=roots)

    assert config is not None
    assert config.verifier_service == "verifier"
    assert config.primary_reward == "score"
    runtime = TaskServiceRuntime(config)
    observed: list[float] = []
    for index, score in enumerate((0.19, 0.83)):
        reward = tmp_path / f"reward-{index}.json"
        reward.write_text(json.dumps({"score": score}))
        payload = runtime._canonical_payload({config.verifier_reward_path: reward})
        observed.append(payload["score"])

    assert observed == [0.19, 0.83]


def test_dynamic_verifier_score_depends_on_sealed_candidate(
    tmp_path: Path,
) -> None:
    scores: list[float] = []
    for name, content in (("short", b"abc"), ("long", b"abcdefghij")):
        case = tmp_path / name
        case.mkdir()
        artifact = ServiceArtifact(
            kind="file",
            source="/workspace/result.txt",
            service="main",
            destination="workspace/result.txt",
            required=True,
            exclude=(),
            max_bytes=1024,
            max_files=1,
            max_depth=1,
            preserve_mode=False,
        )
        runtime, _runner = _runtime_parts(
            case,
            artifacts=(artifact,),
            files={"main-id:/workspace/result.txt": content},
            dynamic_reward=True,
        )
        runtime.start()
        result = runtime.finalize_and_verify()
        assert result is not None
        scores.append(result.payload["score"])
        handoff = runtime.grader_handoff()
        assert (handoff.workspace / "workspace/result.txt").read_bytes() == content

    assert scores == [0.03, 0.1]


def test_capture_failure_is_infrastructure_and_skips_verifier(
    tmp_path: Path,
) -> None:
    captures = (
        CaptureHook(
            service="api",
            command="fail-capture",
            timeout_s=10.0,
            user=None,
            accepted_exit_codes=(0,),
            atomic_destination=None,
            failure_policy="infrastructure",
        ),
    )
    runtime, runner = _runtime_parts(
        tmp_path,
        captures=captures,
        failing_capture="fail-capture",
    )
    runtime.start()

    with pytest.raises(CaptureInfrastructureError, match="status 7"):
        runtime.finalize_and_verify()

    assert any("stop" in command for command in runner.commands)
    assert not any("create" in command for command in runner.commands)


def test_cleanup_is_idempotent_and_stops_daemon_process_group(
    tmp_path: Path,
) -> None:
    runtime, runner = _runtime_parts(tmp_path)
    runtime.start()

    runtime.cleanup()
    runtime.cleanup()

    down_commands = [command for command in runner.commands if "down" in command]
    assert len(down_commands) == 1
    assert runner.stop_count == 1
    assert runtime.closed


def test_preexisting_state_directory_is_never_mutated_or_removed(
    tmp_path: Path,
) -> None:
    runtime, _runner = _runtime_parts(tmp_path)
    runtime.config.operator_roots.state.mkdir()
    runtime.config.state_dir.mkdir()
    marker = runtime.config.state_dir / "operator-owned"
    marker.write_text("keep")

    with pytest.raises(ServiceSecurityError, match="pre-existing"):
        runtime.start()
    runtime.cleanup()

    assert marker.read_text() == "keep"


def test_artifact_depth_and_binary_mode_semantics_are_enforced(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "tree"
    (nested / "one" / "two").mkdir(parents=True)
    (nested / "one" / "two" / "result.txt").write_text("value")

    with pytest.raises(CaptureInfrastructureError, match="depth 1"):
        TaskServiceRuntime._measure_and_digest(nested, max_depth=1)

    case = tmp_path / "binary"
    case.mkdir()
    artifact = ServiceArtifact(
        kind="binary",
        source="/workspace/tool",
        service="main",
        destination="workspace/tool",
        required=True,
        exclude=(),
        max_bytes=1024,
        max_files=1,
        max_depth=1,
        preserve_mode=True,
    )
    runtime, _runner = _runtime_parts(
        case,
        artifacts=(artifact,),
        verifier=False,
        files={"main-id:/workspace/tool": b"binary"},
    )
    runtime.start()
    assert runtime.finalize_and_verify() is None
    handoff = runtime.grader_handoff()

    assert handoff.workspace.joinpath("workspace/tool").stat().st_mode & 0o777 == 0o644
    manifest = json.loads(handoff.manifest.read_text())
    assert manifest["artifacts"][0]["mode"] == 0o644


def test_artifact_byte_limit_is_enforced_during_tar_extraction(
    tmp_path: Path,
) -> None:
    artifact = ServiceArtifact(
        kind="file",
        source="/workspace/oversized.bin",
        service="main",
        destination="workspace/oversized.bin",
        required=True,
        exclude=(),
        max_bytes=8,
        max_files=1,
        max_depth=1,
        preserve_mode=False,
    )
    runtime, _runner = _runtime_parts(
        tmp_path,
        artifacts=(artifact,),
        verifier=False,
        files={"main-id:/workspace/oversized.bin": b"x" * 1024},
    )
    runtime.start()

    with pytest.raises(CaptureInfrastructureError, match="exceeds 8 bytes"):
        runtime.finalize_and_verify()

    assert not list(runtime.config.sealed_dir.rglob("oversized.bin"))


def test_artifact_tar_limits_are_checked_before_extraction(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "artifact.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name in ("tree/one.txt", "tree/nested/two.txt"):
            member = tarfile.TarInfo(name)
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "destination"
    artifact = ServiceArtifact(
        kind="tree",
        source="/workspace/tree",
        service="main",
        destination="workspace/tree",
        required=True,
        exclude=(),
        max_bytes=10,
        max_files=1,
        max_depth=1,
        preserve_mode=False,
    )

    with pytest.raises(CaptureInfrastructureError, match="files|depth"):
        TaskServiceRuntime._extract_artifact_archive(
            archive_path,
            destination,
            artifact,
        )

    assert not destination.exists()


def test_subprocess_artifact_stream_is_stopped_at_byte_limit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stream.bin"
    result = SubprocessCommandRunner().stream_to_file(
        (sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 1000000)"),
        env=os.environ,
        output_path=output,
        timeout_s=5.0,
        max_bytes=1024,
        max_error_bytes=1024,
    )

    assert result.stdout_truncated
    assert output.stat().st_size == 1024


def test_task_local_sse_mcp_becomes_ready_and_resolves_service_dns(
    tmp_path: Path,
) -> None:
    endpoint = ToolEndpoint(
        name="browser",
        transport="sse",
        service="api",
        url="http://api:3080/sse",
        command=(),
        readiness_kind=None,
        readiness_service="api",
        readiness_command=None,
        readiness_url=None,
        readiness_host=None,
        readiness_port=None,
        readiness_timeout_s=10.0,
        readiness_interval_s=0.1,
    )
    runtime, _runner = _runtime_parts(tmp_path, tools=(endpoint,))
    runtime.start()

    assert runtime.tools.readiness("browser").ready
    assert runtime._resolve_tool_sse_url(endpoint) == "http://172.30.0.10:3080/sse"
