from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_upload_docker_source():
    script = REPO_ROOT / "scripts" / "upload_docker_source.py"
    spec = importlib.util.spec_from_file_location("upload_docker_source", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_problem_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    problem = repo / "problems" / "toy"
    (problem / "environment").mkdir(parents=True)
    (problem / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (problem / "task.toml").write_text("[task]\nname = 'toy'\n")
    (problem / "solution").mkdir()
    (problem / "solution" / "solve.sh").write_text("#!/usr/bin/env bash\n")
    (problem / ".alignerr").mkdir()
    (problem / ".alignerr" / "build_proof.json").write_text("{}\n")
    (problem / "baselines").mkdir()
    (problem / "baselines" / "baseline.py").write_text("pass\n")
    (problem / "data-generation").mkdir()
    (problem / "data-generation" / "generate.py").write_text("pass\n")
    (problem / "scorer" / "data").mkdir(parents=True)
    (problem / "scorer" / "data" / "truth.json").write_text("{}\n")
    (repo / "grader").mkdir()
    (repo / ".git").mkdir()
    return repo, problem


def test_build_tarball_includes_extra_files_at_archive_root(tmp_path: Path) -> None:
    module = _load_upload_docker_source()
    repo, _problem = _make_problem_repo(tmp_path)
    generated = tmp_path / "problems-metadata.json"
    generated.write_text('{"problem_set": {"problems": []}}\n')
    notice = tmp_path / "NOTICE.md"
    notice.write_text("# NOTICE\n\n## Deploy-Time Mounts\n")
    tar_path = tmp_path / "source.tar.gz"

    module.build_tarball(
        str(repo),
        "problems/toy",
        str(tar_path),
        extra_files=[
            (str(generated), "problems-metadata.json"),
            (str(notice), "NOTICE.md"),
        ],
    )

    with tarfile.open(tar_path) as tar:
        names = set(tar.getnames())
    assert "Dockerfile" in names
    assert "problems/toy/task.toml" in names
    assert "problems/toy/solution/solve.sh" in names
    assert "problems/toy/.alignerr/build_proof.json" in names
    assert "problems/toy/baselines/baseline.py" in names
    assert "problems/toy/data-generation/generate.py" in names
    assert "problems/toy/scorer/data/truth.json" in names
    assert "grader" in names
    assert "problems-metadata.json" in names
    assert "NOTICE.md" in names
    assert ".git" not in names


def test_require_digest_pinned_rejects_tag_only_refs() -> None:
    module = _load_upload_docker_source()
    with pytest.raises(SystemExit, match="digest-pinned"):
        module.require_digest_pinned("registry.example/repo:latest")
    module.require_digest_pinned("registry.example/repo@sha256:" + ("a" * 64))


def test_require_includes_fails_when_extra_file_missing(tmp_path: Path) -> None:
    module = _load_upload_docker_source()
    repo, _problem = _make_problem_repo(tmp_path)
    tar_path = tmp_path / "source.tar.gz"
    with pytest.raises(SystemExit, match="required include missing"):
        module.build_tarball(
            str(repo),
            "problems/toy",
            str(tar_path),
            extra_files=[(str(tmp_path / "missing.json"), "problems-metadata.json")],
            require_includes=True,
        )


def test_req_treats_bare_timeout_and_oserror_as_soft_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urlopen can raise TimeoutError/OSError outside URLError on py3.13+."""
    module = _load_upload_docker_source()

    def raise_timeout(*_a, **_k):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(module.urllib.request, "urlopen", raise_timeout)
    status, raw = module._req("GET", "https://example.test/x", headers={})
    assert status == 0
    assert b"timed out" in raw

    def raise_oserror(*_a, **_k):
        raise OSError("connection reset")

    monkeypatch.setattr(module.urllib.request, "urlopen", raise_oserror)
    status, raw = module._req("GET", "https://example.test/x", headers={})
    assert status == 0
    assert b"connection reset" in raw


def test_upload_retries_when_req_hits_bare_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_upload_docker_source()
    tar_path = tmp_path / "source.tar.gz"
    tar_path.write_bytes(b"\x1f\x8b" + b"0" * 100)
    calls = {"n": 0}

    def flaky_upload(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate _req soft-failing on TimeoutError (status 0 → upload 1).
            return 1, None
        return 0, "img-retry"

    monkeypatch.setattr(module, "source_verified", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "upload", flaky_upload)
    monkeypatch.setattr(module, "verify_download", lambda _id: True)
    monkeypatch.setattr(module.time, "sleep", lambda *_a, **_k: None)

    assert (
        module.upload_with_retries(
            "registry.example/repo@sha256:" + ("d" * 64),
            "docker build .",
            "desc",
            str(tar_path),
            tar_path.stat().st_size,
            retries=3,
        )
        == 0
    )
    assert calls["n"] == 2


def test_upload_with_retries_succeeds_after_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_upload_docker_source()
    tar_path = tmp_path / "source.tar.gz"
    tar_path.write_bytes(b"\x1f\x8b" + b"0" * 100)

    calls = {"upload": 0, "verify": 0}

    def fake_upload(*_args, **_kwargs):
        calls["upload"] += 1
        if calls["upload"] == 1:
            return 1, None
        return 0, "img-123"

    def fake_verify(image_id: str) -> bool:
        calls["verify"] += 1
        return image_id == "img-123"

    monkeypatch.setattr(module, "source_verified", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "upload", fake_upload)
    monkeypatch.setattr(module, "verify_download", fake_verify)
    monkeypatch.setattr(module.time, "sleep", lambda *_a, **_k: None)

    assert (
        module.upload_with_retries(
            "registry.example/repo@sha256:" + ("b" * 64),
            "docker build .",
            "desc",
            str(tar_path),
            tar_path.stat().st_size,
            retries=3,
        )
        == 0
    )
    assert calls["upload"] == 2
    assert calls["verify"] >= 1


def test_upload_with_retries_fails_when_never_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_upload_docker_source()
    tar_path = tmp_path / "source.tar.gz"
    tar_path.write_bytes(b"\x1f\x8b" + b"0" * 100)

    monkeypatch.setattr(module, "source_verified", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "upload", lambda *_a, **_k: (0, "img-bad"))
    monkeypatch.setattr(module, "verify_download", lambda *_a, **_k: False)
    monkeypatch.setattr(module.time, "sleep", lambda *_a, **_k: None)

    assert (
        module.upload_with_retries(
            "registry.example/repo@sha256:" + ("c" * 64),
            "docker build .",
            "desc",
            str(tar_path),
            tar_path.stat().st_size,
            retries=2,
        )
        == 1
    )


def test_upload_with_retries_verifies_once_per_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flaky second verify must not overturn a successful first probe."""
    module = _load_upload_docker_source()
    tar_path = tmp_path / "source.tar.gz"
    tar_path.write_bytes(b"\x1f\x8b" + b"0" * 100)
    verify_calls = {"n": 0}

    def flaky_second_verify(_image_id: str) -> bool:
        verify_calls["n"] += 1
        # First call succeeds; any accidental second call would fail.
        return verify_calls["n"] == 1

    monkeypatch.setattr(module, "source_verified", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "upload", lambda *_a, **_k: (0, "img-once"))
    monkeypatch.setattr(module, "verify_download", flaky_second_verify)
    monkeypatch.setattr(module.time, "sleep", lambda *_a, **_k: None)

    assert (
        module.upload_with_retries(
            "registry.example/repo@sha256:" + ("e" * 64),
            "docker build .",
            "desc",
            str(tar_path),
            tar_path.stat().st_size,
            retries=2,
        )
        == 0
    )
    assert verify_calls["n"] == 1


def test_already_registered_skips_only_when_downloadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_upload_docker_source()
    image = "registry.example/repo@sha256:" + ("d" * 64)

    monkeypatch.setattr(
        module,
        "find_registered",
        lambda _name: {"id": "rec-1", "image_name": image},
    )
    monkeypatch.setattr(module, "verify_download", lambda _id: True)
    assert module.source_verified(image) == "rec-1"

    monkeypatch.setattr(module, "verify_download", lambda _id: False)
    assert module.source_verified(image) is None


def test_main_dry_run_rejects_non_digest_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_upload_docker_source()
    repo, _problem = _make_problem_repo(tmp_path)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "upload_docker_source.py",
            "--image-name",
            "registry.example/repo:tag-only",
            "--problem-dir",
            "problems/toy",
            "--repo-root",
            str(repo),
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit, match="digest-pinned"):
        module.main()
