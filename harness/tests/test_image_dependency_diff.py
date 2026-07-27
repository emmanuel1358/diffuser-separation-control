"""Tests for the authoritative dependency-channel gate (image diff).

The static Dockerfile scan is a convenience layer; these cover the check that
actually guarantees the contract, so they are written against what a build
*produced* rather than how a Dockerfile was written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from alignerr_plugin.validators.task import image_deps


def _packages(
    *,
    apt: tuple[str, ...] = (),
    venv: tuple[str, ...] = (),
    grading: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
    closure: dict[str, tuple[str, ...]] | None = None,
    unparseable: dict[str, tuple[str, ...]] | None = None,
) -> image_deps.ImagePackages:
    return image_deps.ImagePackages(
        apt=frozenset(apt),
        installed={
            "venv": frozenset(venv),
            "grading": frozenset(grading),
            "env": frozenset(env),
        },
        closure={
            label: frozenset((closure or {}).get(label, ()))
            for label in ("venv", "grading", "env")
        },
        unparseable={
            label: tuple((unparseable or {}).get(label, ()))
            for label in ("venv", "grading", "env")
        },
    )


def _task_dir(tmp_path: Path, **channels: str) -> Path:
    for relative, body in (
        ("environment/requirements.txt", channels.get("requirements")),
        ("environment/apt.txt", channels.get("apt")),
        ("scorer/requirements.txt", channels.get("scorer")),
        ("scorer/env-requirements.txt", channels.get("scorer_env")),
    ):
        if body is None:
            continue
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return tmp_path


def test_undeclared_venv_package_is_rejected_and_named(tmp_path: Path) -> None:
    """The property the static scan could not deliver.

    `RUN sh -c 'pip install humanize'` hides the command from any text scan, but
    the package is in the image either way, so the diff still sees it.
    """
    problem_dir = _task_dir(tmp_path, requirements="")
    issues = image_deps.undeclared_package_issues(
        _packages(venv=("numpy",)),
        _packages(venv=("numpy", "humanize")),
        problem_dir,
    )
    assert len(issues) == 1
    assert "humanize" in issues[0]
    assert "environment/requirements.txt" in issues[0]


def test_transitive_dependencies_of_a_declared_root_are_allowed(
    tmp_path: Path,
) -> None:
    """Declaring gymnasium must not require declaring farama-notifications."""
    problem_dir = _task_dir(tmp_path, requirements="gymnasium\n")
    issues = image_deps.undeclared_package_issues(
        _packages(venv=("numpy",)),
        _packages(
            venv=("numpy", "gymnasium", "farama-notifications", "cloudpickle"),
            closure={
                "venv": ("gymnasium", "farama-notifications", "cloudpickle", "numpy")
            },
        ),
        problem_dir,
    )
    assert issues == []


def test_a_package_outside_the_declared_closure_still_fails(tmp_path: Path) -> None:
    """A real dependency tree must not become cover for an undeclared package."""
    problem_dir = _task_dir(tmp_path, requirements="gymnasium\n")
    issues = image_deps.undeclared_package_issues(
        _packages(),
        _packages(
            venv=("gymnasium", "cloudpickle", "humanize"),
            closure={"venv": ("gymnasium", "cloudpickle")},
        ),
        problem_dir,
    )
    assert len(issues) == 1
    assert "humanize" in issues[0]
    assert "cloudpickle" not in issues[0]


def test_redundant_declaration_of_a_base_package_passes(tmp_path: Path) -> None:
    """Declaring what the base already ships is deliberate and self-documenting.

    It produces no delta at all, so it must not be mistaken for a problem in
    either direction.
    """
    problem_dir = _task_dir(tmp_path, requirements="numpy\nopenseespy\n")
    issues = image_deps.undeclared_package_issues(
        _packages(venv=("numpy", "openseespy")),
        _packages(venv=("numpy", "openseespy"), closure={"venv": ("numpy",)}),
        problem_dir,
    )
    assert issues == []


def test_a_non_runtime_interpreter_produces_no_delta(tmp_path: Path) -> None:
    """A conda/solver env installs outside the venv, so it is invisible here.

    That is the intended outcome: those installs have no channel to declare
    them in, and the runtime venv the agent sees is unchanged.
    """
    problem_dir = _task_dir(tmp_path, requirements="")
    issues = image_deps.undeclared_package_issues(
        _packages(venv=("numpy",)),
        _packages(venv=("numpy",)),
        problem_dir,
    )
    assert issues == []


@pytest.mark.parametrize(
    ("tree", "channel_file", "channel_text"),
    [
        ("grading", "scorer", "scorer/requirements.txt"),
        ("env", "scorer_env", "scorer/env-requirements.txt"),
    ],
)
def test_private_target_trees_are_diffed_too(
    tmp_path: Path, tree: str, channel_file: str, channel_text: str
) -> None:
    """The root-only --target trees get the same treatment as the venv."""
    undeclared = image_deps.undeclared_package_issues(
        _packages(),
        _packages(**{tree: ("humanize",)}),
        _task_dir(tmp_path, **{channel_file: ""}),
    )
    assert len(undeclared) == 1
    assert "humanize" in undeclared[0]
    assert channel_text in undeclared[0]

    declared = image_deps.undeclared_package_issues(
        _packages(),
        _packages(**{tree: ("humanize",)}, closure={tree: ("humanize",)}),
        _task_dir(tmp_path, **{channel_file: "humanize\n"}),
    )
    assert declared == []


def test_undeclared_apt_package_is_rejected(tmp_path: Path) -> None:
    problem_dir = _task_dir(tmp_path, apt="ngspice\n")
    issues = image_deps.undeclared_package_issues(
        _packages(apt=("bash",)),
        _packages(apt=("bash", "ngspice", "sl")),
        problem_dir,
    )
    assert len(issues) == 1
    assert "sl" in issues[0]
    assert "ngspice" not in issues[0]
    assert "environment/apt.txt" in issues[0]


def test_a_declared_spec_with_no_resolvable_name_is_reported(tmp_path: Path) -> None:
    """A bare VCS URL installs a package the channel never names.

    Neither passing nor failing on the delta would mean anything, so the
    author is told to use direct reference form instead.
    """
    problem_dir = _task_dir(tmp_path, requirements="git+https://example.test/x.git\n")
    issues = image_deps.undeclared_package_issues(
        _packages(),
        _packages(
            venv=("x",),
            unparseable={"venv": ("git+https://example.test/x.git",)},
        ),
        problem_dir,
    )
    assert len(issues) == 1
    assert "resolvable distribution name" in issues[0]


def test_probe_output_is_read_from_its_marker_line() -> None:
    """An image may print from sitecustomize; the payload is still findable."""
    payload = {"apt": ["bash"], "trees": {}}
    stdout = f"a warning\nLBX_PACKAGE_PROBE {json.dumps(payload)}\n"
    assert image_deps._parse_probe_output(stdout, "img")["apt"] == ["bash"]

    with pytest.raises(image_deps.ProbeError):
        image_deps._parse_probe_output("no marker here", "img")


def test_declared_roots_keep_extras(tmp_path: Path) -> None:
    """`foo[bar]` and `foo` have different closures, so the spec must survive."""
    problem_dir = _task_dir(tmp_path, requirements="gymnasium[box2d]>=1.0\n# note\n")
    assert image_deps.declared_roots(problem_dir)["venv"] == ["gymnasium[box2d]>=1.0"]


def test_declared_roots_follow_requirement_includes(tmp_path: Path) -> None:
    """`uv pip install -r` honors `-r` includes, so the gate must too.

    Dropping the line would leave the included packages in the delta with no
    root to justify them -- a legitimate task failing the authoritative check.
    """
    problem_dir = _task_dir(
        tmp_path, requirements="-r extra.txt\n--index-url https://example.test\nnumpy\n"
    )
    (problem_dir / "environment" / "extra.txt").write_text("gymnasium\n-r deeper.txt\n")
    (problem_dir / "environment" / "deeper.txt").write_text("humanize\n")

    assert image_deps.declared_roots(problem_dir)["venv"] == [
        "gymnasium",
        "humanize",
        "numpy",
    ]


def test_declared_roots_survive_a_self_including_file(tmp_path: Path) -> None:
    problem_dir = _task_dir(tmp_path, requirements="-r requirements.txt\nnumpy\n")
    assert image_deps.declared_roots(problem_dir)["venv"] == ["numpy"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("-e git+https://example.test/x.git#egg=widget", "widget"),
        ("--editable git+https://example.test/x.git#egg=widget&subdir=p", "widget"),
        # No #egg=: nothing here names a distribution, so it is surfaced as-is
        # and reported unresolvable rather than silently dropped.
        ("-e ./local-pkg", "./local-pkg"),
    ],
)
def test_declared_roots_surface_editables(
    tmp_path: Path, line: str, expected: str
) -> None:
    problem_dir = _task_dir(tmp_path, requirements=f"{line}\n")
    assert image_deps.declared_roots(problem_dir)["venv"] == [expected]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("ngspice", "ngspice"),
        ("ngspice=44", "ngspice"),
        ("ngspice:amd64", "ngspice"),
        ("ngspice/trixie-backports", "ngspice"),
        ("libstdc++6", "libstdc++6"),
        ("python3.11", "python3.11"),
    ],
)
def test_apt_qualifiers_are_stripped_for_matching(spec: str, expected: str) -> None:
    """apt accepts a pin that dpkg never reports back under that spelling."""
    assert image_deps.apt_package_name(spec) == expected


def test_a_pinned_apt_declaration_matches_the_installed_package(
    tmp_path: Path,
) -> None:
    problem_dir = _task_dir(tmp_path, apt="ngspice=44\n")
    issues = image_deps.undeclared_package_issues(
        _packages(apt=("bash",)),
        _packages(apt=("bash", "ngspice")),
        problem_dir,
    )
    assert issues == []


class _FakeDistribution:
    def __init__(self, requires: list[str]) -> None:
        self.metadata = self
        self._requires = requires

    def get_all(self, key: str) -> list[str]:
        return self._requires if key == "Requires-Dist" else []


@pytest.fixture(name="probe")
def _probe() -> dict:
    return image_deps.probe_helper_namespace()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("numpy", ("numpy", frozenset())),
        ("gymnasium[box2d,jax]>=1.0", ("gymnasium", frozenset({"box2d", "jax"}))),
        ("Foo_Bar", ("foo-bar", frozenset())),
        ("pkg @ git+https://example.test/x.git", ("pkg", frozenset())),
        # Installed metadata is full of the parenthesised form; dropping it
        # would silently prune real edges out of the closure.
        ("urllib3 (>=1.21.1,<3)", ("urllib3", frozenset())),
        ("requests[socks] (>=2)", ("requests", frozenset({"socks"}))),
        ("git+https://example.test/x.git", None),
        ("./local/wheel.whl", None),
        ("https://example.test/x.whl", None),
    ],
)
def test_probe_parses_requirement_names(
    probe: dict, spec: str, expected: tuple[str, frozenset[str]] | None
) -> None:
    parsed = probe["parse_requirement"](spec)
    assert (parsed[:2] if parsed else None) == expected


def test_probe_closure_follows_only_requested_extras(probe: dict) -> None:
    """The reason the closure is not simply every name in Requires-Dist.

    gymnasium's optional extras reach torch and tensorflow; admitting them
    would let a raw-installed torch pass as a "transitive dependency".
    """
    index = {
        "gymnasium": (
            _FakeDistribution(
                [
                    "numpy>=1.21",
                    "cloudpickle>=1.2",
                    'torch>=1.0; extra == "jax"',
                    'box2d-py==2.3.5 ; extra == "box2d"',
                ]
            ),
            "1.0",
        ),
        "numpy": (_FakeDistribution([]), "2.0"),
        "cloudpickle": (_FakeDistribution([]), "3.0"),
        "torch": (_FakeDistribution(["tensorflow"]), "2.0"),
        "box2d-py": (_FakeDistribution([]), "2.3.5"),
    }

    reached, unparseable = probe["closure"](["gymnasium"], index)
    assert reached == ["cloudpickle", "gymnasium", "numpy"]
    assert unparseable == []

    with_extra, _ = probe["closure"](["gymnasium[box2d]"], index)
    assert "box2d-py" in with_extra
    assert "torch" not in with_extra


def test_probe_closure_follows_parenthesized_requirements(probe: dict) -> None:
    """A dropped edge here rejects a legitimate task, so it must not be dropped."""
    index = {
        "requests": (_FakeDistribution(["urllib3 (>=1.21.1,<3)", "idna (<4)"]), "2.0"),
        "urllib3": (_FakeDistribution([]), "2.0"),
        "idna": (_FakeDistribution([]), "3.0"),
    }
    reached, _ = probe["closure"](["requests"], index)
    assert reached == ["idna", "requests", "urllib3"]


def test_probe_closure_evaluates_non_extra_markers_permissively(probe: dict) -> None:
    """A platform marker only ever hides a package that was not installed.

    Admitting the name costs nothing, and evaluating markers faithfully would
    mean reimplementing PEP 508 inside the probe.
    """
    index = {
        "root": (_FakeDistribution(['pywin32; sys_platform == "win32"']), "1.0"),
    }
    reached, _ = probe["closure"](["root"], index)
    assert reached == ["pywin32", "root"]


def test_probe_closure_terminates_on_a_dependency_cycle(probe: dict) -> None:
    index = {
        "a": (_FakeDistribution(["b"]), "1"),
        "b": (_FakeDistribution(["a"]), "1"),
    }
    reached, _ = probe["closure"](["a"], index)
    assert reached == ["a", "b"]


def test_probe_closure_reports_unparseable_roots(probe: dict) -> None:
    reached, unparseable = probe["closure"](["git+https://example.test/x.git"], {})
    assert reached == []
    assert unparseable == ["git+https://example.test/x.git"]
