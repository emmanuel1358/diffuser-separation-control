"""Authoritative dependency-channel gate: diff the built image against the base.

The static Dockerfile scan in ``validator.py`` reads how an install is *written*,
which is a losing game -- ``RUN sh -c 'pip install x'``, an alias, a shim, a
variable, exec form. This module reads what the build *produced* instead:
enumerate the distributions in the runtime venv (and the two private target
trees) plus the dpkg set in the base image and in the built task image, and hold
the author to declaring the difference. No spelling of an install command can
evade it, because it never looks at the command.

Transitive dependencies are the reason a naive diff does not work: declaring
``gymnasium`` also installs ``farama-notifications`` and ``cloudpickle``, which
no author writes down. The closure of the declared roots is resolved from the
``Requires-Dist`` metadata *inside the built image*, so it reflects what was
actually installed for the actual markers and extras rather than what an index
says today.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

RUNTIME_VENV_PYTHON = "/opt/lbx-runtime/.venv/bin/python"
GRADING_DEPS_DIR = "/mcp_server/grading_deps"
ENV_DEPS_DIR = "/mcp_server/env_deps"

# Channel file -> the tree install-task-deps.sh routes it to.
AGENT_REQUIREMENTS = Path("environment") / "requirements.txt"
AGENT_APT = Path("environment") / "apt.txt"
GRADING_REQUIREMENTS = Path("scorer") / "requirements.txt"
ENV_REQUIREMENTS = Path("scorer") / "env-requirements.txt"

_PROBE_TIMEOUT_SECONDS = 180
_PROBE_MARKER = "LBX_PACKAGE_PROBE "

# Runs under the runtime venv's own interpreter, as root, in a throwaway
# container. Emits the installed sets plus the dependency closure of whatever
# roots it is given, so one container run answers everything about one image.
#
# Split into pure helpers and a driver so the closure rules -- the subtle part --
# can be exec'd and tested directly on the host instead of only through a
# container. `_PACKAGE_PROBE` is the two concatenated.
_PROBE_HELPERS = r"""
import json
import os
import re
import subprocess
import sys

import importlib.metadata as md

GRADING_DEPS_DIR = "/mcp_server/grading_deps"
ENV_DEPS_DIR = "/mcp_server/env_deps"
NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
EXTRAS_RE = re.compile(r"^\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*\[([^\]]*)\]")
MARKER_EXTRA_RE = re.compile("extra\\s*==\\s*['\"]([^'\"]+)['\"]")
# What may legally follow a distribution name: extras, a version specifier
# (bare or parenthesised -- installed metadata is full of `urllib3 (>=1.21.1,<3)`
# and dropping those would silently prune the closure), a direct reference, or
# nothing. Without this, `git+https://host/x.git` matches NAME_RE as the
# distribution "git".
REMAINDER_RE = re.compile(r"^\s*($|[\[(<>=!~,;@])")


def canon(name):
    return re.sub(r"[-_.]+", "-", str(name)).strip().lower()


# (name, extras) of a PEP 508 requirement, or None if it has no parseable name
# (a bare URL or local path).
def parse_requirement(text):
    text = str(text)
    marker = ""
    if ";" in text:
        text, marker = text.split(";", 1)
    match = NAME_RE.match(text)
    if not match or not REMAINDER_RE.match(text[match.end() :]):
        return None
    extras_match = EXTRAS_RE.match(text)
    extras = frozenset(
        canon(part) for part in (extras_match.group(1).split(",") if extras_match else [])
        if part.strip()
    )
    return canon(match.group(1)), extras, marker


# Optional dependencies are only installed when their extra was requested, so
# following `extra == "..."` requirements the author did not ask for would let
# the closure swallow most of PyPI -- gymnasium's extras alone reach torch and
# tensorflow. Non-extra markers (sys_platform, python_version) are deliberately
# NOT evaluated: an unsatisfied one leaves the package uninstalled, so it never
# appears in the diff and admitting its name costs nothing.
def marker_allows(marker, active_extras):
    wanted = [canon(name) for name in MARKER_EXTRA_RE.findall(marker)]
    return not wanted or any(name in active_extras for name in wanted)


# canonical name -> (Distribution, version), for the venv (path=None) or a
# private --target tree.
def index_for(path):
    found = {}
    try:
        dists = md.distributions() if path is None else md.distributions(path=[path])
    except Exception:
        return found
    for dist in dists:
        try:
            name = dist.metadata["Name"]
            version = dist.version or ""
        except Exception:
            continue
        if name:
            found.setdefault(canon(name), (dist, version))
    return found


# Walk Requires-Dist from the declared roots over (name, requested extras)
# nodes, so `foo[bar]` admits bar's dependencies and a plain `foo` does not.
def closure(roots, index):
    visited = set()
    reached = set()
    queue = []
    unparseable = []
    for root in roots:
        parsed = parse_requirement(root)
        if parsed is None:
            unparseable.append(str(root))
            continue
        queue.append((parsed[0], parsed[1]))
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        name, extras = node
        reached.add(name)
        entry = index.get(name)
        if entry is None:
            continue
        try:
            requires = entry[0].metadata.get_all("Requires-Dist") or []
        except Exception:
            requires = []
        for requirement in requires:
            parsed = parse_requirement(requirement)
            if parsed is None:
                continue
            dep_name, dep_extras, marker = parsed
            if marker_allows(marker, extras):
                queue.append((dep_name, dep_extras))
    return sorted(reached), sorted(unparseable)


# Explicitly-installed apt packages. `apt-get install foo` marks foo manual and
# everything it drags in auto, so this is the apt analogue of the pip closure:
# an author declaring ngspice need not also declare libngspice0.
def apt_manual():
    try:
        completed = subprocess.run(
            ["apt-mark", "showmanual"], capture_output=True, text=True, check=False
        )
    except OSError:
        return []
    return sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
"""

_PROBE_MAIN = r"""
roots = json.loads(os.environ.get("LBX_DECLARED_ROOTS") or "{}")
result = {"apt": apt_manual(), "trees": {}}
for label, path in (
    ("venv", None),
    ("grading", GRADING_DEPS_DIR),
    ("env", ENV_DEPS_DIR),
):
    if path is not None and not os.path.isdir(path):
        result["trees"][label] = {"installed": {}, "closure": [], "unparseable": []}
        continue
    index = index_for(path)
    reached, unparseable = closure(roots.get(label) or [], index)
    result["trees"][label] = {
        "installed": {name: entry[1] for name, entry in index.items()},
        "closure": reached,
        "unparseable": unparseable,
    }
sys.stdout.write("LBX_PACKAGE_PROBE " + json.dumps(result) + "\n")
"""

_PACKAGE_PROBE = _PROBE_HELPERS + _PROBE_MAIN


def probe_helper_namespace() -> dict:
    """The probe's pure helpers, exec'd locally so tests can drive them."""
    namespace: dict = {}
    exec(_PROBE_HELPERS, namespace)  # noqa: S102
    return namespace


class ProbeError(RuntimeError):
    """The package probe could not be run against an image."""


@dataclass(frozen=True)
class ImagePackages:
    """What one image has installed, plus the closure of the declared roots."""

    apt: frozenset[str]
    installed: dict[str, frozenset[str]] = field(default_factory=dict)
    closure: dict[str, frozenset[str]] = field(default_factory=dict)
    # Declared specs with no resolvable distribution name (a bare VCS URL or
    # local path), per tree. Their contribution to the delta is unknowable.
    unparseable: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict) -> ImagePackages:
        trees = payload.get("trees") or {}
        return cls(
            apt=frozenset(payload.get("apt") or ()),
            installed={
                label: frozenset((tree.get("installed") or {}).keys())
                for label, tree in trees.items()
            },
            closure={
                label: frozenset(tree.get("closure") or ())
                for label, tree in trees.items()
            },
            unparseable={
                label: tuple(tree.get("unparseable") or ())
                for label, tree in trees.items()
            },
        )


def probe_image_packages(
    image_ref: str, declared_roots: dict[str, list[str]] | None = None
) -> ImagePackages:
    """Enumerate an image's installed packages by running a probe inside it."""
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "root",
            "--env",
            f"LBX_DECLARED_ROOTS={json.dumps(declared_roots or {})}",
            "--entrypoint",
            RUNTIME_VENV_PYTHON,
            image_ref,
            "-c",
            _PACKAGE_PROBE,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ProbeError(f"package probe failed for {image_ref}: {detail}")
    return ImagePackages.from_payload(_parse_probe_output(completed.stdout, image_ref))


def _parse_probe_output(stdout: str, image_ref: str) -> dict:
    """Pull the probe's JSON off its marker line.

    An image is free to print from sitecustomize or a shell profile, so the
    payload is delimited rather than assumed to be the whole of stdout.
    """
    for line in reversed(stdout.splitlines()):
        if line.startswith(_PROBE_MARKER):
            try:
                return json.loads(line[len(_PROBE_MARKER) :])
            except ValueError as exc:
                raise ProbeError(
                    f"package probe emitted bad JSON for {image_ref}: {exc}"
                ) from exc
    raise ProbeError(f"package probe produced no output for {image_ref}")


def base_packages_cached(base_ref: str, cache_dir: Path | None = None) -> ImagePackages:
    """Probe the base image, memoized on its image ID.

    The base is the same for every task on a machine and its package set only
    changes when the image itself does, so this pays the container run once per
    base build rather than once per validation.
    """
    image_id = _docker_image_id(base_ref)
    cache_path = (
        cache_dir / f"{_slug(base_ref)}-{_slug(image_id)}.json"
        if cache_dir and image_id
        else None
    )
    if cache_path and cache_path.is_file():
        try:
            return ImagePackages.from_payload(json.loads(cache_path.read_text()))
        except (OSError, ValueError):
            pass  # a corrupt cache entry is just a cache miss
    packages = probe_image_packages(base_ref)
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(_to_payload(packages)))
        except OSError:
            pass  # caching is an optimization, never a failure
    return packages


def _to_payload(packages: ImagePackages) -> dict:
    return {
        "apt": sorted(packages.apt),
        "trees": {
            label: {"installed": {name: "" for name in sorted(names)}, "closure": []}
            for label, names in packages.installed.items()
        },
    }


def _docker_image_id(image_ref: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image_ref],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")[:80]


def declared_roots(problem_dir: Path) -> dict[str, list[str]]:
    """Declared requirement specs per tree, keyed as the probe expects.

    Full specs, not bare names: ``foo[bar]`` and ``foo`` have different
    dependency closures, and the probe is what parses them.
    """
    return {
        label: _requirement_roots(problem_dir / path)
        for label, path in (
            ("venv", AGENT_REQUIREMENTS),
            ("grading", GRADING_REQUIREMENTS),
            ("env", ENV_REQUIREMENTS),
        )
    }


# In a requirements file `#` only starts a comment at line start or after
# whitespace; the `#egg=name` fragment of a URL is part of the spec.
_REQUIREMENT_COMMENT_RE = re.compile(r"(?:^|\s)#")


def _strip_requirement_comment(raw: str) -> str:
    match = _REQUIREMENT_COMMENT_RE.search(raw)
    return (raw[: match.start()] if match else raw).strip()


def _requirement_roots(path: Path, _seen: frozenset[Path] = frozenset()) -> list[str]:
    """Every spec ``uv pip install -r path`` would act on, includes followed.

    This deliberately does not reuse the validator's ``_requirement_specs``,
    which drops every ``-`` line. That is right for the overlap check it serves,
    but here a dropped line means a package the installer installed and this
    gate cannot account for -- a legitimate task failing the diff. ``-r``
    includes are followed and ``-e`` editables are surfaced by name; other pip
    options install nothing and are ignored.
    """
    if not path.is_file():
        return []
    resolved = path.resolve()
    if resolved in _seen:
        return []  # a requirements file that includes itself
    seen = _seen | {resolved}

    specs: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_requirement_comment(raw)
        if not line:
            continue
        if not line.startswith("-"):
            specs.append(line)
            continue
        # `-r file` / `--requirement=file`. Split on whitespace first: an `=`
        # may belong to the argument (a URL's `#egg=` fragment), so only an `=`
        # inside the option word itself is a separator.
        parts = line.split(None, 1)
        keyword = parts[0]
        argument = parts[1].strip() if len(parts) > 1 else ""
        if "=" in keyword:
            keyword, _, argument = keyword.partition("=")
        if keyword in ("-r", "--requirement") and argument:
            specs.extend(_requirement_roots(path.parent / argument, seen))
        elif keyword in ("-e", "--editable") and argument:
            # An editable's distribution name is only knowable from `#egg=`;
            # without it the probe reports it as unresolvable rather than
            # guessing, which is the same treatment a bare URL gets.
            _, _, egg = argument.partition("#egg=")
            specs.append(egg.split("&")[0] if egg else argument)
    return specs


# `apt-get install` accepts a version pin or a release/arch qualifier that is
# not part of the package name apt-mark reports back: ngspice=44,
# ngspice/trixie-backports, ngspice:amd64.
_APT_QUALIFIER_RE = re.compile(r"[=/:].*$")


def declared_apt(problem_dir: Path) -> set[str]:
    path = problem_dir / AGENT_APT
    if not path.is_file():
        return set()
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.add(apt_package_name(line))
    return names


def apt_package_name(spec: str) -> str:
    return _APT_QUALIFIER_RE.sub("", spec.strip())


# Which channel an author should add an undeclared package to, per tree.
_TREE_CHANNEL = {
    "venv": (
        f"{AGENT_REQUIREMENTS.as_posix()} (agent-visible)",
        "the runtime venv",
    ),
    "grading": (
        f"{GRADING_REQUIREMENTS.as_posix()} (grader-only)",
        GRADING_DEPS_DIR,
    ),
    "env": (
        f"{ENV_REQUIREMENTS.as_posix()} (hidden-env-only)",
        ENV_DEPS_DIR,
    ),
}


def undeclared_package_issues(
    base: ImagePackages, task: ImagePackages, problem_dir: Path
) -> list[str]:
    """Packages the task image added that its channels do not account for.

    The allowed set per tree is the dependency closure of that tree's declared
    roots, resolved from metadata in the built image, so a single declared
    package with a deep tree passes cleanly. A declared package the base already
    provides produces no delta and is simply not mentioned -- redundant
    declarations are deliberate and must never fail.
    """
    issues: list[str] = []
    for label, (channel, where) in _TREE_CHANNEL.items():
        added = task.installed.get(label, frozenset()) - base.installed.get(
            label, frozenset()
        )
        if task.unparseable.get(label):
            # A bare URL / local-path requirement installs a distribution whose
            # name is not in the spec, so nothing here can tell its packages
            # apart from an undeclared one. Say so instead of failing the task
            # on a legitimate declaration or passing it on a fiction.
            issues.append(
                f"{channel} declares {', '.join(task.unparseable[label])} without a "
                "resolvable distribution name, so the built image cannot be "
                "checked against it. Use PEP 508 direct reference form "
                "(`name @ git+https://...`) so the package this installs is "
                "named in the channel."
            )
            continue
        undeclared = sorted(added - task.closure.get(label, frozenset()))
        if undeclared:
            issues.append(
                f"{', '.join(undeclared)} installed into {where} by the task "
                f"image but not declared in any dependency channel. Add "
                f"{'them' if len(undeclared) > 1 else 'it'} to {channel}, or to "
                "the channel matching who may import "
                f"{'them' if len(undeclared) > 1 else 'it'}, and install via "
                "`RUN /opt/lbx-runtime/install-task-deps.sh /tmp/task-deps`. "
                "Comparing the built image against the base is what makes this "
                "check independent of how the install was written."
            )

    # apt names are compared literally apart from the qualifier apt accepts but
    # does not report back: `-`/`_`/`.`/`+` are meaningful in them (python3.11,
    # libstdc++6), unlike in pip distribution names.
    declared = declared_apt(problem_dir)
    base_apt = {apt_package_name(name) for name in base.apt}
    added_apt = sorted(
        {apt_package_name(name) for name in task.apt} - base_apt - declared
    )
    if added_apt:
        issues.append(
            f"apt package(s) {', '.join(added_apt)} explicitly installed by the "
            f"task image but not declared in {AGENT_APT.as_posix()}. Declare them "
            "there one per line and install via `RUN /opt/lbx-runtime/"
            "install-task-deps.sh /tmp/task-deps`. Only explicitly-installed "
            "packages are compared, so dependencies pulled in by a declared "
            "package need no declaration of their own, and a build toolchain "
            "installed and purged within a single RUN leaves nothing behind."
        )
    return issues
