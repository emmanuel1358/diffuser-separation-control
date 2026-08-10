#!/usr/bin/env bash
set -euo pipefail
trap 'echo "install-common.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

export DEBIAN_FRONTEND=noninteractive
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
export RUNTIME_VENV_DIR="${RUNTIME_VENV_DIR:-/opt/lbx-runtime/.venv}"

echo ":: install-common: installing apt packages"
apt-get update
apt-get install -y --no-install-recommends \
  bash \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  graphviz \
  libegl1 \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  libngspice0-dev \
  libosmesa6 \
  libsndfile1 \
  ngspice \
  pkg-config \
  tmux
rm -rf /var/lib/apt/lists/*

# Keep interactive shells and tmux panes aligned with the image runtime. Debian's
# /etc/profile can reset PATH for login shells, so bake the runtime venv back in
# before task authors or agent tools fall through to system Python.
runtime_path="${RUNTIME_VENV_DIR}/bin:${PATH}"
{
  echo "set -g default-shell /bin/bash"
  echo "set -g default-command /bin/bash"
  echo "set-environment -g PATH \"${runtime_path}\""
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    echo "set-environment -g LD_LIBRARY_PATH \"${LD_LIBRARY_PATH}\""
  fi
} > /etc/tmux.conf

cat > /etc/lbx-runtime-venv-path.sh <<EOF
case ":\${PATH}:" in
  *:"${RUNTIME_VENV_DIR}/bin":*) ;;
  *) PATH="${RUNTIME_VENV_DIR}/bin:\${PATH}" ;;
esac
export PATH
EOF
chmod 0644 /etc/lbx-runtime-venv-path.sh
install -m 0644 /etc/lbx-runtime-venv-path.sh /etc/profile.d/lbx-runtime-venv-path.sh

for shell_startup in /etc/profile /etc/bash.bashrc; do
  touch "${shell_startup}"
  if ! grep -Fq "/etc/lbx-runtime-venv-path.sh" "${shell_startup}"; then
    cat >> "${shell_startup}" <<'EOF'

# Keep Labelbox task runtime Python before system Python.
if [ -r /etc/lbx-runtime-venv-path.sh ]; then
  . /etc/lbx-runtime-venv-path.sh
fi
EOF
  fi
done

# CPU and GPU images run the Taiga runtime on Python 3.13; the TPU flavor pins
# 3.12 (jax[tpu]/jaxlib wheels target 3.12). Override with PYTHON_VERSION.
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
echo ":: install-common: installing Python ${PYTHON_VERSION}"
mkdir -p "${UV_PYTHON_INSTALL_DIR}" "$(dirname "${RUNTIME_VENV_DIR}")"
uv python install "${PYTHON_VERSION}"
python_bin="$(uv python find "${PYTHON_VERSION}")"

mkdir -p /mcp_server /tmp/output /data /grader/data /workdir /runtime /tmp/hf-cache/hub
# Taiga tools run as a non-root uid. Keep public work/output directories
# writable even when task Dockerfiles do not repair ownership themselves.
# /tmp/hf-cache is HF_HOME (see base Dockerfiles): agent-writable cache and the
# parent for deploy-time read-only Hugging Face weight mounts (preloaded_files),
# so from_pretrained resolves mounted weights without a network fetch.
chmod 0777 /tmp/output /workdir /tmp/hf-cache /tmp/hf-cache/hub

# Unprivileged account that the rubric server's agent-facing tools and the
# grader's PolicyWorker drop to. Real /etc/passwd entry + writable home so
# tools that read $HOME (numpy/mujoco/pip caches, shell history, …) work.
#
# uid/gid 1000 must be reclaimed first: Ubuntu 24.04 (noble) ships a default
# `ubuntu` user at 1000:1000, where 22.04 (jammy) left the id free. Without
# this, groupadd fails with "GID '1000' already exists" and every noble base
# build dies here. The uid is not negotiable -- Taiga runs task tools under
# 1000 and the grader drops to it by number, so the account must be `agent`
# rather than whatever the OS happened to put there.
echo ":: install-common: creating agent user (uid 1000)"
existing_user="$(getent passwd 1000 | cut -d: -f1 || true)"
if [[ -n "${existing_user}" && "${existing_user}" != "agent" ]]; then
  echo ":: install-common: reclaiming uid 1000 from '${existing_user}'"
  userdel -r "${existing_user}" 2>/dev/null || userdel "${existing_user}"
fi
existing_group="$(getent group 1000 | cut -d: -f1 || true)"
if [[ -n "${existing_group}" && "${existing_group}" != "agent" ]]; then
  echo ":: install-common: reclaiming gid 1000 from '${existing_group}'"
  groupdel "${existing_group}"
fi
groupadd -g 1000 agent
useradd -u 1000 -g 1000 -m -d /home/agent -s /bin/bash agent

echo ":: install-common: creating runtime venv"
uv venv --python "${python_bin}" "${RUNTIME_VENV_DIR}"

runtime_constraints=()
if [[ -f /tmp/base/requirements-runtime.txt ]]; then
  echo ":: install-common: installing pinned rubric/grader runtime dependencies"
  uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache -r /tmp/base/requirements-runtime.txt
  runtime_constraints=(-c /tmp/base/requirements-runtime.txt)
fi

torch_constraint_args=()
torch_install_args=()

echo ":: install-common: installing rubric package"
uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache --no-deps -e /mcp_server

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  echo ":: install-common: installing torch packages from ${TORCH_INDEX_URL}"
  read -r -a torch_packages <<< "${TORCH_PACKAGES:-torch torchvision torchaudio}"
  # CUDA wheels use local versions like +cu130; uv's first-index safety can hide
  # them once PyPI has a plain `torch` release, so CUDA bases opt into best-match.
  torch_index_strategy="${TORCH_INDEX_STRATEGY:-first-index}"
  torch_install_args=(
    --index-url "${TORCH_INDEX_URL}"
    --extra-index-url https://pypi.org/simple
    --index-strategy "${torch_index_strategy}"
  )
  torch_constraints=/tmp/base/requirements-torch.txt
  : > "${torch_constraints}"
  for pkg in "${torch_packages[@]}"; do
    if [[ "${pkg}" == *"=="* ]]; then
      echo "${pkg}" >> "${torch_constraints}"
    fi
  done
  if [[ -s "${torch_constraints}" ]]; then
    torch_constraint_args=(-c "${torch_constraints}")
    torch_install_args+=("${torch_constraint_args[@]}")
  fi
  uv pip install \
    --python "${RUNTIME_VENV_DIR}/bin/python" \
    --no-cache \
    "${torch_install_args[@]}" \
    "${torch_packages[@]}"
fi

# The slim flavor sets SKIP_COMMON_REQUIREMENTS=1 to skip the heavy ML stack.
if [[ "${SKIP_COMMON_REQUIREMENTS:-0}" != "1" ]]; then
  echo ":: install-common: installing common requirements"
  uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache "${runtime_constraints[@]}" "${torch_constraint_args[@]}" -r /tmp/base/requirements-common.txt
else
  echo ":: install-common: SKIP_COMMON_REQUIREMENTS=1 -> skipping heavy ML stack (slim base)"
fi

if [[ -f /tmp/base/requirements-solvers.txt ]]; then
  echo ":: install-common: installing numerical-solver stack"
  uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache "${runtime_constraints[@]}" "${torch_constraint_args[@]}" -r /tmp/base/requirements-solvers.txt

  # HARD-WON FINDING (do not re-learn): PySpice 1.5 fails every solve against
  # ngspice 42 -- the version noble ships -- over a banner, not over any real
  # incompatibility.
  #
  # NgSpiceShared._send_char classifies every line ngspice writes to stderr that
  # does not start with "Warning:" as a hard error and sets _error_in_stderr; the
  # next exec_command() then raises NgSpiceCommandError("Command 'run' failed")
  # even though the analysis already ran and returned its data rows. ngspice 42
  # is built with the KLU direct solver and announces the solver it chose on
  # stderr at the start of every analysis:
  #     Using SPARSE 1.3 as Direct Linear Solver
  # which turned a working 10V divider .op into a failure on the 24.04 bases.
  # 42 is alone in this: jammy's 36 was built without KLU and wrote nothing at
  # all to stderr, and the CPU base's trixie 44 is KLU-enabled but emits no
  # banner, so only the CUDA 13 move onto noble exposed it.
  #
  # No version escapes it. PySpice 1.5 (May 2021) is the last release on PyPI and
  # upstream master holds no code changes past it, so nothing newer knows about
  # ngspice 42; and 42+ds-3build1 is the only ngspice anywhere in the noble
  # archive (universe, updates, backports, security, proposed), so apt can move
  # neither back to 36 nor forward to 44. Asking for KLU explicitly with
  # `.options klu` only changes the wording of the banner, not its stream.
  #
  # So relax the classification for that one banner. No numerics change: the
  # simulation was already correct. The patch asserts on its anchor so that a
  # PySpice bump which moves this code fails the build loudly rather than
  # silently leaving every SPICE solve broken.
  echo ":: install-common: patching PySpice for the ngspice 42 solver banner"
  "${RUNTIME_VENV_DIR}/bin/python" - <<'PYSPICE_NGSPICE42_PATCH'
import pathlib
import sys
import sysconfig

target = (
    pathlib.Path(sysconfig.get_paths()["purelib"])
    / "PySpice"
    / "Spice"
    / "NgSpice"
    / "Shared.py"
)
if not target.exists():
    print(f":: PySpice not installed at {target}, nothing to patch")
    sys.exit(0)

source = target.read_text()
if "lbx: ngspice >= 42" in source:
    print(":: PySpice already patched")
    sys.exit(0)

anchor = "            if content.startswith('Warning:'):\n"
replacement = (
    "            # lbx: ngspice >= 42 announces its direct linear solver on stderr\n"
    "            # at the start of every analysis. That is informational, not a\n"
    "            # failure. See base/install-common.sh.\n"
    "            _lbx_solver_banner = content.strip().endswith('as Direct Linear Solver')\n"
    "            if content.startswith('Warning:') or _lbx_solver_banner:\n"
)
found = source.count(anchor)
if found != 1:
    raise SystemExit(
        f"expected exactly one stderr-classification anchor in {target}, found "
        f"{found}: PySpice changed shape, re-check the ngspice >= 42 patch"
    )
target.write_text(source.replace(anchor, replacement))
print(f":: patched {target}")
PYSPICE_NGSPICE42_PATCH
fi

if [[ -f /tmp/base/install-solvers-heavy.sh ]]; then
  echo ":: install-common: installing heavy binary solver engines"
  bash /tmp/base/install-solvers-heavy.sh
fi

if [[ -n "${BASE_EXTRA_REQUIREMENTS:-}" && -f "${BASE_EXTRA_REQUIREMENTS}" ]]; then
  echo ":: install-common: installing extra requirements from ${BASE_EXTRA_REQUIREMENTS}"
  uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache "${runtime_constraints[@]}" "${torch_constraint_args[@]}" -r "${BASE_EXTRA_REQUIREMENTS}"
fi

# HARD-WON FINDING (do not re-learn): two NCCL distributions own the same file,
# so whichever is installed last silently wins.
#
# torch 2.9.1+cu130 requires nvidia-nccl-cu13==2.27.7. xgboost (in
# requirements-gpu.txt) requires nvidia-nccl-cu12, unpinned, which resolves to a
# CUDA 12.9 build. Both wheels list nvidia/nccl/lib/libnccl.so.2 and
# nvidia/nccl/include/nccl.h in their RECORD and unpack them into the same
# site-packages, so they do not coexist -- they overwrite each other. The extras
# install above runs after the torch install, so the CUDA 12.9 library always
# landed last, and libtorch_cuda.so (DT_NEEDED libnccl.so.2, RPATH
# $ORIGIN/../../nvidia/nccl/lib) then loaded a CUDA 12.9 NCCL into a CUDA 13
# process for every collective.
#
# Reinstalling nvidia-nccl-cu13 here, after the extras, re-extracts the CUDA 13
# build over the CUDA 12 one. Both distributions stay installed, so xgboost
# keeps its declared dependency and its multi-GPU path, and only the shared .so
# changes hands. This must stay ordered after every install that can pull in a
# competing NCCL; nothing below it may install one.
#
# The expected version comes from torch's own metadata rather than a literal, so
# a torch bump cannot leave a stale pin behind. The check reads the library that
# is actually on disk: torch.cuda.nccl.version() is useless here because it
# returns the compile-time constant from torch's bundled nccl.h and reports the
# pinned version on a clobbered image and a clean one alike.
#
# Verify first, repair only if the check fails, then verify again. That keeps
# this a true no-op on flavors that were never clobbered (cpu, tpu and
# cuda-graphics, which has no xgboost), while a repair that does not take fails
# the build instead of shipping quietly.
nccl_verify() {  # $1 = version torch pins; non-zero if that is not what is installed
  NCCL_EXPECTED_VERSION="$1" "${RUNTIME_VENV_DIR}/bin/python" - <<'NCCL_VERIFY'
import ctypes
import importlib.metadata as md
import os
import re
import sysconfig
from pathlib import Path

expected = os.environ["NCCL_EXPECTED_VERSION"]

try:
    installed = md.version("nvidia-nccl-cu13")
except md.PackageNotFoundError:
    raise SystemExit(f"torch pins nvidia-nccl-cu13=={expected} but it is not installed")
if installed != expected:
    raise SystemExit(f"nvidia-nccl-cu13 {installed} is installed, torch pins {expected}")

lib = Path(sysconfig.get_paths()["purelib"]) / "nvidia/nccl/lib/libnccl.so.2"
if not lib.exists():
    raise SystemExit(f"{lib} is missing; libtorch_cuda.so cannot resolve NCCL")

# The build string baked into the library names the CUDA it was built for: a
# clobbered image reads "2.30.7+cuda12.9" while the metadata still claims 2.27.7.
# Finding no build string is a failure, not a pass -- it would mean this check no
# longer knows how to read the library and is about to wave a clobber through.
builds = sorted(
    {m.decode() for m in re.findall(rb"\d+\.\d+\.\d+\+cuda\d+\.\d+", lib.read_bytes())}
)
if not builds:
    raise SystemExit(
        f"no NCCL build string found in {lib}: cannot tell which CUDA it was "
        "built for, so a cu12/cu13 clobber cannot be ruled out"
    )
wrong = [
    build
    for build in builds
    if build.partition("+cuda")[0] != expected
    or not build.partition("+cuda")[2].startswith("13.")
]
if wrong:
    raise SystemExit(
        f"{lib} carries the {wrong} build string(s) but torch needs "
        f"{expected}+cuda13.x (see the nvidia-nccl-cu12 clobber note in "
        "base/install-common.sh)"
    )

# Then load it and ask it directly. NCCL >= 2.9 encodes its version as
# major*10000 + minor*100 + patch; older releases used major*1000.
code = ctypes.c_int()
ctypes.CDLL(str(lib)).ncclGetVersion(ctypes.byref(code))
divisor = 10000 if code.value >= 20000 else 1000
loaded = (code.value // divisor, (code.value % divisor) // 100, code.value % 100)
if loaded != tuple(int(part) for part in expected.split(".")[:3]):
    raise SystemExit(
        f"ncclGetVersion() on {lib} reports {'.'.join(map(str, loaded))}, "
        f"but torch pins {expected}"
    )

print(f":: {lib} is {builds[0]} and reports {'.'.join(map(str, loaded))}")
NCCL_VERIFY
}

echo ":: install-common: checking which NCCL torch will load at runtime"
# Via a file rather than $(...): bash 3.2, which is still what macOS ships and
# what the `bash -n` guard in harness/tests/test_base_image.py runs, mis-parses a
# heredoc inside a command substitution.
nccl_pin_file="$(mktemp)"
"${RUNTIME_VENV_DIR}/bin/python" - > "${nccl_pin_file}" <<'NCCL_PIN_FROM_TORCH'
import importlib.metadata as md
import re

try:
    requirements = md.requires("torch") or []
except md.PackageNotFoundError:
    raise SystemExit(0)  # No torch in this flavor, so nothing here needs NCCL.

# Every base image is Linux, so the `platform_system == "Linux"` marker that
# torch puts on these pins always holds.
pins = {
    match.group(1)
    for requirement in requirements
    if (match := re.match(r"\s*nvidia[-_]nccl[-_]cu13\s*==\s*([^\s;,]+)", requirement))
}
if not pins:
    raise SystemExit(0)  # CPU-only torch builds bundle no CUDA 13 NCCL.
if len(pins) != 1:
    raise SystemExit(
        "expected exactly one nvidia-nccl-cu13==<version> pin in the torch "
        f"metadata, found {sorted(pins)}: torch changed how it declares NCCL, "
        "re-check the NCCL repair in base/install-common.sh"
    )
print(pins.pop())
NCCL_PIN_FROM_TORCH
nccl_pin="$(cat "${nccl_pin_file}")"
rm -f "${nccl_pin_file}"

if [[ -z "${nccl_pin}" ]]; then
  echo ":: install-common: torch pins no CUDA 13 NCCL here; nothing to repair"
elif nccl_verify "${nccl_pin}"; then
  echo ":: install-common: NCCL already matches torch's pin (${nccl_pin})"
else
  echo ":: install-common: NCCL was overwritten by another distribution; reinstalling nvidia-nccl-cu13==${nccl_pin}"
  uv pip install \
    --python "${RUNTIME_VENV_DIR}/bin/python" \
    --no-cache \
    "${torch_install_args[@]}" \
    --reinstall-package nvidia-nccl-cu13 \
    "nvidia-nccl-cu13==${nccl_pin}"
  # Unguarded on purpose: `set -e` turns a repair that did not take into a build
  # failure rather than an image that quietly loads the wrong NCCL.
  nccl_verify "${nccl_pin}"
fi

if [[ -f /runtime/grading/pyproject.toml ]]; then
  echo ":: install-common: installing grading runtime"
  uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache --no-deps -e /runtime/grading
fi

# Taiga tools and task scripts expect these names on PATH. Use a wrapper for
# python: CPython follows symlinks to the base interpreter, which bypasses the
# venv and drops baked packages like numpy from sys.path.
rm -f /usr/local/bin/python /usr/local/bin/python3
cat > /usr/local/bin/python <<'PYTHON_WRAPPER'
#!/bin/sh
exec /opt/lbx-runtime/.venv/bin/python "$@"
PYTHON_WRAPPER
chmod 0755 /usr/local/bin/python
ln -sf /usr/local/bin/python /usr/local/bin/python3

# `uv venv` does not seed pip, so the venv had no pip module and
# /usr/local/bin/pip was a dangling symlink. Install pip into the runtime venv
# so `python -m pip` resolves, then point both pip names at it: the base image
# ships /usr/local/bin/pip3 for the system CPython at /usr/local, whose
# site-packages the runtime venv cannot see. Agents that ran `pip list` to
# check for numpy/mujoco otherwise saw a bare "command not found" and
# concluded the packages were missing when they were already installed.
uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache pip
rm -f /usr/local/bin/pip /usr/local/bin/pip3
ln -sf "${RUNTIME_VENV_DIR}/bin/pip" /usr/local/bin/pip
ln -sf "${RUNTIME_VENV_DIR}/bin/pip" /usr/local/bin/pip3
ln -sf "${RUNTIME_VENV_DIR}/bin/rubric" /usr/local/bin/rubric

# Ship the task dependency-channel installer at a stable path so every native
# task Dockerfile can install its declared channels without vendoring the
# root-only hardening steps for grader/env deps.
if [[ -f /tmp/base/install-task-deps.sh ]]; then
  echo ":: install-common: installing task dependency-channel installer"
  install -m 0755 /tmp/base/install-task-deps.sh \
    "$(dirname "${RUNTIME_VENV_DIR}")/install-task-deps.sh"
fi

# Taiga's Bash/editor tools run as uid 1000 and need Python on PATH, but the
# MCP server tree contains grader-private data on squashfs mounts where runtime
# chmod backstops cannot repair permissions. Keep the runtime venv outside the
# private tree, then make /mcp_server root-only at image build time.
chmod -R a+rX "${UV_PYTHON_INSTALL_DIR}" "${RUNTIME_VENV_DIR}"
chmod 0755 "$(dirname "${RUNTIME_VENV_DIR}")"
chown -R root:root /mcp_server
chmod -R 0700 /mcp_server

cat > /runtime/run_grader.py <<'PY'
#!/usr/bin/env python
from grader_runner.run_grader import main

raise SystemExit(main())
PY
chmod +x /runtime/run_grader.py
