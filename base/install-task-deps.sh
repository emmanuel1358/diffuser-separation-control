#!/usr/bin/env bash
# Install a native task's declared dependency channels.
#
# Native tasks declare dependencies as files rather than task.toml fields so that
# private package names never reach the agent-visible /task/task.toml. Each
# channel installs to a different place, and that placement is the isolation
# boundary:
#
#   environment/apt.txt          apt packages (system-wide, agent-visible)
#   environment/requirements.txt pip into /opt/lbx-runtime/.venv (agent-visible)
#   scorer/requirements.txt      pip into /mcp_server/grading_deps (root-only);
#                                the grader worker prepends it to sys.path before
#                                loading compute_score, so the grader can import
#                                a scoring/reference library the agent cannot see
#   scorer/env-requirements.txt  pip into /mcp_server/env_deps (root-only); the
#                                hidden env server prepends it before loading
#                                env.py, so env-only deps are reachable ONLY
#                                through the RPC
#
# /mcp_server is chmod 0700 root, so the uid-1000 agent cannot read either
# private tree. Overlap between the agent-visible and private channels is
# rejected at authoring time by the task validator; installing the same package
# into both would silently defeat the isolation.
#
# Usage: install-task-deps.sh <task-src-root>
# Missing channel files are skipped, so every native task can call this
# unconditionally.
set -euo pipefail
trap 'echo "install-task-deps.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SRC="${1:-}"
if [[ -z "${SRC}" || ! -d "${SRC}" ]]; then
  echo "install-task-deps.sh: usage: install-task-deps.sh <task-src-root>" >&2
  exit 2
fi

RUNTIME_VENV_DIR="${RUNTIME_VENV_DIR:-/opt/lbx-runtime/.venv}"
VENV_PYTHON="${RUNTIME_VENV_DIR}/bin/python"

apt_file="${SRC}/environment/apt.txt"
pip_file="${SRC}/environment/requirements.txt"
grading_file="${SRC}/scorer/requirements.txt"
env_file="${SRC}/scorer/env-requirements.txt"

# Install a requirements file into a root-only --target dir and seal it. Sealing
# here (not in a later layer) keeps the private tree unreadable in every layer
# that contains it.
install_private() {
  local requirements="$1" target="$2" label="$3"
  echo ":: install-task-deps: installing ${label} into ${target}"
  mkdir -p "${target}"
  # --target installs a self-contained tree; the consuming runtime puts it on
  # sys.path itself rather than relying on the venv. --python still has to name
  # the runtime interpreter: it selects the ABI wheels are resolved for, and the
  # grader worker / env server import this tree under ${VENV_PYTHON}. Without it
  # uv picks whatever interpreter it discovers -- UV_SYSTEM_PYTHON=1 on the CUDA
  # bases, or a system python that an apt.txt package pulled in above -- and a
  # C-extension wheel built for that ABI fails to import at grading time.
  env -u UV_SYSTEM_PYTHON uv pip install \
    --python "${VENV_PYTHON}" --target "${target}" --no-cache -r "${requirements}"
  chown -R root:root "${target}"
  find "${target}" -type d -exec chmod 0700 {} +
  find "${target}" -type f -exec chmod 0600 {} +
}

if [[ -f "${apt_file}" ]]; then
  # One package per line; blank lines and # comments ignored.
  mapfile -t apt_packages < <(sed -e 's/#.*//' -e 's/[[:space:]]\+//g' "${apt_file}" | grep -v '^$' || true)
  if ((${#apt_packages[@]})); then
    echo ":: install-task-deps: installing apt packages: ${apt_packages[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "${apt_packages[@]}"
    rm -rf /var/lib/apt/lists/*
  fi
fi

if [[ -f "${pip_file}" ]]; then
  echo ":: install-task-deps: installing agent-visible requirements into ${RUNTIME_VENV_DIR}"
  # env -u UV_SYSTEM_PYTHON: the CUDA bases export UV_SYSTEM_PYTHON=1, which
  # would otherwise redirect this install away from the runtime venv.
  env -u UV_SYSTEM_PYTHON uv pip install --python "${VENV_PYTHON}" --no-cache -r "${pip_file}"
fi

if [[ -f "${grading_file}" ]]; then
  install_private "${grading_file}" /mcp_server/grading_deps "grader-only requirements"
fi

if [[ -f "${env_file}" ]]; then
  install_private "${env_file}" /mcp_server/env_deps "hidden-env-only requirements"
fi

# Re-seal the private root in case a channel created it fresh.
if [[ -d /mcp_server ]]; then
  chown root:root /mcp_server
  chmod 0700 /mcp_server
fi

echo ":: install-task-deps: done"
