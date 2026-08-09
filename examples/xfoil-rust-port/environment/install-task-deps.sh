#!/usr/bin/env bash
# Task-local copy of current-main base/install-task-deps.sh.
#
# The feature branch's cached base predates the installed
# /opt/lbx-runtime/install-task-deps.sh helper. Keeping this exact task-local
# channel router makes the example build against both that base and current
# main without a shared runtime edit or raw Dockerfile install.
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

install_private() {
  local requirements="$1" target="$2" label="$3"
  echo ":: install-task-deps: installing ${label} into ${target}"
  mkdir -p "${target}"
  env -u UV_SYSTEM_PYTHON uv pip install \
    --python "${VENV_PYTHON}" --target "${target}" --no-cache -r "${requirements}"
  chown -R root:root "${target}"
  find "${target}" -type d -exec chmod 0700 {} +
  find "${target}" -type f -exec chmod 0600 {} +
}

if [[ -f "${apt_file}" ]]; then
  mapfile -t apt_packages < <(
    sed -e 's/#.*//' -e 's/[[:space:]]\+//g' "${apt_file}" |
      grep -v '^$' || true
  )
  if ((${#apt_packages[@]})); then
    echo ":: install-task-deps: installing apt packages: ${apt_packages[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "${apt_packages[@]}"
    rm -rf /var/lib/apt/lists/*
  fi
fi

if [[ -f "${pip_file}" ]]; then
  echo ":: install-task-deps: installing agent-visible requirements"
  env -u UV_SYSTEM_PYTHON uv pip install \
    --python "${VENV_PYTHON}" --no-cache -r "${pip_file}"
fi

if [[ -f "${grading_file}" ]]; then
  install_private \
    "${grading_file}" \
    /mcp_server/grading_deps \
    "grader-only requirements"
fi

if [[ -f "${env_file}" ]]; then
  install_private \
    "${env_file}" \
    /mcp_server/env_deps \
    "hidden-env-only requirements"
fi

if [[ -d /mcp_server ]]; then
  chown root:root /mcp_server
  chmod 0700 /mcp_server
fi

echo ":: install-task-deps: done"
