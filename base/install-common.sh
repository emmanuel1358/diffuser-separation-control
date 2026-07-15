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
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  libngspice0-dev \
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
echo ":: install-common: creating agent user (uid 1000)"
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

echo ":: install-common: installing rubric package"
uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache --no-deps -e /mcp_server

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  echo ":: install-common: installing torch packages from ${TORCH_INDEX_URL}"
  read -r -a torch_packages <<< "${TORCH_PACKAGES:-torch torchvision torchaudio}"
  # CUDA wheels use local versions like +cu124; uv's first-index safety can hide
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
fi

if [[ -f /tmp/base/install-solvers-heavy.sh ]]; then
  echo ":: install-common: installing heavy binary solver engines"
  bash /tmp/base/install-solvers-heavy.sh
fi

if [[ -n "${BASE_EXTRA_REQUIREMENTS:-}" && -f "${BASE_EXTRA_REQUIREMENTS}" ]]; then
  echo ":: install-common: installing extra requirements from ${BASE_EXTRA_REQUIREMENTS}"
  uv pip install --python "${RUNTIME_VENV_DIR}/bin/python" --no-cache "${runtime_constraints[@]}" "${torch_constraint_args[@]}" -r "${BASE_EXTRA_REQUIREMENTS}"
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
ln -sf "${RUNTIME_VENV_DIR}/bin/pip" /usr/local/bin/pip
ln -sf "${RUNTIME_VENV_DIR}/bin/rubric" /usr/local/bin/rubric

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
