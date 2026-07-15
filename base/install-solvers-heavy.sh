#!/usr/bin/env bash
# Install the heavy binary numerical-solver engines into the base image.
#
# Runs on the Debian CPU/TPU bases and the Ubuntu GPU bases. Engines that ship
# as conda-forge binaries are
# installed into isolated conda envs (separate envs are required because their
# MPI stacks conflict; isolated + serial/nompi also matches the determinism
# playbook). Pure-python engines go into the shared runtime venv.
#
# Exposed for scorers via:
#   * /usr/local/bin symlinks for single CLIs (ccx, SU2_*, openroad, xfoil, avl)
#   * /opt/solver-envs/<engine>/bin/python for python-API engines (meep)
#   * /etc/solver-envs.d/<engine>.sh activation snippets (OpenFOAM needs its env)
set -euo pipefail
trap 'echo "install-solvers-heavy.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

export DEBIAN_FRONTEND=noninteractive

VENV_PY="${RUNTIME_VENV_DIR:-/opt/lbx-runtime/.venv}/bin/python"
CONDA_DIR=/opt/conda
ENVS_DIR=/opt/solver-envs
ACT_DIR=/etc/solver-envs.d
mkdir -p "${ENVS_DIR}" "${ACT_DIR}" /usr/local/bin

arch="$(uname -m)"

echo ":: heavy: installing build/runtime apt deps"
apt-get update
apt-get install -y --no-install-recommends \
  gfortran \
  cmake \
  swig \
  libx11-dev \
  bzip2
rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 1) Miniforge (conda) — cross-distro prebuilt binary engines.
# ---------------------------------------------------------------------------
if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
  echo ":: heavy: installing miniforge"
  curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh" -o /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "${CONDA_DIR}"
  rm -f /tmp/miniforge.sh
fi
CONDA="${CONDA_DIR}/bin/conda"
"${CONDA}" config --system --set remote_max_retries 5 || true
"${CONDA}" config --system --set remote_read_timeout_secs 120 || true

conda_create() {
  local label="$1"
  local prefix="$2"
  shift 2

  local max_attempts="${CONDA_CREATE_ATTEMPTS:-3}"
  local attempt status
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    echo ":: heavy: creating ${label} env (attempt ${attempt}/${max_attempts})"
    if "${CONDA}" create -y -p "${prefix}" -c conda-forge "$@"; then
      return 0
    fi
    status=$?
    if ((attempt == max_attempts)); then
      return "${status}"
    fi
    echo ":: heavy: ${label} conda create failed with ${status}; cleaning partial state before retry" >&2
    rm -rf "${prefix}"
    "${CONDA}" clean -afy || true
    sleep $((attempt * 10))
  done
}

# ---------------------------------------------------------------------------
# 2) Isolated conda envs (Meep / SU2 / OpenFOAM / CalculiX).
# ---------------------------------------------------------------------------
conda_create "Meep (FDTD, nompi)" "${ENVS_DIR}/meep" "pymeep=*=nompi*"
conda_create "SU2 (CFD)" "${ENVS_DIR}/su2" su2
conda_create "OpenFOAM (CFD)" "${ENVS_DIR}/openfoam" openfoam
conda_create "CalculiX (FEM)" "${ENVS_DIR}/calculix" calculix

echo ":: heavy: pruning conda caches"
"${CONDA}" clean -afy

# Expose single-binary CLIs on PATH for subprocess calls from scorers.
for b in SU2_CFD SU2_SOL SU2_DEF SU2_GEO; do
  [[ -x "${ENVS_DIR}/su2/bin/${b}" ]] && ln -sf "${ENVS_DIR}/su2/bin/${b}" "/usr/local/bin/${b}"
done
ln -sf "${ENVS_DIR}/calculix/bin/ccx" /usr/local/bin/ccx

# OpenFOAM is environment-heavy (WM_PROJECT_DIR etc.); ship an activation snippet
# the scorer can `source` before invoking simpleFoam/blockMesh/etc.
foam_bashrc="$(find "${ENVS_DIR}/openfoam" -name bashrc -path '*etc*' | head -1 || true)"
if [[ -n "${foam_bashrc}" ]]; then
  cat > "${ACT_DIR}/openfoam.sh" <<EOF
_lbx_openfoam_had_nounset=0
_lbx_openfoam_had_errexit=0
case "\$-" in
  *e*) _lbx_openfoam_had_errexit=1; set +e ;;
esac
case "\$-" in
  *u*) _lbx_openfoam_had_nounset=1; set +u ;;
esac
source '${foam_bashrc}'
if [ "\${_lbx_openfoam_had_errexit}" = "1" ]; then
  set -e
fi
if [ "\${_lbx_openfoam_had_nounset}" = "1" ]; then
  set -u
fi
unset _lbx_openfoam_had_errexit _lbx_openfoam_had_nounset
EOF
fi

# ---------------------------------------------------------------------------
# 3) OpenSeesPy (structural FEM) — pure-python wheel, works on py3.13.
# ---------------------------------------------------------------------------
echo ":: heavy: installing OpenSeesPy into the runtime venv"
uv pip install --python "${VENV_PY}" --no-cache openseespy

echo ":: heavy: keeping conda tree world-readable for the non-root tool user"
chmod -R a+rX "${CONDA_DIR}" "${ENVS_DIR}"

echo ":: install-solvers-heavy: done"
