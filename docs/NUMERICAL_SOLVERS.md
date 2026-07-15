# Numerical Solvers (CFD and Structures Task Types)

This template ships an open-source **numerical-solver stack** baked into the
shared base image so non-MuJoCo tasks (`task_type = "cfd"` or
`task_type = "structures"`) can reuse the existing deterministic-scoring
contract with zero harness changes. See
[`../numerical_solver_dataset_proposal.md`](../numerical_solver_dataset_proposal.md)
for the domain rationale.

The contract is unchanged: `compute_score(workspace, trajectory, private)`,
artifacts to `/tmp/output`, and enum-backed task metadata in `task.toml`.
Most solver tasks use `reward_type = "multi_deterministic_rubrics"` with an
oracle score of `1.0` and naive baseline near `0.0`.

For `cfd` and `structures` tasks, `instruction.md` must be
**solver-agnostic**. The grader may still own private cases and hidden operating
conditions, and the oracle/scorer may still run the real solver. The core
solution implementation (`solution/solve.sh` or solution files it calls) must
include solver-specific runnable material. The prompt itself must not
mention solvers, require solver commands, name solver software, ask for solver
evidence, or hint at solver/tool implementation details. It should still include
physics-grounded scoring details in solver-agnostic terms.

QA checks this split:

- pre-Taiga prompt QA blocks solver/tool/provenance leaks in `instruction.md`
  and verifies solver-specific runnable material in the submitted ground-truth
  oracle path when applicable;
- hidden-side validation, ground-truth proof, task validation, and image smoke
  tests verify the real solver path works where the grader/oracle uses it.

## What's in the base image

These engines are installed for **every** CPU and canonical ML GPU task — both
the CPU base (`base/cpu/Dockerfile`, Debian `python:3.13-slim`) and the clean GPU
base (`base/gpu/Dockerfile`, `nvidia/cuda:...ubuntu22.04`) run the same
[`../base/install-common.sh`](../base/install-common.sh), which installs
[`../base/requirements-solvers.txt`](../base/requirements-solvers.txt). They are
pinned, pip-installable, CPU-friendly, and deterministic. Each was verified to
import **and** run a real solve inside both built base images (and on
`linux/amd64` + `arm64`) — see
[`../base/tests/test_solvers.py`](../base/tests/test_solvers.py).

`task.toml` selects which base a task's image is built from:
`[environment].required_resources` picks one Taiga resource enum. With
`base_flavor = "auto"`, CPU enums use the CPU base, H100 enums use the clean GPU
base, graphics H100 enums use `cuda-graphics`, and TPU enums use the TPU base.
OpenROAD tasks must set `base_flavor = "gpu-openroad"` with a non-graphics H100
resource enum. Per-task `environment/Dockerfile`s are
`FROM ${BASE_IMAGE}:${BASE_TAG}` and only add packages, so every final task image
inherits the selected shared stack.

| Domain | Library | Import name | Use | apt deps |
| --- | --- | --- | --- | --- |
| CFD & Aerodynamics | `aerosandbox==4.2.9` | `aerosandbox` | Vortex-lattice + airfoil aero (pulls `casadi`, `neuralfoil`) | — |
| Analog Circuits | `PySpice==1.5` | `PySpice` | SPICE DC/AC/transient via ngspice | `ngspice`, `libngspice0-dev` |
| Structural / FEM | `scikit-fem==12.0.1` | `skfem` | General FEM (`K u = F`, modal) | — |
| Structural / frames | `PyniteFEA==3.0.0` | `Pynite` | 3D frame/truss FEA | — |
| Structural / frames | `anastruct==1.6.2` | `anastruct` | 2D frame/truss FEA | — |
| Electromagnetics | `scikit-rf==1.12.0` | `skrf` | RF networks & S-parameters | — |
| Reacting flow / thermo | `cantera==3.2.0` | `cantera` | Chemical kinetics, equilibrium, 1-D flames | — |
| Thermo properties | `CoolProp==7.2.0` | `CoolProp` | Fluid-property lookups (cycle tasks) | — |
| EDA | `klayout==0.30.9` | `klayout.db` | Layout I/O, DRC, reviewer figures | — |

Versions are pinned on purpose: the determinism playbook requires "same
submission → same score", and a solver version bump can shift the reference
field a task grades against. Bump deliberately and re-run the smoke test.

## Heavy binary engines — also in the base

The heavier engines are installed too, by
[`../base/install-solvers-heavy.sh`](../base/install-solvers-heavy.sh) (invoked
from `install-common.sh`). They are not plain pip wheels, so each is installed
via the most reliable cross-distro mechanism and exposed for scorers to call.

| Engine | Domain | Install mechanism | How a scorer invokes it | Bases |
| --- | --- | --- | --- | --- |
| **OpenSeesPy** | Structural FEM (nonlinear) | pip into the runtime venv (works on py3.13) | `python -c "import openseespy.opensees as ops; ..."` | CPU + GPU |
| **Meep** | EM / photonics (FDTD) | conda env `pymeep` (`nompi`) at `/opt/solver-envs/meep` | subprocess `"/opt/solver-envs/meep/bin/python"` | CPU + GPU |
| **SU2** | High-fidelity CFD (RANS) | conda env at `/opt/solver-envs/su2`; CLIs symlinked | `SU2_CFD config.cfg` (on PATH) | CPU + GPU |
| **OpenFOAM** | High-fidelity CFD | conda env at `/opt/solver-envs/openfoam` | `bash -lc 'source /etc/solver-envs.d/openfoam.sh && blockMesh ... && checkMesh ...'` | CPU + GPU |
| **CalculiX** | 3D continuum FEM | conda env; `ccx` symlinked onto PATH | `ccx jobname` (on PATH) | CPU + GPU |
| **OpenROAD** | EDA placement/route | `gpu-openroad` overlay copies from `openroad/orfs` (+ `or-tools` libs) | `openroad -exit script.tcl` (on PATH) | **gpu-openroad only** |

Why these mechanics:

- The CFD/EM engines conflict over MPI in a single conda env, so each gets its
  own isolated env — which also matches the determinism playbook's "run serial /
  fixed decomposition" rule (Meep is the `nompi` build).
- OpenROAD ships only as a large Ubuntu-built tree (`icu70`, `libpython3.10`,
  bundled `or-tools`/`scip`/`highs`). It is isolated in `gpu-openroad` so normal
  ML GPU tasks stay on the clean CUDA/PyTorch stack. EDA tasks opt in with
  `base_flavor = "gpu-openroad"`.

### Verify the stack

```bash
# CPU base -> 14/14 (OpenROAD reported as SKIP, it is GPU-only)
docker build -f base/cpu/Dockerfile -t lbx-tasks-base:runtime-ml-core-py313-local .
docker run --rm -v "$PWD/base/tests/test_solvers.py:/tmp/test_solvers.py:ro" \
    lbx-tasks-base:runtime-ml-core-py313-local python /tmp/test_solvers.py

# Clean GPU base -> 14/14 (OpenROAD reported as SKIP). Builds without a GPU;
# every engine in the smoke test is CPU, so it runs anywhere -- only CUDA
# workloads need a GPU.
docker build -f base/gpu/Dockerfile -t lbx-tasks-base-gpu:runtime-ml-core-py313-local .
docker run --rm -v "$PWD/base/tests/test_solvers.py:/tmp/test_solvers.py:ro" \
    lbx-tasks-base-gpu:runtime-ml-core-py313-local python /tmp/test_solvers.py

# OpenROAD overlay -> 15/15.
docker build -f base/gpu-openroad/Dockerfile \
    -t lbx-tasks-base-gpu-openroad:runtime-ml-core-py313-local .
docker run --rm -v "$PWD/base/tests/test_solvers.py:/tmp/test_solvers.py:ro" \
    lbx-tasks-base-gpu-openroad:runtime-ml-core-py313-local python /tmp/test_solvers.py
```

Reference values (identical across arch, a good determinism signal):
aerosandbox VLM `CL=0.4324`, pyspice `Vout=5.000V`, cantera `Tad=2225.5K`,
coolprop water `Tsat=373.12K`, meep FDTD `Ez=-0.26517`.

> Note: PySpice 1.5 prints `Unsupported Ngspice version 44` (on the Debian CPU
> base) and a couple of `SyntaxWarning`s from its docstrings. Both are cosmetic
> — the simulation runs and returns correct results.
>
> The CPU base ships ngspice **44** (Debian) and the GPU base ships ngspice
> **36** (Ubuntu 22.04). The divider smoke test returns `Vout=5.000V` on both.
> This version skew is a non-issue for determinism because SPICE tasks should
> pick a CPU `required_resources` enum and therefore build from the CPU base. If
> you ever need a SPICE task on the GPU base, pin/verify ngspice behavior
> explicitly.

### Solver-agnostic instructions with solver-backed oracles

Every numerical-solver task must put solver-specific runnable material in the
core solution implementation (`solution/solve.sh` or solution files it calls).
Public `/data` may contain ordinary task inputs and optional debug assets, but it
is not the blocking solver-validation surface.

For OpenFOAM tasks, the oracle path should include material such as:

```bash
python solution/openfoam_case.py /tmp/output/design.json /tmp/oracle_openfoam_case
bash -lc 'source /etc/solver-envs.d/openfoam.sh && blockMesh -case /tmp/oracle_openfoam_case && checkMesh -case /tmp/oracle_openfoam_case'
```

For OpenSeesPy tasks, the oracle path should include material that imports
`openseespy.opensees as ops`, builds or repairs a model, calls
`ops.analyze(...)`, and writes the reference result.

Do not require transcript checks or a submitted `solver_evidence.json` artifact.
The submitted oracle source, hidden oracle/scorer, ground-truth proof, and image
smoke tests are the solver validation surface.

### Propagation to the production base images

`base/` is part of the mothership `shared_paths` synced by
[`sync-shared-from-template.yml`](https://github.com/Alignerr-Code-Labeling/lbx-rl-tasks-iso-mothership/blob/main/.github/workflows/sync-shared-from-template.yml)
in `lbx-rl-tasks-iso-mothership`,
so the production CPU and GPU base images are built from these exact files
(`requirements-solvers.txt` + `install-solvers-heavy.sh` + the Dockerfiles).
After this change lands, the **mothership must rebuild both base images and bump
`BASE_IMAGE_TAG`** (in its `exporters/taiga.py`) so exported tasks pick up the
new base. The sync workflow preserves the existing tag; it does not bump it
automatically. Local harness runs already use the updated `base/` files
directly, so no tag bump is needed for local authoring.

Heads-up on size/time: the heavy engines (conda envs + OpenROAD copy) add
several GB and minutes to each base build, and the GPU base now pulls
`openroad/orfs` at build time. This is the cost of baking everything into the
shared base rather than per-task layers.

## Still per-task (not baked into the base)

Four engines are intentionally left out of the shared base because they are
fragile source builds, niche, or cannot be runtime-verified here without a GPU.
Add them in the task's `environment/Dockerfile` (`FROM ${BASE_IMAGE}`). Recipes
are starting points; validate per task with
`uv run lbx-rl-harness run --runtime ground-truth`.

| Engine | Domain | Why per-task | Recipe |
| --- | --- | --- | --- |
| **XFOIL** | 2D airfoil viscous-inviscid | No maintained pip/conda CLI (`xfoil` pip is dead on py3.13); fragile gfortran build | build from `web.mit.edu/drela/.../xfoil6.99.tgz` with `gfortran` + `libx11-dev` |
| **AVL** | Vortex-lattice (Drela) | Same Fortran/X11 build situation as XFOIL | build from `web.mit.edu/drela/.../avl3.36.tgz` |
| **PyNEC** | EM antennas (MoM) | pip sdist drops its SWIG wrapper on py3.13; needs autotools + SWIG source build | build `github.com/tmolteno/necpp` (`./build.sh`, then `python/` bindings) |
| **DREAMPlace** | EDA placement (GPU) | Custom CUDA ops; build + run require an actual GPU (can't verify in CI here) | GPU base + source build against the base's CUDA 12.4 torch |

> AeroSandbox (VLM) + NeuralFoil (an ML XFOIL surrogate) are already in the base,
> so most airfoil/wing aero tasks do not need the XFOIL/AVL CLIs.
>
> `ahkab` (a pure-python SPICE alt) is deliberately excluded everywhere: it is
> unmaintained and imports the removed `imp` module, so it fails on py3.13. Use
> the in-base PySpice + ngspice path instead.

### Example: a task that needs XFOIL (built from source)

```dockerfile
ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/airfoil-xfoil-polar
FROM ${BASE_IMAGE}:${BASE_TAG}
ARG PROBLEM_DIR=problems/airfoil-xfoil-polar
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gfortran make libx11-dev curl \
 && curl -fsSL https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz | tar xz -C /opt \
 && echo "build /opt/Xfoil/plotlib then /opt/Xfoil/bin with FC=gfortran; cp xfoil /usr/local/bin" \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workdir
RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
# Copy task data, scorer/data, scorer/, and task metadata as in the starters.
# The shared grading package is already installed in the selected base image.
```

### Example: a task that uses an in-base engine (no extra install)

```dockerfile
ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/sallen-key-lowpass
FROM ${BASE_IMAGE}:${BASE_TAG}
ARG PROBLEM_DIR=problems/sallen-key-lowpass
WORKDIR /workdir
RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
# PySpice/ngspice, Cantera, scikit-fem, Meep, SU2, OpenFOAM, CalculiX,
# OpenSeesPy are already in the base; nothing else to install.
COPY ${PROBLEM_DIR}/data/ /data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +
COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
```

For every heavy engine follow the cross-cutting **Determinism Playbook** in the
proposal: pin the solver version, run serial / fixed decomposition, use fixed
iteration counts, ship pre-built meshes/inputs in `scorer/data/`, grade on
tolerances, and treat divergence / NaN / non-convergence as a `0`. Also keep
the submitted ground-truth oracle path solver-backed through the core solution
implementation, without exposing solver requirements in `instruction.md`.

## Installed packages (full inventory)

Generated from the built base images (`importlib.metadata` for the runtime venv,
`conda list` per heavy-engine env, plus the system binaries). Regenerate after
any base change.

### Heavy-engine conda envs (`/opt/solver-envs/<env>`)

| Env | Package | Version | Env Python |
| --- | --- | --- | --- |
| `meep` | `pymeep` (nompi) | 1.33.0 | 3.13 |
| `su2` | `su2` | 8.5.0 | 3.11 |
| `openfoam` | `openfoam` | 2412 | n/a (binaries) |
| `calculix` | `calculix` (`ccx`) | 2.23 | n/a (binaries) |

### System binaries

| Binary | Version | Source | Bases |
| --- | --- | --- | --- |
| `ngspice` / `libngspice0` | 44 (Debian CPU) / 36 (Ubuntu GPU) | apt | CPU + GPU |
| `ccx` (CalculiX) | 2.23 | conda env, symlinked | CPU + GPU |
| `SU2_CFD` / `SU2_SOL` / `SU2_DEF` | 8.5.0 | conda env, symlinked | CPU + GPU |
| `simpleFoam` / `pimpleFoam` / `blockMesh` / `topoSet` / `createBaffles` (OpenFOAM) | v2412 | conda env (`source /etc/solver-envs.d/openfoam.sh`) | CPU + GPU |
| `openroad` | 26Q2 | multi-stage copy from `openroad/orfs` | GPU only |

### Runtime venv — pip packages (`/opt/lbx-runtime/.venv`, 191 total)

The runtime venv carries the shared ML stack plus the pip-installable solver
libraries. The solver-relevant entries:

| Package | Version | Domain |
| --- | --- | --- |
| `AeroSandbox` (+ `casadi`, `NeuralFoil`) | 4.2.9 (3.7.2 / 0.3.2) | CFD / aero |
| `PySpice` | 1.5 | analog circuits |
| `scikit-fem` | 12.0.1 | structural FEM |
| `PyNiteFEA` | 3.0.0 | frames / trusses |
| `anastruct` | 1.6.2 | frames / trusses (2D) |
| `openseespy` (+ `openseespylinux`) | 3.8.0.0 | structural FEM (nonlinear) |
| `scikit-rf` | 1.12.0 | electromagnetics (RF) |
| `cantera` | 3.2.0 | reacting flow / thermo |
| `CoolProp` | 7.2.0 | thermo properties |
| `klayout` | 0.30.9 | EDA layout / DRC |
| `pin` (Pinocchio) | 3.9.0 | rigid-body dynamics |
| `numpy` / `scipy` / `matplotlib` | 2.4.6 / 1.17.1 / 3.10.9 | shared numerics + plotting |
| `mujoco` | 3.9.0 | (robotics base) |
| `torch` / `torchvision` / `torchaudio` | 2.12.0 / 0.27.0 / 2.11.0 | ML base |

<details>
<summary>Complete runtime venv package list (191)</summary>

```text
absl-py==2.4.0
AeroSandbox==4.2.9
albucore==0.0.24
albumentations==2.0.8
alembic==1.18.4
anastruct==1.6.2
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
attrs==26.1.0
audioop-lts==0.2.2
audioread==3.1.0
braceexpand==0.1.7
cantera==3.2.0
casadi==3.7.2
catboost==1.2.10
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
cmeel==0.60.1
cmeel-assimp==6.0.5
cmeel-boost==1.90.0
cmeel-console-bridge==1.0.2.3
cmeel-octomap==1.10.0
cmeel-qhull==8.0.2.1
cmeel-tinyxml2==11.0.0
cmeel-urdfdom==6.0.0
cmeel-zlib==1.3.2
coal==3.0.3
colorlog==6.10.1
contourpy==1.3.3
CoolProp==7.2.0
cuda-bindings==13.3.1
cuda-pathfinder==1.5.5
cuda-toolkit==13.0.2
cycler==0.12.1
decorator==5.3.1
dicom2nifti==2.6.2
dill==0.4.1
donfig==0.8.1.post1
eigenpy==3.13.0
einops==0.8.2
etils==1.14.0
fastapi==0.136.3
filelock==3.29.1
fonttools==4.63.0
fsspec==2026.4.0
glfw==2.10.0
google-crc32c==1.8.0
graphviz==0.21
greenlet==3.5.1
h11==0.16.0
h5py==3.16.0
hf-xet==1.5.0
httpcore==1.0.9
httpx==0.28.1
httpx-sse==0.4.3
huggingface_hub==1.17.0
idna==3.18
ImageIO==2.37.3
importlib_resources==7.1.0
Jinja2==3.1.6
joblib==1.5.3
json-numpy==2.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.0
klayout==0.30.9
laspy==2.7.0
lazy-loader==0.5
lbx-rl-tasks-grading==0.1.0
libcoal==3.0.3
libpinocchio==4.0.0
librosa==0.11.0
lightgbm==4.6.0
llvmlite==0.47.0
Mako==1.3.12
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mcp==1.9.4
mdurl==0.1.2
mido==1.3.3
monai==1.5.2
mpmath==1.3.0
msgpack==1.1.2
mujoco==3.9.0
narwhals==2.22.0
networkx==3.6.1
NeuralFoil==0.3.2
nibabel==5.4.2
numba==0.65.1
numcodecs==0.16.5
numpy==2.4.6
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
opencv-python-headless==4.13.0.92
openseespy==3.8.0.0
openseespylinux==3.8.0.0
optuna==4.9.0
packaging==26.2
pandas==3.0.3
pillow==12.2.0
pin==4.0.0
platformdirs==4.10.0
plotly==6.8.0
ply==3.11
pooch==1.9.0
pretty_midi==0.2.11
prettytable==3.17.0
pyarrow==24.0.0
pycparser==3.0
pydantic==2.13.4
pydantic-settings==2.14.1
pydantic_core==2.46.4
pydicom==3.0.2
Pygments==2.20.0
PyNiteFEA==3.0.0
PyOpenGL==3.1.10
pyparsing==3.3.2
PySpice==1.5
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-gdcm==3.2.6
python-multipart==0.0.32
PyYAML==6.0.3
referencing==0.37.0
regex==2026.5.9
requests==2.34.2
rich==15.0.0
rpds-py==2026.5.1
ruamel.yaml==0.19.1
rubric==0.1.0
safetensors==0.7.0
scikit-fem==12.0.1
scikit-image==0.26.0
scikit-learn==1.9.0
scikit-rf==1.12.0
scipy==1.17.1
seaborn==0.13.2
setuptools==81.0.0
shellingham==1.5.4
simpleitk==2.5.5
simsimd==6.5.16
six==1.17.0
sortedcontainers==2.4.0
soundfile==0.13.1
soxr==1.1.0
SQLAlchemy==2.0.50
sse-starlette==3.4.4
standard-aifc==3.13.0
standard-chunk==3.13.0
standard-sunau==3.13.0
starlette==1.2.1
stringzilla==4.6.1
sympy==1.14.0
threadpoolctl==3.6.0
tifffile==2026.6.1
timm==1.0.27
tokenizers==0.22.2
torch==2.12.0
torchaudio==2.11.0
torchvision==0.27.0
tqdm==4.67.3
transformers==5.10.2
triton==3.7.0
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.7.0
uvicorn==0.49.0
wcwidth==0.7.0
webdataset==1.0.2
xgboost-cpu==3.2.0
zarr==3.2.1
zipp==4.1.0
zstandard==0.25.0
```

</details>

> The GPU base adds CUDA `torch` wheels and `openroad`; otherwise the solver
> inventory matches the CPU base.
