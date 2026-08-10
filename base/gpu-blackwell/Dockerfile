# Canonical ML GPU base for Blackwell tasks.
#
# Targets CUDA 13.0 / PyTorch cu130 — the same CUDA and torch as base/gpu and
# base/cuda-graphics — and exports sm_120 architecture flags so task-specific
# CUDA extensions build for B100/B200/GB200-class GPUs. Only the arch flags and
# the -devel image (nvcc, for those extension builds) differ from base/gpu.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
# Pinned explicitly. These were previously bare `torch torchvision torchaudio`,
# so the resolved torch floated with whatever was newest at build time and two
# builds of the same commit could bake different torch versions.
ARG TORCH_PACKAGES="torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130"
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG CMAKE_CUDA_ARCHITECTURES=120
ARG CUDAARCHS=120

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV TORCH_PACKAGES="${TORCH_PACKAGES}"
ENV TORCH_INDEX_STRATEGY=unsafe-best-match
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-gpu.txt
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
ENV CUDAARCHS=${CUDAARCHS}
# Same gVisor + nvproxy limitation as base/gpu: the graphics RM ioctls are not
# forwarded, so a real GL/EGL context cannot allocate. Render on CPU via osmesa
# instead of probing into a backend that fails at draw time.
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-gpu.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh base/install-task-deps.sh /tmp/base/
RUN /tmp/base/install-common.sh

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
