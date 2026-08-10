# Canonical ML GPU base for H100/Hopper tasks. OpenROAD/EDA lives in the
# gpu-openroad overlay so normal ML tasks do not inherit the EDA ABI stack.
#
# CUDA 13.0.x, NOT a higher 13.minor: Taiga's H100 runners carry NVIDIA driver
# 580.82.07, whose nvidia-smi advertises "CUDA Version: 13.0" (read from a live
# Taiga H100 container). CUDA 13.0 is therefore natively supported rather than
# leaning on minor-version compatibility, and cu130 is the newest PyTorch wheel
# index that exists at all (cu131/cu133 return 403).
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_PACKAGES="torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130"

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV TORCH_PACKAGES="${TORCH_PACKAGES}"
ENV TORCH_INDEX_STRATEGY=unsafe-best-match
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-gpu.txt
# Taiga's gVisor + nvproxy nodes do not forward the graphics RM ioctls, so a real
# GPU GL/EGL context cannot allocate (see base/cuda-graphics/Dockerfile). Render
# on CPU via osmesa instead of probing into a backend that fails at draw time.
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
