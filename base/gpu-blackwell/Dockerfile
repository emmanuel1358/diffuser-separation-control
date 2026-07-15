# Canonical ML GPU base for Blackwell tasks.
#
# Targets CUDA 12.8 / PyTorch cu128 and exports sm_120 architecture flags so
# task-specific CUDA extensions build for B100/B200/GB200-class GPUs.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_PACKAGES="torch torchvision torchaudio"
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
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-gpu.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh /tmp/base/
RUN /tmp/base/install-common.sh

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
