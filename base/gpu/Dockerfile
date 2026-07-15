# Canonical ML GPU base for H100/Hopper tasks. OpenROAD/EDA lives in the
# gpu-openroad overlay so normal ML tasks do not inherit the EDA ABI stack.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_PACKAGES="torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124"

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV TORCH_PACKAGES="${TORCH_PACKAGES}"
ENV TORCH_INDEX_STRATEGY=unsafe-best-match
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-gpu.txt
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
