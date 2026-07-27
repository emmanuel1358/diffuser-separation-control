FROM python:3.13-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Pinned, and pinned to the +cpu local version specifically.
#
# These were previously bare `torch torchvision torchaudio` (install-common's
# default), which floated with whatever was newest at build time and, worse,
# resolved to the CUDA wheel: uv searches --extra-index-url (PyPI) before
# --index-url, so PyPI's plain `torch` won over this index's +cpu build. The CPU
# base was shipping the whole nvidia-* stack -- cuDNN, cuBLAS, cuSPARSE and a
# 200 MB NCCL -- in an image that can never have a GPU.
#
# 2.13.0 / 0.28.0 / 2.11.0 is exactly what floated in before this pin (verified
# by resolving with the same index arguments), so this is not a version change;
# it only drops the CUDA payload and stops two builds of the same commit from
# baking different torch versions.
ARG TORCH_PACKAGES="torch==2.13.0+cpu torchvision==0.28.0+cpu torchaudio==2.11.0+cpu"

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV TORCH_PACKAGES="${TORCH_PACKAGES}"
# Reaching a +cpu local version needs best-match for the same reason the CUDA
# bases need it for +cu130: under uv's default first-index safety, PyPI carries a
# plain `torch` at this version, so the +cpu build on this index is never even
# considered and the pin fails to resolve.
ENV TORCH_INDEX_STRATEGY=unsafe-best-match
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cpu.txt
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache
# Pin the headless software rasterizer instead of letting MuJoCo probe: there is
# no display and no GPU here, so glfw and EGL cannot produce a context and
# mujoco.Renderer would fail. osmesa renders on CPU.
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cpu.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh base/install-task-deps.sh /tmp/base/
RUN /tmp/base/install-common.sh

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
