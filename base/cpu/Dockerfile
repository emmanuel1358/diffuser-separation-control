FROM python:3.13-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cpu.txt
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cpu.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh /tmp/base/
RUN /tmp/base/install-common.sh

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
