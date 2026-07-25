# tpu base flavor: jax[tpu] on Python 3.12 (jaxlib/libtpu wheels target 3.12).
#
# After the JAX install, base/clear_execstack.py strips the executable-stack flag
# off jaxlib/libtpu .so files: jaxlib ships PT_GNU_STACK as RWX, and the hardened
# grading loader refuses an executable stack (`import jax` fails with "cannot
# enable executable stack ...", jax-ml/jax#26781). The flag is a packaging
# artifact; JAX does not need an executable stack.
FROM python:3.12-slim

# TPU pins Python 3.12; install-common honors PYTHON_VERSION.
ARG PYTHON_VERSION=3.12
ARG LIBTPU_INDEX_URL=https://storage.googleapis.com/jax-releases/libtpu_releases.html

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV PYTHON_VERSION=${PYTHON_VERSION}
# Single-chip libtpu config for the grading process.
ENV ALLOW_MULTIPLE_LIBTPU_LOAD=1
ENV TPU_LOG_DIR=/var/log/libtpu
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh /tmp/base/
COPY base/requirements-tpu.txt base/clear_execstack.py /tmp/base/
RUN mkdir -p "${TPU_LOG_DIR}" && /tmp/base/install-common.sh

# jax[tpu] from the libtpu releases index, then clear the exec-stack flag.
RUN uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache \
      --extra-index-url "${LIBTPU_INDEX_URL}" \
      -r /tmp/base/requirements-tpu.txt \
 && /opt/lbx-runtime/.venv/bin/python /tmp/base/clear_execstack.py

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
