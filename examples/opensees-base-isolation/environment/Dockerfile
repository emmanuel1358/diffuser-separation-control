ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=examples/opensees-base-isolation
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=examples/opensees-base-isolation

ENV MPLBACKEND=Agg

WORKDIR /workdir

RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output

RUN env -u UV_SYSTEM_PYTHON uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache \
    openseespy \
    numpy

# Public files the agent may read (world-readable at /data).
COPY ${PROBLEM_DIR}/data/ /data/

# Private grading fixtures + grader code (root-owned, unreadable to the agent).
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/

# Root-readable backup of the shared model so the grader can import it even
# without /data on its import path.
COPY --chown=root:root \
    ${PROBLEM_DIR}/data/public_isolation_model.py \
    /mcp_server/public_data/

RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \
    && chmod 0555 /mcp_server/public_data \
    && chmod 0444 /mcp_server/public_data/*.py

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/

ENV PATH="/usr/local/bin:${PATH}"

# Make the bare `python` / `python3` the grading venv interpreter (which carries
# openseespy + numpy).
RUN set -eux; \
    test -x /opt/lbx-runtime/.venv/bin/python; \
    rm -f /usr/local/bin/python /usr/local/bin/python3; \
    printf '%s\n' '#!/bin/sh' 'exec /opt/lbx-runtime/.venv/bin/python "$@"' > /usr/local/bin/python; \
    cp /usr/local/bin/python /usr/local/bin/python3; \
    chmod 0755 /usr/local/bin/python /usr/local/bin/python3; \
    python --version; \
    python -c "import openseespy.opensees as ops; print('openseespy import ok')"
