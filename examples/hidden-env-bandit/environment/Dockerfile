ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=.
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=.

WORKDIR /workdir

RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
# grading ships the env_server package (python -m env_server) + load_env_module
# + the sandboxed policy loader. numpy + msgpack already live in the base image.
# data/env_client.py -> /data/env_client.py (agent-visible client).
COPY ${PROBLEM_DIR}/data/ /data/
# scorer/data/env.py -> /mcp_server/data/env.py (root-only; the hidden env).
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
