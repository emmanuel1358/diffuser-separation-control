ARG BASE_IMAGE=lbx-tasks-base-gpu
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=.
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=.

WORKDIR /workdir

# Declared dependency channels. Each file is optional; the installer routes each
# one to its isolation boundary:
#   environment/apt.txt          apt packages (agent-visible)
#   environment/requirements.txt pip into the runtime venv (agent-visible)
#   scorer/requirements.txt      pip into root-only /mcp_server/grading_deps
#   scorer/env-requirements.txt  pip into root-only /mcp_server/env_deps
COPY ${PROBLEM_DIR}/environment/ /tmp/task-deps/environment/
COPY ${PROBLEM_DIR}/scorer/ /tmp/task-deps/scorer/
RUN /opt/lbx-runtime/install-task-deps.sh /tmp/task-deps && rm -rf /tmp/task-deps

RUN mkdir -p /mcp_server/data /mcp_server/calibration /workdir /tmp/output \
    && chown -R 1000:1000 /workdir /tmp/output

# Public data the agent reads at /data, read-only.
COPY ${PROBLEM_DIR}/data/ /data/
RUN chown -R root:root /data \
    && find /data -type d -exec chmod 0555 {} + \
    && find /data -type f -exec chmod 0444 {} +

# Held-out data and the grader itself. Both root-only: the uid-1000 agent must
# not be able to read the answer key or the scoring code.
COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
# Author-generated calibration lock. Trusted CI mounts the promoted lock over
# /mcp_server/calibration at deploy; the .author-source marker makes the rubric
# server refuse to grade with this fallback when the mount is required. The glob
# keeps a freshly scaffolded task buildable before the first calibration run.
COPY --chown=root:root ${PROBLEM_DIR}/calibration.lock.jso[n] /mcp_server/calibration/
RUN [ -f /mcp_server/calibration/calibration.lock.json ] \
    && printf 'author-image-fallback\n' > /mcp_server/calibration/.author-source \
    || true
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader /mcp_server/calibration \
    && find /mcp_server/data /mcp_server/grader /mcp_server/calibration -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader /mcp_server/calibration -type f -exec chmod 0600 {} + \
    && chmod 0700 /mcp_server

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
