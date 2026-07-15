# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE=lbx-tasks-base-mlenvs-gpu
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR

FROM --platform=linux/amd64 python:3.12-slim AS task-src
ARG PROBLEM_DIR
COPY ${PROBLEM_DIR}/ /src/
RUN mkdir -p /data-src/public /data-src/private && \
    if [ -d /src/data/public ]; then cp -a /src/data/public/. /data-src/public/; fi && \
    if [ -d /src/data/private ]; then cp -a /src/data/private/. /data-src/private/; fi && \
    if [ ! -f /src/test_file.py ]; then echo "ERROR: ML_Envs task missing test_file.py" >&2; exit 1; fi

FROM ${BASE_IMAGE}:${BASE_TAG}

ARG APT_EXTRAS=""
RUN if [ -n "$APT_EXTRAS" ]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends $APT_EXTRAS && \
      rm -rf /var/lib/apt/lists/*; \
    fi

ARG DEPENDENCIES=""
RUN if [ -n "$DEPENDENCIES" ]; then \
      uv pip install --system --break-system-packages --no-cache $DEPENDENCIES; \
    fi

# Env-server-only deps (env/hybrid tasks): root-only /mcp_server/env_deps on the
# env server's sys.path. /mcp_server is chmod 0700 root (below), so the uid-1000
# agent cannot import them directly -- reachable ONLY through the RPC.
ARG ENV_DEPENDENCIES=""
RUN if [ -n "$ENV_DEPENDENCIES" ]; then \
      mkdir -p /mcp_server/env_deps && \
      uv pip install --target /mcp_server/env_deps --no-cache $ENV_DEPENDENCIES && \
      chown -R root:root /mcp_server/env_deps && \
      find /mcp_server/env_deps -type d -exec chmod 0700 {} + && \
      find /mcp_server/env_deps -type f -exec chmod 0600 {} + ; \
    fi

# Grader-only deps (any task type): root-only /mcp_server/grading_deps, prepended
# to the grader worker's sys.path before compute_score loads. 0700 /mcp_server
# (below) keeps them invisible to the uid-1000 agent -- for a scoring/reference
# library that would leak the intended approach if it were agent-visible.
ARG GRADING_DEPENDENCIES=""
RUN if [ -n "$GRADING_DEPENDENCIES" ]; then \
      mkdir -p /mcp_server/grading_deps && \
      uv pip install --target /mcp_server/grading_deps --no-cache $GRADING_DEPENDENCIES && \
      chown -R root:root /mcp_server/grading_deps && \
      find /mcp_server/grading_deps -type d -exec chmod 0700 {} + && \
      find /mcp_server/grading_deps -type f -exec chmod 0600 {} + ; \
    fi

COPY --from=task-src /src/test_file.py /mcp_server/grader/compute_score.py
COPY --from=task-src /src/prompt.md /task/prompt.md
RUN chown -R root:root /mcp_server/grader \
 && find /mcp_server/grader -type d -exec chmod 0700 {} + \
 && find /mcp_server/grader -type f -exec chmod 0600 {} +

# Bake [environment].hidden_env into /task/task.toml so the runtime env-server
# gate activates the hidden env server for env/hybrid tasks. Agent-visible; holds
# no held-out truth.
ARG HIDDEN_ENV=""
RUN mkdir -p /task \
 && printf '[environment]\nhidden_env = "%s"\n' "$HIDDEN_ENV" > /task/task.toml

RUN mkdir -p /mcp_server/data
COPY --from=task-src /data-src/private /mcp_server/data
RUN chown -R root:root /mcp_server/data \
 && find /mcp_server/data -type d -exec chmod 0700 {} + \
 && find /mcp_server/data -type f -exec chmod 0600 {} +

RUN mkdir -p /data
COPY --from=task-src /data-src/public /data
RUN chown -R root:root /data \
 && find /data -type d -exec chmod 0555 {} + \
 && find /data -type f -exec chmod 0444 {} +

# Agent-writable working + submission dirs (uid 1000). The mlenvs bases don't run
# install-common.sh, so without this the agent could not write to /tmp/output.
RUN mkdir -p /tmp/output /workdir \
 && chown -R 1000:1000 /tmp/output /workdir \
 && chmod 0777 /tmp/output /workdir

RUN chmod 0700 /mcp_server
