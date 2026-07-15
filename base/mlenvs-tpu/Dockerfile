# syntax=docker/dockerfile:1.4

FROM --platform=linux/amd64 python:3.12-slim

ENV IS_SANDBOX="yes" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOURCE_DATE_EPOCH=0 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/home/claude \
    PATH="/root/.local/bin:${PATH}:/home/claude/.local/bin" \
    USERNAME="model" \
    UV_CACHE_DIR=/root/.cache/uv \
    XDG_CACHE_HOME=/root/.cache \
    PIP_NO_CACHE_DIR=1

RUN groupadd -g 1000 model && useradd -m -u 1000 -g 1000 model

# The rubric MCP + grader drop privileges to the uid-1000 account by NAME,
# defaulting to "agent" (the native bases' account, created by install-common.sh
# which the mlenvs bases never run). Here that account is "model", so bake the
# identity by NUMERIC uid/gid + name: with RUBRIC_AGENT_UID/GID set,
# grading._agent_identity takes the getpwuid(1000) branch and drops correctly
# without any platform-injected override -- otherwise every privilege drop (model
# loaders, kfold worker, hidden-env sandbox) AND the agent's own bash/editor tools
# raise "user 'agent' not found". HOME is pinned to the image's /home/claude
# (already on PATH as /home/claude/.local/bin) so the dropped policy worker's
# user-site matches where the agent pip-installs --user.
ENV RUBRIC_AGENT_UID=1000 \
    RUBRIC_AGENT_GID=1000 \
    RUBRIC_AGENT_USER=model \
    RUBRIC_AGENT_HOME=/home/claude

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        tmux \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN { \
        echo "set -g default-shell /bin/bash"; \
        echo "set -g default-command /bin/bash"; \
        echo "set-environment -g PATH \"${PATH}\""; \
        [ -n "${LD_LIBRARY_PATH:-}" ] && echo "set-environment -g LD_LIBRARY_PATH \"${LD_LIBRARY_PATH}\""; \
        true; \
    } > /etc/tmux.conf

RUN HOME=/root sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

RUN mkdir -p /mcp_server
WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/

# Install rubric/grader runtime deps EXPLICITLY minus numpy, then the local
# packages --no-deps, so the JAX-resolved numpy is preserved (a plain `-e`
# install would force the rubric pyproject's pinned numpy==2.4.4).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        "mcp[cli]==1.27.1" "mujoco==3.8.0" "pydantic==2.13.4" "typer==0.25.1" \
        "httpx==0.28.1" "json-numpy==2.1.1" "msgpack==1.2.0" && \
    uv pip install --system --break-system-packages --no-deps \
        -e /mcp_server -e /runtime/grading

RUN printf '%s\n' \
    '#!/usr/bin/env python' \
    'from grader_runner.run_grader import main' \
    '' \
    'raise SystemExit(main())' \
    > /runtime/run_grader.py && \
    chmod +x /runtime/run_grader.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        "jax[tpu]==0.5.2" "jaxlib==0.5.1"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        "jax==0.5.2" "jaxlib==0.5.1" \
        flax optax chex orbax-checkpoint equinox \
        pandas scipy statsmodels safetensors

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        "jax==0.5.2" "jaxlib==0.5.1" \
        brax mujoco-mjx gymnax

COPY base/clear_execstack.py /tmp/clear_execstack.py
RUN python3 /tmp/clear_execstack.py && rm -f /tmp/clear_execstack.py

RUN python3 -c "import jax, jaxlib, flax, optax, chex, orbax.checkpoint, equinox, brax, mujoco.mjx, gymnax; \
assert jax.__version__ == '0.5.2', jax.__version__; \
print('jax', jax.__version__, 'jaxlib', jaxlib.__version__, '+ ecosystem OK')"

RUN mkdir -p /var/log/libtpu && chmod 1777 /var/log/libtpu
ENV ALLOW_MULTIPLE_LIBTPU_LOAD=1 \
    TPU_LOG_DIR=/var/log/libtpu

RUN mkdir -p /mcp_server/task && \
    chown -R root:root /mcp_server && \
    chmod -R 0700 /mcp_server
RUN mkdir -p /home/claude && chown -R model:model /home/claude
RUN mkdir -p /mnt/skills && chown 1000:1000 /mnt/skills

CMD ["rubric", "mcp"]

ARG TPU_HASH=unknown
LABEL taiga.tpu.hash="${TPU_HASH}"
LABEL taiga.base.flavor="tpu"
LABEL lbx.base.flavor="mlenvs-tpu"
