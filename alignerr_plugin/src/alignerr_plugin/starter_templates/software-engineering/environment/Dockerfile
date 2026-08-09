ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=.
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=.
ARG AGENT_UID=1000
ARG AGENT_GID=1000

USER root
WORKDIR /workdir

COPY ${PROBLEM_DIR}/environment/ /tmp/task-deps/environment/
COPY ${PROBLEM_DIR}/scorer/ /tmp/task-deps/scorer/
RUN /opt/lbx-runtime/install-task-deps.sh /tmp/task-deps \
    && rm -rf /tmp/task-deps

RUN mkdir -p \
        /mcp_server/data \
        /mcp_server/grader \
        /opt/software-engineering-starter \
        /tmp/output/repo \
        /workdir \
    && chown -R ${AGENT_UID}:${AGENT_GID} /tmp/output /workdir

COPY --chown=root:root ${PROBLEM_DIR}/starter/ /opt/software-engineering-starter/
COPY --chown=${AGENT_UID}:${AGENT_GID} ${PROBLEM_DIR}/starter/ /tmp/output/repo/
RUN chmod -R a+rX /opt/software-engineering-starter

COPY --chown=root:root ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chown=root:root ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} + \
    && ! su -s /bin/sh agent -c \
        'test -r /mcp_server/data/hidden_cases.json' \
    && ! su -s /bin/sh agent -c \
        'test -r /mcp_server/data/candidate_driver.py' \
    && ! su -s /bin/sh agent -c \
        'test -r /mcp_server/grader/compute_score.py'

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/

ENV PYTHONDONTWRITEBYTECODE=1
