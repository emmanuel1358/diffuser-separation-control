ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local

FROM ${BASE_IMAGE}:${BASE_TAG}

COPY problems/diffuser-separation-control/data/ /data/
COPY problems/diffuser-separation-control/scorer/ /mcp_server/grader/
COPY problems/diffuser-separation-control/solution/ /workspace/solution/

# Harden private grader permissions
RUN chmod -R go-rwx /mcp_server/grader