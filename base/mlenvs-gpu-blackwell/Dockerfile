# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE=lbx-tasks-base-mlenvs-gpu
ARG BASE_TAG=runtime-ml-core-py313-local
FROM ${BASE_IMAGE}:${BASE_TAG}

RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 \
        torchvision==0.22.0 \
        torchaudio==2.7.0

ARG BASE_BLACKWELL_HASH=unknown
LABEL taiga.base_blackwell.hash="${BASE_BLACKWELL_HASH}"
LABEL taiga.base_blackwell.torch="2.7.0+cu128"
LABEL lbx.base.flavor="mlenvs-gpu-blackwell"
