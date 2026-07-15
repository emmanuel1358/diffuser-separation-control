# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE=lbx-tasks-base-mlenvs-cuda-graphics
ARG BASE_TAG=runtime-ml-core-py313-local
FROM ${BASE_IMAGE}:${BASE_TAG}

RUN echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 \
        cuda-cudart-dev-12-8 \
        cuda-nvrtc-dev-12-8 \
        cuda-cccl-12-8 && \
    rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sfn /usr/local/cuda-12.8 /usr/local/cuda && \
    /usr/local/cuda/bin/nvcc --version

RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 \
        torchvision==0.22.0 \
        torchaudio==2.7.0

RUN PYSITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" && \
    for inc in "$PYSITE"/nvidia/*/include; do \
        [ -d "$inc" ] && cp -anL "$inc"/. /usr/local/cuda/include/ 2>/dev/null || true; \
    done && \
    for lib in "$PYSITE"/nvidia/*/lib; do \
        [ -d "$lib" ] && ln -sf "$lib"/lib*.so* /usr/local/cuda/lib64/ 2>/dev/null || true; \
    done && \
    sed -E -i '/__device_builtin__.*\b(sinpi|sinpif|cospi|cospif)[[:space:]]*\(/{/noexcept/!s/\)[[:space:]]*;/) noexcept(true);/}' \
        /usr/local/cuda/include/crt/math_functions.h && \
    test -f /usr/local/cuda/include/cusparse.h

ENV FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="9.0;10.0+PTX;12.0+PTX"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y pytorch3d 2>/dev/null || true && \
    pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

RUN rm -rf /tmp/.cache/torch_extensions && \
    mkdir -p /tmp/.cache/torch_extensions && \
    chmod -R 1777 /tmp/.cache/torch_extensions

ARG CUDA_GRAPHICS_BLACKWELL_HASH=unknown
LABEL taiga.cuda_graphics_blackwell.hash="${CUDA_GRAPHICS_BLACKWELL_HASH}"
LABEL taiga.cuda_graphics_blackwell.parent_arch_list="9.0;12.0+PTX"
LABEL taiga.cuda_graphics_blackwell.torch="2.7.0+cu128"
LABEL lbx.base.flavor="mlenvs-cuda-graphics-blackwell"
