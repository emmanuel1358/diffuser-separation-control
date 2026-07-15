# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE=lbx-tasks-base-mlenvs-gpu
ARG BASE_TAG=runtime-ml-core-py313-local
FROM ${BASE_IMAGE}:${BASE_TAG}

RUN echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    M="$(apt-cache pkgnames cuda-nvcc-12- | grep -E '^cuda-nvcc-12-[0-9]+$' | sort -V | tail -1)" && \
    echo "Selected nvcc package: ${M:-<none>}" && test -n "$M" && \
    MINOR="${M#cuda-nvcc-}" && \
    apt-get install -y --no-install-recommends \
        "cuda-nvcc-${MINOR}" \
        "cuda-cudart-dev-${MINOR}" \
        "cuda-nvrtc-dev-${MINOR}" \
        "cuda-cccl-${MINOR}" && \
    rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/* && \
    CUDA_REAL="/usr/local/cuda-$(echo "$MINOR" | tr - .)" && \
    echo "CUDA toolkit installed at: $CUDA_REAL" && test -d "$CUDA_REAL" && \
    ln -sfn "$CUDA_REAL" /usr/local/cuda && \
    sed -E -i '/__device_builtin__.*\b(sinpi|sinpif|cospi|cospif)[[:space:]]*\(/{/noexcept/!s/\)[[:space:]]*;/) noexcept(true);/}' \
        /usr/local/cuda/include/crt/math_functions.h && \
    PYSITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" && \
    for inc in "$PYSITE"/nvidia/*/include; do \
        [ -d "$inc" ] && cp -anL "$inc"/. /usr/local/cuda/include/ 2>/dev/null || true; \
    done && \
    for lib in "$PYSITE"/nvidia/*/lib; do \
        [ -d "$lib" ] && ln -sf "$lib"/lib*.so* /usr/local/cuda/lib64/ 2>/dev/null || true; \
    done && \
    /usr/local/cuda/bin/nvcc --version

ENV CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    TORCH_EXTENSIONS_DIR=/tmp/.cache/torch_extensions \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

RUN --mount=type=cache,target=/root/.cache/uv \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/* && \
    uv pip install --system --break-system-packages setuptools wheel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglu1-mesa \
        libglx0 \
        libglvnd0 \
        libgles2 \
        libvulkan1 \
        libosmesa6 \
        libglfw3 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxfixes3 \
        libxi6 \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 \
        libxxf86vm1 \
        libxkbcommon0 \
        libsm6 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --break-system-packages \
        gsplat \
        nerfacc

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        open3d \
        trimesh \
        point-cloud-utils \
        kornia \
        pyrender \
        "imageio[ffmpeg]" \
        av && \
    uv pip install --system --break-system-packages \
        pyopengl==3.1.7

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        pyg-lib \
        torch-geometric \
        -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
        kaolin==0.18.0 \
        -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system --break-system-packages \
        mujoco \
        warp-lang

RUN mkdir -p /tmp/.cache/torch_extensions && chmod -R 1777 /tmp/.cache/torch_extensions

CMD ["rubric", "mcp"]

ARG CUDA_GRAPHICS_HASH=unknown
LABEL taiga.cuda_graphics.hash="${CUDA_GRAPHICS_HASH}"
LABEL lbx.base.flavor="mlenvs-cuda-graphics"
