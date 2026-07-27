# cuda-graphics base flavor: CUDA-native rasterization (pytorch3d / nvdiffrast).
#
# HARD-WON DEPLOYMENT FINDING (do not re-learn): on Taiga's gVisor + nvproxy
# H100 nodes, GPU GRAPHICS rendering does NOT work. A real NVIDIA GL/EGL context
# can be created, but the first GPU graphics-memory allocation fails
# (glBufferData -> GL_OUT_OF_MEMORY, FBOs come back GL_FRAMEBUFFER_UNSUPPORTED)
# because nvproxy does not forward the datacenter-Hopper graphics RM ioctls. That
# gate is upstream-only and not tunable from the image. Registering NVIDIA
# EGL/Vulkan ICDs was tried and reverted. The ONLY deployable GPU render path is
# CUDA-native rasterization (pytorch3d/nvdiffrast), so this flavor rides the
# compute GPU profile and never advertises +graphics. GL-based renderers
# (mujoco, PyOpenGL) therefore have to rasterize on CPU through osmesa, which is
# what the backend pins further down select.
#
# Uses the CUDA -devel image (not -runtime): pytorch3d is not on PyPI and
# compiles CUDA extensions from source, so it needs nvcc + headers. It must be
# the same CUDA minor that torch was built against, because those extensions
# link against this base's torch.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

# Python 3.13, in step with the other CUDA bases. This was pinned to 3.12 for
# the old torch-2.6.0+cu124 matrix, where kaolin, the PyG binaries and open3d
# all stopped at cp312. On cu130 that is no longer true: PyG publishes cp313
# wheels for all of torch_scatter/sparse/cluster/spline_conv + pyg_lib on
# data.pyg.org/whl/torch-2.9.1+cu130. kaolin and open3d were dropped instead of
# holding the whole base back -- kaolin has no cu130 index at all (its newest is
# torch-2.8.0_cu128, cp39-cp312) and open3d's latest release (0.19.0) is
# cp38-cp312. Re-add either only once it ships cp313 for this CUDA.
ARG PYTHON_VERSION=3.13
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_PACKAGES="torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130"
ARG TORCH_CUDA_ARCH_LIST=9.0

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV TORCH_PACKAGES="${TORCH_PACKAGES}"
ENV TORCH_INDEX_STRATEGY=unsafe-best-match
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cuda-graphics.txt
# Docker build has no visible GPU, so PyTorch cannot infer H100 arch flags.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# CUDA-native raster reads the device through compute APIs and does not go
# through PyOpenGL at all, so the PyOpenGL/MuJoCo backend only decides what the
# GL-based renderers in this image do. It used to say egl, which contradicted
# the header above and left MUJOCO_GL unset: mujoco then probed, picked glfw,
# and mujoco.Renderer died with "the DISPLAY environment variable is missing"
# followed by "an OpenGL platform library has not been loaded into this
# process". osmesa rasterizes on CPU (libosmesa6 is installed by
# install-common.sh on every flavor), and matches every other base.
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cuda-graphics.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh base/install-task-deps.sh /tmp/base/
RUN /tmp/base/install-common.sh

# nvdiffrast and pytorch3d are not on PyPI; install from source against the
# torch+CUDA that install-common put in the venv. pytorch3d compiles CUDA
# extensions (FORCE_CUDA=1, nvcc from the -devel base, ninja). --no-build-isolation
# so the build sees the already-installed torch.
#
# NVCC_FLAGS=-static-global-template-stub=false is what makes pytorch3d link
# under CUDA 13 (HARD-WON FINDING, do not drop it). nvcc >= 12.8 flipped the
# default of -static-global-template-stub to true, which gives every __global__
# function template instantiation internal linkage in whole-program mode.
# pytorch3d's pulsar renderer instantiates its kernel templates in one
# translation unit and calls them from another, so the link died with a wall of
# "undefined reference to pulsar::Renderer::calc_signature<true>" and
# "hidden symbol ... isn't defined / final link failed: bad value". nvcc says so
# itself first, as warning #20280-D against renderer.h, and names this flag as
# the fix. pytorch3d's setup.py appends $NVCC_FLAGS to its nvcc args, so this is
# the supported way in. The alternative it suggests, -rdc=true, would force
# separate compilation on the whole extension.
RUN uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache ninja \
 && FORCE_CUDA=1 \
    NVCC_FLAGS="-static-global-template-stub=false" \
    uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache \
      --no-build-isolation \
      "git+https://github.com/NVlabs/nvdiffrast.git" \
      "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# The PyG binaries ship per-(torch, CUDA) wheel indexes rather than PyPI builds,
# so they need their own -f index matched to this base's torch 2.9.1+cu130.
RUN uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache \
      torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
      -f https://data.pyg.org/whl/torch-2.9.1+cu130.html \
 && uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache torch-geometric

# kaolin is intentionally NOT installed. It publishes no cu130 wheel index (the
# newest is torch-2.8.0_cu128) and no cp313 wheels on any index, so it cannot be
# satisfied on this base's CUDA or Python. No task in this repo imports it.

# gsplat and nerfacc are sdist-only and JIT-compile CUDA kernels on first use;
# give them a world-writable extension cache so the uid-1000 agent can compile
# at runtime instead of failing on a root-owned cache dir.
ENV TORCH_EXTENSIONS_DIR=/tmp/.cache/torch_extensions
RUN mkdir -p /tmp/.cache/torch_extensions && chmod -R 1777 /tmp/.cache/torch_extensions

# Fail the build here rather than at task runtime if the graphics stack cannot
# import against this base's torch.
RUN /opt/lbx-runtime/.venv/bin/python -c "\
import torch, pytorch3d, nvdiffrast, torch_geometric, kornia, trimesh; \
print('graphics stack OK on torch', torch.__version__)"

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
