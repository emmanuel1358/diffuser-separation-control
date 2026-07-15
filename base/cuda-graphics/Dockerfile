# cuda-graphics base flavor: CUDA-native rasterization (pytorch3d / nvdiffrast).
#
# HARD-WON DEPLOYMENT FINDING (do not re-learn): on Taiga's gVisor + nvproxy
# H100 nodes, GPU GRAPHICS rendering does NOT work. A real NVIDIA GL/EGL context
# can be created, but the first GPU graphics-memory allocation fails
# (glBufferData -> GL_OUT_OF_MEMORY, FBOs come back GL_FRAMEBUFFER_UNSUPPORTED)
# because nvproxy does not forward the datacenter-Hopper graphics RM ioctls. That
# gate is upstream-only and not tunable from the image. Registering NVIDIA
# EGL/Vulkan ICDs was tried and reverted. The ONLY deployable render path is
# CUDA-native rasterization (pytorch3d/nvdiffrast), so this flavor rides the
# compute GPU profile and never advertises +graphics.
#
# Uses the CUDA -devel image (not -runtime): pytorch3d is not on PyPI and
# compiles CUDA extensions from source, so it needs nvcc + headers.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# pytorch3d/nvdiffrast wheels target a specific (python, torch, cuda) matrix. If
# the matching wheels are unavailable for 3.13, set PYTHON_VERSION=3.12 here.
ARG PYTHON_VERSION=3.13
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_PACKAGES="torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124"
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
# CUDA-native raster reads the device through compute APIs; no GL/EGL ICDs.
ENV PYOPENGL_PLATFORM=egl
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cuda-graphics.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh /tmp/base/
RUN /tmp/base/install-common.sh

# nvdiffrast and pytorch3d are not on PyPI; install from source against the
# torch+CUDA that install-common put in the venv. pytorch3d compiles CUDA
# extensions (FORCE_CUDA=1, nvcc from the -devel base, ninja). --no-build-isolation
# so the build sees the already-installed torch.
RUN uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache ninja \
 && FORCE_CUDA=1 uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache \
      --no-build-isolation \
      "git+https://github.com/NVlabs/nvdiffrast.git" \
      "git+https://github.com/facebookresearch/pytorch3d.git@stable"

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
