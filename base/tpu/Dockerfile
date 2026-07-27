# tpu base flavor: jax[tpu] on Python 3.12 (jaxlib/libtpu wheels target 3.12).
#
# After the JAX install, base/clear_execstack.py strips the executable-stack flag
# off jaxlib/libtpu .so files: jaxlib ships PT_GNU_STACK as RWX, and the hardened
# grading loader refuses an executable stack (`import jax` fails with "cannot
# enable executable stack ...", jax-ml/jax#26781). The flag is a packaging
# artifact; JAX does not need an executable stack.
FROM python:3.12-slim

# TPU pins Python 3.12; install-common honors PYTHON_VERSION.
ARG PYTHON_VERSION=3.12
ARG LIBTPU_INDEX_URL=https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Torch is pinned to a CPU-only build even though this flavor has no torch of
# its own: timm and transformers (requirements-common.txt) depend on it, so it
# arrived transitively from PyPI, which serves the CUDA wheel. A TPU image was
# therefore carrying the full nvidia-* stack -- cuDNN, cuBLAS, a 200 MB NCCL --
# for an accelerator it does not have and cannot use. Installing it explicitly
# here also moves it ahead of requirements-common, and install-common turns the
# == pins into constraints for that install, so the transitive resolve can no
# longer pull the CUDA build back in.
#
# Same versions as base/cpu (both have cp312 and cp313 wheels), so the two
# flavors stay in step. torchaudio is deliberately not included: nothing on this
# flavor has ever pulled it in, and this pin is meant to preserve the package
# set, not extend it.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES="torch==2.13.0+cpu torchvision==0.28.0+cpu"

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/opt/lbx-runtime/.venv/bin:/usr/local/bin:${PATH}"
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}
ENV TORCH_PACKAGES="${TORCH_PACKAGES}"
# A +cpu local version is invisible under uv's default first-index safety once
# PyPI carries a plain `torch` at the same version; see base/cpu/Dockerfile.
ENV TORCH_INDEX_STRATEGY=unsafe-best-match
# Single-chip libtpu config for the grading process.
# No display and no render-capable accelerator; osmesa rasterizes on CPU.
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV ALLOW_MULTIPLE_LIBTPU_LOAD=1
ENV TPU_LOG_DIR=/var/log/libtpu
# Agent-writable HF cache; also the parent for deploy-time read-only weight mounts.
ENV HF_HOME=/tmp/hf-cache

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-solvers.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh base/install-solvers-heavy.sh base/install-task-deps.sh /tmp/base/
COPY base/requirements-tpu.txt base/clear_execstack.py /tmp/base/
RUN mkdir -p "${TPU_LOG_DIR}" && /tmp/base/install-common.sh

# jax[tpu] from the libtpu releases index, then clear the exec-stack flag.
RUN uv pip install --python /opt/lbx-runtime/.venv/bin/python --no-cache \
      --extra-index-url "${LIBTPU_INDEX_URL}" \
      -r /tmp/base/requirements-tpu.txt \
 && /opt/lbx-runtime/.venv/bin/python /tmp/base/clear_execstack.py

# Fail the build here rather than at task runtime if a resolve silently moved
# jax off the libtpu-matched pin, or if an ecosystem package cannot import.
RUN /opt/lbx-runtime/.venv/bin/python -c "\
import jax, jaxlib, flax, optax, chex, orbax.checkpoint, equinox, brax, mujoco.mjx, gymnax; \
assert jax.__version__ == '0.5.2', jax.__version__; \
assert jaxlib.__version__ == '0.5.1', jaxlib.__version__; \
print('jax', jax.__version__, 'jaxlib', jaxlib.__version__, '+ ecosystem OK')"

CMD ["/opt/lbx-runtime/.venv/bin/rubric", "mcp"]
